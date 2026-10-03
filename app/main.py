import asyncio
import hashlib
import hmac
import secrets
import uuid
from contextlib import asynccontextmanager

import asyncpg
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import Settings
from app.database import ApiKeyRepository, PostgresKeyStore, PostgresUsageStore, UsageRepository, migrate_schema
from app.dependencies import DatabaseProbe, GatewayServices
from app.errors import ollama_error, openai_error
from app.routers.ollama import create_ollama_router
from app.routers.openai import create_openai_router
from app.upstream import InferenceAdmission, UpstreamClient

SETTINGS = Settings.from_env()
DEFAULT_MODEL = SETTINGS.default_model
BOOTSTRAP_API_KEY = SETTINGS.bootstrap_api_key
ADMIN_USERNAME = SETTINGS.admin_username
ADMIN_PASSWORD = SETTINGS.admin_password

security = HTTPBasic()


def hash_key(value: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(value.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return f"{salt.hex()}:{digest.hex()}"


async def initialize_database(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
              id UUID PRIMARY KEY,
              name TEXT NOT NULL UNIQUE,
              key_hash TEXT NOT NULL,
              enabled BOOLEAN NOT NULL DEFAULT TRUE,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              last_used_at TIMESTAMPTZ
            );
            CREATE TABLE IF NOT EXISTS usage_events (
              id UUID PRIMARY KEY,
              api_key_id UUID REFERENCES api_keys(id),
              request_id UUID NOT NULL,
              model_alias TEXT NOT NULL,
              upstream_model TEXT NOT NULL,
              prompt_tokens INTEGER NOT NULL DEFAULT 0,
              completion_tokens INTEGER NOT NULL DEFAULT 0,
              duration_ms INTEGER NOT NULL,
              status_code INTEGER NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS usage_events_created_at_idx ON usage_events(created_at DESC);
            CREATE INDEX IF NOT EXISTS usage_events_api_key_id_idx ON usage_events(api_key_id);
        """)
        existing = await conn.fetchval("SELECT id FROM api_keys WHERE name = 'bootstrap'")
        if not existing:
            await conn.execute(
                "INSERT INTO api_keys (id, name, key_hash) VALUES ($1, $2, $3)",
                uuid.uuid4(), "bootstrap", hash_key(BOOTSTRAP_API_KEY),
            )


def create_app(settings: Settings, services: GatewayServices | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if services is not None:
            application.state.services = services
            yield
            return

        pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
        http = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_seconds))
        application.state.pool = pool
        application.state.http = http
        await initialize_database(pool)
        async with pool.acquire() as connection:
            await migrate_schema(connection)
        upstream = UpstreamClient(http, settings.ollama_url)
        application.state.services = GatewayServices(
            database=DatabaseProbe(pool),
            upstream=upstream,
            keys=ApiKeyRepository(PostgresKeyStore(pool)),
            usage=UsageRepository(PostgresUsageStore(pool)),
            admission=InferenceAdmission(max_active=1, max_queue=settings.max_queue),
        )
        try:
            yield
        finally:
            await http.aclose()
            await pool.close()

    application = FastAPI(
        title="LLM Gateway",
        version="0.2.0",
        description="Authenticated OpenAI-compatible and Ollama-native inference gateway.",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "system", "description": "Health and readiness endpoints."},
            {"name": "openai", "description": "OpenAI-compatible API."},
            {"name": "ollama", "description": "Ollama-native read and inference API."},
            {"name": "admin", "description": "Gateway administration."},
        ],
    )

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(HTTPException)
    async def protocol_http_exception(request: Request, exc: HTTPException):
        if request.url.path.startswith("/api/"):
            response = ollama_error(
                exc.status_code,
                str(exc.detail),
                request_id=getattr(request.state, "request_id", None),
            )
            if exc.headers:
                response.headers.update(exc.headers)
            return response
        if request.url.path.startswith("/v1/"):
            response = openai_error(
                exc.status_code,
                str(exc.detail),
                error_type=(
                    "authentication_error"
                    if exc.status_code == 401
                    else "permission_error"
                    if exc.status_code == 403
                    else "invalid_request_error"
                ),
                code="authentication_error" if exc.status_code == 401 else None,
                request_id=getattr(request.state, "request_id", None),
            )
            if exc.headers:
                response.headers.update(exc.headers)
            return response
        return await http_exception_handler(request, exc)

    @application.exception_handler(RequestValidationError)
    async def protocol_validation_exception(
        request: Request, exc: RequestValidationError
    ):
        if request.url.path.startswith("/api/"):
            return ollama_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "Invalid request",
                request_id=getattr(request.state, "request_id", None),
            )
        if request.url.path.startswith("/v1/"):
            return openai_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "Invalid request",
                code="invalid_request",
                request_id=getattr(request.state, "request_id", None),
            )
        return await request_validation_exception_handler(request, exc)

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready", tags=["system"])
    async def ready(request: Request):
        active_services: GatewayServices = request.app.state.services
        database_ready, ollama_ready = await asyncio.gather(
            active_services.database.is_ready(),
            active_services.upstream.is_ready(),
        )
        payload = {
            "status": "ready" if database_ready and ollama_ready else "not_ready",
            "database": "ok" if database_ready else "unavailable",
            "ollama": "ok" if ollama_ready else "unavailable",
        }
        if not (database_ready and ollama_ready):
            from fastapi.responses import JSONResponse

            return JSONResponse(payload, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return payload

    application.include_router(create_ollama_router(settings))
    application.include_router(create_openai_router(settings))

    return application


app = create_app(SETTINGS)


def admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    valid_user = hmac.compare_digest(credentials.username.encode(), ADMIN_USERNAME.encode())
    valid_password = hmac.compare_digest(credentials.password.encode(), ADMIN_PASSWORD.encode())
    if not (valid_user and valid_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

@app.get("/admin", response_class=HTMLResponse)
async def dashboard(_: str = Depends(admin)) -> str:
    async with app.state.pool.acquire() as conn:
        totals = await conn.fetchrow("SELECT count(*) AS requests, coalesce(sum(prompt_tokens),0) AS prompt, coalesce(sum(completion_tokens),0) AS completion, coalesce(round(avg(duration_ms)),0) AS latency FROM usage_events WHERE created_at >= now() - interval '24 hours'")
        keys = await conn.fetch("SELECT k.name, k.enabled, count(e.id) FILTER (WHERE e.created_at >= now() - interval '24 hours') AS requests, coalesce(sum(e.prompt_tokens) FILTER (WHERE e.created_at >= now() - interval '24 hours'),0) AS prompt_tokens, coalesce(sum(e.completion_tokens) FILTER (WHERE e.created_at >= now() - interval '24 hours'),0) AS completion_tokens, count(e.id) FILTER (WHERE e.created_at >= now() - interval '24 hours' AND e.status_code >= 400) AS failures, coalesce(round(avg(e.duration_ms) FILTER (WHERE e.created_at >= now() - interval '24 hours')),0) AS latency, max(e.created_at) AS last_used FROM api_keys k LEFT JOIN usage_events e ON e.api_key_id = k.id GROUP BY k.id ORDER BY last_used DESC NULLS LAST")
    rows = ''.join(f"<tr class='text-slate-300'><td class='px-5 py-4 font-medium text-white'>{row['name']}</td><td class='px-5 py-4'><span class='rounded-full px-2 py-1 text-xs font-medium {'bg-emerald-400/10 text-emerald-300' if row['enabled'] else 'bg-slate-800 text-slate-400'}'>{'Active' if row['enabled'] else 'Disabled'}</span></td><td class='px-5 py-4'>{row['requests']:,}</td><td class='px-5 py-4'>{row['prompt_tokens']:,} / {row['completion_tokens']:,}</td><td class='px-5 py-4'>{row['failures']:,}</td><td class='px-5 py-4'>{row['latency']:,} ms</td><td class='px-5 py-4 text-slate-400'>{row['last_used'].strftime('%Y-%m-%d %H:%M') if row['last_used'] else 'Never'}</td></tr>" for row in keys)
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>llm-gateway · Usage</title><script src='https://cdn.tailwindcss.com'></script></head><body class='min-h-screen bg-slate-950 text-slate-100'><main class='mx-auto max-w-6xl px-6 py-10'><header class='mb-10 flex flex-col gap-5 border-b border-slate-800 pb-8 sm:flex-row sm:items-end sm:justify-between'><div><div class='mb-3 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-emerald-300'><span class='h-2 w-2 rounded-full bg-emerald-400'></span>Gateway online</div><h1 class='text-3xl font-semibold tracking-tight text-white'>llm-gateway</h1><p class='mt-2 text-slate-400'>Private inference usage and API-key activity.</p></div><div class='rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-sm text-slate-400'><span class='font-medium text-slate-200'>Current model</span><br>default → {DEFAULT_MODEL}</div></header><section class='grid gap-4 sm:grid-cols-2 lg:grid-cols-3'><article class='rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-sm'><p class='text-sm font-medium text-slate-400'>Requests</p><p class='mt-3 text-3xl font-semibold text-white'>{totals['requests']:,}</p><p class='mt-2 text-xs text-slate-500'>Last 24 hours</p></article><article class='rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-sm'><p class='text-sm font-medium text-slate-400'>Tokens processed</p><p class='mt-3 text-3xl font-semibold text-white'>{totals['prompt'] + totals['completion']:,}</p><p class='mt-2 text-xs text-slate-500'>{totals['prompt']:,} prompt · {totals['completion']:,} completion</p></article><article class='rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-sm'><p class='text-sm font-medium text-slate-400'>Average latency</p><p class='mt-3 text-3xl font-semibold text-white'>{totals['latency']:,}<span class='ml-1 text-base font-medium text-slate-400'>ms</span></p><p class='mt-2 text-xs text-slate-500'>Completed requests, last 24 hours</p></article></section><section class='mt-8 overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 shadow-sm'><div class='flex items-center justify-between border-b border-slate-800 px-5 py-4'><div><h2 class='font-semibold text-white'>API key usage</h2><p class='mt-1 text-sm text-slate-400'>Usage is recorded without storing request content.</p></div><div class='flex items-center gap-3'><button type='button' onclick='window.location.reload()' class='rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-slate-500 hover:bg-slate-700'>Refresh</button><span class='rounded-full bg-slate-800 px-3 py-1 text-xs font-medium text-slate-300'>{len(keys)} active key{'s' if len(keys) != 1 else ''}</span></div></div><div class='overflow-x-auto'><table class='w-full min-w-[950px] text-left text-sm'><thead class='border-b border-slate-800 bg-slate-900 text-xs uppercase tracking-wider text-slate-500'><tr><th class='px-5 py-3 font-medium'>Key</th><th class='px-5 py-3 font-medium'>Status</th><th class='px-5 py-3 font-medium'>Requests</th><th class='px-5 py-3 font-medium'>Input / output</th><th class='px-5 py-3 font-medium'>Failures</th><th class='px-5 py-3 font-medium'>Avg latency</th><th class='px-5 py-3 font-medium'>Last used</th></tr></thead><tbody class='divide-y divide-slate-800'>{rows}</tbody></table></div></section><p class='mt-6 text-center text-xs text-slate-600'>llm.ultracodes.io · refreshed on page load</p></main></body></html>"""
