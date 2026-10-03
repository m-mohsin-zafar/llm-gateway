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
from app.routers.admin import create_admin_router
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
    application.include_router(create_admin_router(settings))

    return application


app = create_app(SETTINGS)
