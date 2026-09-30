import asyncio
import hashlib
import hmac
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

DATABASE_URL = os.environ["DATABASE_URL"]
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "qwen3:4b")
BOOTSTRAP_API_KEY = os.environ["BOOTSTRAP_API_KEY"]
ADMIN_USERNAME = os.environ["ADMIN_USERNAME"]
ADMIN_PASSWORD = os.environ["ADMIN_PASSWORD"]
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "180"))
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "8192"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "1024"))

security = HTTPBasic()


def hash_key(value: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(value.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return f"{salt.hex()}:{digest.hex()}"


def verify_key(value: str, encoded: str) -> bool:
    try:
        salt_hex, digest_hex = encoded.split(":", 1)
        calculated = hash_key(value, bytes.fromhex(salt_hex)).split(":", 1)[1]
        return hmac.compare_digest(calculated, digest_hex)
    except (ValueError, AttributeError):
        return False


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4)
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS))
    await initialize_database(app.state.pool)
    yield
    await app.state.http.aclose()
    await app.state.pool.close()


app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False


async def api_key(x_api_key: str | None = Header(default=None)) -> asyncpg.Record:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key")
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, name, key_hash FROM api_keys WHERE enabled = TRUE")
        for row in rows:
            if verify_key(x_api_key, row["key_hash"]):
                await conn.execute("UPDATE api_keys SET last_used_at = now() WHERE id = $1", row["id"])
                return row
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    valid_user = hmac.compare_digest(credentials.username.encode(), ADMIN_USERNAME.encode())
    valid_password = hmac.compare_digest(credentials.password.encode(), ADMIN_PASSWORD.encode())
    if not (valid_user and valid_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials", headers={"WWW-Authenticate": "Basic"})
    return credentials.username


def resolve_model(name: str) -> str:
    if name not in {"default", DEFAULT_MODEL}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only the default model is currently enabled")
    return DEFAULT_MODEL


async def log_usage(key_id: uuid.UUID | None, request_id: uuid.UUID, prompt: int, completion: int, duration_ms: int, status_code: int) -> None:
    async with app.state.pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO usage_events (id, api_key_id, request_id, model_alias, upstream_model, prompt_tokens, completion_tokens, duration_ms, status_code)
               VALUES ($1, $2, $3, 'default', $4, $5, $6, $7, $8)""",
            uuid.uuid4(), key_id, request_id, DEFAULT_MODEL, prompt, completion, duration_ms, status_code,
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models(_: asyncpg.Record = Depends(api_key)) -> dict[str, Any]:
    return {"object": "list", "data": [{"id": "default", "object": "model", "owned_by": "llm-gateway"}]}


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest, _: Request, key: asyncpg.Record = Depends(api_key)) -> dict[str, Any]:
    if payload.stream:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Streaming is not enabled in the initial deployment")
    model = resolve_model(payload.model)
    request_id = uuid.uuid4()
    started = time.perf_counter()
    options: dict[str, Any] = {"num_ctx": MAX_CONTEXT_TOKENS, "num_predict": min(payload.max_tokens or MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)}
    if payload.temperature is not None:
        options["temperature"] = payload.temperature
    try:
        response = await app.state.http.post(f"{OLLAMA_URL}/api/chat", json={"model": model, "messages": [message.model_dump() for message in payload.messages], "stream": False, "think": False, "options": options})
        response.raise_for_status()
        upstream = response.json()
    except httpx.HTTPError as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        await log_usage(key["id"], request_id, 0, 0, duration_ms, 502)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Ollama inference failed") from exc
    duration_ms = int((time.perf_counter() - started) * 1000)
    prompt_tokens = int(upstream.get("prompt_eval_count", 0))
    completion_tokens = int(upstream.get("eval_count", 0))
    await log_usage(key["id"], request_id, prompt_tokens, completion_tokens, duration_ms, 200)
    return {
        "id": f"chatcmpl-{request_id}", "object": "chat.completion", "created": int(time.time()), "model": "default",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": upstream["message"]["content"]}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


@app.get("/admin", response_class=HTMLResponse)
async def dashboard(_: str = Depends(admin)) -> str:
    async with app.state.pool.acquire() as conn:
        totals = await conn.fetchrow("SELECT count(*) AS requests, coalesce(sum(prompt_tokens),0) AS prompt, coalesce(sum(completion_tokens),0) AS completion, coalesce(round(avg(duration_ms)),0) AS latency FROM usage_events WHERE created_at >= now() - interval '24 hours'")
        keys = await conn.fetch("SELECT k.name, count(e.id) AS requests, coalesce(sum(e.prompt_tokens + e.completion_tokens),0) AS tokens, max(e.created_at) AS last_used FROM api_keys k LEFT JOIN usage_events e ON e.api_key_id = k.id GROUP BY k.id ORDER BY last_used DESC NULLS LAST")
    rows = ''.join(f"<tr><td>{row['name']}</td><td>{row['requests']}</td><td>{row['tokens']}</td><td>{row['last_used'] or 'never'}</td></tr>" for row in keys)
    return f"""<!doctype html><html><head><title>llm-gateway usage</title><style>body{{font-family:system-ui;max-width:900px;margin:40px auto}}.cards{{display:flex;gap:16px}}.card{{border:1px solid #ddd;padding:16px;border-radius:8px;min-width:140px}}table{{width:100%;border-collapse:collapse;margin-top:24px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}</style></head><body><h1>llm-gateway</h1><p>Last 24 hours</p><div class=cards><div class=card>Requests<br><b>{totals['requests']}</b></div><div class=card>Tokens<br><b>{totals['prompt'] + totals['completion']}</b></div><div class=card>Avg latency<br><b>{totals['latency']} ms</b></div></div><h2>API keys</h2><table><tr><th>Name</th><th>Requests</th><th>Tokens</th><th>Last used</th></tr>{rows}</table></body></html>"""
