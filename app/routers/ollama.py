from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.auth import ApiKeyPrincipal, require_scope
from app.config import Settings
from app.errors import ollama_error
from app.models.native import ChatRequest, EmbedRequest, GenerateRequest, ShowRequest
from app.upstream import (
    InvalidUpstreamResponse,
    QueueFull,
    UpstreamTimeout,
    UpstreamUnavailable,
    UsageContext,
)


def create_ollama_router(settings: Settings) -> APIRouter:
    router = APIRouter(tags=["ollama"])

    def request_id(request: Request) -> str | None:
        return getattr(request.state, "request_id", None)

    def resolve_model(model: str) -> str | None:
        if model in {"default", settings.default_model}:
            return settings.default_model
        return None

    def prepare_payload(model: Any) -> tuple[dict[str, Any] | None, JSONResponse | None]:
        payload = model.model_dump(exclude_none=True)
        resolved = resolve_model(payload.get("model", "default"))
        if resolved is None:
            return None, ollama_error(400, "Model is not enabled")
        payload["model"] = resolved
        options = dict(payload.get("options") or {})
        for name, ceiling in (
            ("num_ctx", settings.max_context_tokens),
            ("num_predict", settings.max_output_tokens),
        ):
            if name not in options:
                continue
            value = options[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return None, ollama_error(
                    400, f"options.{name} must be greater than zero"
                )
            options[name] = min(value, ceiling)
        if options:
            payload["options"] = options
        return payload, None

    def usage_values(
        request: Request,
        principal: ApiKeyPrincipal,
        endpoint: str,
        model_alias: str,
        upstream_model: str,
    ) -> dict[str, Any]:
        return {
            "api_key_id": principal.id,
            "request_id": request_id(request) or "unknown",
            "protocol": "ollama",
            "endpoint": endpoint,
            "model_alias": model_alias,
            "upstream_model": upstream_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "duration_ms": 0,
            "status_code": 0,
        }

    async def record_usage(repository: Any, values: dict[str, Any]) -> None:
        if repository is None:
            return
        try:
            await repository.record(**values)
        except Exception:
            # Observability must never corrupt the protocol response.
            return

    def upstream_failure(request: Request, error: Exception) -> JSONResponse:
        if isinstance(error, QueueFull):
            return ollama_error(
                429,
                str(error),
                request_id=request_id(request),
                retry_after=error.retry_after,
            )
        if isinstance(error, UpstreamTimeout):
            return ollama_error(
                504, "Ollama request timed out", request_id=request_id(request)
            )
        if isinstance(error, (UpstreamUnavailable, InvalidUpstreamResponse)):
            return ollama_error(
                502, "Ollama is unavailable", request_id=request_id(request)
            )
        raise error

    @asynccontextmanager
    async def admission(services):
        if services.admission is None:
            yield
            return
        async with services.admission.acquire():
            yield

    async def buffered_inference(
        request: Request,
        principal: ApiKeyPrincipal,
        endpoint: str,
        payload_model: Any,
    ):
        payload, error = prepare_payload(payload_model)
        if error is not None:
            error.headers["X-Request-ID"] = request_id(request) or ""
            return error
        assert payload is not None
        services = request.app.state.services
        started = time.perf_counter()
        try:
            async with admission(services):
                response = await services.upstream.request_json("POST", endpoint, payload)
        except (QueueFull, UpstreamTimeout, UpstreamUnavailable, InvalidUpstreamResponse) as exc:
            return upstream_failure(request, exc)
        values = usage_values(
            request,
            principal,
            endpoint,
            payload_model.model,
            payload["model"],
        )
        if isinstance(response.data, dict):
            values["prompt_tokens"] = int(response.data.get("prompt_eval_count", 0))
            values["completion_tokens"] = int(response.data.get("eval_count", 0))
        values["duration_ms"] = int((time.perf_counter() - started) * 1000)
        values["status_code"] = response.status_code
        return JSONResponse(
            response.data,
            status_code=response.status_code,
            headers=response.headers,
            background=BackgroundTask(record_usage, services.usage, values),
        )

    async def streaming_inference(
        request: Request,
        principal: ApiKeyPrincipal,
        endpoint: str,
        payload_model: Any,
    ):
        payload, error = prepare_payload(payload_model)
        if error is not None:
            error.headers["X-Request-ID"] = request_id(request) or ""
            return error
        assert payload is not None
        services = request.app.state.services
        values = usage_values(
            request,
            principal,
            endpoint,
            payload_model.model,
            payload["model"],
        )
        usage = UsageContext(services.usage, values) if services.usage else None

        slot = admission(services)
        try:
            await slot.__aenter__()
        except QueueFull as exc:
            return upstream_failure(request, exc)

        upstream_stream = services.upstream.stream("POST", endpoint, payload, usage)

        async def close_and_release(exc_info=(None, None, None)):
            try:
                close = getattr(upstream_stream, "aclose", None)
                if close is not None:
                    await close()
            finally:
                await slot.__aexit__(*exc_info)

        try:
            first_chunk = await upstream_stream.__anext__()
        except StopAsyncIteration:
            await close_and_release()

            async def empty_chunks():
                if False:
                    yield b""

            return StreamingResponse(
                empty_chunks(), media_type="application/x-ndjson"
            )
        except (UpstreamTimeout, UpstreamUnavailable, InvalidUpstreamResponse) as exc:
            await close_and_release(sys.exc_info())
            return upstream_failure(request, exc)

        async def chunks():
            try:
                yield first_chunk
                async for chunk in upstream_stream:
                    yield chunk
            finally:
                await close_and_release(sys.exc_info())

        return StreamingResponse(chunks(), media_type="application/x-ndjson")

    async def read_call(
        request: Request,
        principal: ApiKeyPrincipal,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        model_alias: str = "default",
    ):
        services = request.app.state.services
        started = time.perf_counter()
        try:
            response = await services.upstream.request_json(method, endpoint, payload)
        except (UpstreamTimeout, UpstreamUnavailable, InvalidUpstreamResponse) as exc:
            return upstream_failure(request, exc)
        values = usage_values(
            request, principal, endpoint, model_alias, settings.default_model
        )
        values["duration_ms"] = int((time.perf_counter() - started) * 1000)
        values["status_code"] = response.status_code
        return JSONResponse(
            response.data,
            status_code=response.status_code,
            headers=response.headers,
            background=BackgroundTask(record_usage, services.usage, values),
        )

    @router.post("/api/chat")
    async def chat(
        payload: ChatRequest,
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("ollama:inference")),
    ):
        if payload.stream:
            return await streaming_inference(request, principal, "/api/chat", payload)
        return await buffered_inference(request, principal, "/api/chat", payload)

    @router.post("/api/generate")
    async def generate(
        payload: GenerateRequest,
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("ollama:inference")),
    ):
        if payload.stream:
            return await streaming_inference(request, principal, "/api/generate", payload)
        return await buffered_inference(request, principal, "/api/generate", payload)

    @router.post("/api/embed")
    async def embed(
        payload: EmbedRequest,
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("ollama:inference")),
    ):
        return await buffered_inference(request, principal, "/api/embed", payload)

    @router.get("/api/tags")
    async def tags(
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("ollama:read")),
    ):
        return await read_call(request, principal, "GET", "/api/tags")

    @router.post("/api/show")
    async def show(
        payload: ShowRequest,
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("ollama:read")),
    ):
        prepared, error = prepare_payload(payload)
        if error is not None:
            return error
        return await read_call(
            request, principal, "POST", "/api/show", prepared, payload.model
        )

    @router.get("/api/ps")
    async def ps(
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("ollama:read")),
    ):
        return await read_call(request, principal, "GET", "/api/ps")

    @router.get("/api/version")
    async def version(
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("ollama:read")),
    ):
        return await read_call(request, principal, "GET", "/api/version")

    return router
