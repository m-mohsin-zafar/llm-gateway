from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.auth import ApiKeyPrincipal, require_scope
from app.config import Settings
from app.errors import openai_error
from app.models.openai import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ResponseRequest,
)
from app.upstream import (
    InvalidUpstreamResponse,
    QueueFull,
    UpstreamTimeout,
    UpstreamUnavailable,
    UsageContext,
)


def create_openai_router(settings: Settings) -> APIRouter:
    router = APIRouter(tags=["openai"])

    def request_id(request: Request) -> str | None:
        return getattr(request.state, "request_id", None)

    def resolve_model(model: str) -> str | None:
        if model in {"default", settings.default_model}:
            return settings.default_model
        return None

    def prepare_payload(payload_model: Any, endpoint: str):
        payload = payload_model.model_dump(exclude_none=True)
        resolved = resolve_model(payload.get("model", "default"))
        if resolved is None:
            return None, openai_error(
                400,
                "Model is not enabled",
                code="model_not_found",
                request_id=None,
            )
        payload["model"] = resolved
        if endpoint in {"/v1/chat/completions", "/v1/completions"}:
            requested = [
                value
                for value in (
                    payload.pop("max_tokens", None),
                    payload.pop("max_completion_tokens", None),
                )
                if value is not None
            ]
            payload["max_tokens"] = min(
                min(requested) if requested else settings.max_output_tokens,
                settings.max_output_tokens,
            )
        elif endpoint == "/v1/responses":
            payload.pop("store", None)
            payload.pop("previous_response_id", None)
            payload.pop("conversation", None)
            payload["max_output_tokens"] = min(
                payload.get("max_output_tokens", settings.max_output_tokens),
                settings.max_output_tokens,
            )
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
            "protocol": "openai",
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
            return

    def failure(request: Request, error: Exception) -> JSONResponse:
        if isinstance(error, QueueFull):
            return openai_error(
                429,
                str(error),
                error_type="rate_limit_error",
                code="queue_full",
                request_id=request_id(request),
                retry_after=error.retry_after,
            )
        if isinstance(error, UpstreamTimeout):
            return openai_error(
                504,
                "Ollama request timed out",
                error_type="server_error",
                code="upstream_timeout",
                request_id=request_id(request),
            )
        if isinstance(error, UpstreamUnavailable):
            return openai_error(
                503,
                "Ollama is unavailable",
                error_type="server_error",
                code="upstream_unavailable",
                request_id=request_id(request),
            )
        if isinstance(error, InvalidUpstreamResponse):
            return openai_error(
                502,
                "Ollama returned an invalid response",
                error_type="server_error",
                code="invalid_upstream_response",
                request_id=request_id(request),
            )
        raise error

    @asynccontextmanager
    async def admission(services):
        if services.admission is None:
            yield
            return
        async with services.admission.acquire():
            yield

    def observe_usage(values: dict[str, Any], data: dict[str, Any]) -> None:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return
        values["prompt_tokens"] = int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        values["completion_tokens"] = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )

    async def buffered(
        request: Request,
        principal: ApiKeyPrincipal,
        endpoint: str,
        payload_model: Any,
    ):
        payload, error = prepare_payload(payload_model, endpoint)
        if error is not None:
            return error
        assert payload is not None
        services = request.app.state.services
        started = time.perf_counter()
        try:
            async with admission(services):
                response = await services.upstream.request_json("POST", endpoint, payload)
        except (QueueFull, UpstreamTimeout, UpstreamUnavailable, InvalidUpstreamResponse) as exc:
            return failure(request, exc)
        values = usage_values(
            request, principal, endpoint, payload_model.model, payload["model"]
        )
        if isinstance(response.data, dict):
            observe_usage(values, response.data)
        values["duration_ms"] = int((time.perf_counter() - started) * 1000)
        values["status_code"] = response.status_code
        response_data = response.data
        if response.status_code >= 400:
            upstream_error = (
                response_data.get("error")
                if isinstance(response_data, dict)
                else None
            )
            if isinstance(upstream_error, dict) and isinstance(
                upstream_error.get("message"), str
            ) and "://" not in upstream_error["message"]:
                pass
            else:
                message = (
                    upstream_error
                    if isinstance(upstream_error, str) and "://" not in upstream_error
                    else "Ollama rejected the request"
                )
                response_data = {
                    "error": {
                        "message": message,
                        "type": "invalid_request_error",
                        "code": "upstream_rejected",
                    }
                }
        return JSONResponse(
            response_data,
            status_code=response.status_code,
            headers=response.headers,
            background=BackgroundTask(record_usage, services.usage, values),
        )

    async def streaming(
        request: Request,
        principal: ApiKeyPrincipal,
        endpoint: str,
        payload_model: Any,
    ):
        payload, error = prepare_payload(payload_model, endpoint)
        if error is not None:
            return error
        assert payload is not None
        services = request.app.state.services
        values = usage_values(
            request, principal, endpoint, payload_model.model, payload["model"]
        )
        usage = UsageContext(services.usage, values) if services.usage else None
        slot = admission(services)
        try:
            await slot.__aenter__()
        except QueueFull as exc:
            return failure(request, exc)

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

            return StreamingResponse(empty_chunks(), media_type="text/event-stream")
        except (UpstreamTimeout, UpstreamUnavailable, InvalidUpstreamResponse) as exc:
            await close_and_release(sys.exc_info())
            return failure(request, exc)

        pending = bytearray()

        def inspect_chunk(chunk: bytes) -> None:
            pending.extend(chunk)
            while b"\n\n" in pending:
                event, _, remainder = pending.partition(b"\n\n")
                pending[:] = remainder
                for line in event.splitlines():
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == b"[DONE]":
                        continue
                    try:
                        parsed = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(parsed, dict):
                        observe_usage(values, parsed)

        async def chunks():
            try:
                inspect_chunk(first_chunk)
                yield first_chunk
                async for chunk in upstream_stream:
                    inspect_chunk(chunk)
                    yield chunk
            finally:
                await close_and_release(sys.exc_info())

        return StreamingResponse(chunks(), media_type="text/event-stream")

    @router.get("/v1/models")
    async def models(_: ApiKeyPrincipal = Depends(require_scope("openai"))):
        return {
            "object": "list",
            "data": [
                {
                    "id": "default",
                    "object": "model",
                    "created": 0,
                    "owned_by": "ollama",
                }
            ],
        }

    @router.get("/v1/models/{model}")
    async def model(
        model: str,
        request: Request,
        _: ApiKeyPrincipal = Depends(require_scope("openai")),
    ):
        if resolve_model(model) is None:
            return openai_error(
                404,
                "Model is not enabled",
                code="model_not_found",
                request_id=request_id(request),
            )
        return {
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": "ollama",
        }

    @router.post("/v1/chat/completions")
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("openai")),
    ):
        if payload.stream:
            return await streaming(request, principal, "/v1/chat/completions", payload)
        return await buffered(request, principal, "/v1/chat/completions", payload)

    @router.post("/v1/completions")
    async def completions(
        payload: CompletionRequest,
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("openai")),
    ):
        if payload.stream:
            return await streaming(request, principal, "/v1/completions", payload)
        return await buffered(request, principal, "/v1/completions", payload)

    @router.post("/v1/responses")
    async def responses(
        payload: ResponseRequest,
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("openai")),
    ):
        if (
            payload.store
            or payload.previous_response_id is not None
            or payload.conversation is not None
        ):
            return openai_error(
                400,
                "Stateful Responses features are not supported",
                code="unsupported_parameter",
                request_id=request_id(request),
            )
        if payload.stream:
            return await streaming(request, principal, "/v1/responses", payload)
        return await buffered(request, principal, "/v1/responses", payload)

    @router.post("/v1/embeddings")
    async def embeddings(
        payload: EmbeddingRequest,
        request: Request,
        principal: ApiKeyPrincipal = Depends(require_scope("openai")),
    ):
        return await buffered(request, principal, "/v1/embeddings", payload)

    return router
