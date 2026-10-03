from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.auth import ApiKeyPrincipal
from app.dependencies import GatewayServices
from app.upstream import InferenceAdmission, UpstreamResponse


class OpenAIKeyRepository:
    async def authenticate(self, token: str):
        scopes = {
            "openai-token": frozenset({"openai"}),
            "native-token": frozenset({"ollama:read", "ollama:inference"}),
        }.get(token)
        if scopes is None:
            return None
        return ApiKeyPrincipal(uuid.uuid4(), "test", token, scopes)


class OpenAIUpstream:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.status_code = 200
        self.responses = {
            "/v1/chat/completions": {
                "id": "chatcmpl-upstream",
                "object": "chat.completion",
                "model": "qwen3:4b",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
            "/v1/completions": {
                "id": "cmpl-upstream",
                "object": "text_completion",
                "choices": [{"index": 0, "text": "hello", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
            "/v1/responses": {
                "id": "resp-upstream",
                "object": "response",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            },
            "/v1/embeddings": {
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        }
        self.stream_chunks = [
            b'data: {"id":"chatcmpl-upstream","choices":[{"delta":{"content":"hel"}}]}\n\n',
            b'data: {"id":"chatcmpl-upstream","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n',
            b"data: [DONE]\n\n",
        ]

    async def is_ready(self):
        return True

    async def request_json(self, method, path, payload=None):
        self.calls.append((method, path, payload, False))
        if self.failure is not None:
            raise self.failure
        return UpstreamResponse(
            self.status_code,
            self.responses[path],
            {"content-type": "application/json"},
        )

    async def stream(self, method, path, payload, usage=None):
        self.calls.append((method, path, payload, True))
        if self.failure is not None:
            raise self.failure
        try:
            for chunk in self.stream_chunks:
                yield chunk
        finally:
            if usage is not None:
                values = dict(usage.values)
                values["duration_ms"] = 1
                values["status_code"] = 200
                await usage.repository.record(**values)


class OpenAIUsage:
    def __init__(self):
        self.events = []

    async def record(self, **values):
        self.events.append(values)


@pytest.fixture
def openai_api(settings, fake_database):
    from app.main import create_app

    upstream = OpenAIUpstream()
    usage = OpenAIUsage()
    services = GatewayServices(
        database=fake_database,
        upstream=upstream,
        keys=OpenAIKeyRepository(),
        usage=usage,
        admission=InferenceAdmission(max_active=1, max_queue=1),
    )
    application = create_app(settings, services)
    with TestClient(application) as client:
        yield client, upstream, usage


def auth(token="openai-token"):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    ("path", "body", "upstream_path"),
    [
        ("/v1/chat/completions", {"model": "default", "messages": [{"role": "user", "content": "Hi"}]}, "/v1/chat/completions"),
        ("/v1/completions", {"model": "default", "prompt": "Hi"}, "/v1/completions"),
        ("/v1/responses", {"model": "default", "input": "Hi"}, "/v1/responses"),
        ("/v1/embeddings", {"model": "default", "input": "Hi"}, "/v1/embeddings"),
    ],
)
def test_openai_inference_routes_delegate_to_matching_upstream_path(
    openai_api, path, body, upstream_path
):
    client, upstream, _ = openai_api

    response = client.post(path, json=body, headers=auth())

    assert response.status_code == 200
    assert upstream.calls[-1][1] == upstream_path
    assert upstream.calls[-1][2]["model"] == "qwen3:4b"


def test_models_expose_only_the_configured_alias(openai_api):
    client, upstream, _ = openai_api

    listed = client.get("/v1/models", headers=auth())
    retrieved = client.get("/v1/models/default", headers=auth())
    real_name = client.get("/v1/models/qwen3:4b", headers=auth())
    rejected = client.get("/v1/models/gemma3:4b", headers=auth())

    assert listed.json() == {
        "object": "list",
        "data": [{"id": "default", "object": "model", "created": 0, "owned_by": "ollama"}],
    }
    assert retrieved.json()["id"] == "default"
    assert real_name.json()["id"] == "qwen3:4b"
    assert rejected.status_code == 404
    assert set(rejected.json()) == {"error"}
    assert upstream.calls == []


def test_chat_preserves_tools_structured_output_images_logprobs_and_reasoning(openai_api):
    client, upstream, usage = openai_api
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
        "logprobs": True,
        "top_logprobs": 2,
        "reasoning_effort": "low",
        "stream_options": {"include_usage": True},
        "future_documented_field": "preserved",
    }

    response = client.post("/v1/chat/completions", json=payload, headers=auth())

    assert response.status_code == 200
    forwarded = upstream.calls[-1][2]
    for field in (
        "messages", "tools", "tool_choice", "response_format", "logprobs",
        "top_logprobs", "reasoning_effort", "stream_options", "future_documented_field",
    ):
        assert forwarded[field] == payload[field]
    assert response.json() == upstream.responses["/v1/chat/completions"]
    assert usage.events[-1]["prompt_tokens"] == 5
    assert usage.events[-1]["completion_tokens"] == 2
    assert "messages" not in usage.events[-1]


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_chat_output_token_fields_converge_on_server_ceiling(openai_api, field):
    client, upstream, _ = openai_api

    response = client.post(
        "/v1/chat/completions",
        json={"model": "default", "messages": [{"role": "user", "content": "Hi"}], field: 99999},
        headers=auth(),
    )

    assert response.status_code == 200
    forwarded = upstream.calls[-1][2]
    assert forwarded["max_tokens"] == 2048
    assert "max_completion_tokens" not in forwarded


def test_responses_is_stateless_and_bounds_output_tokens(openai_api):
    client, upstream, _ = openai_api
    payload = {
        "model": "default",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
        "instructions": "Be concise",
        "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
        "tool_choice": "auto",
        "text": {"format": {"type": "json_object"}},
        "reasoning": {"effort": "low"},
        "max_output_tokens": 99999,
    }

    response = client.post("/v1/responses", json=payload, headers=auth())

    assert response.status_code == 200
    forwarded = upstream.calls[-1][2]
    assert forwarded["max_output_tokens"] == 2048
    assert "store" not in forwarded
    assert "previous_response_id" not in forwarded
    assert "conversation" not in forwarded
    for field in ("input", "instructions", "tools", "tool_choice", "text", "reasoning"):
        assert forwarded[field] == payload[field]


@pytest.mark.parametrize(
    "unsupported",
    [
        {"store": True},
        {"previous_response_id": "resp_previous"},
        {"conversation": "conv_123"},
    ],
)
def test_stateful_responses_fields_are_rejected(openai_api, unsupported):
    client, upstream, _ = openai_api

    response = client.post(
        "/v1/responses",
        json={"model": "default", "input": "Hi", **unsupported},
        headers=auth(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert upstream.calls == []


def test_chat_stream_preserves_sse_done_and_records_usage_trailer(openai_api):
    client, upstream, usage = openai_api

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers=auth(),
    ) as response:
        received = b"".join(response.iter_raw())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert received == b"".join(upstream.stream_chunks)
    assert received.endswith(b"data: [DONE]\n\n")
    assert usage.events[-1]["prompt_tokens"] == 5
    assert usage.events[-1]["completion_tokens"] == 2


def test_openai_stream_queue_and_transport_failures_are_mapped_before_sse_headers(openai_api):
    from contextlib import asynccontextmanager

    from app.upstream import QueueFull, UpstreamUnavailable

    client, upstream, _ = openai_api

    class FullAdmission:
        @asynccontextmanager
        async def acquire(self):
            raise QueueFull(retry_after=4)
            yield

    client.app.state.services.admission = FullAdmission()
    queued = client.post(
        "/v1/chat/completions",
        json={"model": "default", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        headers=auth(),
    )
    client.app.state.services.admission = InferenceAdmission(max_active=1, max_queue=1)
    upstream.failure = UpstreamUnavailable("private detail")
    unavailable = client.post(
        "/v1/chat/completions",
        json={"model": "default", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        headers=auth(),
    )

    assert queued.status_code == 429
    assert queued.headers["retry-after"] == "4"
    assert queued.json()["error"]["type"] == "rate_limit_error"
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "upstream_unavailable"


def test_nonstandard_upstream_error_is_normalized_to_openai_error_object(openai_api):
    client, upstream, _ = openai_api
    upstream.status_code = 400
    upstream.responses["/v1/embeddings"] = {"error": "unsupported input"}

    response = client.post(
        "/v1/embeddings",
        json={"model": "default", "input": "Hi"},
        headers=auth(),
    )

    assert response.status_code == 400
    assert isinstance(response.json()["error"], dict)
    assert response.json()["error"]["message"] == "unsupported input"


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/v1/models", None),
        ("get", "/v1/models/default", None),
        ("post", "/v1/chat/completions", {"model": "default", "messages": [{"role": "user", "content": "Hi"}]}),
        ("post", "/v1/completions", {"model": "default", "prompt": "Hi"}),
        ("post", "/v1/responses", {"model": "default", "input": "Hi"}),
        ("post", "/v1/embeddings", {"model": "default", "input": "Hi"}),
    ],
)
def test_non_openai_scope_cannot_call_any_v1_route(openai_api, method, path, body):
    client, upstream, _ = openai_api

    response = client.request(method, path, json=body, headers=auth("native-token"))

    assert response.status_code == 403
    assert set(response.json()) == {"error"}
    assert upstream.calls == []


def test_openai_local_auth_validation_model_and_upstream_errors_use_error_object(openai_api):
    from app.upstream import UpstreamUnavailable

    client, upstream, _ = openai_api
    missing_auth = client.get("/v1/models")
    invalid_body = client.post("/v1/chat/completions", json={}, headers=auth())
    invalid_model = client.post(
        "/v1/embeddings",
        json={"model": "gemma3:4b", "input": "Hi"},
        headers=auth(),
    )
    upstream.failure = UpstreamUnavailable("http://127.0.0.1:11434/private")
    unavailable = client.post(
        "/v1/embeddings",
        json={"model": "default", "input": "Hi"},
        headers=auth(),
    )

    for response in (missing_auth, invalid_body, invalid_model, unavailable):
        assert set(response.json()) == {"error"}
    assert missing_auth.status_code == 401
    assert missing_auth.json()["error"]["type"] == "authentication_error"
    assert invalid_body.status_code == 422
    assert invalid_model.status_code == 400
    assert unavailable.status_code == 503
    assert "127.0.0.1" not in unavailable.text


def test_openai_routes_are_registered_once_without_legacy_duplicates(openai_api):
    client, _, _ = openai_api
    paths = [
        route.path for route in client.app.routes
        if route.path in {"/v1/models", "/v1/chat/completions"}
    ]

    assert paths.count("/v1/models") == 1
    assert paths.count("/v1/chat/completions") == 1

    from app.main import app as production_app

    production_paths = [route.path for route in production_app.routes]
    assert production_paths.count("/v1/models") == 1
    assert production_paths.count("/v1/chat/completions") == 1
