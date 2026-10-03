from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.auth import ApiKeyPrincipal
from app.dependencies import GatewayServices
from app.upstream import InferenceAdmission, UpstreamResponse


class NativeKeyRepository:
    async def authenticate(self, token: str):
        scopes = {
            "inference-token": frozenset({"ollama:inference"}),
            "read-token": frozenset({"ollama:read"}),
            "full-token": frozenset({"ollama:inference", "ollama:read"}),
        }.get(token)
        if scopes is None:
            return None
        return ApiKeyPrincipal(uuid.uuid4(), "test", token, scopes)


class NativeUpstream:
    def __init__(self):
        self.calls = []
        self.response = {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": "hello"},
            "prompt_eval_count": 4,
            "eval_count": 2,
            "done": True,
        }
        self.stream_chunks = [
            b'{"message":{"content":"hel"},"done":false}\n',
            b'{"message":{"content":"lo"},"done":true}\n',
        ]
        self.stream_error = None

    async def is_ready(self):
        return True

    async def request_json(self, method, path, payload=None):
        self.calls.append((method, path, payload, False))
        return UpstreamResponse(200, self.response, {"content-type": "application/json"})

    async def stream(self, method, path, payload, usage=None):
        self.calls.append((method, path, payload, True))
        if self.stream_error is not None:
            raise self.stream_error
        for chunk in self.stream_chunks:
            yield chunk


class NativeUsage:
    def __init__(self):
        self.events = []

    async def record(self, **values):
        self.events.append(values)


@pytest.fixture
def native_api(settings, fake_database):
    from app.main import create_app

    upstream = NativeUpstream()
    usage = NativeUsage()
    services = GatewayServices(
        database=fake_database,
        upstream=upstream,
        keys=NativeKeyRepository(),
        usage=usage,
        admission=InferenceAdmission(max_active=1, max_queue=1),
    )
    application = create_app(settings, services)
    with TestClient(application) as client:
        yield client, upstream, usage


def auth(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    ("method", "path", "body", "token", "upstream_path"),
    [
        ("post", "/api/chat", {"model": "default", "messages": [{"role": "user", "content": "Hi"}], "stream": False}, "inference-token", "/api/chat"),
        ("post", "/api/generate", {"model": "default", "prompt": "Hi", "stream": False}, "inference-token", "/api/generate"),
        ("post", "/api/embed", {"model": "default", "input": "Hi"}, "inference-token", "/api/embed"),
        ("get", "/api/tags", None, "read-token", "/api/tags"),
        ("post", "/api/show", {"model": "default"}, "read-token", "/api/show"),
        ("get", "/api/ps", None, "read-token", "/api/ps"),
        ("get", "/api/version", None, "read-token", "/api/version"),
    ],
)
def test_explicit_native_routes_forward_to_matching_upstream_path(
    native_api, method, path, body, token, upstream_path
):
    client, upstream, _ = native_api

    response = client.request(method, path, json=body, headers=auth(token))

    assert response.status_code == 200
    assert upstream.calls[-1][1] == upstream_path


def test_buffered_chat_maps_default_model_and_preserves_documented_fields(native_api):
    client, upstream, usage = native_api
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Find a USB-C hub"}],
        "stream": False,
        "tools": [{"type": "function", "function": {"name": "search"}}],
        "format": {"type": "object", "properties": {"answer": {"type": "string"}}},
        "think": False,
        "keep_alive": "10m",
        "logprobs": True,
        "top_logprobs": 3,
        "future_documented_field": {"enabled": True},
        "options": {"temperature": 0.2, "mirostat": 1, "num_ctx": 4096, "num_predict": 512},
    }

    response = client.post("/api/chat", json=payload, headers=auth("inference-token"))

    assert response.status_code == 200
    forwarded = upstream.calls[-1][2]
    assert forwarded["model"] == "qwen3:4b"
    for field in ("messages", "tools", "format", "think", "keep_alive", "logprobs", "top_logprobs"):
        assert forwarded[field] == payload[field]
    assert forwarded["options"] == payload["options"]
    assert forwarded["future_documented_field"] == {"enabled": True}
    assert response.json() == upstream.response
    assert usage.events[-1]["protocol"] == "ollama"
    assert usage.events[-1]["endpoint"] == "/api/chat"
    assert "messages" not in usage.events[-1]


@pytest.mark.parametrize("path,body", [
    ("/api/chat", {"model": "default", "messages": [{"role": "user", "content": "Hi"}]}),
    ("/api/generate", {"model": "default", "prompt": "Hi"}),
])
def test_native_streaming_preserves_ndjson_bytes(native_api, path, body):
    client, upstream, _ = native_api

    with client.stream("POST", path, json=body, headers=auth("inference-token")) as response:
        received = b"".join(response.iter_raw())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert received == b"".join(upstream.stream_chunks)
    assert upstream.calls[-1][3] is True


def test_native_limits_clamp_high_values_and_reject_non_positive_values(native_api):
    client, upstream, _ = native_api
    base = {"model": "default", "prompt": "Hi", "stream": False}

    clamped = client.post(
        "/api/generate",
        json={**base, "options": {"num_ctx": 99999, "num_predict": 99999}},
        headers=auth("inference-token"),
    )
    rejected = client.post(
        "/api/generate",
        json={**base, "options": {"num_predict": 0}},
        headers=auth("inference-token"),
    )

    assert clamped.status_code == 200
    assert upstream.calls[-1][2]["options"]["num_ctx"] == 8192
    assert upstream.calls[-1][2]["options"]["num_predict"] == 2048
    assert rejected.status_code == 400
    assert rejected.json() == {"error": "options.num_predict must be greater than zero"}


def test_unapproved_model_is_rejected_without_upstream_call(native_api):
    client, upstream, _ = native_api

    response = client.post(
        "/api/embed",
        json={"model": "gemma3:4b", "input": "Hi"},
        headers=auth("inference-token"),
    )

    assert response.status_code == 400
    assert response.json() == {"error": "Model is not enabled"}
    assert upstream.calls == []


def test_read_scope_cannot_invoke_inference(native_api):
    client, upstream, _ = native_api

    response = client.post(
        "/api/chat",
        json={"model": "default", "messages": [{"role": "user", "content": "Hi"}], "stream": False},
        headers=auth("read-token"),
    )

    assert response.status_code == 403
    assert response.json() == {"error": "API key lacks required scope: ollama:inference"}
    assert upstream.calls == []


def test_native_validation_errors_use_native_shape(native_api):
    client, upstream, _ = native_api

    response = client.post(
        "/api/chat",
        json={"model": "default", "messages": []},
        headers=auth("inference-token"),
    )

    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert upstream.calls == []


def test_full_inference_queue_returns_native_429_before_streaming(native_api):
    from contextlib import asynccontextmanager

    from app.upstream import QueueFull

    client, upstream, _ = native_api

    class FullAdmission:
        @asynccontextmanager
        async def acquire(self):
            raise QueueFull(retry_after=3)
            yield

    client.app.state.services.admission = FullAdmission()
    response = client.post(
        "/api/generate",
        json={"model": "default", "prompt": "Hi", "stream": True},
        headers=auth("inference-token"),
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "3"
    assert response.json() == {"error": "Inference queue is full"}
    assert upstream.calls == []


def test_stream_transport_failure_returns_native_error_before_headers(native_api):
    from app.upstream import UpstreamUnavailable

    client, upstream, _ = native_api
    upstream.stream_error = UpstreamUnavailable("private upstream detail")

    response = client.post(
        "/api/chat",
        json={"model": "default", "messages": [{"role": "user", "content": "Hi"}]},
        headers=auth("inference-token"),
    )

    assert response.status_code == 502
    assert response.json() == {"error": "Ollama is unavailable"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/pull"),
        ("post", "/api/create"),
        ("post", "/api/copy"),
        ("delete", "/api/delete"),
        ("post", "/api/push"),
        ("post", "/api/arbitrary"),
    ],
)
def test_model_management_and_arbitrary_native_paths_are_not_exposed(native_api, method, path):
    client, upstream, _ = native_api

    response = client.request(method, path, json={}, headers=auth("full-token"))

    assert response.status_code == 404
    assert upstream.calls == []
