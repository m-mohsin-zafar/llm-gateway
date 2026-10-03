from __future__ import annotations

import importlib.util

import pytest

ERRORS_EXIST = importlib.util.find_spec("app.errors") is not None
requires_errors = pytest.mark.skipif(not ERRORS_EXIST, reason="errors not implemented")


def test_protocol_error_helpers_exist():
    assert ERRORS_EXIST


@requires_errors
def test_openai_error_uses_standard_shape_and_request_id():
    from app.errors import openai_error

    response = openai_error(
        503,
        "Ollama unavailable",
        error_type="server_error",
        code="upstream_unavailable",
        request_id="req-123",
    )

    assert response.status_code == 503
    assert response.headers["x-request-id"] == "req-123"
    assert response.body == (
        b'{"error":{"message":"Ollama unavailable","type":"server_error",'
        b'"code":"upstream_unavailable"}}'
    )


@requires_errors
def test_ollama_error_uses_native_shape_and_retry_after():
    from app.errors import ollama_error

    response = ollama_error(
        429, "Inference queue is full", request_id="req-456", retry_after=2
    )

    assert response.status_code == 429
    assert response.headers["x-request-id"] == "req-456"
    assert response.headers["retry-after"] == "2"
    assert response.body == b'{"error":"Inference queue is full"}'

@requires_errors
def test_every_response_gets_a_gateway_request_id(client):
    generated = client.get("/health")
    preserved = client.get("/health", headers={"X-Request-ID": "req-client-123"})

    assert generated.headers["x-request-id"]
    assert preserved.headers["x-request-id"] == "req-client-123"
