from __future__ import annotations

import asyncio
import importlib.util

import httpx
import pytest

UPSTREAM_EXISTS = importlib.util.find_spec("app.upstream") is not None
requires_upstream = pytest.mark.skipif(
    not UPSTREAM_EXISTS, reason="upstream transport is not implemented"
)


def test_upstream_transport_module_exists():
    assert UPSTREAM_EXISTS


@requires_upstream
@pytest.mark.asyncio
async def test_request_json_returns_filtered_headers_and_parsed_body():
    from app.upstream import UpstreamClient

    async def handler(request: httpx.Request):
        assert request.url == "http://127.0.0.1:11434/api/version"
        return httpx.Response(
            200,
            json={"version": "0.35.0"},
            headers={
                "Content-Type": "application/json",
                "X-Ollama-Internal": "http://127.0.0.1:11434/private",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    upstream = UpstreamClient(client, "http://127.0.0.1:11434")

    response = await upstream.request_json("GET", "/api/version")

    assert response.status_code == 200
    assert response.data == {"version": "0.35.0"}
    assert response.headers == {"content-type": "application/json"}
    await client.aclose()


@requires_upstream
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "expected_type"),
    [
        (httpx.ConnectError("failed"), "UpstreamUnavailable"),
        (httpx.ReadTimeout("slow"), "UpstreamTimeout"),
    ],
)
async def test_request_json_maps_transport_failures(exception, expected_type):
    import app.upstream as module

    async def handler(_request: httpx.Request):
        raise exception

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    upstream = module.UpstreamClient(client, "http://127.0.0.1:11434")

    with pytest.raises(getattr(module, expected_type)):
        await upstream.request_json("GET", "/api/version")

    await client.aclose()


@requires_upstream
@pytest.mark.asyncio
async def test_request_json_rejects_invalid_upstream_json_without_leaking_url():
    from app.upstream import InvalidUpstreamResponse, UpstreamClient

    async def handler(_request: httpx.Request):
        return httpx.Response(200, text="http://127.0.0.1:11434/private")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    upstream = UpstreamClient(client, "http://127.0.0.1:11434")

    with pytest.raises(InvalidUpstreamResponse) as error:
        await upstream.request_json("GET", "/api/version")

    assert "127.0.0.1" not in str(error.value)
    await client.aclose()


class ClosingStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class RecordingUsage:
    def __init__(self):
        self.events = []

    async def record(self, **event):
        self.events.append(event)


@requires_upstream
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type,chunks",
    [
        ("text/event-stream", [b"data: {\"x\":1}\n\n", b"data: [DONE]\n\n"]),
        ("application/x-ndjson", [b'{"message":"a"}\n', b'{"done":true}\n']),
    ],
)
async def test_stream_preserves_sse_and_ndjson_bytes(content_type, chunks):
    from app.upstream import UpstreamClient

    stream = ClosingStream(chunks)

    async def handler(_request: httpx.Request):
        return httpx.Response(200, headers={"Content-Type": content_type}, stream=stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    upstream = UpstreamClient(client, "http://127.0.0.1:11434")

    received = [chunk async for chunk in upstream.stream("POST", "/api/chat", {})]

    assert received == chunks
    assert stream.closed is True
    await client.aclose()


@requires_upstream
@pytest.mark.asyncio
async def test_stream_usage_failure_does_not_break_response_forwarding():
    from app.upstream import UpstreamClient, UsageContext

    class FailingUsage:
        async def record(self, **_event):
            raise RuntimeError("database unavailable")

    stream = ClosingStream([b'{"done":true}\n'])

    async def handler(_request: httpx.Request):
        return httpx.Response(200, stream=stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    upstream = UpstreamClient(client, "http://127.0.0.1:11434")
    usage = UsageContext(FailingUsage(), {"request_id": "req-usage"})

    received = [chunk async for chunk in upstream.stream("POST", "/api/chat", {}, usage)]

    assert received == [b'{"done":true}\n']
    assert stream.closed is True
    await client.aclose()


@requires_upstream
@pytest.mark.asyncio
async def test_disconnected_stream_closes_upstream_releases_slot_and_records_failure():
    from app.upstream import InferenceAdmission, UpstreamClient, UsageContext

    stream = ClosingStream([b"first", b"second"])
    usage = RecordingUsage()

    async def handler(_request: httpx.Request):
        return httpx.Response(200, stream=stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    upstream = UpstreamClient(client, "http://127.0.0.1:11434")
    admission = InferenceAdmission(max_active=1, max_queue=0)
    context = UsageContext(
        repository=usage,
        values={
            "api_key_id": None,
            "request_id": "req-stream",
            "protocol": "ollama",
            "endpoint": "/api/chat",
            "model_alias": "default",
            "upstream_model": "qwen3:4b",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "duration_ms": 0,
        },
    )

    async with admission.acquire():
        response_stream = upstream.stream("POST", "/api/chat", {}, context)
        assert await response_stream.__anext__() == b"first"
        await response_stream.aclose()

    assert stream.closed is True
    assert admission.active == 0
    assert usage.events[0]["status_code"] == 499
    await client.aclose()


@requires_upstream
@pytest.mark.asyncio
async def test_inference_admission_rejects_when_active_and_queue_are_full():
    from app.upstream import InferenceAdmission, QueueFull

    admission = InferenceAdmission(max_active=1, max_queue=1)

    async with admission.acquire():
        waiter_started = asyncio.Event()

        async def wait_for_slot():
            waiter_started.set()
            async with admission.acquire():
                return True

        waiter = asyncio.create_task(wait_for_slot())
        await waiter_started.wait()
        await asyncio.sleep(0)

        with pytest.raises(QueueFull) as error:
            async with admission.acquire():
                pass

    assert error.value.retry_after == 2
    assert await waiter is True
