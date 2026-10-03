from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx


class UpstreamError(Exception):
    """Base class for failures that are safe to map at the protocol boundary."""


class UpstreamUnavailable(UpstreamError):
    pass


class UpstreamTimeout(UpstreamError):
    pass


class InvalidUpstreamResponse(UpstreamError):
    pass


class QueueFull(UpstreamError):
    def __init__(self, retry_after: int = 2) -> None:
        super().__init__("Inference queue is full")
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class UpstreamResponse:
    status_code: int
    data: Any
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class UsageContext:
    repository: Any
    values: dict[str, Any]


class UpstreamClient:
    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")

    async def is_ready(self) -> bool:
        try:
            response = await self.request_json("GET", "/api/version")
            return response.status_code == 200
        except UpstreamError:
            return False

    async def request_json(
        self, method: str, path: str, payload: Any | None = None
    ) -> UpstreamResponse:
        try:
            response = await self.client.request(
                method,
                f"{self.base_url}{path}",
                json=payload if payload is not None else None,
            )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeout("Ollama request timed out") from exc
        except httpx.RequestError as exc:
            raise UpstreamUnavailable("Ollama is unavailable") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidUpstreamResponse("Ollama returned invalid JSON") from exc
        headers = self._safe_headers(response.headers)
        return UpstreamResponse(response.status_code, data, headers)

    async def stream(
        self,
        method: str,
        path: str,
        payload: Any,
        usage: UsageContext | None = None,
    ) -> AsyncIterator[bytes]:
        request = self.client.build_request(
            method, f"{self.base_url}{path}", json=payload
        )
        try:
            response = await self.client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeout("Ollama request timed out") from exc
        except httpx.RequestError as exc:
            raise UpstreamUnavailable("Ollama is unavailable") from exc

        complete = False
        started = time.perf_counter()
        try:
            async for chunk in response.aiter_raw():
                yield chunk
            complete = True
        finally:
            await response.aclose()
            if usage is not None:
                values = dict(usage.values)
                values["duration_ms"] = int((time.perf_counter() - started) * 1000)
                values["status_code"] = response.status_code if complete else 499
                await usage.repository.record(**values)

    @staticmethod
    def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
        allowed = {"content-type", "cache-control", "retry-after"}
        return {key: value for key, value in headers.items() if key in allowed}


class InferenceAdmission:
    def __init__(self, max_active: int = 1, max_queue: int = 8) -> None:
        if max_active < 1 or max_queue < 0:
            raise ValueError("Invalid inference admission limits")
        self.max_active = max_active
        self.max_queue = max_queue
        self.active = 0
        self.waiting = 0
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def acquire(self):
        admitted = False
        async with self._condition:
            if self.active >= self.max_active:
                if self.waiting >= self.max_queue:
                    raise QueueFull()
                self.waiting += 1
                try:
                    await self._condition.wait_for(
                        lambda: self.active < self.max_active
                    )
                finally:
                    self.waiting -= 1
            self.active += 1
            admitted = True
        try:
            yield
        finally:
            if admitted:
                async with self._condition:
                    self.active -= 1
                    self._condition.notify(1)
