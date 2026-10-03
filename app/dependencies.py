from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import asyncpg
import httpx


class ReadinessProbe(Protocol):
    async def is_ready(self) -> bool: ...


@dataclass(slots=True)
class GatewayServices:
    database: ReadinessProbe
    upstream: ReadinessProbe
    keys: Any | None = None
    usage: Any | None = None


class DatabaseProbe:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def is_ready(self) -> bool:
        try:
            async with self.pool.acquire() as connection:
                return await connection.fetchval("SELECT 1") == 1
        except (asyncpg.PostgresError, OSError):
            return False


class OllamaProbe:
    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self.client = client
        self.base_url = base_url

    async def is_ready(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/api/version")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
