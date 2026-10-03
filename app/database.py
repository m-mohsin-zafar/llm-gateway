from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol

import asyncpg

from app.auth import ApiKeyPrincipal
from app.models.admin import ApiKeyMetadata, IssuedApiKey

ALL_SCOPES = frozenset({"openai", "ollama:inference", "ollama:read"})


def hash_secret(value: str, salt: bytes | None = None) -> str:
    actual_salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        value.encode(), salt=actual_salt, n=16384, r=8, p=1, dklen=32
    )
    return f"{actual_salt.hex()}:{digest.hex()}"


def verify_secret(value: str, encoded: str) -> bool:
    try:
        salt_hex, digest_hex = encoded.split(":", 1)
        calculated = hash_secret(value, bytes.fromhex(salt_hex)).split(":", 1)[1]
        return hmac.compare_digest(calculated, digest_hex)
    except (ValueError, AttributeError):
        return False


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    id: uuid.UUID
    public_id: str | None
    name: str
    secret_hash: str
    scopes: frozenset[str]
    enabled: bool
    created_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    rotated_from_id: uuid.UUID | None = None
    last_used_at: datetime | None = None

    @classmethod
    def legacy(cls, name: str, secret_hash: str) -> "ApiKeyRecord":
        return cls(
            id=uuid.uuid4(),
            public_id=None,
            name=name,
            secret_hash=secret_hash,
            scopes=ALL_SCOPES,
            enabled=True,
            created_at=datetime.now(timezone.utc),
        )

    def metadata(self) -> ApiKeyMetadata:
        return ApiKeyMetadata(
            id=self.id,
            public_id=self.public_id,
            name=self.name,
            scopes=self.scopes,
            enabled=self.enabled,
            created_at=self.created_at,
            expires_at=self.expires_at,
            revoked_at=self.revoked_at,
            rotated_from_id=self.rotated_from_id,
            last_used_at=self.last_used_at,
        )


class KeyStore(Protocol):
    async def insert_key(self, record: ApiKeyRecord) -> None: ...
    async def get_key(self, public_id: str) -> ApiKeyRecord | None: ...
    async def list_legacy_keys(self) -> list[ApiKeyRecord]: ...
    async def list_keys(self) -> list[ApiKeyRecord]: ...
    async def update_key(self, record: ApiKeyRecord) -> None: ...


class ApiKeyRepository:
    def __init__(self, store: KeyStore) -> None:
        self.store = store

    async def create(
        self,
        name: str,
        scopes: set[str] | frozenset[str],
        expires_at: datetime | None = None,
        rotated_from_id: uuid.UUID | None = None,
    ) -> IssuedApiKey:
        normalized_scopes = frozenset(scopes)
        if not normalized_scopes or not normalized_scopes <= ALL_SCOPES:
            raise ValueError("Invalid API key scopes")
        public_id = secrets.token_hex(6)
        raw_secret = secrets.token_urlsafe(32)
        token = f"llmgw_{public_id}_{raw_secret}"
        record = ApiKeyRecord(
            id=uuid.uuid4(),
            public_id=public_id,
            name=name,
            secret_hash=hash_secret(raw_secret),
            scopes=normalized_scopes,
            enabled=True,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            rotated_from_id=rotated_from_id,
        )
        await self.store.insert_key(record)
        return IssuedApiKey(secret=token, metadata=record.metadata())

    async def authenticate(self, token: str) -> ApiKeyPrincipal | None:
        now = datetime.now(timezone.utc)
        record: ApiKeyRecord | None = None
        secret = token
        if token.startswith("llmgw_"):
            parts = token.split("_", 2)
            if len(parts) == 3:
                _, public_id, secret = parts
                record = await self.store.get_key(public_id)
                candidates = [record] if record else []
            else:
                candidates = await self.store.list_legacy_keys()
        else:
            candidates = await self.store.list_legacy_keys()

        for candidate in candidates:
            if candidate is None or not self._active(candidate, now):
                continue
            if verify_secret(secret if candidate.public_id else token, candidate.secret_hash):
                updated = replace(candidate, last_used_at=now)
                await self.store.update_key(updated)
                return ApiKeyPrincipal(
                    id=candidate.id,
                    public_id=candidate.public_id,
                    name=candidate.name,
                    scopes=candidate.scopes,
                )
        return None

    async def list(self) -> list[ApiKeyMetadata]:
        return [record.metadata() for record in await self.store.list_keys()]

    async def rotate(self, public_id: str) -> IssuedApiKey:
        current = await self._required(public_id)
        return await self.create(
            f"{current.name}-rotation-{secrets.token_hex(3)}",
            current.scopes,
            current.expires_at,
            rotated_from_id=current.id,
        )

    async def set_enabled(self, public_id: str, enabled: bool) -> ApiKeyMetadata:
        current = await self._required(public_id)
        updated = replace(current, enabled=enabled)
        await self.store.update_key(updated)
        return updated.metadata()

    async def revoke(self, public_id: str) -> ApiKeyMetadata:
        current = await self._required(public_id)
        updated = replace(current, enabled=False, revoked_at=datetime.now(timezone.utc))
        await self.store.update_key(updated)
        return updated.metadata()

    async def _required(self, public_id: str) -> ApiKeyRecord:
        record = await self.store.get_key(public_id)
        if record is None:
            raise KeyError(public_id)
        return record

    @staticmethod
    def _active(record: ApiKeyRecord, now: datetime) -> bool:
        return (
            record.enabled
            and record.revoked_at is None
            and (record.expires_at is None or record.expires_at > now)
        )


class PostgresKeyStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def insert_key(self, record: ApiKeyRecord) -> None:
        await self.pool.execute(
            """INSERT INTO api_keys
               (id, public_id, name, key_hash, scopes, enabled, created_at,
                expires_at, revoked_at, rotated_from_id, last_used_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
            record.id,
            record.public_id,
            record.name,
            record.secret_hash,
            list(record.scopes),
            record.enabled,
            record.created_at,
            record.expires_at,
            record.revoked_at,
            record.rotated_from_id,
            record.last_used_at,
        )

    async def get_key(self, public_id: str) -> ApiKeyRecord | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM api_keys WHERE public_id = $1", public_id
        )
        return self._record(row) if row else None

    async def list_legacy_keys(self) -> list[ApiKeyRecord]:
        rows = await self.pool.fetch(
            "SELECT * FROM api_keys WHERE public_id IS NULL AND enabled = TRUE"
        )
        return [self._record(row) for row in rows]

    async def list_keys(self) -> list[ApiKeyRecord]:
        rows = await self.pool.fetch("SELECT * FROM api_keys ORDER BY created_at DESC")
        return [self._record(row) for row in rows]

    async def update_key(self, record: ApiKeyRecord) -> None:
        await self.pool.execute(
            """UPDATE api_keys SET scopes=$2, enabled=$3, expires_at=$4,
               revoked_at=$5, last_used_at=$6 WHERE id=$1""",
            record.id,
            list(record.scopes),
            record.enabled,
            record.expires_at,
            record.revoked_at,
            record.last_used_at,
        )

    @staticmethod
    def _record(row) -> ApiKeyRecord:
        return ApiKeyRecord(
            id=row["id"],
            public_id=row["public_id"],
            name=row["name"],
            secret_hash=row["key_hash"],
            scopes=frozenset(row["scopes"] or ALL_SCOPES),
            enabled=row["enabled"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            revoked_at=row["revoked_at"],
            rotated_from_id=row["rotated_from_id"],
            last_used_at=row["last_used_at"],
        )


@dataclass(frozen=True, slots=True)
class UsageEvent:
    api_key_id: uuid.UUID | None
    request_id: uuid.UUID | str
    protocol: str
    endpoint: str
    model_alias: str
    upstream_model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: int
    status_code: int


class UsageStore(Protocol):
    async def insert_usage(self, event: UsageEvent) -> None: ...


class UsageRepository:
    def __init__(self, store: UsageStore) -> None:
        self.store = store

    async def record(self, **values) -> None:
        await self.store.insert_usage(UsageEvent(**values))


class PostgresUsageStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def insert_usage(self, event: UsageEvent) -> None:
        await self.pool.execute(
            """INSERT INTO usage_events
               (id, api_key_id, request_id, protocol, endpoint, model_alias,
                upstream_model, prompt_tokens, completion_tokens, duration_ms,
                status_code)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
            uuid.uuid4(), event.api_key_id, event.request_id, event.protocol,
            event.endpoint, event.model_alias, event.upstream_model,
            event.prompt_tokens, event.completion_tokens, event.duration_ms,
            event.status_code,
        )


async def migrate_schema(connection) -> None:
    await connection.execute(
        """
        ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS public_id TEXT;
        ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scopes TEXT[] NOT NULL
            DEFAULT ARRAY['openai','ollama:inference','ollama:read']::TEXT[];
        ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;
        ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ;
        ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rotated_from_id UUID
            REFERENCES api_keys(id);
        CREATE UNIQUE INDEX IF NOT EXISTS api_keys_public_id_idx
            ON api_keys(public_id) WHERE public_id IS NOT NULL;
        ALTER TABLE usage_events ADD COLUMN IF NOT EXISTS protocol TEXT NOT NULL
            DEFAULT 'openai';
        ALTER TABLE usage_events ADD COLUMN IF NOT EXISTS endpoint TEXT NOT NULL
            DEFAULT '/v1/chat/completions';
        """
    )
