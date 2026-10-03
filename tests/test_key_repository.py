from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone

import pytest

DATABASE_EXISTS = importlib.util.find_spec("app.database") is not None
requires_database = pytest.mark.skipif(
    not DATABASE_EXISTS, reason="key repository is not implemented"
)


def test_key_repository_module_exists():
    assert DATABASE_EXISTS


class MemoryKeyStore:
    def __init__(self):
        self.records = {}
        self.legacy = []
        self.schema_sql = ""

    async def insert_key(self, record):
        self.records[record.public_id] = record

    async def get_key(self, public_id):
        return self.records.get(public_id)

    async def list_legacy_keys(self):
        return list(self.legacy)

    async def list_keys(self):
        return list(self.records.values()) + list(self.legacy)

    async def update_key(self, record):
        if record.public_id:
            self.records[record.public_id] = record


@requires_database
@pytest.mark.asyncio
async def test_create_returns_secret_once_and_stores_only_hash():
    from app.database import ApiKeyRepository

    store = MemoryKeyStore()
    repository = ApiKeyRepository(store)

    issued = await repository.create("marketplace", {"openai", "ollama:read"})
    stored = store.records[issued.metadata.public_id]

    assert issued.secret.startswith(f"llmgw_{issued.metadata.public_id}_")
    assert issued.secret not in repr(stored)
    assert stored.secret_hash
    assert stored.secret_hash != issued.secret
    assert issued.metadata.scopes == frozenset({"openai", "ollama:read"})


@requires_database
@pytest.mark.asyncio
async def test_authenticate_uses_public_id_and_rejects_bad_disabled_revoked_expired_keys():
    from app.database import ApiKeyRepository

    store = MemoryKeyStore()
    repository = ApiKeyRepository(store)
    issued = await repository.create("client", {"openai"})

    assert (await repository.authenticate(issued.secret)).name == "client"
    assert await repository.authenticate(issued.secret + "bad") is None

    await repository.set_enabled(issued.metadata.public_id, False)
    assert await repository.authenticate(issued.secret) is None

    await repository.set_enabled(issued.metadata.public_id, True)
    await repository.revoke(issued.metadata.public_id)
    assert await repository.authenticate(issued.secret) is None

    expired = await repository.create(
        "expired",
        {"openai"},
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert await repository.authenticate(expired.secret) is None


@requires_database
@pytest.mark.asyncio
async def test_rotate_creates_distinct_key_and_revocation_is_immediate():
    from app.database import ApiKeyRepository

    store = MemoryKeyStore()
    repository = ApiKeyRepository(store)
    original = await repository.create("sales", {"openai"})
    replacement = await repository.rotate(original.metadata.public_id)

    assert replacement.secret != original.secret
    assert replacement.metadata.rotated_from_id == original.metadata.id
    assert await repository.authenticate(original.secret) is not None
    assert await repository.authenticate(replacement.secret) is not None

    await repository.revoke(original.metadata.public_id)
    assert await repository.authenticate(original.secret) is None
    assert await repository.authenticate(replacement.secret) is not None


@requires_database
@pytest.mark.asyncio
async def test_legacy_bootstrap_hash_remains_authenticatable():
    from app.database import ApiKeyRecord, ApiKeyRepository, hash_secret

    store = MemoryKeyStore()
    store.legacy.append(
        ApiKeyRecord.legacy(
            name="bootstrap",
            secret_hash=hash_secret("existing-bootstrap-secret"),
        )
    )
    repository = ApiKeyRepository(store)

    principal = await repository.authenticate("existing-bootstrap-secret")

    assert principal is not None
    assert principal.name == "bootstrap"
    assert principal.scopes == frozenset(
        {"openai", "ollama:inference", "ollama:read"}
    )


@requires_database
@pytest.mark.asyncio
async def test_legacy_bootstrap_with_llmgw_prefix_remains_authenticatable():
    from app.database import ApiKeyRecord, ApiKeyRepository, hash_secret

    legacy_token = "llmgw_3c036368181f5ec90b2b7cf76779a2ce"
    store = MemoryKeyStore()
    store.legacy.append(
        ApiKeyRecord.legacy(
            name="bootstrap",
            secret_hash=hash_secret(legacy_token),
        )
    )
    repository = ApiKeyRepository(store)

    principal = await repository.authenticate(legacy_token)

    assert principal is not None
    assert principal.name == "bootstrap"


@requires_database
@pytest.mark.asyncio
async def test_additive_migration_preserves_existing_rows_and_usage_history():
    from app.database import migrate_schema

    class RecordingConnection:
        def __init__(self):
            self.sql = []

        async def execute(self, statement):
            self.sql.append(statement)

    connection = RecordingConnection()

    await migrate_schema(connection)

    sql = "\n".join(connection.sql).upper()
    assert "ADD COLUMN IF NOT EXISTS PUBLIC_ID" in sql
    assert "ADD COLUMN IF NOT EXISTS SCOPES" in sql
    assert "ADD COLUMN IF NOT EXISTS PROTOCOL" in sql
    assert "ADD COLUMN IF NOT EXISTS ENDPOINT" in sql
    assert "ALTER COLUMN REQUEST_ID TYPE TEXT" in sql
    assert "DROP TABLE" not in sql
    assert "TRUNCATE" not in sql
    assert "DELETE FROM API_KEYS" not in sql
    assert "DELETE FROM USAGE_EVENTS" not in sql

@requires_database
@pytest.mark.asyncio
async def test_usage_repository_records_metadata_without_request_content():
    from app.database import UsageRepository

    class UsageStore:
        def __init__(self):
            self.events = []

        async def insert_usage(self, event):
            self.events.append(event)

    store = UsageStore()
    repository = UsageRepository(store)

    await repository.record(
        api_key_id=None,
        request_id="req-123",
        protocol="openai",
        endpoint="/v1/chat/completions",
        model_alias="default",
        upstream_model="qwen3:4b",
        prompt_tokens=12,
        completion_tokens=8,
        duration_ms=250,
        status_code=200,
    )

    event = store.events[0]
    assert event.protocol == "openai"
    assert event.endpoint == "/v1/chat/completions"
    assert event.prompt_tokens == 12
    assert event.completion_tokens == 8
    assert not hasattr(event, "prompt")
    assert not hasattr(event, "response")
