from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.dependencies import GatewayServices
from app.models.admin import ApiKeyMetadata, IssuedApiKey


class AdminKeys:
    def __init__(self):
        self.metadata = ApiKeyMetadata(id=uuid.uuid4(), public_id="public123", name="service", scopes=frozenset({"openai"}), enabled=True, created_at=datetime.now(timezone.utc))

    async def list(self): return [self.metadata]
    async def create(self, name, scopes, expires_at): return IssuedApiKey(secret="llmgw_never-render-this", metadata=self.metadata.model_copy(update={"name": name, "scopes": frozenset(scopes)}))
    async def rotate(self, public_id): return IssuedApiKey(secret="llmgw_rotate-only-once", metadata=self.metadata)
    async def set_enabled(self, public_id, enabled): self.metadata = self.metadata.model_copy(update={"enabled": enabled}); return self.metadata
    async def revoke(self, public_id): self.metadata = self.metadata.model_copy(update={"enabled": False}); return self.metadata


def test_admin_template_and_key_lifecycle(settings, fake_database, fake_upstream):
    from app.main import create_app

    app = create_app(settings, GatewayServices(database=fake_database, upstream=fake_upstream, keys=AdminKeys()))
    basic = ("admin", "test-password")
    with TestClient(app) as client:
        denied = client.get("/admin")
        page = client.get("/admin", auth=basic)
        blocked = client.post("/admin/api-keys", auth=basic, json={"name":"new", "scopes":["openai"]})
        created = client.post("/admin/api-keys", auth=basic, headers={"X-Requested-With":"llm-gateway-admin"}, json={"name":"new", "scopes":["openai"]})
        listed = client.get("/admin/api-keys", auth=basic)
    assert denied.status_code == 401
    assert blocked.status_code == 403
    assert created.status_code == 201 and "secret" in created.json()
    assert "llmgw_never-render-this" not in page.text and "secret" not in listed.json()[0]
    for label in ("Overview", "API Keys", "Integration Guide", "Reference", "Pydantic AI", "LangChain", "LangGraph", "think: false"):
        assert label in page.text
    assert 'id="create-key"' in page.text
    assert 'id="key-form"' in page.text
    assert 'id="requests-total"' in page.text
