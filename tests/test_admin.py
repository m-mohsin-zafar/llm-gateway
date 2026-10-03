from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.dependencies import GatewayServices
from app.models.admin import ApiKeyMetadata, IssuedApiKey
from app.upstream import UpstreamResponse


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


class PlaygroundUpstream:
    async def is_ready(self): return True

    async def request_json(self, method, path, payload):
        assert method == "POST"
        assert path == "/v1/chat/completions"
        assert payload == {
            "model": "qwen3:4b",
            "messages": [{"role": "user", "content": "Give me one test idea."}],
            "max_tokens": 256,
        }
        return UpstreamResponse(200, {"choices": [{"message": {"content": "Test invalid input."}}]}, {})


def test_admin_playground_runs_a_bounded_openai_prompt(settings, fake_database):
    from app.main import create_app

    app = create_app(settings, GatewayServices(database=fake_database, upstream=PlaygroundUpstream(), keys=AdminKeys()))
    with TestClient(app) as client:
        response = client.post(
            "/admin/playground",
            auth=("admin", "test-password"),
            headers={"X-Requested-With": "llm-gateway-admin"},
            json={"protocol": "openai", "prompt": "Give me one test idea."},
        )

    assert response.status_code == 200
    assert response.json()["output"] == "Test invalid input."
    assert response.json()["status_code"] == 200
    assert response.json()["latency_ms"] >= 0
