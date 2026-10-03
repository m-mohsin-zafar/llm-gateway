from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

class FakeDatabase:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready

    async def is_ready(self) -> bool:
        return self.ready

class FakeUpstream:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready

    async def is_ready(self) -> bool:
        return self.ready

@pytest.fixture(autouse=True)
def gateway_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("BOOTSTRAP_API_KEY", "llmgw_test_bootstrap")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")

@pytest.fixture
def settings():
    from app.config import Settings

    return Settings(
        database_url="postgresql://test:test@localhost/test",
        ollama_url="http://127.0.0.1:11434",
        default_model="qwen3:4b",
        bootstrap_api_key="llmgw_test_bootstrap",
        admin_username="admin",
        admin_password="test-password",
        request_timeout_seconds=30,
        max_context_tokens=8192,
        max_output_tokens=2048,
        max_queue=8,
    )

@pytest.fixture
def fake_database():
    return FakeDatabase()

@pytest.fixture
def fake_upstream():
    return FakeUpstream()

@pytest.fixture
def app(settings, fake_database, fake_upstream):
    from app.dependencies import GatewayServices
    from app.main import create_app

    return create_app(
        settings,
        GatewayServices(database=fake_database, upstream=fake_upstream),
    )

@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client
