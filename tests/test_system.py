from __future__ import annotations
import importlib.util

import pytest
from pydantic import ValidationError

FOUNDATION_EXISTS = importlib.util.find_spec("app.config") is not None
requires_foundation = pytest.mark.skipif(
    not FOUNDATION_EXISTS,
    reason="gateway application foundation is not implemented",
)


def test_gateway_application_foundation_exists():
    assert FOUNDATION_EXISTS


@requires_foundation
def test_health_reports_process_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
@requires_foundation
def test_api_documentation_is_enabled(client, path):
    response = client.get(path)

    assert response.status_code == 200


@requires_foundation
def test_readiness_reports_database_and_ollama_separately(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "database": "ok",
        "ollama": "ok",
    }


@requires_foundation
def test_readiness_is_unavailable_when_one_dependency_is_down(
    settings, fake_database, fake_upstream
):
    from fastapi.testclient import TestClient

    from app.dependencies import GatewayServices
    from app.main import create_app

    fake_upstream.ready = False
    app = create_app(
        settings,
        GatewayServices(database=fake_database, upstream=fake_upstream),
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "database": "ok",
        "ollama": "unavailable",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_timeout_seconds", 0),
        ("max_context_tokens", 0),
        ("max_output_tokens", -1),
        ("max_queue", -1),
    ],
)
@requires_foundation
def test_settings_reject_non_positive_resource_limits(settings, field, value):
    from app.config import Settings

    values = settings.model_dump()
    values[field] = value

    with pytest.raises(ValidationError):
        Settings(**values)
