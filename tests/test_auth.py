from __future__ import annotations

import importlib.util
import uuid

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

AUTH_EXISTS = importlib.util.find_spec("app.auth") is not None
requires_auth = pytest.mark.skipif(not AUTH_EXISTS, reason="scoped auth is not implemented")


def test_scoped_authentication_module_exists():
    assert AUTH_EXISTS


@requires_auth
@pytest.mark.parametrize(
    ("authorization", "x_api_key", "expected"),
    [
        ("Bearer llmgw_abc_secret", None, "llmgw_abc_secret"),
        (None, "llmgw_abc_secret", "llmgw_abc_secret"),
        ("Bearer llmgw_abc_secret", "llmgw_abc_secret", "llmgw_abc_secret"),
    ],
)
def test_parse_api_key_accepts_standard_and_legacy_headers(
    authorization, x_api_key, expected
):
    from app.auth import parse_api_key

    assert parse_api_key(authorization, x_api_key) == expected


@requires_auth
@pytest.mark.parametrize(
    ("authorization", "x_api_key"),
    [
        (None, None),
        ("Basic abc", None),
        ("Bearer", None),
        ("Bearer first", "second"),
    ],
)
def test_parse_api_key_rejects_missing_malformed_or_conflicting_headers(
    authorization, x_api_key
):
    from fastapi import HTTPException

    from app.auth import parse_api_key

    with pytest.raises(HTTPException) as error:
        parse_api_key(authorization, x_api_key)

    assert error.value.status_code == 401


class StubKeyRepository:
    def __init__(self, principal=None):
        self.principal = principal

    async def authenticate(self, token: str):
        return self.principal if token == "valid-token" else None


@requires_auth
def test_require_scope_authenticates_bearer_and_enforces_scope(
    settings, fake_database, fake_upstream
):
    from app.auth import ApiKeyPrincipal, require_scope
    from app.dependencies import GatewayServices
    from app.main import create_app

    principal = ApiKeyPrincipal(
        id=uuid.uuid4(),
        public_id="client123",
        name="test-client",
        scopes=frozenset({"openai"}),
    )
    services = GatewayServices(
        database=fake_database,
        upstream=fake_upstream,
        keys=StubKeyRepository(principal),
    )
    app = create_app(settings, services)

    @app.get("/protected")
    async def protected(key=Depends(require_scope("openai"))):
        return {"name": key.name}

    with TestClient(app) as client:
        allowed = client.get(
            "/protected", headers={"Authorization": "Bearer valid-token"}
        )
        denied = client.get(
            "/protected", headers={"Authorization": "Bearer invalid-token"}
        )

    assert allowed.status_code == 200
    assert allowed.json() == {"name": "test-client"}
    assert denied.status_code == 401


@requires_auth
def test_require_scope_rejects_authenticated_key_without_scope(
    settings, fake_database, fake_upstream
):
    from app.auth import ApiKeyPrincipal, require_scope
    from app.dependencies import GatewayServices
    from app.main import create_app

    principal = ApiKeyPrincipal(
        id=uuid.uuid4(),
        public_id="reader123",
        name="reader",
        scopes=frozenset({"ollama:read"}),
    )
    app = create_app(
        settings,
        GatewayServices(
            database=fake_database,
            upstream=fake_upstream,
            keys=StubKeyRepository(principal),
        ),
    )

    @app.get("/protected")
    async def protected(_=Depends(require_scope("openai"))):
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get(
            "/protected", headers={"Authorization": "Bearer valid-token"}
        )

    assert response.status_code == 403


@requires_auth
def test_openapi_declares_bearer_and_legacy_api_key_schemes(
    settings, fake_database, fake_upstream
):
    from app.auth import require_scope
    from app.dependencies import GatewayServices
    from app.main import create_app

    app = create_app(
        settings,
        GatewayServices(database=fake_database, upstream=fake_upstream),
    )

    @app.get("/protected")
    async def protected(_=Depends(require_scope("openai"))):
        return {"ok": True}

    schema = app.openapi()
    schemes = schema["components"]["securitySchemes"]

    assert schemes["BearerAuth"] == {"type": "http", "scheme": "bearer"}
    assert schemes["ApiKeyAuth"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
    }
