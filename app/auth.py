from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth")
api_key_scheme = APIKeyHeader(
    name="X-API-Key", auto_error=False, scheme_name="ApiKeyAuth"
)


@dataclass(frozen=True, slots=True)
class ApiKeyPrincipal:
    id: uuid.UUID
    public_id: str | None
    name: str
    scopes: frozenset[str]


class KeyAuthenticator(Protocol):
    async def authenticate(self, token: str) -> ApiKeyPrincipal | None: ...


def unauthorized(detail: str = "Invalid API key") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def parse_api_key(authorization: str | None, x_api_key: str | None) -> str:
    bearer: str | None = None
    if authorization:
        scheme, separator, credentials = authorization.partition(" ")
        if scheme.lower() != "bearer" or not separator or not credentials.strip():
            raise unauthorized("Malformed Authorization header")
        bearer = credentials.strip()
    legacy = x_api_key.strip() if x_api_key and x_api_key.strip() else None
    if bearer and legacy and bearer != legacy:
        raise unauthorized("Conflicting API key headers")
    token = bearer or legacy
    if not token:
        raise unauthorized("Missing API key")
    return token


def require_scope(scope: str):
    async def dependency(
        request: Request,
        bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
        legacy: str | None = Security(api_key_scheme),
    ) -> ApiKeyPrincipal:
        authorization = (
            f"{bearer.scheme} {bearer.credentials}" if bearer is not None else None
        )
        token = parse_api_key(authorization, legacy)
        repository: KeyAuthenticator | None = getattr(
            request.app.state.services, "keys", None
        )
        if repository is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable",
            )
        principal = await repository.authenticate(token)
        if principal is None:
            raise unauthorized()
        if scope not in principal.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key lacks required scope: {scope}",
            )
        return principal

    return dependency
