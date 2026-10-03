from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from app.config import Settings
from app.models.admin import CreateApiKeyRequest, UpdateApiKeyRequest


def create_admin_router(settings: Settings) -> APIRouter:
    router = APIRouter(tags=["admin"], include_in_schema=False)
    templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
    basic = HTTPBasic()

    def authenticate(credentials: HTTPBasicCredentials = Depends(basic)) -> None:
        if not (hmac.compare_digest(credentials.username, settings.admin_username) and hmac.compare_digest(credentials.password, settings.admin_password)):
            raise HTTPException(401, "Invalid admin credentials", headers={"WWW-Authenticate": "Basic"})

    def csrf(x_requested_with: str | None = Header(default=None)) -> None:
        if x_requested_with != "llm-gateway-admin":
            raise HTTPException(403, "Missing admin CSRF header")

    def keys(request: Request):
        return request.app.state.services.keys

    @router.get("/admin", response_class=HTMLResponse)
    async def dashboard(request: Request, _: None = Depends(authenticate)):
        overview = {"requests": 0, "tokens": 0, "latency": 0, "errors": 0}
        usage_by_key = {}
        pool = getattr(request.app.state, "pool", None)
        if pool is not None:
            async with pool.acquire() as connection:
                totals = await connection.fetchrow("SELECT count(*) requests, coalesce(sum(prompt_tokens + completion_tokens), 0) tokens, coalesce(round(avg(duration_ms)), 0) latency, count(*) filter (where status_code >= 400) errors FROM usage_events WHERE created_at >= now() - interval '24 hours'")
                overview = dict(totals)
                rows = await connection.fetch("SELECT api_key_id, count(*) requests, coalesce(sum(prompt_tokens), 0) prompt, coalesce(sum(completion_tokens), 0) completion, count(*) filter (where status_code >= 400) errors, coalesce(round(avg(duration_ms)), 0) latency FROM usage_events WHERE created_at >= now() - interval '24 hours' GROUP BY api_key_id")
                usage_by_key = {row["api_key_id"]: dict(row) for row in rows}
        return templates.TemplateResponse(request, "admin.html", {"settings": settings, "keys": await keys(request).list(), "overview": overview, "usage_by_key": usage_by_key})

    @router.get("/admin/api-keys")
    async def list_keys(request: Request, _: None = Depends(authenticate)):
        return [key.model_dump(mode="json") for key in await keys(request).list()]

    @router.post("/admin/api-keys", status_code=201)
    async def create(payload: CreateApiKeyRequest, request: Request, _: None = Depends(authenticate), __: None = Depends(csrf)):
        return (await keys(request).create(payload.name, payload.scopes, payload.expires_at)).model_dump(mode="json")

    @router.post("/admin/api-keys/{public_id}/rotate", status_code=201)
    async def rotate(public_id: str, request: Request, _: None = Depends(authenticate), __: None = Depends(csrf)):
        return (await keys(request).rotate(public_id)).model_dump(mode="json")

    @router.patch("/admin/api-keys/{public_id}")
    async def update(public_id: str, payload: UpdateApiKeyRequest, request: Request, _: None = Depends(authenticate), __: None = Depends(csrf)):
        return (await keys(request).set_enabled(public_id, payload.enabled)).model_dump(mode="json")

    @router.post("/admin/api-keys/{public_id}/revoke")
    async def revoke(public_id: str, request: Request, _: None = Depends(authenticate), __: None = Depends(csrf)):
        return (await keys(request).revoke(public_id)).model_dump(mode="json")

    return router
