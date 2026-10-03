from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ApiKeyMetadata(BaseModel):
    id: uuid.UUID
    public_id: str | None
    name: str
    scopes: frozenset[str]
    enabled: bool
    created_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    rotated_from_id: uuid.UUID | None = None
    last_used_at: datetime | None = None


class IssuedApiKey(BaseModel):
    secret: str = Field(repr=False)
    metadata: ApiKeyMetadata


class CreateApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: set[str] = Field(min_length=1)
    expires_at: datetime | None = None


class UpdateApiKeyRequest(BaseModel):
    enabled: bool


class PlaygroundRequest(BaseModel):
    protocol: Literal["openai", "ollama"] = "openai"
    prompt: str = Field(min_length=1, max_length=12000)
    think: bool = False
    max_tokens: int = Field(default=256, ge=1, le=512)
