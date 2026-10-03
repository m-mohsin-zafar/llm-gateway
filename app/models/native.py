from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NativeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "default"
    options: dict[str, Any] | None = None
    keep_alive: str | int | None = None


class ChatRequest(NativeRequest):
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = None
    format: str | dict[str, Any] | None = None
    think: bool | str | None = None
    stream: bool = True
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0)


class GenerateRequest(NativeRequest):
    prompt: str = ""
    suffix: str | None = None
    images: list[str] | None = None
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    raw: bool | None = None
    format: str | dict[str, Any] | None = None
    think: bool | str | None = None
    stream: bool = True


class EmbedRequest(NativeRequest):
    input: str | list[str]
    truncate: bool | None = None
    dimensions: int | None = Field(default=None, gt=0)


class ShowRequest(NativeRequest):
    verbose: bool | None = None
