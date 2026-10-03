from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OpenAIRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "default"


class ChatCompletionRequest(OpenAIRequest):
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0)
    reasoning_effort: str | None = None


class CompletionRequest(OpenAIRequest):
    prompt: Any
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)


class ResponseRequest(OpenAIRequest):
    input: Any
    instructions: str | None = None
    stream: bool = False
    max_output_tokens: int | None = Field(default=None, gt=0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    text: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    store: bool = False
    previous_response_id: str | None = None
    conversation: Any | None = None


class EmbeddingRequest(OpenAIRequest):
    input: Any
    encoding_format: str | None = None
    dimensions: int | None = Field(default=None, gt=0)
