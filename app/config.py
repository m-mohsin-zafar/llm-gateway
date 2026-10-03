from __future__ import annotations

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    database_url: str
    ollama_url: str = "http://127.0.0.1:11434"
    default_model: str = "qwen3:4b"
    bootstrap_api_key: str
    admin_username: str
    admin_password: str
    request_timeout_seconds: float = Field(default=180, gt=0)
    max_context_tokens: int = Field(default=8192, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)
    max_queue: int = Field(default=8, ge=0)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.environ["DATABASE_URL"],
            ollama_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            default_model=os.environ.get("DEFAULT_MODEL", "qwen3:4b"),
            bootstrap_api_key=os.environ["BOOTSTRAP_API_KEY"],
            admin_username=os.environ["ADMIN_USERNAME"],
            admin_password=os.environ["ADMIN_PASSWORD"],
            request_timeout_seconds=float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "180")),
            max_context_tokens=int(os.environ.get("MAX_CONTEXT_TOKENS", "8192")),
            max_output_tokens=int(os.environ.get("MAX_OUTPUT_TOKENS", "2048")),
            max_queue=int(os.environ.get("OLLAMA_MAX_QUEUE", "8")),
        )
