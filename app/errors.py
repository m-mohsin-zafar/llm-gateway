from __future__ import annotations

from fastapi.responses import JSONResponse


def _headers(request_id: str | None, retry_after: int | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-ID"] = request_id
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return headers


def openai_error(
    status_code: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    request_id: str | None = None,
    retry_after: int | None = None,
) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "code": code}},
        status_code=status_code,
        headers=_headers(request_id, retry_after),
    )


def ollama_error(
    status_code: int,
    message: str,
    *,
    request_id: str | None = None,
    retry_after: int | None = None,
) -> JSONResponse:
    return JSONResponse(
        {"error": message},
        status_code=status_code,
        headers=_headers(request_id, retry_after),
    )
