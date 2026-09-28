"""Error responses of the REST API.

Errors the API reports itself are JSON objects ``{"error": <code>,
"detail": <message>}``; IML validation failures add the offending
``issues``. Request bodies that do not match the endpoint's schema are
FastAPI's standard 422 response (``{"detail": [...]}``), as are unusable
``words`` or ``transcript`` fields of audio-to-iml (``loc`` names the field).
A rejected ``NaN`` or ``Infinity`` in a JSON body (which Python's JSON
parser accepts) is echoed as the string ``"nan"`` or ``"inf"``, since JSON
cannot hold it.

========  ============================  ==========================================
Status    ``error``                     Cause
========  ============================  ==========================================
400       ``iml_parse_error``           The IML is not well-formed XML.
400       ``validation_error``          The IML breaks a spec rule (``issues``).
400       ``conversion_error``          The IML cannot be converted or
                                        synthesized (e.g. an unknown voice, or
                                        audio longer than PP_MAX_SYNTH_SECONDS).
400       ``audio_processing_error``    The upload cannot be read or analysed,
                                        or is longer than PP_MAX_AUDIO_SECONDS.
400       ``profile_error``             The ``profile`` of audio-to-iml is not
                                        valid JSON or not a valid prosody profile.
413       ``payload_too_large``         The body exceeds PP_MAX_UPLOAD_MB.
413       ``text_too_large``            A text field exceeds PP_MAX_TEXT_CHARS, or
                                        the ``words`` of audio-to-iml exceed
                                        PP_MAX_WORDS_CHARS.
415       ``unsupported_media_type``    audio-to-iml was not sent as
                                        multipart/form-data.
429       ``rate_limited``              Over PP_RATE_LIMIT (see ``Retry-After``).
500       ``internal_error``            A bug; the details are in the server log.
503       ``server_busy``               Audio conversion or synthesis: every
                                        worker is busy and PP_MAX_QUEUED_JOBS
                                        requests are waiting (``Retry-After``).
========  ============================  ==========================================
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from prosody_protocol.exceptions import (
    AudioProcessingError,
    ConversionError,
    IMLParseError,
    IMLValidationError,
    ProfileError,
    ProsodyProtocolError,
)
from prosody_protocol.validator import ValidationIssue


class ValidationIssueResponse(BaseModel):
    """One validator finding. ``column`` is only known for XML syntax errors (V1)."""

    severity: str
    rule: str
    message: str
    line: int | None = None
    column: int | None = None


def issue_fields(issue: ValidationIssue) -> dict[str, Any]:
    """The :class:`ValidationIssueResponse` fields of a validator issue."""
    return {
        "severity": issue.severity,
        "rule": issue.rule,
        "message": issue.message,
        "line": issue.line,
        "column": issue.column,
    }


class ErrorResponse(BaseModel):
    """Body of every error response the API produces itself."""

    error: str = Field(description="Machine-readable error code, e.g. `validation_error`.")
    detail: str = Field(description="Human-readable explanation.")
    issues: list[ValidationIssueResponse] | None = Field(
        default=None,
        description="For `validation_error`: the spec rules the IML breaks.",
    )


class APIError(Exception):
    """An error the API reports directly, with its own status and ``error`` code."""

    def __init__(
        self,
        status_code: int,
        error: str,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error = error
        self.detail = detail
        self.headers = headers


def error_response(
    status_code: int,
    error: str,
    detail: str,
    *,
    headers: dict[str, str] | None = None,
    issues: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    """A JSON response with an :class:`ErrorResponse` body."""
    content: dict[str, Any] = {"error": error, "detail": detail}
    if issues is not None:
        content["issues"] = issues
    return JSONResponse(status_code=status_code, content=content, headers=headers)


#: OpenAPI ``responses`` shared by the POST endpoints.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "model": ErrorResponse,
        "description": "The input cannot be processed (see `error` and `detail`).",
    },
    413: {
        "model": ErrorResponse,
        "description": "The request body or a text field is larger than the server allows.",
    },
    429: {
        "model": ErrorResponse,
        "description": "Rate limit exceeded; retry after `Retry-After` seconds.",
        "headers": {
            "Retry-After": {
                "description": "Seconds until the client may send another request.",
                "schema": {"type": "integer"},
            }
        },
    },
}

#: OpenAPI ``responses`` of the endpoints that run in a worker process.
BUSY_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {
        "model": ErrorResponse,
        "description": (
            "Every worker is busy and the queue is full; retry after `Retry-After` seconds."
        ),
        "headers": {
            "Retry-After": {
                "description": "Seconds to wait before retrying.",
                "schema": {"type": "integer"},
            }
        },
    },
}

# Checked in order: the first matching class decides the ``error`` code.
_SDK_ERRORS: tuple[tuple[type[ProsodyProtocolError], str], ...] = (
    (IMLParseError, "iml_parse_error"),
    (ConversionError, "conversion_error"),
    (AudioProcessingError, "audio_processing_error"),
    (ProfileError, "profile_error"),
    (ProsodyProtocolError, "prosody_protocol_error"),
)


def _json_safe(value: Any) -> Any:
    """*value* with the non-finite floats JSON cannot hold as strings (``"nan"``)."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers that turn exceptions into :class:`ErrorResponse` bodies."""

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI's own handler, except that a rejected NaN or Infinity (which
        # the error echoes as its input) no longer makes the response a 500.
        return JSONResponse(
            status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))}
        )

    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError) -> JSONResponse:
        return error_response(exc.status_code, exc.error, exc.detail, headers=exc.headers)

    @app.exception_handler(IMLValidationError)
    async def validation_error(request: Request, exc: IMLValidationError) -> JSONResponse:
        issues = [issue_fields(issue) for issue in exc.issues]
        return error_response(400, "validation_error", str(exc), issues=issues)

    async def sdk_error(request: Request, exc: ProsodyProtocolError) -> JSONResponse:
        code = next(code for cls, code in _SDK_ERRORS if isinstance(exc, cls))
        return error_response(400, code, str(exc))

    for cls, _code in _SDK_ERRORS:
        app.exception_handler(cls)(sdk_error)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        # Starlette re-raises after sending this, so the traceback is logged.
        return error_response(
            500, "internal_error", "Internal server error. The details are in the server log."
        )
