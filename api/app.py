"""FastAPI application for the Prosody Protocol REST API.

Endpoints defined in spec:
  POST /v1/convert/audio-to-iml
  POST /v1/synthesize
  POST /v1/validate
  POST /v1/convert/text-to-iml
  POST /v1/convert/iml-to-ssml
  GET  /v1/health
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from prosody_protocol import __version__
from prosody_protocol.exceptions import (
    ConversionError,
    IMLParseError,
    IMLValidationError,
    ProfileError,
    ProsodyProtocolError,
)

from .config import Settings
from .routes import convert, synthesize, validate

settings = Settings()

app = FastAPI(
    title="Prosody Protocol API",
    description="REST API for the Intent Markup Language (IML) SDK.",
    version=__version__,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class UploadSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds the configured maximum."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "payload_too_large",
                    "detail": (
                        f"Request body ({int(content_length)} bytes) exceeds "
                        f"maximum allowed size ({self.max_bytes} bytes)."
                    ),
                },
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter based on client IP.

    For production use behind a reverse proxy, prefer nginx/Traefik rate
    limiting instead. This middleware is a safety net for direct exposure.
    """

    def __init__(self, app: ASGIApp, requests_per_minute: int) -> None:
        super().__init__(app)
        self.rpm = requests_per_minute
        self._window: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if self.rpm <= 0:
            return await call_next(request)

        import time

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._window.setdefault(client_ip, [])

        # Remove entries older than 60 seconds
        cutoff = now - 60
        window[:] = [t for t in window if t > cutoff]

        if len(window) >= self.rpm:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": f"Rate limit exceeded ({self.rpm} requests/minute).",
                },
            )

        window.append(now)
        return await call_next(request)


app.add_middleware(UploadSizeLimitMiddleware, max_bytes=settings.max_upload_bytes)

if settings.rate_limit_per_minute > 0:
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)

# CORS: only allow configured origins. Empty list → no cross-origin access.
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

app.include_router(convert.router, prefix="/v1/convert", tags=["convert"])
app.include_router(synthesize.router, prefix="/v1", tags=["synthesize"])
app.include_router(validate.router, prefix="/v1", tags=["validate"])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@app.exception_handler(IMLParseError)
async def iml_parse_error_handler(request: Request, exc: IMLParseError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "iml_parse_error", "detail": str(exc)},
    )


@app.exception_handler(ConversionError)
async def conversion_error_handler(request: Request, exc: ConversionError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "conversion_error", "detail": str(exc)},
    )


@app.exception_handler(IMLValidationError)
async def iml_validation_error_handler(
    request: Request, exc: IMLValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "validation_error", "detail": str(exc)},
    )


@app.exception_handler(ProfileError)
async def profile_error_handler(request: Request, exc: ProfileError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "profile_error", "detail": str(exc)},
    )


@app.exception_handler(ProsodyProtocolError)
async def prosody_protocol_error_handler(
    request: Request, exc: ProsodyProtocolError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "prosody_protocol_error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
