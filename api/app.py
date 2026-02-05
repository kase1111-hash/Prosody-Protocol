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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_methods=["*"],
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
