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

from prosody_protocol import __version__

# FastAPI import deferred -- requires the [api] extra.
# This module will fail at import time without it, which is intentional:
# you should only run the API server with the api extra installed.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
