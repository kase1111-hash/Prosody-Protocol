"""FastAPI application for the Prosody Protocol REST API.

Endpoints:
  POST /v1/convert/audio-to-iml
  POST /v1/convert/text-to-iml
  POST /v1/convert/iml-to-ssml
  POST /v1/convert/iml-to-prompt
  POST /v1/synthesize
  POST /v1/validate
  GET  /v1/health

``app`` is configured from the environment (see
:class:`~prosody_protocol.server.config.Settings`); :func:`create_app` builds
an application from explicit settings.
"""

from __future__ import annotations

import functools
import importlib.util
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prosody_protocol import __version__

from .config import Settings
from .deps import SettingsDep
from .errors import install_error_handlers
from .jobs import JobRunner
from .middleware import RateLimitMiddleware, UploadSizeLimitMiddleware
from .routes import convert, synthesize, validate

__all__ = [
    "RateLimitMiddleware",
    "UploadSizeLimitMiddleware",
    "app",
    "create_app",
    "settings",
]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class Capabilities(BaseModel):
    """Optional backends present on this server."""

    whisper: bool = Field(
        description="Speech recognition for audio-to-iml; without it the text is placeholders."
    )
    espeak_ng: bool = Field(
        description='Speech synthesis; without it /v1/synthesize renders "tones" only.'
    )
    ffmpeg: bool = Field(description="Decoding of OGG/Opus, WebM, M4A and similar uploads.")


class Limits(BaseModel):
    """Request limits this server enforces."""

    max_upload_bytes: int = Field(description="Largest request body (audio-to-iml uploads).")
    max_json_bytes: int = Field(
        description="Largest body of any other request (the JSON endpoints)."
    )
    max_text_chars: int
    max_words_chars: int
    max_synth_seconds: float
    max_audio_seconds: float
    rate_limit_per_minute: int = Field(description="0 means unlimited.")


class HealthResponse(BaseModel):
    status: str
    version: str
    capabilities: Capabilities
    limits: Limits


@functools.lru_cache(maxsize=1)
def _capabilities() -> Capabilities:
    # Detected once: find_spec locates whisper without importing PyTorch.
    return Capabilities(
        whisper=importlib.util.find_spec("whisper") is not None,
        espeak_ng=shutil.which("espeak-ng") is not None,
        ffmpeg=shutil.which("ffmpeg") is not None,
    )


health_router = APIRouter()


@health_router.get("/v1/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Liveness check, with the server's optional backends and limits.

    Not rate limited.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
        capabilities=_capabilities(),
        limits=Limits(
            max_upload_bytes=settings.max_upload_bytes,
            max_json_bytes=settings.json_body_limit,
            max_text_chars=settings.max_text_chars,
            max_words_chars=settings.max_words_chars,
            max_synth_seconds=settings.max_synth_seconds,
            max_audio_seconds=settings.max_audio_seconds,
            rate_limit_per_minute=settings.rate_limit_per_minute,
        ),
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    yield
    jobs: JobRunner = application.state.jobs
    await run_in_threadpool(jobs.shutdown)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application; *settings* default to the environment."""
    settings = settings if settings is not None else Settings()
    application = FastAPI(
        title="Prosody Protocol API",
        description="REST API for the Intent Markup Language (IML) SDK.",
        version=__version__,
        lifespan=_lifespan,
    )
    application.state.settings = settings
    application.state.jobs = JobRunner(settings.max_concurrent_jobs, settings.max_queued_jobs)

    # Middleware added last runs first: CORS, then rate limit, then size limit.
    application.add_middleware(
        UploadSizeLimitMiddleware,
        max_bytes=settings.max_upload_bytes,
        max_json_bytes=settings.json_body_limit,
    )
    if settings.rate_limit_per_minute > 0:
        application.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.rate_limit_per_minute,
            trusted_proxies=settings.trusted_proxy_networks(),
        )
    # CORS: only allow configured origins. Empty list → no cross-origin access.
    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["Content-Disposition", "Retry-After", "X-Prosody-Engine"],
        )

    application.include_router(convert.router, prefix="/v1/convert", tags=["convert"])
    application.include_router(synthesize.router, prefix="/v1", tags=["synthesize"])
    application.include_router(validate.router, prefix="/v1", tags=["validate"])
    application.include_router(health_router, tags=["health"])
    install_error_handlers(application)
    return application


settings = Settings()
app = create_app(settings)
