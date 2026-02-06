"""API server configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Application settings, configurable via environment variables.

    Environment variables:
        PP_HOST: Server bind address (default "0.0.0.0")
        PP_PORT: Server port (default 8000)
        PP_DEBUG: Enable debug mode ("1" or "true")
        PP_CORS_ORIGINS: Comma-separated allowed origins (default: none, reject cross-origin)
        PP_MAX_UPLOAD_MB: Maximum upload size in megabytes (default 50)
        PP_RATE_LIMIT: Requests per minute per client (default 60, 0 = unlimited)
    """

    host: str = field(default_factory=lambda: os.getenv("PP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PP_PORT", "8000")))
    debug: bool = field(
        default_factory=lambda: os.getenv("PP_DEBUG", "").lower() in ("1", "true")
    )
    cors_origins: list[str] = field(default_factory=lambda: _parse_cors())
    max_upload_size_mb: int = field(
        default_factory=lambda: int(os.getenv("PP_MAX_UPLOAD_MB", "50"))
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("PP_RATE_LIMIT", "60"))
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


def _parse_cors() -> list[str]:
    raw = os.getenv("PP_CORS_ORIGINS", "")
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]
