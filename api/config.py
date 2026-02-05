"""API server configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list[str] | None = None
    max_upload_size_mb: int = 50
