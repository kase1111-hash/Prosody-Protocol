"""API server configuration."""

from __future__ import annotations

import ipaddress
import math
import os
from dataclasses import dataclass, field

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

#: The highest TCP port number.
MAX_PORT = 65_535


@dataclass
class Settings:
    """Application settings, configurable via environment variables.

    Environment variables:
        PP_HOST: Server bind address (default "127.0.0.1")
        PP_PORT: Server port, 1-65535 (default 8000)
        PP_DEBUG: Enable debug mode ("1" or "true")
        PP_CORS_ORIGINS: Comma-separated allowed origins (default: none, reject cross-origin)
        PP_MAX_UPLOAD_MB: Maximum request body size in megabytes, counted on the
            bytes actually received, so chunked uploads are limited too (default 50)
        PP_RATE_LIMIT: Requests per minute per client (default 60, 0 = unlimited).
            ``/v1/health`` is not rate limited.
        PP_TRUSTED_PROXIES: Comma-separated IP addresses or CIDR networks of
            reverse proxies whose ``X-Forwarded-For`` header names the client
            for rate limiting (default: none, so the limiter keys on the
            connecting address and ignores ``X-Forwarded-For``)
        PP_MAX_TEXT_CHARS: Maximum length, in characters, of each text field
            of a request (``iml``, ``text``, ``context``, ``instruction``, and the
            ``transcript`` and ``profile`` fields of audio-to-iml) (default 100000)
        PP_MAX_WORDS_CHARS: Maximum size of the ``words`` field of audio-to-iml
            (word timings JSON), in characters, or bytes when it is sent as a
            file (default 1000000: the full response of any supported speech
            recogniser for PP_MAX_AUDIO_SECONDS of speech, with room to spare).
            Words may overlap by at most 500 ms, so the analysis they cost is
            bounded by the audio's length plus a small amount per word.
        PP_MAX_SYNTH_SECONDS: Maximum duration of the audio ``/v1/synthesize``
            produces; longer documents are rejected before synthesis (default 120)
        PP_MAX_AUDIO_SECONDS: Maximum duration of an audio upload; longer audio
            is rejected before it is decoded in full, since a small compressed
            file can hold hours of audio (default 600)
        PP_MAX_CONCURRENT_JOBS: Audio conversions and syntheses run at the same
            time, each in its own worker process (default 2)
        PP_MAX_QUEUED_JOBS: Audio conversions and syntheses that may wait for a
            free worker; further requests are refused with 503 (default 8)

    Invalid values, from the environment or passed directly, raise
    :class:`ValueError` naming the setting and its variable.
    """

    host: str = field(default_factory=lambda: os.getenv("PP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PP_PORT", 8000))
    debug: bool = field(
        default_factory=lambda: os.getenv("PP_DEBUG", "").lower() in ("1", "true")
    )
    cors_origins: list[str] = field(default_factory=lambda: _parse_cors())
    max_upload_size_mb: int = field(default_factory=lambda: _env_int("PP_MAX_UPLOAD_MB", 50))
    rate_limit_per_minute: int = field(default_factory=lambda: _env_int("PP_RATE_LIMIT", 60))
    trusted_proxies: list[str] = field(default_factory=lambda: _env_list("PP_TRUSTED_PROXIES"))
    max_text_chars: int = field(default_factory=lambda: _env_int("PP_MAX_TEXT_CHARS", 100_000))
    max_words_chars: int = field(
        default_factory=lambda: _env_int("PP_MAX_WORDS_CHARS", 1_000_000)
    )
    max_synth_seconds: float = field(
        default_factory=lambda: _env_float("PP_MAX_SYNTH_SECONDS", 120.0)
    )
    max_concurrent_jobs: int = field(default_factory=lambda: _env_int("PP_MAX_CONCURRENT_JOBS", 2))
    max_audio_seconds: float = field(
        default_factory=lambda: _env_float("PP_MAX_AUDIO_SECONDS", 600.0)
    )
    max_queued_jobs: int = field(default_factory=lambda: _env_int("PP_MAX_QUEUED_JOBS", 8))

    def __post_init__(self) -> None:
        # Fail at startup, not on the first request that needs the value.
        for name, (variable, minimum) in _INT_SETTINGS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"{name} ({variable}) must be an integer >= {minimum}, got {value!r}"
                )
        if self.port > MAX_PORT:
            raise ValueError(f"port (PP_PORT) must be at most {MAX_PORT}, got {self.port!r}")
        for name, variable in _DURATION_SETTINGS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not (
                0 < value < math.inf
            ):
                raise ValueError(f"{name} ({variable}) must be a positive number, got {value!r}")
        self.trusted_proxy_networks()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def trusted_proxy_networks(self) -> tuple[IPNetwork, ...]:
        """``trusted_proxies`` parsed as networks (a bare address is a /32 or /128)."""
        networks: list[IPNetwork] = []
        for entry in self.trusted_proxies:
            try:
                networks.append(ipaddress.ip_network(entry, strict=False))
            except ValueError as exc:
                raise ValueError(
                    f"PP_TRUSTED_PROXIES: {entry!r} is not an IP address or CIDR network"
                ) from exc
        return tuple(networks)


# Numeric settings checked by Settings.__post_init__: integers with their
# variable and minimum, and durations (positive, finite seconds).
_INT_SETTINGS: dict[str, tuple[str, int]] = {
    "port": ("PP_PORT", 1),
    "max_upload_size_mb": ("PP_MAX_UPLOAD_MB", 1),
    "rate_limit_per_minute": ("PP_RATE_LIMIT", 0),
    "max_text_chars": ("PP_MAX_TEXT_CHARS", 1),
    "max_words_chars": ("PP_MAX_WORDS_CHARS", 1),
    "max_concurrent_jobs": ("PP_MAX_CONCURRENT_JOBS", 1),
    "max_queued_jobs": ("PP_MAX_QUEUED_JOBS", 0),
}
_DURATION_SETTINGS: dict[str, str] = {
    "max_synth_seconds": "PP_MAX_SYNTH_SECONDS",
    "max_audio_seconds": "PP_MAX_AUDIO_SECONDS",
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_cors() -> list[str]:
    return _env_list("PP_CORS_ORIGINS")
