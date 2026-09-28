"""Prosody Protocol REST API server (FastAPI).

Requires the ``api`` extra (see README "Install").

Run it with ``python -m prosody_protocol.server``, or with any ASGI server:
the application object is ``prosody_protocol.server.app:app``, and
:func:`prosody_protocol.server.app.create_app` builds one from explicit
:class:`~prosody_protocol.server.config.Settings`. Audio conversion and
synthesis run in worker processes (see :mod:`prosody_protocol.server.jobs`).
"""

from __future__ import annotations

from .._install import install_hint


def run(host: str | None = None, port: int | None = None) -> None:
    """Start the API server with uvicorn.

    ``host`` and ``port`` default to the ``PP_HOST`` / ``PP_PORT``
    environment variables (see :class:`prosody_protocol.server.config.Settings`).
    A *port* outside 0-65535 raises :class:`ValueError`; 0 lets the system
    choose a free port.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "The REST API requires the api extra. Install with: " + install_hint("api")
        ) from exc

    from .config import MAX_PORT, Settings

    if port is not None and not 0 <= port <= MAX_PORT:
        raise ValueError(f"port must be between 0 and {MAX_PORT}, got {port}")
    settings = Settings()
    uvicorn.run(
        "prosody_protocol.server.app:app",
        host=host or settings.host,
        port=settings.port if port is None else port,
        log_level="debug" if settings.debug else "info",
    )
