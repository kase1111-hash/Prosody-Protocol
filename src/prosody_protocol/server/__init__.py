"""Prosody Protocol REST API server (FastAPI).

Requires the ``api`` extra: ``pip install 'prosody-protocol[api]'``.

Run it with ``python -m prosody_protocol.server``, or with any ASGI server:
the application object is ``prosody_protocol.server.app:app``, and
:func:`prosody_protocol.server.app.create_app` builds one from explicit
:class:`~prosody_protocol.server.config.Settings`. Audio conversion and
synthesis run in worker processes (see :mod:`prosody_protocol.server.jobs`).
"""

from __future__ import annotations


def run(host: str | None = None, port: int | None = None) -> None:
    """Start the API server with uvicorn.

    ``host`` and ``port`` default to the ``PP_HOST`` / ``PP_PORT``
    environment variables (see :class:`prosody_protocol.server.config.Settings`).
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "The REST API requires the api extra. "
            "Install with: pip install 'prosody-protocol[api]'"
        ) from exc

    from .config import Settings

    settings = Settings()
    uvicorn.run(
        "prosody_protocol.server.app:app",
        host=host or settings.host,
        port=port or settings.port,
        log_level="debug" if settings.debug else "info",
    )
