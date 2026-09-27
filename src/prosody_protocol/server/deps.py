"""Request dependencies shared by the routes: settings, worker jobs, text limits."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from .config import Settings
from .errors import APIError
from .jobs import JobRunner


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_jobs(request: Request) -> JobRunner:
    jobs: JobRunner = request.app.state.jobs
    return jobs


SettingsDep = Annotated[Settings, Depends(get_settings)]
JobsDep = Annotated[JobRunner, Depends(get_jobs)]


def check_text_length(settings: Settings, **fields: str | None) -> None:
    """Raise a 413 :class:`APIError` if a text field is over PP_MAX_TEXT_CHARS."""
    for name, value in fields.items():
        if value is not None and len(value) > settings.max_text_chars:
            raise APIError(
                413,
                "text_too_large",
                f"Field {name!r} has {len(value)} characters; the maximum is "
                f"{settings.max_text_chars} (PP_MAX_TEXT_CHARS).",
            )
