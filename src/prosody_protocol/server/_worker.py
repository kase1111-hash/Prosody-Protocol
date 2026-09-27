"""Code that runs in the API's worker processes (see :mod:`.jobs`).

Only the SDK is imported here, so a worker starts without loading FastAPI.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.connection
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

from prosody_protocol.audio_to_iml import AudioToIML, ConversionResult
from prosody_protocol.exceptions import (
    AudioProcessingError,
    ConversionError,
    DatasetError,
    IMLParseError,
    IMLValidationError,
    ProfileError,
    ProsodyProtocolError,
    TrainingError,
)
from prosody_protocol.iml_to_audio import IMLToAudio
from prosody_protocol.validator import ValidationIssue

T = TypeVar("T")

# The Whisper models this worker has loaded, by model name. Every converter
# uses this one dict (see _converter), so a worker loads each model once,
# whatever languages its requests ask for.
_whisper_models: dict[str, Any] = {}

_SDK_ERRORS: tuple[type[ProsodyProtocolError], ...] = (
    IMLParseError,
    IMLValidationError,
    ProfileError,
    AudioProcessingError,
    ConversionError,
    DatasetError,
    TrainingError,
    ProsodyProtocolError,
)


class JobError(Exception):
    """An SDK exception raised in a worker, in a form that survives pickling.

    Only the class name, message and validation issues are sent back, so
    the API does not depend on every SDK exception rebuilding itself from a
    pickle: one that failed to would break the whole worker pool.
    """

    def __init__(self, kind: str, message: str, issues: tuple[ValidationIssue, ...] = ()) -> None:
        super().__init__(kind, message, issues)
        self.kind = kind
        self.message = message
        self.issues = issues

    @classmethod
    def from_exception(cls, exc: ProsodyProtocolError) -> JobError:
        kind = next(k for k in type(exc).__mro__ if k in _SDK_ERRORS)
        issues = exc.issues if isinstance(exc, IMLValidationError) else ()
        return cls(kind.__name__, str(exc), issues)

    def rebuild(self) -> ProsodyProtocolError:
        """The SDK exception this stands for."""
        if self.kind == IMLValidationError.__name__:
            return IMLValidationError(self.message, self.issues)
        kind = next((k for k in _SDK_ERRORS if k.__name__ == self.kind), ProsodyProtocolError)
        return kind(self.message)


def exit_with_parent() -> None:
    """Worker initializer: end this process when the server process ends.

    Covers servers that are killed without shutting the pool down, which
    would otherwise leave workers waiting for jobs forever.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    sentinel = parent.sentinel

    def watch() -> None:
        multiprocessing.connection.wait([sentinel])
        os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def call(func: Callable[..., T], *args: object) -> T:
    """Run ``func(*args)``, turning SDK exceptions into :class:`JobError`."""
    try:
        return func(*args)
    except ProsodyProtocolError as exc:
        raise JobError.from_exception(exc) from None


def _converter(language: str | None, max_duration_s: float) -> AudioToIML:
    """A converter for one request, sharing this worker's Whisper models.

    AudioToIML keeps the models it loads in its private ``_whisper_models``
    dict, one per instance; a converter per language would load a model per
    language. Replacing the dict is checked by
    ``test_whisper_model_is_loaded_once_per_worker``.
    """
    converter = AudioToIML(language=language, max_duration_s=max_duration_s)
    converter._whisper_models = _whisper_models
    return converter


def convert_audio(
    path: str, language: str | None, display_name: str, max_duration_s: float
) -> ConversionResult:
    """Convert the audio file at *path*; errors name *display_name* instead.

    Audio longer than *max_duration_s* seconds is rejected before it is
    decoded in full.
    """
    try:
        return _converter(language, max_duration_s).convert_detailed(path)
    except AudioProcessingError as exc:
        # Name the client's file, not the server's temporary copy.
        message = str(exc).replace(str(Path(path).resolve()), display_name)
        raise AudioProcessingError(message.replace(path, display_name)) from exc


def synthesize(
    iml: str,
    voice: str | None,
    engine: Literal["auto", "espeak", "tones"],
    max_duration_s: float,
    strict: bool,
) -> tuple[bytes, str]:
    """Synthesize *iml*; returns the WAV bytes and the engine used."""
    synth = IMLToAudio(voice=voice, engine=engine, max_duration_s=max_duration_s, strict=strict)
    return synth.synthesize(iml), synth.backend
