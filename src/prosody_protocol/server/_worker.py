"""Code that runs in the API's worker processes (see :mod:`.jobs`).

Only the SDK is imported here, so a worker starts without loading FastAPI.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import multiprocessing.connection
import os
import signal
import threading
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from prosody_protocol._types import WordAlignment
from prosody_protocol.audio_to_iml import AudioToIML, ConversionResult
from prosody_protocol.exceptions import (
    AudioProcessingError,
    ConversionError,
    DatasetError,
    IMLParseError,
    IMLValidationError,
    ProfileError,
    ProsodyProtocolError,
    SpeechRecognitionError,
    TrainingError,
)
from prosody_protocol.iml_to_audio import IMLToAudio
from prosody_protocol.profiles import ProsodyProfile
from prosody_protocol.validator import ValidationIssue

T = TypeVar("T")

# The Whisper models this worker has loaded, by model name. Every converter
# uses this one dict (see _converter), so a worker loads each model once,
# whatever languages its requests ask for.
_whisper_models: dict[str, Any] = {}


class SpeechRecognitionUnavailable(AudioProcessingError):
    """The server's speech recognition cannot run (its Whisper model did not load).

    A problem of the server, not of the upload: reported as 503.
    """


class SpeechRecognitionFailed(AudioProcessingError):
    """The server's speech recognition failed on audio that was read fine.

    A problem of the server, not of the upload: reported as 500.
    """


# How AudioToIML reports Whisper failures (audio_to_iml.AudioToIML._transcribe).
_STT_LOAD_FAILED = "Cannot load Whisper model"


_SDK_ERRORS: tuple[type[ProsodyProtocolError], ...] = (
    SpeechRecognitionUnavailable,
    SpeechRecognitionFailed,
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


# Messages between the server and a worker process (see serve). The server
# sends ``(func, args)``, or ``None`` to stop the worker; the worker answers
# STARTED as soon as it has the job, then ``(OK, result)``,
# ``(RAISED, (exception, traceback text))``, or ``(FAILED, traceback text)``
# when the answer cannot be pickled.
STARTED = "started"
OK = "ok"
RAISED = "raised"
FAILED = "failed"


def serve(conn: multiprocessing.connection.Connection) -> None:
    """Main loop of a worker process: run the jobs *conn* delivers, one at a time.

    The STARTED answer tells the server that a job reached this worker, so
    if the worker dies, the server knows whether the job was running (it
    fails) or never started (it runs elsewhere). Answers are written to the
    pipe before :meth:`~multiprocessing.connection.Connection.send` returns,
    so they reach the server even when the worker is killed right after.
    """
    # Ctrl-C in a terminal reaches the whole process group; the server stops
    # its workers itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    exit_with_parent()
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return  # The server closed its end.
        if job is None:
            return
        conn.send(STARTED)
        func, args = job
        try:
            answer: tuple[str, Any] = (OK, call(func, *args))
        except Exception as exc:  # Reported to the server, which re-raises it.
            answer = (RAISED, (exc, traceback.format_exc()))
        try:
            conn.send(answer)
        except Exception:  # A result or exception that cannot be pickled.
            conn.send((FAILED, traceback.format_exc()))


@dataclass(frozen=True)
class ConvertOptions:
    """What an audio-to-iml request asks for, checked by the route.

    Everything here is picklable: it is sent to a worker process.
    """

    max_duration_s: float
    language: str | None = None
    words: tuple[WordAlignment, ...] | None = None
    transcript: str | None = None
    profile: ProsodyProfile | None = None
    stt_model: str = "base"


def _converter(options: ConvertOptions, calibration: Sequence[str] = ()) -> AudioToIML:
    """A converter for one request, sharing this worker's Whisper models.

    AudioToIML keeps the models it loads in its private ``_whisper_models``
    dict, one per instance; a converter per language would load a model per
    language. Replacing the dict is checked by
    ``test_whisper_model_is_loaded_once_per_worker``.
    """
    calibration_audio: Any = None
    if len(calibration) == 1:
        calibration_audio = calibration[0]
    elif calibration:
        calibration_audio = tuple(calibration)
    converter = AudioToIML(
        stt_model=options.stt_model,
        language=options.language,
        max_duration_s=options.max_duration_s,
        profile=options.profile,
        calibration_audio=calibration_audio,
    )
    converter._whisper_models = _whisper_models
    return converter


def convert_audio(
    path: str,
    display_name: str,
    options: ConvertOptions,
    calibration: Sequence[tuple[str, str]] = (),
) -> ConversionResult:
    """Convert the audio file at *path*; errors name *display_name* instead.

    *calibration* holds ``(path, display_name)`` pairs of calibration
    recordings of the same speaker. Audio longer than
    ``options.max_duration_s`` seconds is rejected before it is decoded in
    full. A Whisper failure raises :class:`SpeechRecognitionUnavailable` or
    :class:`SpeechRecognitionFailed`, since it is not the upload's fault.
    """
    names = [(path, display_name), *calibration]
    try:
        result = _converter(options, [p for p, _ in calibration]).convert_detailed(
            path, words=options.words, transcript=options.transcript
        )
    except AudioProcessingError as exc:
        message = _client_names(str(exc), names)
        raise _server_side(exc, message) or AudioProcessingError(message) from exc
    return dataclasses.replace(
        result, warnings=tuple(_client_names(note, names) for note in result.warnings)
    )


def _client_names(text: str, names: Sequence[tuple[str, str]]) -> str:
    """*text* with the server's temporary copies named as the client's files."""
    for temporary, name in names:
        text = text.replace(str(Path(temporary).resolve()), name)
        text = text.replace(temporary, name)
    return text


def _server_side(exc: AudioProcessingError, message: str) -> AudioProcessingError | None:
    """The server-side error *exc* stands for, if it is a speech recognition failure.

    The SDK raises :class:`SpeechRecognitionError` for both a model that cannot
    be loaded (the service is unavailable: 503) and a failed transcription
    (500); its message prefix tells them apart.
    """
    if not isinstance(exc, SpeechRecognitionError):
        return None
    if message.startswith(_STT_LOAD_FAILED):
        return SpeechRecognitionUnavailable(message)
    return SpeechRecognitionFailed(message)


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
