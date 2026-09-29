"""Custom exception hierarchy for the prosody_protocol SDK."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .validator import ValidationIssue


class ProsodyProtocolError(Exception):
    """Base exception for all prosody_protocol errors."""


class IMLParseError(ProsodyProtocolError):
    """Raised when IML XML cannot be parsed."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None) -> None:
        self.line = line
        self.column = column
        location = ""
        if line is not None:
            location = f" (line {line}"
            if column is not None:
                location += f", column {column}"
            location += ")"
        super().__init__(f"{message}{location}")


class IMLValidationError(ProsodyProtocolError):
    """Raised when an IML document fails validation.

    ``issues`` holds the :class:`~prosody_protocol.validator.ValidationIssue`
    objects that made the document invalid (empty when none were given).
    :meth:`ValidationResult.raise_for_errors
    <prosody_protocol.validator.ValidationResult.raise_for_errors>` raises
    this with the document's errors attached.
    """

    def __init__(self, message: str, issues: Sequence[ValidationIssue] = ()) -> None:
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        super().__init__(message)


class ProfileError(ProsodyProtocolError):
    """Raised when a prosody profile cannot be loaded or applied."""


class AudioProcessingError(ProsodyProtocolError):
    """Raised when audio processing fails."""


class SpeechRecognitionError(AudioProcessingError):
    """Raised when built-in speech recognition (Whisper) fails: its model
    cannot be loaded, or transcription fails.

    A subclass of :class:`AudioProcessingError`, so code that catches that
    catches this too; catch this to tell a recognizer failure from audio
    that cannot be read or analysed. Messages start with ``Cannot load
    Whisper model`` or ``Whisper transcription failed``. (Whisper required
    but not installed is an :class:`AudioProcessingError`: a setup problem,
    not a failed recognition.)
    """


class ConversionError(ProsodyProtocolError):
    """Raised when format conversion fails (e.g., IML to SSML)."""


class DatasetError(ProsodyProtocolError):
    """Raised when dataset loading or validation fails."""


class TrainingError(ProsodyProtocolError):
    """Raised when model training or evaluation fails."""
