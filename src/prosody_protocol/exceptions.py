"""Custom exception hierarchy for the prosody_protocol SDK."""


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
    """Raised when IML document fails validation."""


class ProfileError(ProsodyProtocolError):
    """Raised when a prosody profile cannot be loaded or applied."""


class AudioProcessingError(ProsodyProtocolError):
    """Raised when audio processing fails."""


class ConversionError(ProsodyProtocolError):
    """Raised when format conversion fails (e.g., IML to SSML)."""


class DatasetError(ProsodyProtocolError):
    """Raised when dataset loading or validation fails."""


class TrainingError(ProsodyProtocolError):
    """Raised when model training or evaluation fails."""
