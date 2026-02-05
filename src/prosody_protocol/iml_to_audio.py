"""IMLToAudio -- synthesize speech from IML markup.

Spec reference: Section 3 (IML tags drive synthesis parameters).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


class IMLToAudio:
    """Synthesize speech audio from IML markup."""

    def __init__(
        self,
        voice: str = "en_US-female-medium",
        engine: Literal["coqui", "piper", "elevenlabs"] = "coqui",
    ) -> None:
        self.voice = voice
        self.engine = engine

    def synthesize(self, iml_string: str) -> bytes:
        """Synthesize IML to raw audio bytes (WAV format)."""
        raise NotImplementedError

    def synthesize_to_file(self, iml_string: str, output_path: str | Path) -> None:
        """Synthesize IML and write to an audio file."""
        raise NotImplementedError
