"""AudioToIML -- convert audio files to IML-annotated transcripts.

Orchestrates STT, prosody analysis, emotion classification, and IML assembly.
Spec reference: Sections 3-4.
"""

from __future__ import annotations

from pathlib import Path

from .models import IMLDocument


class AudioToIML:
    """Convert audio files to IML markup."""

    def __init__(
        self,
        stt_model: str = "base",
        include_extended: bool = False,
    ) -> None:
        self.stt_model = stt_model
        self.include_extended = include_extended

    def convert(self, audio_path: str | Path) -> str:
        """Convert an audio file to an IML XML string."""
        raise NotImplementedError

    def convert_to_doc(self, audio_path: str | Path) -> IMLDocument:
        """Convert an audio file to a parsed :class:`IMLDocument`."""
        raise NotImplementedError
