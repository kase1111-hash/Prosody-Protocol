"""ProsodyAnalyzer -- extract acoustic features from audio.

Uses parselmouth (Praat) and librosa to measure F0, intensity,
speech rate, jitter, shimmer, HNR, and voice quality.
Spec reference: Section 4 (extended attributes).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordAlignment:
    """A word with its time boundaries in the audio."""

    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SpanFeatures:
    """Acoustic features measured over a span of audio."""

    start_ms: int
    end_ms: int
    text: str
    f0_mean: float | None = None
    f0_range: tuple[float, float] | None = None
    f0_contour: list[float] | None = None
    intensity_mean: float | None = None
    intensity_range: float | None = None
    speech_rate: float | None = None
    jitter: float | None = None
    shimmer: float | None = None
    hnr: float | None = None
    quality: str | None = None


class ProsodyAnalyzer:
    """Analyze acoustic prosody from audio given word-level alignments."""

    def analyze(
        self, audio_path: str, alignments: list[WordAlignment]
    ) -> list[SpanFeatures]:
        """Extract prosodic features for each aligned span."""
        raise NotImplementedError
