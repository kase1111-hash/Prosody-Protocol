"""Dependency-free data types shared by the analysis and assembly modules.

These live apart from :mod:`prosody_protocol.prosody_analyzer` so that the
assembler and emotion classifier can be imported without numpy or
parselmouth installed.
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


@dataclass
class PauseInterval:
    """A detected silence gap between speech segments."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
