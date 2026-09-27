"""Dependency-free data types shared by the analysis and assembly modules.

These live apart from :mod:`prosody_protocol.prosody_analyzer` so that the
assembler and emotion classifier can be imported without numpy or
parselmouth installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordAlignment:
    """A word with its time boundaries in the audio.

    ``word`` is a token as the speech recogniser produced it; it may carry
    surrounding whitespace (Whisper's leading space) or be punctuation only.
    Times are in milliseconds from the start of the audio.
    """

    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SpanFeatures:
    """Acoustic features measured over a span of audio.

    Units match the extended attributes of spec Section 4. Every measurement
    is ``None`` when it could not be made (e.g. pitch in an unvoiced span).

    Attributes
    ----------
    f0_mean, f0_range:
        Mean and ``(min, max)`` fundamental frequency in Hz.
    f0_contour:
        The voiced F0 samples in Hz, in time order across the span, with
        pitch-tracking octave jumps removed.
    intensity_mean, intensity_range:
        Mean level and dynamic range in dB, over the non-silent frames only.
    speech_rate:
        Syllables per second of speaking time around the span; ``None``
        when there is too little speech to estimate it.
    jitter, shimmer:
        Local jitter and shimmer in percent (``1.2`` means 1.2 %).
    hnr:
        Harmonics-to-noise ratio in dB.
    quality:
        A voice quality from the spec vocabulary (``modal``, ``breathy``,
        ``creaky``, ``tense``, ...).
    """

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
    """A detected silence gap between speech segments, in milliseconds."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
