"""Dependency-free data types shared by the analysis and assembly modules.

These live apart from :mod:`prosody_protocol.prosody_analyzer` so that the
assembler, emotion classifier, profiles and word-timing adapters can be
imported without numpy or parselmouth installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordAlignment:
    """A word with its time boundaries in the audio.

    ``word`` is a token as the speech recognizer produced it; it may carry
    surrounding whitespace (Whisper's leading space) or be punctuation only.
    Times are in milliseconds from the start of the audio.
    :func:`~prosody_protocol.alignment.load_word_timings` and
    :func:`~prosody_protocol.alignment.parse_word_timings` make them from
    the output of common speech-to-text services.
    """

    word: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SpanFeatures:
    """Acoustic features measured over a span of audio.

    Units match the extended attributes of spec Section 4. Every measurement
    is ``None`` when it could not be made (e.g. pitch in an unvoiced span,
    or a span outside the audio). The attributes are described as
    :meth:`ProsodyAnalyzer.analyze
    <prosody_protocol.prosody_analyzer.ProsodyAnalyzer.analyze>` measures
    them; its docstring gives the method.

    Attributes
    ----------
    start_ms, end_ms, text:
        The span's word and times (ms) as the :class:`WordAlignment` gave
        them; the measurements cover the part inside the audio.
    f0_mean, f0_range:
        Mean and ``(min, max)`` fundamental frequency in Hz.
    f0_contour:
        The voiced F0 samples in Hz, one every 10 ms in time order across
        the span, with pitch-tracking octave jumps removed.
    intensity_mean, intensity_range:
        Mean level (averaged as power) and dynamic range in dB, over the
        non-silent frames only.
    speech_rate:
        Syllables per second of speaking time (pauses excluded) in a window
        of at least one second around the span; ``None`` when the span is
        unvoiced or there is too little speech around it to estimate a rate.
    jitter, shimmer:
        Local jitter and shimmer in percent (``1.2`` means 1.2 %); ``None``
        for an unvoiced span.
    hnr:
        Harmonics-to-noise ratio in dB; ``None`` for an unvoiced span.
    quality:
        Voice quality relative to the speaker's usual voice in the rest of
        the same recording, in the spec 3.2 vocabulary: ``creaky``,
        ``harsh``, ``breathy``, or ``modal`` when the span shows none of
        those deviations (the speaker's usual phonation, whatever it is).
        The analyzer never yields ``tense`` or ``whispery``. ``None`` when
        it cannot tell: less than 100 ms of voicing in the span, too little
        voiced speech elsewhere in the recording to compare with (always
        the case for a span covering the whole recording), or deviations
        that fit none of the labels.
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
    """A detected silent stretch of audio.

    Times are milliseconds from the start of the audio. The stretch is a
    pause in the speech or silence at the start or end of the recording;
    :meth:`ProsodyAnalyzer.detect_pauses
    <prosody_protocol.prosody_analyzer.ProsodyAnalyzer.detect_pauses>`
    returns both.
    """

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
