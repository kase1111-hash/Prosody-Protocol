"""ProsodyAnalyzer -- extract acoustic features from audio.

Uses parselmouth (Praat) and librosa to measure F0, intensity,
speech rate, jitter, shimmer, HNR, and voice quality.
Spec reference: Section 4 (extended attributes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from .exceptions import AudioProcessingError


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float) -> float | None:
    """Return *value* if it is a normal finite number, else ``None``."""
    if np.isnan(value) or np.isinf(value):
        return None
    return float(value)


def _extract_f0(sound: parselmouth.Sound, t_start: float, t_end: float) -> tuple[
    float | None, tuple[float, float] | None, list[float] | None
]:
    """Extract F0 features for a time span using Praat pitch tracking."""
    pitch_obj = call(sound, "To Pitch", 0.0, 75.0, 600.0)

    # Get all pitch values in the span at 10 ms steps.
    step = 0.01
    times = np.arange(t_start, t_end, step)
    values: list[float] = []
    for t in times:
        v = call(pitch_obj, "Get value at time", float(t), "Hertz", "Linear")
        if not np.isnan(v):
            values.append(float(v))

    if not values:
        return None, None, None

    f0_mean = float(np.mean(values))
    f0_range = (float(min(values)), float(max(values)))
    return f0_mean, f0_range, values


def _extract_intensity(
    sound: parselmouth.Sound, t_start: float, t_end: float
) -> tuple[float | None, float | None]:
    """Extract mean and range of intensity (dB) for a span."""
    intensity_obj = call(sound, "To Intensity", 100.0, 0.0, "yes")

    step = 0.01
    times = np.arange(t_start, t_end, step)
    values: list[float] = []
    for t in times:
        v = call(intensity_obj, "Get value at time", float(t), "cubic")
        if not np.isnan(v):
            values.append(float(v))

    if not values:
        return None, None

    return float(np.mean(values)), float(max(values) - min(values))


def _extract_jitter_shimmer_hnr(
    sound: parselmouth.Sound, t_start: float, t_end: float
) -> tuple[float | None, float | None, float | None]:
    """Extract voice quality measures from a span using Praat PointProcess."""
    part = sound.extract_part(t_start, t_end, parselmouth.WindowShape.HANNING, 1.0, False)
    if part.duration < 0.05:
        return None, None, None

    try:
        point_process = call(part, "To PointProcess (periodic, cc)", 75.0, 600.0)
    except Exception:
        return None, None, None

    jitter: float | None = None
    shimmer: float | None = None
    hnr: float | None = None

    try:
        jitter = _safe_float(
            call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        )
    except Exception:
        pass

    try:
        shimmer = _safe_float(
            call([part, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
        )
    except Exception:
        pass

    try:
        harmonicity = call(part, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
        hnr = _safe_float(call(harmonicity, "Get mean", 0.0, 0.0))
    except Exception:
        pass

    return jitter, shimmer, hnr


def _classify_quality(
    jitter: float | None, shimmer: float | None, hnr: float | None
) -> str | None:
    """Derive voice quality label from jitter/shimmer/HNR thresholds."""
    if jitter is None and shimmer is None and hnr is None:
        return None

    # Heuristic thresholds based on clinical voice literature.
    if hnr is not None and hnr < 7.0:
        return "breathy"
    if jitter is not None and jitter > 0.02:
        return "creaky"
    if shimmer is not None and shimmer > 0.12:
        return "tense"
    return "modal"


def _estimate_speech_rate(
    sound: parselmouth.Sound, t_start: float, t_end: float
) -> float | None:
    """Estimate syllables per second using intensity-peak counting.

    This is a simplified approach: we count intensity peaks above a
    threshold, treating each as roughly one syllable.
    """
    duration = t_end - t_start
    if duration < 0.05:
        return None

    part = sound.extract_part(t_start, t_end, parselmouth.WindowShape.HANNING, 1.0, False)
    intensity_obj = call(part, "To Intensity", 100.0, 0.0, "yes")

    # Sample intensity values.
    step = 0.005
    times = np.arange(0, part.duration, step)
    values = []
    for t in times:
        v = call(intensity_obj, "Get value at time", float(t), "cubic")
        if not np.isnan(v):
            values.append(v)

    if len(values) < 3:
        return None

    arr = np.array(values)
    threshold = np.mean(arr) - 5.0  # dB below mean

    # Count peaks: points higher than both neighbors and above threshold.
    peaks = 0
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > threshold:
            peaks += 1

    if duration > 0:
        return float(peaks / duration)
    return None


def detect_pauses(
    sound: parselmouth.Sound,
    min_pause_ms: int = 200,
    rms_threshold_db: float = -40.0,
) -> list[PauseInterval]:
    """Detect silence gaps in the audio using intensity thresholds.

    Returns a list of :class:`PauseInterval` for gaps longer than
    *min_pause_ms* milliseconds.
    """
    intensity_obj = call(sound, "To Intensity", 100.0, 0.0, "yes")

    step = 0.01  # 10 ms
    times = np.arange(0, sound.duration, step)
    silent_start: float | None = None
    pauses: list[PauseInterval] = []

    for t in times:
        val = call(intensity_obj, "Get value at time", float(t), "cubic")
        is_silent = np.isnan(val) or val < rms_threshold_db

        if is_silent and silent_start is None:
            silent_start = t
        elif not is_silent and silent_start is not None:
            gap_ms = int((t - silent_start) * 1000)
            if gap_ms >= min_pause_ms:
                pauses.append(PauseInterval(
                    start_ms=int(silent_start * 1000),
                    end_ms=int(t * 1000),
                ))
            silent_start = None

    # Handle trailing silence.
    if silent_start is not None:
        gap_ms = int((sound.duration - silent_start) * 1000)
        if gap_ms >= min_pause_ms:
            pauses.append(PauseInterval(
                start_ms=int(silent_start * 1000),
                end_ms=int(sound.duration * 1000),
            ))

    return pauses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ProsodyAnalyzer:
    """Analyze acoustic prosody from audio given word-level alignments."""

    def analyze(
        self, audio_path: str | Path, alignments: list[WordAlignment]
    ) -> list[SpanFeatures]:
        """Extract prosodic features for each aligned span.

        Parameters
        ----------
        audio_path:
            Path to a WAV audio file.
        alignments:
            Word-level time boundaries from an STT engine.

        Returns
        -------
        list[SpanFeatures]
            One :class:`SpanFeatures` per alignment entry.
        """
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        try:
            sound = parselmouth.Sound(str(path))
        except Exception as exc:
            raise AudioProcessingError(f"Cannot load audio: {exc}") from exc

        results: list[SpanFeatures] = []
        for alignment in alignments:
            t_start = alignment.start_ms / 1000.0
            t_end = alignment.end_ms / 1000.0

            # Clamp to audio boundaries.
            t_start = max(0.0, t_start)
            t_end = min(sound.duration, t_end)
            if t_end <= t_start:
                results.append(SpanFeatures(
                    start_ms=alignment.start_ms,
                    end_ms=alignment.end_ms,
                    text=alignment.word,
                ))
                continue

            f0_mean, f0_range, f0_contour = _extract_f0(sound, t_start, t_end)
            intensity_mean, intensity_range = _extract_intensity(sound, t_start, t_end)
            jitter, shimmer, hnr = _extract_jitter_shimmer_hnr(sound, t_start, t_end)
            quality = _classify_quality(jitter, shimmer, hnr)
            speech_rate = _estimate_speech_rate(sound, t_start, t_end)

            results.append(SpanFeatures(
                start_ms=alignment.start_ms,
                end_ms=alignment.end_ms,
                text=alignment.word,
                f0_mean=f0_mean,
                f0_range=f0_range,
                f0_contour=f0_contour,
                intensity_mean=intensity_mean,
                intensity_range=intensity_range,
                speech_rate=speech_rate,
                jitter=jitter,
                shimmer=shimmer,
                hnr=hnr,
                quality=quality,
            ))

        return results

    def detect_pauses(
        self,
        audio_path: str | Path,
        min_pause_ms: int = 200,
    ) -> list[PauseInterval]:
        """Detect silence pauses in the audio."""
        path = Path(audio_path)
        if not path.exists():
            raise AudioProcessingError(f"Audio file not found: {path}")

        sound = parselmouth.Sound(str(path))
        return detect_pauses(sound, min_pause_ms=min_pause_ms)
