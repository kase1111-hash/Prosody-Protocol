"""ProsodyAnalyzer -- extract acoustic features from audio.

Uses parselmouth (Praat) and numpy to measure F0, intensity,
speech rate, jitter, shimmer, HNR, and voice quality.
Spec reference: Section 4 (extended attributes).

Each recording is analysed once: the pitch track, intensity contour,
glottal pulses and harmonicity are computed for the whole file and then
sliced per word, so the cost grows linearly with the length of the audio.

Silence is judged relative to the recording itself: a frame is silent when
it is far below the file's speech level (or close to its noise floor), and
frames of exact digital silence are always silent. Intensity statistics
ignore silent frames.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import TypeVar

try:
    import numpy as np
    import numpy.typing as npt
    import parselmouth
    from parselmouth.praat import call
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "Audio analysis requires numpy and praat-parselmouth. "
        "Install with: pip install 'prosody-protocol[audio]'"
    ) from exc

from ._types import PauseInterval, SpanFeatures, WordAlignment
from .exceptions import AudioProcessingError

__all__ = [
    "DEFAULT_MIN_PAUSE_MS",
    "DEFAULT_SILENCE_THRESHOLD_DB",
    "PauseInterval",
    "ProsodyAnalyzer",
    "SpanFeatures",
    "WordAlignment",
    "detect_pauses",
]

_T = TypeVar("_T")
_FloatArray = npt.NDArray[np.float64]
_BoolArray = npt.NDArray[np.bool_]

# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------

# Pitch search range (Hz) for F0, glottal pulses and harmonicity.
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 600.0

# Hop between analysis frames (pitch, intensity, silence), in seconds.
FRAME_STEP_S = 0.01

# Praat's intensity window is 3.2 / INTENSITY_MIN_PITCH_HZ seconds long.
INTENSITY_MIN_PITCH_HZ = 100.0

# A frame is silent when its level is more than this many dB below the
# file's speech level (the 99th percentile of frame levels), as with the
# silence threshold of Praat's "To TextGrid (silences)".
DEFAULT_SILENCE_THRESHOLD_DB = 25.0

# Silent stretches shorter than this are not reported as pauses. The spec
# treats pauses under 200 ms as ordinary speech rhythm (Section 3.3).
DEFAULT_MIN_PAUSE_MS = 200

# Shorter recordings cannot be pitch-tracked.
MIN_AUDIO_DURATION_S = 0.1

# Containers ffmpeg may open (demuxer names). Playlists and scripts such as
# HLS and concat are left out: they make ffmpeg read the other files they
# name, so an uploaded playlist could expose any audio on the machine.
_FFMPEG_FORMATS = (
    "ogg", "matroska", "webm", "mov", "mp4", "m4a", "3gp", "3g2", "mj2",
    "mp3", "aac", "flac", "wav", "aiff", "caf", "amr", "au", "w64",
)
# ffmpeg decodes to 16 kHz mono: ample for prosody, which lies well below
# 8 kHz, and a third of the memory of 48 kHz Opus.
_FFMPEG_SAMPLE_RATE = 16_000
# Decoding a whole hour of audio takes ffmpeg about ten seconds.
_FFMPEG_TIMEOUT_S = 300.0

_SPEECH_LEVEL_PERCENTILE = 99.0
_NOISE_FLOOR_PERCENTILE = 5.0
# In recordings whose noise floor lies above the relative threshold, frames
# within this many dB of the noise floor are silent too ...
_NOISE_MARGIN_DB = 6.0
# ... but never frames within this many dB of the speech level.
_MIN_SPEECH_MARGIN_DB = 10.0
# Sounding bursts shorter than this between silences (clicks, lip smacks)
# do not interrupt a pause.
_MIN_SOUNDING_MS = 30
# Frames with less power than this (samples read as Pa) are digital silence.
_DIGITAL_SILENCE_POWER = 1e-12
# Praat reports digital silence as -300 dB; reuse that value for our frames.
_DIGITAL_SILENCE_DB = -300.0
# Praat's intensity reference: samples are read as Pa, re 20 uPa.
_REFERENCE_POWER = 4e-10

# Octave-jump guard: a voiced F0 sample more than this ratio away from the
# running median of its neighbours is halved or doubled when that brings it
# back into range, and dropped otherwise.
_OCTAVE_JUMP_RATIO = 1.6
# Neighbours on each side for that running median (150 ms window).
_F0_MEDIAN_HALF_WINDOW = 7

# Syllable nuclei (de Jong & Wempe 2009): voiced intensity peaks separated
# from their neighbours by dips of at least this depth.
_NUCLEUS_MIN_DIP_DB = 2.0
# Speech rate is counted over this much speaking time (pauses excluded)
# around a span, looking no further than _MAX_RATE_WINDOW_S in total; with
# less than _MIN_RATE_SPEAKING_S of speech it is unknown.
_RATE_WINDOW_S = 1.0
_MAX_RATE_WINDOW_S = 4.0
_MIN_RATE_SPEAKING_S = 0.25

# A stretch of audio is speech only if it has at least this much voicing.
_MIN_VOICED_S = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float) -> float | None:
    """Return *value* if it is a normal finite number, else ``None``."""
    if np.isnan(value) or np.isinf(value):
        return None
    return float(value)


def _praat_message(exc: Exception) -> str:
    """First line of a Praat error, which carries the useful part."""
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def _run_praat(what: str, func: Callable[[], _T]) -> _T:
    """Run a whole-file Praat analysis, turning Praat errors into
    :class:`AudioProcessingError`."""
    try:
        return func()
    except parselmouth.PraatError as exc:
        raise AudioProcessingError(
            f"Praat {what} analysis failed: {_praat_message(exc)}"
        ) from exc


def _query(func: Callable[[], float]) -> float | None:
    """Run a per-span Praat query; undefined results and errors give ``None``."""
    try:
        return _safe_float(func())
    except parselmouth.PraatError:
        return None


def _ffmpeg_error(stderr: str, returncode: int) -> str:
    """The first line of ffmpeg's error output, without memory addresses."""
    for line in stderr.splitlines():
        if line.strip():
            return re.sub(r" @ 0x[0-9a-fA-F]+", "", line.strip())
    return f"exit status {returncode}"


def _decode_with_ffmpeg(
    path: Path, praat_error: str | None = None, max_duration_s: float | None = None
) -> parselmouth.Sound:
    """Decode audio with ffmpeg to 16 kHz mono (OGG/Opus, WebM, M4A, MP3, ...).

    *praat_error* says why Praat could not read the file, if it was tried.
    Only the audio containers in :data:`_FFMPEG_FORMATS` are opened, and at
    most a second more than *max_duration_s* is decoded.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioProcessingError(
            f"Cannot read audio file {path} ({praat_error}). WAV, AIFF, FLAC and MP3 are "
            "read directly; for other formats such as OGG/Opus, WebM or M4A, install "
            "ffmpeg and make sure it is on PATH, or convert the file to WAV first."
        )
    with tempfile.TemporaryDirectory(prefix="prosody-protocol-") as tmp:
        wav_path = Path(tmp) / "decoded.wav"
        command = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-protocol_whitelist", "file", "-format_whitelist", ",".join(_FFMPEG_FORMATS),
            # The file: prefix stops ffmpeg from reading the name as a protocol.
            "-i", f"file:{path.resolve()}",
        ]
        if max_duration_s is not None:
            # Decode just enough to tell that the audio is too long.
            command += ["-t", f"{max_duration_s + 1.0:.3f}"]
        command += [
            "-vn", "-ac", "1", "-ar", str(_FFMPEG_SAMPLE_RATE),
            "-c:a", "pcm_f32le", "-f", "wav", str(wav_path),
        ]
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=_FFMPEG_TIMEOUT_S
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioProcessingError(
                f"Cannot read audio file {path}: ffmpeg did not finish decoding it "
                f"within {_FFMPEG_TIMEOUT_S:.0f} s"
            ) from exc
        except OSError as exc:
            raise AudioProcessingError(f"Cannot run ffmpeg to decode {path}: {exc}") from exc
        if proc.returncode != 0 or not wav_path.is_file():
            detail = f"ffmpeg: {_ffmpeg_error(proc.stderr, proc.returncode)}"
            if praat_error:
                detail = f"Praat: {praat_error}; {detail}"
            raise AudioProcessingError(
                f"Cannot read audio file {path}: not a supported audio format ({detail})"
            )
        try:
            return parselmouth.Sound(str(wav_path))
        except parselmouth.PraatError as exc:
            raise AudioProcessingError(
                f"Cannot read audio file {path}: {_praat_message(exc)}"
            ) from exc


def _looks_like_mp3(path: Path) -> bool:
    """Whether the file starts like MPEG audio: an ID3 tag or a frame sync.

    Praat's MP3 reader crashes the whole process (SIGFPE) on some damaged
    or crafted files of this kind, so they are decoded with ffmpeg instead
    whenever it is available.
    """
    with path.open("rb") as f:
        head = f.read(3)
    return head.startswith(b"ID3") or (
        len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0
    )


def _header_duration(path: Path) -> float | None:
    """Duration (s) from the file header alone, without reading the samples;
    ``None`` when Praat cannot open the file as a long sound."""
    try:
        return float(call(call("Open long sound file", str(path)), "Get total duration"))
    except parselmouth.PraatError:
        return None


def _check_duration(path: Path, duration_s: float, max_duration_s: float | None) -> None:
    if max_duration_s is not None and duration_s > max_duration_s:
        raise AudioProcessingError(
            f"Audio is longer than max_duration_s={max_duration_s:g} s: {path}"
        )


def _load_sound(audio_path: str | Path, max_duration_s: float | None = None) -> parselmouth.Sound:
    """Read *audio_path* as a mono Sound.

    Any failure (missing file, directory, empty or non-audio file, audio too
    short to analyse, or longer than *max_duration_s* seconds) raises
    :class:`AudioProcessingError`. The length limit is checked from the
    file header, or while decoding, before the whole file is loaded. A
    truncated file, whose header promises more samples than it holds, is
    decoded by ffmpeg, which reads just the samples that are there (Praat
    would pad it with silence); without ffmpeg it is an error.
    """
    path = Path(audio_path)
    if not path.exists():
        raise AudioProcessingError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioProcessingError(f"Audio path is not a file: {path}")
    if path.stat().st_size == 0:
        raise AudioProcessingError(f"Audio file is empty: {path}")

    try:
        mp3 = _looks_like_mp3(path)
    except OSError as exc:
        raise AudioProcessingError(f"Cannot read audio file {path}: {exc}") from exc
    if mp3 and shutil.which("ffmpeg") is not None:
        sound = _decode_with_ffmpeg(path, max_duration_s=max_duration_s)
    else:
        if max_duration_s is not None:
            header_duration = _header_duration(path)
            if header_duration is not None:
                _check_duration(path, header_duration, max_duration_s)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", "File too small", parselmouth.PraatWarning)
                sound = parselmouth.Sound(str(path))
        except parselmouth.PraatWarning:
            # Praat would pad the missing samples with silence; ffmpeg decodes
            # only the samples that are there.
            sound = _decode_with_ffmpeg(path, "the file is truncated", max_duration_s)
        except Exception as exc:  # parselmouth.PraatError for formats Praat cannot read
            sound = _decode_with_ffmpeg(path, _praat_message(exc).rstrip("."), max_duration_s)

    _check_duration(path, float(sound.duration), max_duration_s)
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    if sound.duration < MIN_AUDIO_DURATION_S:
        raise AudioProcessingError(
            f"Audio is too short to analyse ({sound.duration * 1000:.0f} ms; at least "
            f"{MIN_AUDIO_DURATION_S * 1000:.0f} ms is needed): {path}"
        )
    return sound


def _checked_max_duration(max_duration_s: float | None) -> float | None:
    """Validate a ``max_duration_s`` argument (``None`` means no limit)."""
    if max_duration_s is None:
        return None
    if not (0 < max_duration_s < math.inf):
        raise ValueError(
            f"max_duration_s must be a positive number or None, got {max_duration_s!r}"
        )
    return float(max_duration_s)


def _runs(mask: _BoolArray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index ranges where *mask* is true."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _remove_octave_jumps(f0: _FloatArray) -> _FloatArray:
    """Correct pitch-tracker octave errors in *f0* (NaN where unvoiced).

    Each voiced sample is compared with the running median of the voiced
    samples around it. Samples off by more than :data:`_OCTAVE_JUMP_RATIO`
    are halved or doubled; if that does not bring them back they are dropped.
    """
    voiced = np.flatnonzero(~np.isnan(f0))
    if voiced.size < 3:
        return f0
    values = f0[voiced]
    k = _F0_MEDIAN_HALF_WINDOW
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(values, k, mode="edge"), 2 * k + 1)
    reference = np.median(windows, axis=1)

    corrected = values.copy()
    ratio = values / reference
    corrected[ratio > _OCTAVE_JUMP_RATIO] /= 2.0
    corrected[ratio < 1.0 / _OCTAVE_JUMP_RATIO] *= 2.0
    ratio = corrected / reference
    corrected[(ratio > _OCTAVE_JUMP_RATIO) | (ratio < 1.0 / _OCTAVE_JUMP_RATIO)] = np.nan

    cleaned: _FloatArray = f0.copy()
    cleaned[voiced] = corrected
    return cleaned


def _peak_indices(values: _FloatArray, min_dip: float) -> list[int]:
    """Indices of peaks separated from neighbouring peaks by dips of at least
    *min_dip* (the start and end of *values* count as dips)."""
    peaks: list[int] = []
    high, high_at = -np.inf, -1
    low = np.inf
    looking_for_peak = True
    for i, v in enumerate(values):
        if v > high:
            high, high_at = v, i
        low = min(low, v)
        if looking_for_peak:
            if v < high - min_dip:
                peaks.append(high_at)
                low = v
                looking_for_peak = False
        elif v > low + min_dip:
            high, high_at = v, i
            looking_for_peak = True
    if looking_for_peak and high_at >= 0:
        peaks.append(high_at)
    return peaks


def _classify_quality(
    jitter: float | None, shimmer: float | None, hnr: float | None
) -> str | None:
    """Derive voice quality label from jitter/shimmer (%) and HNR (dB)."""
    if jitter is None and shimmer is None and hnr is None:
        return None

    # Heuristic thresholds based on clinical voice literature.
    if hnr is not None and hnr < 7.0:
        return "breathy"
    if jitter is not None and jitter > 2.0:
        return "creaky"
    if shimmer is not None and shimmer > 12.0:
        return "tense"
    return "modal"


class _AudioAnalysis:
    """Whole-file analyses of one recording, computed once and sliced per span.

    Each analysis is computed on first use, so pause detection alone does
    not pay for glottal pulses, harmonicity or syllable detection.
    """

    def __init__(self, sound: parselmouth.Sound) -> None:
        self.sound = sound
        self.duration_s = float(sound.duration)
        self.duration_ms = int(round(self.duration_s * 1000))

    @classmethod
    def from_path(
        cls, audio_path: str | Path, max_duration_s: float | None = None
    ) -> _AudioAnalysis:
        return cls(_load_sound(audio_path, max_duration_s))

    # -- Silence ------------------------------------------------------------

    @cached_property
    def _frames(self) -> tuple[_FloatArray, _FloatArray, _BoolArray]:
        """Frame start times (s), levels (dB) and digital-silence flags for
        consecutive :data:`FRAME_STEP_S` frames of the waveform."""
        samples = np.asarray(self.sound.values[0], dtype=np.float64)
        rate = float(self.sound.sampling_frequency)
        hop = rate * FRAME_STEP_S
        n_frames = max(1, int(len(samples) // hop))
        starts = np.round(np.arange(n_frames) * hop).astype(np.int64)
        counts = np.diff(np.append(starts, len(samples))).astype(np.float64)
        mean = np.add.reduceat(samples, starts) / counts
        power = np.maximum(np.add.reduceat(samples * samples, starts) / counts - mean**2, 0.0)
        digital_silence = power < _DIGITAL_SILENCE_POWER
        levels = np.full(n_frames, _DIGITAL_SILENCE_DB)
        levels[~digital_silence] = 10.0 * np.log10(power[~digital_silence] / _REFERENCE_POWER)
        return starts / rate, levels, digital_silence

    def silent_frames(
        self,
        silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
        absolute_threshold_db: float | None = None,
    ) -> _BoolArray:
        """Flag the silent frames of the recording.

        A frame is silent when it is digital silence, or when its level is
        more than *silence_threshold_db* below the speech level (99th
        percentile of frame levels). In noisy recordings, whose noise floor
        (5th percentile) lies above that threshold, unvoiced frames within
        6 dB of the floor are silent too, provided they are at least 10 dB
        below the speech level. With *absolute_threshold_db*, frames below
        that level (dB, Praat's intensity scale) are silent instead.
        """
        _, levels, digital_silence = self._frames
        silent: _BoolArray = digital_silence.copy()
        if absolute_threshold_db is not None:
            silent |= levels < absolute_threshold_db
        elif not silent.all():
            speech = float(np.percentile(levels[~silent], _SPEECH_LEVEL_PERCENTILE))
            floor = float(np.percentile(levels, _NOISE_FLOOR_PERCENTILE))
            silent |= levels < speech - silence_threshold_db
            noise = min(floor + _NOISE_MARGIN_DB, speech - _MIN_SPEECH_MARGIN_DB)
            silent |= (levels < noise) & ~self._frame_voicing

        # Brief sounds inside silence (clicks, lip smacks) do not end a pause.
        min_frames = int(round(_MIN_SOUNDING_MS / 1000 / FRAME_STEP_S))
        for start, end in _runs(~silent):
            if end - start < min_frames and (start > 0 or end < len(silent)):
                silent[start:end] = True
        return silent

    @cached_property
    def _silent(self) -> _BoolArray:
        return self.silent_frames()

    def _frame_ms(self, index: int) -> int:
        """Start of frame *index* in ms (the end of the audio past the last frame)."""
        starts = self._frames[0]
        if index >= len(starts):
            return self.duration_ms
        return int(round(starts[index] * 1000))

    def _silent_at(self, times: _FloatArray) -> _BoolArray:
        """Whether the silence frame containing each time in *times* is silent."""
        starts = self._frames[0]
        index = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(starts) - 1)
        silent: _BoolArray = self._silent[index]
        return silent

    def pauses(
        self,
        min_pause_ms: int = DEFAULT_MIN_PAUSE_MS,
        silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
        absolute_threshold_db: float | None = None,
    ) -> list[PauseInterval]:
        """Silent stretches of at least *min_pause_ms*, in time order."""
        if silence_threshold_db == DEFAULT_SILENCE_THRESHOLD_DB and absolute_threshold_db is None:
            silent = self._silent
        else:
            silent = self.silent_frames(silence_threshold_db, absolute_threshold_db)
        pauses: list[PauseInterval] = []
        for start, end in _runs(silent):
            start_ms, end_ms = self._frame_ms(start), self._frame_ms(end)
            if end_ms - start_ms >= min_pause_ms:
                pauses.append(PauseInterval(start_ms=start_ms, end_ms=end_ms))
        return pauses

    @cached_property
    def _default_pauses(self) -> tuple[_FloatArray, _FloatArray]:
        """Starts and ends (s) of the pauses found with the default settings."""
        pauses = self.pauses()
        return (
            np.array([p.start_ms / 1000 for p in pauses], dtype=np.float64),
            np.array([p.end_ms / 1000 for p in pauses], dtype=np.float64),
        )

    # -- Pitch ----------------------------------------------------------------

    @cached_property
    def _raw_pitch(self) -> tuple[_FloatArray, _FloatArray]:
        """Frame times (s) and F0 (Hz, NaN where unvoiced) as Praat tracks it."""
        pitch = _run_praat("pitch", lambda: self.sound.to_pitch_ac(
            time_step=FRAME_STEP_S, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ,
        ))
        times = np.asarray(pitch.xs(), dtype=np.float64)
        f0 = np.array(pitch.selected_array["frequency"], dtype=np.float64)
        f0[f0 <= 0] = np.nan
        return times, f0

    @cached_property
    def _frame_voicing(self) -> _BoolArray:
        """Whether Praat found voicing in each :data:`FRAME_STEP_S` frame."""
        starts = self._frames[0]
        times, f0 = self._raw_pitch
        if times.size == 0:
            return np.zeros(len(starts), dtype=np.bool_)
        centres = starts + FRAME_STEP_S / 2
        after = np.clip(np.searchsorted(times, centres), 0, times.size - 1)
        before = np.clip(after - 1, 0, times.size - 1)
        nearest = np.where(
            np.abs(times[before] - centres) <= np.abs(times[after] - centres), before, after
        )
        voiced: _BoolArray = ~np.isnan(f0[nearest]) & (
            np.abs(times[nearest] - centres) <= FRAME_STEP_S
        )
        return voiced

    @cached_property
    def _pitch(self) -> tuple[_FloatArray, _FloatArray]:
        """Frame times (s) and cleaned F0 (Hz, NaN where unvoiced)."""
        times, f0 = self._raw_pitch
        f0 = f0.copy()
        # "Voicing" inside silence is a tracking artefact or distant background.
        f0[self._silent_at(times)] = np.nan
        return times, _remove_octave_jumps(f0)

    def voiced_seconds(self, start_s: float = 0.0, end_s: float | None = None) -> float:
        """Voiced time (s) between *start_s* and *end_s* (default: whole file)."""
        times, f0 = self._pitch
        end = self.duration_s if end_s is None else end_s
        i0, i1 = np.searchsorted(times, [start_s, end])
        return float(np.count_nonzero(~np.isnan(f0[i0:i1]))) * FRAME_STEP_S

    @property
    def has_speech(self) -> bool:
        """Whether the recording contains any voiced sound."""
        return self.voiced_seconds() >= _MIN_VOICED_S

    def voiced_regions(self, min_pause_ms: int = DEFAULT_MIN_PAUSE_MS) -> list[tuple[int, int]]:
        """Stretches of voiced sound between pauses, as ``(start_ms, end_ms)``.

        Each region runs from its first to its last sounding frame; regions
        with less than 50 ms of voicing (noise, clicks, breaths) are left out.
        """
        starts_ms = np.round(self._frames[0] * 1000)
        bounds: list[tuple[int, int]] = []
        cursor = 0
        for pause in self.pauses(min_pause_ms):
            if pause.start_ms > cursor:
                bounds.append((cursor, pause.start_ms))
            cursor = pause.end_ms
        if cursor < self.duration_ms:
            bounds.append((cursor, self.duration_ms))

        regions: list[tuple[int, int]] = []
        for start_ms, end_ms in bounds:
            i0, i1 = np.searchsorted(starts_ms, [start_ms, end_ms])
            sounding = np.flatnonzero(~self._silent[i0:i1])
            if sounding.size == 0:
                continue
            region = (self._frame_ms(i0 + sounding[0]), self._frame_ms(i0 + sounding[-1] + 1))
            if self.voiced_seconds(region[0] / 1000, region[1] / 1000) >= _MIN_VOICED_S:
                regions.append(region)
        return regions

    # -- Intensity and syllables ------------------------------------------------

    @cached_property
    def _intensity(self) -> tuple[_FloatArray, _FloatArray, _BoolArray]:
        """Frame times (s), intensity (dB) and which frames are not silent."""
        intensity = _run_praat("intensity", lambda: self.sound.to_intensity(
            minimum_pitch=INTENSITY_MIN_PITCH_HZ, time_step=FRAME_STEP_S, subtract_mean=True,
        ))
        times = np.asarray(intensity.xs(), dtype=np.float64)
        values = np.asarray(intensity.values[0], dtype=np.float64)
        sounding = ~self._silent_at(times) & np.isfinite(values) & (values > _DIGITAL_SILENCE_DB)
        return times, values, sounding

    @cached_property
    def _nuclei(self) -> _FloatArray:
        """Times (s) of syllable nuclei: voiced, prominent intensity peaks."""
        times, values, sounding = self._intensity
        if times.size == 0:
            return np.zeros(0)
        contour = np.where(sounding, values, _DIGITAL_SILENCE_DB)
        pitch_times, f0 = self._pitch
        nuclei: list[float] = []
        for i in _peak_indices(contour, _NUCLEUS_MIN_DIP_DB):
            if not sounding[i] or pitch_times.size == 0:
                continue
            j = int(np.argmin(np.abs(pitch_times - times[i])))
            if abs(pitch_times[j] - times[i]) <= FRAME_STEP_S and not np.isnan(f0[j]):
                nuclei.append(float(times[i]))
        return np.array(nuclei, dtype=np.float64)

    def _speech_rate(self, start_s: float, end_s: float) -> float | None:
        """Syllables per second of speaking time around a span.

        Nuclei are counted in a window centred on the span that is widened
        (up to :data:`_MAX_RATE_WINDOW_S`) until it holds
        :data:`_RATE_WINDOW_S` of speech, and divided by the window's
        duration minus its pauses. ``None`` when the span has no voicing or
        the window holds too little speech to estimate a rate.
        """
        if self.voiced_seconds(start_s, end_s) <= 0.0:
            return None
        pause_starts, pause_ends = self._default_pauses
        centre = (start_s + end_s) / 2
        half = max((end_s - start_s) / 2, _RATE_WINDOW_S / 2)
        while True:
            lo, hi = max(0.0, centre - half), min(self.duration_s, centre + half)
            paused = np.clip(np.minimum(pause_ends, hi) - np.maximum(pause_starts, lo), 0.0, None)
            speaking = (hi - lo) - float(paused.sum())
            whole_file = lo <= 0.0 and hi >= self.duration_s
            if speaking >= _RATE_WINDOW_S or whole_file or 2 * half >= _MAX_RATE_WINDOW_S:
                break
            half += FRAME_STEP_S * 5
        if speaking < _MIN_RATE_SPEAKING_S:
            return None
        nuclei = self._nuclei
        count = int(np.count_nonzero((nuclei >= lo) & (nuclei < hi)))
        return count / speaking

    # -- Voice quality ------------------------------------------------------------

    @cached_property
    def _point_process(self) -> object:
        return _run_praat("glottal pulse", lambda: call(
            self.sound, "To PointProcess (periodic, cc)", PITCH_FLOOR_HZ, PITCH_CEILING_HZ
        ))

    @cached_property
    def _harmonicity(self) -> object:
        return _run_praat("harmonicity", lambda: call(
            self.sound, "To Harmonicity (cc)", FRAME_STEP_S, PITCH_FLOOR_HZ, 0.1, 1.0
        ))

    def _voice_quality(
        self, start_s: float, end_s: float
    ) -> tuple[float | None, float | None, float | None]:
        """Local jitter (%), local shimmer (%) and mean HNR (dB) over a span."""
        points = self._point_process
        jitter = _query(lambda: call(
            points, "Get jitter (local)", start_s, end_s, 0.0001, 0.02, 1.3
        ))
        shimmer = _query(lambda: call(
            [self.sound, points], "Get shimmer (local)", start_s, end_s, 0.0001, 0.02, 1.3, 1.6
        ))
        hnr = _query(lambda: call(self._harmonicity, "Get mean", start_s, end_s))
        return (
            None if jitter is None else jitter * 100.0,
            None if shimmer is None else shimmer * 100.0,
            hnr,
        )

    # -- Spans --------------------------------------------------------------------

    def span_features(self, alignment: WordAlignment) -> SpanFeatures:
        """Measure the features of one aligned span."""
        start_s = max(0.0, alignment.start_ms / 1000.0)
        end_s = min(self.duration_s, alignment.end_ms / 1000.0)
        if end_s <= start_s:
            return SpanFeatures(
                start_ms=alignment.start_ms, end_ms=alignment.end_ms, text=alignment.word
            )

        f0_mean: float | None = None
        f0_range: tuple[float, float] | None = None
        f0_contour: list[float] | None = None
        pitch_times, f0 = self._pitch
        i0, i1 = np.searchsorted(pitch_times, [start_s, end_s])
        voiced = f0[i0:i1][~np.isnan(f0[i0:i1])]
        if voiced.size:
            f0_mean = float(voiced.mean())
            f0_range = (float(voiced.min()), float(voiced.max()))
            f0_contour = [round(float(v), 1) for v in voiced]

        intensity_mean: float | None = None
        intensity_range: float | None = None
        times, values, sounding = self._intensity
        i0, i1 = np.searchsorted(times, [start_s, end_s])
        levels = values[i0:i1][sounding[i0:i1]]
        if levels.size:
            # Average the power, not the decibels.
            intensity_mean = float(10.0 * np.log10(np.mean(10.0 ** (levels / 10.0))))
            intensity_range = float(levels.max() - levels.min())

        # Jitter, shimmer and HNR describe phonation, so unvoiced spans have none.
        jitter = shimmer = hnr = None
        if voiced.size:
            jitter, shimmer, hnr = self._voice_quality(start_s, end_s)
        return SpanFeatures(
            start_ms=alignment.start_ms,
            end_ms=alignment.end_ms,
            text=alignment.word,
            f0_mean=f0_mean,
            f0_range=f0_range,
            f0_contour=f0_contour,
            intensity_mean=intensity_mean,
            intensity_range=intensity_range,
            speech_rate=self._speech_rate(start_s, end_s),
            jitter=jitter,
            shimmer=shimmer,
            hnr=hnr,
            quality=_classify_quality(jitter, shimmer, hnr),
        )

    def features(self, alignments: Sequence[WordAlignment]) -> list[SpanFeatures]:
        return [self.span_features(alignment) for alignment in alignments]

    def samples(self, sampling_frequency: float) -> npt.NDArray[np.float32]:
        """The waveform resampled to *sampling_frequency*, as float32."""
        sound = self.sound
        if sound.sampling_frequency != sampling_frequency:
            sound = _run_praat("resampling", lambda: self.sound.resample(sampling_frequency))
        return np.asarray(sound.values[0], dtype=np.float32)


def detect_pauses(
    sound: parselmouth.Sound,
    min_pause_ms: int = DEFAULT_MIN_PAUSE_MS,
    rms_threshold_db: float | None = None,
    silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
) -> list[PauseInterval]:
    """Detect silence gaps in *sound*.

    Returns a :class:`PauseInterval` for every silent stretch of at least
    *min_pause_ms* milliseconds, including silence at the start and end.
    Frames more than *silence_threshold_db* below the recording's speech
    level are silent (see :meth:`ProsodyAnalyzer.detect_pauses`). Passing
    *rms_threshold_db* uses that absolute level (dB on Praat's intensity
    scale) as the threshold instead.
    """
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    return _AudioAnalysis(sound).pauses(min_pause_ms, silence_threshold_db, rms_threshold_db)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ProsodyAnalyzer:
    """Analyze acoustic prosody from audio given word-level alignments.

    Audio is read with Praat (WAV, AIFF, FLAC, MP3); other formats such as
    OGG/Opus, WebM and M4A are decoded with ffmpeg when it is on ``PATH``
    (as is MP3 then, which ffmpeg reads more robustly). Every failure to
    read or analyse the audio raises
    :class:`~prosody_protocol.exceptions.AudioProcessingError`.

    Parameters
    ----------
    max_duration_s:
        Optional limit on the length of the audio, in seconds. Longer audio
        is rejected with :class:`~prosody_protocol.exceptions.AudioProcessingError`
        before it is loaded (a small compressed file can decode to hours of
        audio and gigabytes of memory). ``None`` (default) means no limit.
    """

    def __init__(self, *, max_duration_s: float | None = None) -> None:
        self.max_duration_s = _checked_max_duration(max_duration_s)

    def analyze(
        self, audio_path: str | Path, alignments: Sequence[WordAlignment]
    ) -> list[SpanFeatures]:
        """Extract prosodic features for each aligned span.

        Parameters
        ----------
        audio_path:
            Path to the audio file.
        alignments:
            Word-level time boundaries, e.g. from an STT engine.

        Returns
        -------
        list[SpanFeatures]
            One :class:`SpanFeatures` per alignment entry, in the same order
            and with the same ``start_ms``/``end_ms``. Units: ``f0_mean``,
            ``f0_range`` and ``f0_contour`` in Hz (the contour lists the
            voiced F0 samples of the span every 10 ms, with octave jumps
            removed); ``intensity_mean`` and ``intensity_range`` in dB over
            the non-silent part of the span; ``speech_rate`` in syllables
            per second of speaking time in a window of at least one second
            around the span (``None`` without voicing); ``jitter`` and
            ``shimmer`` in percent; ``hnr`` in dB; ``quality`` one of
            ``modal``, ``breathy``, ``creaky``, ``tense``. Features that
            cannot be measured (e.g. F0 of an unvoiced span, or a span
            outside the audio) are ``None``.
        """
        return _AudioAnalysis.from_path(audio_path, self.max_duration_s).features(alignments)

    def detect_pauses(
        self,
        audio_path: str | Path,
        min_pause_ms: int = DEFAULT_MIN_PAUSE_MS,
        silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    ) -> list[PauseInterval]:
        """Detect silent pauses in the audio.

        Silence is relative to the recording: a 10 ms frame is silent when
        its level is more than *silence_threshold_db* below the speech level
        (99th percentile of frame levels), or, in noisy recordings, when it
        is unvoiced and within 6 dB of the noise floor. Exact digital
        silence is always silent. Clicks shorter than 30 ms do not interrupt
        a pause.

        Returns
        -------
        list[PauseInterval]
            Silent stretches of at least *min_pause_ms* milliseconds,
            sorted and non-overlapping, including silence at the start and
            end of the file.
        """
        analysis = _AudioAnalysis.from_path(audio_path, self.max_duration_s)
        return analysis.pauses(min_pause_ms, silence_threshold_db)
