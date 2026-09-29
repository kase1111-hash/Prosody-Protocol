"""ProsodyAnalyzer -- extract acoustic features from audio.

Uses parselmouth (Praat) and numpy to measure F0, intensity,
speech rate, jitter, shimmer, HNR, and voice quality.
Spec reference: Section 4 (extended attributes).

Each recording is analysed once: the pitch track, intensity contour,
glottal pulses and harmonicity are computed for the whole file and then
sliced per word, so the cost grows linearly with the length of the audio.

Audio is analysed as mono at no more than 16 kHz, whatever its sample rate
and channels (see MAX_SAMPLE_RATE_HZ).

Silence is judged relative to the speech around it: a frame is silent when
it is far below the loudest voiced sound near it (or, away from speech,
below the file's speech level; or close to the noise floor), and frames of
exact digital silence are always silent. Intensity statistics ignore silent
frames.

Pitch is tracked in two passes (after Hirst 2011): a first pass over a wide
range (40-800 Hz) finds the speaker's median F0, and the second searches
from one octave below that median to 1.5 octaves above it. A fixed range
misses low voices: below its 75 Hz floor Praat finds no voicing, or an
octave (or more) too high. Sustained, strongly voiced stretches of the
first pass that lie outside the fitted range -- a shout or a surprised rise
far above the speaker's median, creak far below it -- are kept from the
first pass, and their voice quality is measured in a range that takes them
in.

Voice quality labels are relative to the speaker: each span is compared
with the rest of the same recording (see :meth:`ProsodyAnalyzer.analyze`).
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

from ._install import install_hint

try:
    import numpy as np
    import numpy.typing as npt
    import parselmouth
    from parselmouth.praat import call
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "Audio analysis requires numpy and praat-parselmouth. "
        "Install with: " + install_hint("audio")
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

# Pitch search range (Hz) for F0, glottal pulses and harmonicity when the
# recording has too little voicing to find the speaker's own range (Praat's
# defaults).
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 600.0

# The first pitch pass searches this wide range for the voice's F0; the
# second searches from _RANGE_OCTAVES_BELOW octaves below the median of the
# first to _RANGE_OCTAVES_ABOVE above it, widened to take in the 10th and
# 90th percentiles (_RANGE_PERCENTILES) with a margin
# (_RANGE_QUANTILE_MARGIN, as a ratio) -- when two speakers with different
# voices share a recording, the median can be one speaker's -- and kept
# within _MIN_PITCH_FLOOR_HZ and _MAX_PITCH_CEILING_HZ. (The 5th and 95th
# percentiles would let in the spurious high "voicing" a pitch tracker finds
# in fricatives, a few percent of the frames of a low voice.) It needs at
# least _MIN_RANGE_FRAMES voiced frames that are not silent.
_FIRST_PASS_FLOOR_HZ = 40.0
_FIRST_PASS_CEILING_HZ = 800.0
_RANGE_OCTAVES_BELOW = 1.0
_RANGE_OCTAVES_ABOVE = 1.5
_RANGE_PERCENTILES = (10.0, 90.0)
_RANGE_QUANTILE_MARGIN = 1.5
_MIN_PITCH_FLOOR_HZ = 40.0
_MAX_PITCH_CEILING_HZ = 1200.0
_MIN_RANGE_FRAMES = 10

# Excursions: the fitted range leaves out what a speaker rarely does, such as
# a shouted word three times the median F0 or a creaky word far below it.
# A stretch of first-pass voicing (consecutive voiced frames whose F0 changes
# by less than _EXCURSION_MAX_STEP from one frame to the next) that goes
# outside the fitted range is kept from the first pass when it lasts at least
# _MIN_EXCURSION_FRAMES frames and its median voicing strength (Praat's
# normalised autocorrelation) is at least _MIN_EXCURSION_STRENGTH. The
# spurious "voicing" a pitch tracker finds in the fricatives of a low voice
# can be as long and as loud, but it is weak: a strength of 0.45-0.6, where
# a voice has 0.75-1.
_EXCURSION_MAX_STEP = 2.0 ** (3.0 / 12.0)  # 3 semitones per 10 ms frame
_MIN_EXCURSION_FRAMES = 5
_MIN_EXCURSION_STRENGTH = 0.7
# The glottal pulses and harmonicity of a span with an excursion are
# measured on the span and this much audio around it (s), in a range
# widened to take in the excursion.
_EXCURSION_PADDING_S = 0.1

# Audio sampled more slowly than this cannot carry the harmonics that pitch
# tracking and voice quality analysis need (and 10 ms analysis frames need
# many samples each).
MIN_SAMPLE_RATE_HZ = 4000.0

# Audio is analysed as mono at no more than this rate: faster audio is
# resampled to it as it is read (as ffmpeg decodes to it), a few seconds at a
# time, so the memory and time an analysis takes grow with the length of the
# audio alone, not with its sample rate or number of channels (a 29 MB FLAC
# of ten minutes at 192 kHz stereo took 2.8 GB and four minutes). Prosody
# lies well below its 8 kHz Nyquist frequency: F0 is tracked up to 1200 Hz,
# and loudness, syllables, and the glottal pulses that jitter, shimmer and
# HNR are measured on are carried by the harmonics below 5 kHz. Measured on
# the same speech at 44.1/48 kHz and resampled: F0 within 0.1 %, intensity
# within 0.5 dB, speech rate within 1 %, pauses unchanged; HNR 1-3 dB
# higher where the noise above 8 kHz is removed; jitter and shimmer of most
# words within a few tenths and a few points (more where voicing is
# irregular). It is also the rate of speech recognisers such as Whisper.
MAX_SAMPLE_RATE_HZ = 16_000.0
# Such audio is read in blocks of about this many samples (all channels),
# and no shorter than _MIN_READ_BLOCK_S; each block is resampled with
# _RESAMPLE_PADDING_S of audio on either side, so that its edges match
# resampling the whole file to within a few steps of 16-bit audio.
_READ_BLOCK_SAMPLES = 1 << 20
_MIN_READ_BLOCK_S = 1.0
_RESAMPLE_PADDING_S = 0.05

# Hop between analysis frames (pitch, intensity, silence), in seconds.
FRAME_STEP_S = 0.01

# Praat's intensity window is 3.2 / INTENSITY_MIN_PITCH_HZ seconds long.
INTENSITY_MIN_PITCH_HZ = 100.0

# A frame is silent when its level is more than this many dB below the
# speech around it (see _AudioAnalysis.silent_frames), as with the silence
# threshold of Praat's "To TextGrid (silences)".
DEFAULT_SILENCE_THRESHOLD_DB = 25.0

# Silent stretches shorter than this are not reported as pauses. The spec
# treats pauses under 200 ms as ordinary speech rhythm (Section 3.3).
DEFAULT_MIN_PAUSE_MS = 200

# Shorter recordings cannot be pitch-tracked.
MIN_AUDIO_DURATION_S = 0.1

# Praat reads samples as pascals (full scale is 1 Pa, 94 dB SPL). Samples
# beyond 100 000 Pa -- about one atmosphere, 194 dB SPL, the loudest sound
# air can carry -- are not sound but damaged or unscaled data (such as a
# float WAV holding raw 32-bit integers); at about 1e154 their power
# overflows, and every measure comes out empty. Float data on a 16-bit
# integer scale (up to 32768) is still analysed.
MAX_SAMPLE_VALUE = 100_000.0

# Containers ffmpeg may open (demuxer names). Playlists and scripts such as
# HLS and concat are left out: they make ffmpeg read the other files they
# name, so an uploaded playlist could expose any audio on the machine.
_FFMPEG_FORMATS = (
    "ogg", "matroska", "webm", "mov", "mp4", "m4a", "3gp", "3g2", "mj2",
    "mp3", "aac", "flac", "wav", "aiff", "caf", "amr", "au", "w64",
)
# ffmpeg decodes to mono at the analysis rate (see MAX_SAMPLE_RATE_HZ).
_FFMPEG_SAMPLE_RATE = int(MAX_SAMPLE_RATE_HZ)
# Decoding a whole hour of audio takes ffmpeg about ten seconds.
_FFMPEG_TIMEOUT_S = 300.0

_SPEECH_LEVEL_PERCENTILE = 99.0
_NOISE_FLOOR_PERCENTILE = 5.0
# The speech level around a frame is that of the loudest voiced frame within
# this many seconds of it (a syllable or so: the vowels next to a consonant
# or closure, or the words either side of a short pause), never above the
# file's speech level. Voiced frames count when they are no more than the
# silence threshold plus _VOICED_SPEECH_MARGIN_DB below the file's speech
# level: about as far down as Praat's pitch tracker finds voicing at all
# (its silence threshold, 3 % of the peak amplitude, lies about 30 dB down).
# Frames with no such voicing this near are judged against the file's speech
# level. A louder passage elsewhere in the file -- a shout, a laugh, a second
# speaker -- then no longer turns the weak sounds of quieter speech
# (consonants, the onsets and ends of vowels) into silence: false pauses, a
# speech rate counted over too little speaking time, and a level measured
# only on the loudest frames.
_LOCAL_SPEECH_S = 0.25
_VOICED_SPEECH_MARGIN_DB = 5.0
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
# Praat's harmonicity of a silent frame.
_SILENT_HNR_DB = -200.0

# Octave-jump guard: a voiced F0 sample more than this ratio away from the
# running median of its neighbours is halved or doubled when that brings it
# back into range, and dropped otherwise.
_OCTAVE_JUMP_RATIO = 1.6
# Neighbours on each side for that running median (150 ms window).
_F0_MEDIAN_HALF_WINDOW = 7

# Syllable nuclei (de Jong & Wempe 2009) are the peaks of an intensity
# contour with a 64 ms window (minimum pitch 50 Hz, as in their method),
# which does not ripple with the glottal pulses of a low voice ...
_NUCLEUS_INTENSITY_MIN_PITCH_HZ = 50.0
# ... separated from their neighbours by dips of at least this depth ...
_NUCLEUS_MIN_DIP_DB = 2.0
# ... no more than this far below the loudest sound within _NUCLEUS_CONTEXT_S
# on either side (weaker peaks are consonants: nasals, releases) ...
_NUCLEUS_MAX_DROP_DB = 10.0
_NUCLEUS_CONTEXT_S = 0.15
# ... and voiced within this many frames: the loudest moment of a syllable
# can fall just outside its voicing, and the tracker drops a frame here and
# there. Before, a peak had to be voiced in its own frame, which made the
# measured rate depend on the voice's pitch.
_NUCLEUS_VOICING_TOLERANCE_FRAMES = 3
# Speech rate is counted over this much speaking time (pauses excluded)
# around a span, looking no further than _MAX_RATE_WINDOW_S in total; with
# less than _MIN_RATE_SPEAKING_S of speech it is unknown.
_RATE_WINDOW_S = 1.0
_MAX_RATE_WINDOW_S = 4.0
_MIN_RATE_SPEAKING_S = 0.25

# A stretch of audio is speech only if it has at least this much voicing.
_MIN_VOICED_S = 0.05

# Voice quality. Jitter, shimmer and HNR of connected speech vary widely
# between speakers and recordings (shimmer of 10-15 % is ordinary in running
# speech, far above the thresholds for sustained vowels), so a span is
# compared with the speaker's usual voice: the medians over word-sized
# chunks (_QUALITY_CHUNK_S) of the recording's other voiced speech. A span
# needs _MIN_QUALITY_VOICED_FRAMES voiced frames, and the reference at least
# _MIN_QUALITY_REFERENCE_CHUNKS chunks with as many.
_QUALITY_CHUNK_S = 0.3
_MIN_QUALITY_VOICED_FRAMES = 10
_MIN_QUALITY_REFERENCE_CHUNKS = 3
# Deviations from the reference that count: jitter or shimmer this many
# times the usual; HNR this many dB below it; F0 this many semitones below
# the usual (the low pitch of creak).
_QUALITY_PERTURBATION_RATIO = 2.0
_QUALITY_HNR_DROP_DB = 6.0
_QUALITY_CREAK_F0_DROP_ST = 3.0
# The usual values are taken to be at least this jitter and shimmer (%),
# about half the upper limits for healthy sustained vowels, and at most
# this HNR (dB): smaller perturbations and cleaner voices differ only by
# measurement noise (a synthetic voice has jitter near 0 % and HNR near
# 70 dB).
_QUALITY_MIN_USUAL_JITTER = 0.5
_QUALITY_MIN_USUAL_SHIMMER = 2.0
_QUALITY_MAX_USUAL_HNR_DB = 20.0
# A lower HNR only means a breathier voice where the span is well above the
# recording's noise floor; nearer to it, the noise lowers the HNR.
_QUALITY_MIN_SNR_DB = 30.0


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


def _pulses(sound: parselmouth.Sound, floor_hz: float, ceiling_hz: float) -> object:
    """Glottal pulses (a Praat PointProcess) of *sound*, found with a pitch
    search between *floor_hz* and *ceiling_hz*."""
    return call(sound, "To PointProcess (periodic, cc)", floor_hz, ceiling_hz)


def _hnr_track(sound: parselmouth.Sound, floor_hz: float) -> tuple[_FloatArray, _FloatArray]:
    """Frame times (s) and harmonics-to-noise ratio (dB) of *sound*, as
    Praat's "To Harmonicity (cc)" measures it with pitch floor *floor_hz*;
    NaN where Praat finds the audio silent."""
    harmonicity = call(sound, "To Harmonicity (cc)", FRAME_STEP_S, floor_hz, 0.1, 1.0)
    times = np.asarray(harmonicity.xs(), dtype=np.float64)
    hnr = np.array(harmonicity.values[0], dtype=np.float64)
    hnr[hnr == _SILENT_HNR_DB] = np.nan
    return times, hnr


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


def _open_long_sound(path: Path) -> tuple[object, float, float, int] | None:
    """Open *path* as a Praat LongSound, which reads the header and leaves
    the samples on disk: the LongSound, its duration (s), sampling
    frequency (Hz) and number of channels. ``None`` when Praat cannot open
    the file that way (or reads no samples from it)."""
    try:
        long_sound = call("Open long sound file", str(path))
        duration = float(call(long_sound, "Get total duration"))
        rate = float(call(long_sound, "Get sampling frequency"))
        # LongSound has no query for its channels; a few samples tell.
        first = call(long_sound, "Extract part", 0.0, min(duration, 0.001), "yes")
    except parselmouth.PraatError:
        return None
    return long_sound, duration, rate, int(first.n_channels)


def _read_in_blocks(
    long_sound: object, duration_s: float, rate: float, channels: int, source: str
) -> parselmouth.Sound:
    """Read a LongSound as a mono Sound at no more than
    :data:`MAX_SAMPLE_RATE_HZ`, a block at a time.

    Each block is checked with :func:`_check_samples` (before resampling
    can spread a damaged sample), mixed to mono, and resampled with
    :data:`_RESAMPLE_PADDING_S` of audio on either side; its samples are
    then read off at their times in the whole recording. So the samples at
    the native rate are never all in memory, and the result matches reading
    and resampling the whole file to within a few steps of 16-bit audio.
    (Praat's long-sound reader starts FLAC one sample late: 23 us at
    44.1 kHz.)
    """
    target = min(rate, MAX_SAMPLE_RATE_HZ)
    count = max(1, int(round(duration_s * target)))
    out: _FloatArray = np.empty(count, dtype=np.float64)
    step = max(_MIN_READ_BLOCK_S, _READ_BLOCK_SAMPLES / (rate * channels))
    per_block = max(1, int(step * target))  # output samples per block
    for first in range(0, count, per_block):
        last = min(count, first + per_block)
        start, end = first / target, last / target
        part = call(
            long_sound, "Extract part",
            max(0.0, start - _RESAMPLE_PADDING_S), min(duration_s, end + _RESAMPLE_PADDING_S),
            "yes",
        )
        _check_samples(part.values, source)
        if part.n_channels > 1:
            part = part.convert_to_mono()
        if target < rate:
            part = part.resample(target)
        # Sample i of the result lies at (i + 0.5) / target, as in a Sound
        # read from a file; the block's samples lie on that grid, or (at
        # the end of the file) within half a sample of it. Interpolating
        # between them would filter out the upper frequencies.
        values = part.values[0]
        index = np.rint(np.arange(first, last) + 0.5 - part.x1 * target).astype(np.int64)
        out[first:last] = values[np.clip(index, 0, values.size - 1)]
    return parselmouth.Sound(out, sampling_frequency=target)


def _check_duration(path: Path, duration_s: float, max_duration_s: float | None) -> None:
    if max_duration_s is not None and duration_s > max_duration_s:
        raise AudioProcessingError(
            f"Audio is longer than max_duration_s={max_duration_s:g} s: {path}"
        )


def _check_sound(sound: parselmouth.Sound, source: str) -> None:
    """Raise :class:`AudioProcessingError` for audio that cannot be analysed:
    sampled below :data:`MIN_SAMPLE_RATE_HZ`, shorter than
    :data:`MIN_AUDIO_DURATION_S`, or holding samples that are not finite
    numbers (NaN or infinity, as a damaged float WAV can) or are larger
    than :data:`MAX_SAMPLE_VALUE`."""
    rate = float(sound.sampling_frequency)
    if not rate >= MIN_SAMPLE_RATE_HZ:  # also true for NaN
        raise AudioProcessingError(
            f"Audio sampled at {rate:g} Hz cannot be analysed (at least "
            f"{MIN_SAMPLE_RATE_HZ:g} Hz is needed): {source}"
        )
    if sound.duration < MIN_AUDIO_DURATION_S:
        raise AudioProcessingError(
            f"Audio is too short to analyze ({sound.duration * 1000:.0f} ms; at least "
            f"{MIN_AUDIO_DURATION_S * 1000:.0f} ms is needed): {source}"
        )
    _check_samples(sound.values, source)


def _check_samples(values: npt.NDArray[np.float64], source: str) -> None:
    """Raise :class:`AudioProcessingError` when *values* (channels x
    samples) hold samples that are not finite or are larger than
    :data:`MAX_SAMPLE_VALUE`."""
    block = 1 << 20  # check a million samples at a time, not a copy of the whole file
    for start in range(0, values.shape[-1], block):
        samples = values[..., start:start + block]
        if not np.isfinite(samples).all():
            raise AudioProcessingError(
                f"Audio contains samples that are not finite numbers (NaN or infinity): {source}"
            )
        peak = float(np.abs(samples).max())
        if peak > MAX_SAMPLE_VALUE:
            raise AudioProcessingError(
                f"Audio contains samples of {peak:.3g}, far beyond full scale (1.0), which "
                f"cannot be sound (at most {MAX_SAMPLE_VALUE:g} is accepted); the file is "
                f"damaged or holds unscaled data: {source}"
            )


def _load_sound(audio_path: str | Path, max_duration_s: float | None = None) -> parselmouth.Sound:
    """Read *audio_path* as a mono Sound sampled at no more than
    :data:`MAX_SAMPLE_RATE_HZ`.

    Any failure to read it (missing file, directory, empty or non-audio
    file, or audio longer than *max_duration_s* seconds) raises
    :class:`AudioProcessingError`. The length limit is checked from the
    file header, or while decoding, before the whole file is loaded. Audio
    sampled faster than :data:`MAX_SAMPLE_RATE_HZ`, or with more than one
    channel, is resampled and mixed to mono as it is read, a block at a
    time (see :func:`_read_in_blocks`). A truncated file, whose header
    promises more samples than it holds, is decoded by ffmpeg, which reads
    just the samples that are there (Praat would pad it with silence);
    without ffmpeg it is an error. Whether the audio can be analysed is
    checked by :func:`_check_sound`.
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
        sound = _read_with_praat(path, max_duration_s)

    _check_duration(path, float(sound.duration), max_duration_s)
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    if sound.sampling_frequency > MAX_SAMPLE_RATE_HZ and sound.duration >= MIN_AUDIO_DURATION_S:
        # Only audio that Praat reads whole but cannot open as a long sound
        # (too short a sound is left for _check_sound to reject).
        loaded = sound
        sound = _run_praat("resampling", lambda: loaded.resample(MAX_SAMPLE_RATE_HZ))
    return sound


def _read_with_praat(path: Path, max_duration_s: float | None) -> parselmouth.Sound:
    """Read *path* with Praat (WAV, AIFF, FLAC, MP3), or with ffmpeg when
    Praat cannot read it or finds it truncated (see :func:`_load_sound`)."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "File too small", parselmouth.PraatWarning)
            # Opening an MP3 as a long sound warns that its times can be off
            # by tens of milliseconds, as they can when it is read whole.
            warnings.filterwarnings("ignore", "Time measurements in MP3", parselmouth.PraatWarning)
            opened = _open_long_sound(path)
            if opened is not None:
                long_sound, duration, rate, channels = opened
                _check_duration(path, duration, max_duration_s)
                if rate > MAX_SAMPLE_RATE_HZ or channels > 1:
                    return _read_in_blocks(long_sound, duration, rate, channels, str(path))
            return parselmouth.Sound(str(path))
    except AudioProcessingError:
        raise
    except parselmouth.PraatWarning:
        # Praat would pad the missing samples with silence; ffmpeg decodes
        # only the samples that are there.
        return _decode_with_ffmpeg(path, "the file is truncated", max_duration_s)
    except Exception as exc:  # parselmouth.PraatError for formats Praat cannot read
        return _decode_with_ffmpeg(path, _praat_message(exc).rstrip("."), max_duration_s)


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


def _remove_octave_jumps(f0: _FloatArray, keep: _BoolArray | None = None) -> _FloatArray:
    """Correct pitch-tracker octave errors in *f0* (NaN where unvoiced).

    Each voiced sample is compared with the running median of the voiced
    samples around it. Samples off by more than :data:`_OCTAVE_JUMP_RATIO`
    are halved or doubled; if that does not bring them back they are dropped.
    Near the start and end the window holds only the samples that exist
    (repeating the edge sample would let a few errors there outvote the rest).
    Samples flagged in *keep* (excursions checked otherwise) are left as they
    are, but still count in the running median of the others.
    """
    voiced = np.flatnonzero(~np.isnan(f0))
    if voiced.size < 3:
        return f0
    values = f0[voiced]
    k = _F0_MEDIAN_HALF_WINDOW
    padded = np.pad(values, k, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1)
    reference = np.nanmedian(windows, axis=1)

    corrected = values.copy()
    ratio = values / reference
    corrected[ratio > _OCTAVE_JUMP_RATIO] /= 2.0
    corrected[ratio < 1.0 / _OCTAVE_JUMP_RATIO] *= 2.0
    ratio = corrected / reference
    corrected[(ratio > _OCTAVE_JUMP_RATIO) | (ratio < 1.0 / _OCTAVE_JUMP_RATIO)] = np.nan
    if keep is not None:
        kept = keep[voiced]
        corrected[kept] = values[kept]

    cleaned: _FloatArray = f0.copy()
    cleaned[voiced] = corrected
    return cleaned


def _nearest(grid: _FloatArray, times: _FloatArray) -> npt.NDArray[np.int64]:
    """Index of the time in the sorted, non-empty *grid* nearest to each of *times*."""
    if grid.size < 2:
        return np.zeros(times.shape, dtype=np.int64)
    after = np.clip(np.searchsorted(grid, times), 1, grid.size - 1)
    before = after - 1
    nearest = np.where(times - grid[before] <= grid[after] - times, before, after)
    return nearest.astype(np.int64)


def _smooth_runs(f0: _FloatArray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index ranges of consecutive voiced samples
    of *f0* (NaN where unvoiced) whose F0 changes by less than
    :data:`_EXCURSION_MAX_STEP` from one sample to the next."""
    voiced = ~np.isnan(f0)
    with np.errstate(invalid="ignore"):
        step = f0[1:] / f0[:-1]
    joined = np.zeros(f0.size, dtype=np.bool_)  # sample i continues the run of i - 1
    joined[1:] = (step < _EXCURSION_MAX_STEP) & (step > 1.0 / _EXCURSION_MAX_STEP)
    starts = np.flatnonzero(voiced & ~joined)
    ends = np.flatnonzero(voiced & ~np.append(joined[1:], False)) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


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
    span: tuple[float, float, float, float],
    usual: tuple[float, float, float, float],
    snr_db: float,
) -> str | None:
    """Voice quality of a span relative to the speaker's usual voice.

    *span* and *usual* hold jitter (%), shimmer (%), HNR (dB) and median F0
    (Hz); *snr_db* is how far the span's level lies above the recording's
    noise floor. The labels are those of spec Section 3.2:

    - ``creaky``: jitter at least twice the usual, with F0 at least 3
      semitones below the usual (irregular and low: vocal fry);
    - ``harsh``: jitter and shimmer both at least twice the usual
      (irregular in period and amplitude: a rough voice), with HNR less
      than 6 dB below the usual;
    - ``breathy``: HNR at least 6 dB below the usual (more aspiration
      noise, which also raises the measured jitter or shimmer, but not
      both twofold), measured at least 30 dB above the noise floor;
    - ``modal``: none of these deviations, the speaker's usual phonation;
    - ``None`` (unsure): a lower HNR with both perturbations raised
      (breathy or harsh), jitter or shimmer raised without either
      pattern, or a lower HNR close to the noise floor, where noise
      lowers it.

    ``tense`` and ``whispery`` are never returned: jitter, shimmer and HNR
    do not show vocal-fold constriction, and whispered speech has no voicing
    to measure them on.
    """
    jitter, shimmer, hnr, f0 = span
    usual_jitter, usual_shimmer, usual_hnr, usual_f0 = usual
    usual_jitter = max(usual_jitter, _QUALITY_MIN_USUAL_JITTER)
    usual_shimmer = max(usual_shimmer, _QUALITY_MIN_USUAL_SHIMMER)
    usual_hnr = min(usual_hnr, _QUALITY_MAX_USUAL_HNR_DB)
    irregular = jitter >= _QUALITY_PERTURBATION_RATIO * usual_jitter
    rough = shimmer >= _QUALITY_PERTURBATION_RATIO * usual_shimmer
    low = f0 > 0.0 and usual_f0 > 0.0 and 12.0 * math.log2(f0 / usual_f0) <= (
        -_QUALITY_CREAK_F0_DROP_ST
    )
    noisy = hnr <= usual_hnr - _QUALITY_HNR_DROP_DB
    if irregular and low:
        return "creaky"
    if irregular and rough:
        return None if noisy else "harsh"
    if noisy:
        return "breathy" if snr_db >= _QUALITY_MIN_SNR_DB else None
    if irregular or rough:
        return None
    return "modal"


def _sliding_max(values: _FloatArray, half_width: int) -> _FloatArray:
    """The largest of *values* within *half_width* samples of each sample.

    Takes one pass per offset rather than a window view, so the memory used
    is two copies of *values* whatever the width.
    """
    result: _FloatArray = np.array(values, dtype=np.float64)
    for shift in range(1, min(half_width, values.size - 1) + 1):
        np.maximum(result[shift:], values[:-shift], out=result[shift:])
        np.maximum(result[:-shift], values[shift:], out=result[:-shift])
    return result


def _mean_level(levels: _FloatArray) -> float:
    """Mean of decibel *levels*, averaged as power."""
    return float(10.0 * np.log10(np.mean(10.0 ** (levels / 10.0))))


class _AudioAnalysis:
    """Whole-file analyses of one recording, computed once and sliced per span.

    Each analysis is computed on first use, so pause detection alone does
    not pay for glottal pulses, harmonicity or syllable detection.
    """

    def __init__(self, sound: parselmouth.Sound, source: str = "audio") -> None:
        _check_sound(sound, source)
        self.sound = sound
        self.duration_s = float(sound.duration)
        self.duration_ms = int(round(self.duration_s * 1000))

    @classmethod
    def from_path(
        cls, audio_path: str | Path, max_duration_s: float | None = None
    ) -> _AudioAnalysis:
        return cls(_load_sound(audio_path, max_duration_s), str(audio_path))

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
        more than *silence_threshold_db* below the speech around it: the
        loudest voiced frame within 0.25 s, where one lies no more than
        *silence_threshold_db* + 5 dB below the file's speech level (99th
        percentile of frame levels), and otherwise the file's speech level.
        The speech around a frame is never taken to be louder than the
        file's. In noisy recordings, whose noise floor (5th percentile)
        lies above that threshold, unvoiced frames within 6 dB of the floor
        are silent too, provided they are at least 10 dB below the file's
        speech level. With *absolute_threshold_db*, frames below that level
        (dB, Praat's intensity scale) are silent instead.
        """
        _, levels, digital_silence = self._frames
        silent: _BoolArray = digital_silence.copy()
        if absolute_threshold_db is not None:
            silent |= levels < absolute_threshold_db
        elif not silent.all():
            speech = float(np.percentile(levels[~silent], _SPEECH_LEVEL_PERCENTILE))
            floor = float(np.percentile(levels, _NOISE_FLOOR_PERCENTILE))
            around = self._speech_around(speech, silence_threshold_db)
            silent |= levels < around - silence_threshold_db
            noise = min(floor + _NOISE_MARGIN_DB, speech - _MIN_SPEECH_MARGIN_DB)
            silent |= (levels < noise) & ~self._frame_voicing

        # Brief sounds inside silence (clicks, lip smacks) do not end a pause.
        min_frames = int(round(_MIN_SOUNDING_MS / 1000 / FRAME_STEP_S))
        for start, end in _runs(~silent):
            if end - start < min_frames and (start > 0 or end < len(silent)):
                silent[start:end] = True
        return silent

    def _speech_around(self, speech: float, silence_threshold_db: float) -> _FloatArray:
        """The speech level (dB) around each frame (see :meth:`silent_frames`):
        that of the loudest voiced frame within :data:`_LOCAL_SPEECH_S`, of
        those no more than *silence_threshold_db* plus
        :data:`_VOICED_SPEECH_MARGIN_DB` below the file's level *speech*,
        and at most *speech*; *speech* where none is that near."""
        _, levels, digital_silence = self._frames
        speaking = (
            self._frame_voicing
            & ~digital_silence
            & (levels >= speech - silence_threshold_db - _VOICED_SPEECH_MARGIN_DB)
        )
        nearby = _sliding_max(
            np.where(speaking, levels, -np.inf), int(round(_LOCAL_SPEECH_S / FRAME_STEP_S))
        )
        around: _FloatArray = np.where(np.isfinite(nearby), np.minimum(nearby, speech), speech)
        return around

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

    def _track_pitch(
        self, floor_hz: float, ceiling_hz: float
    ) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
        """Frame times (s), F0 (Hz, NaN where unvoiced) and voicing strength
        (0-1) as Praat tracks them between *floor_hz* and *ceiling_hz*."""
        pitch = _run_praat("pitch", lambda: self.sound.to_pitch_ac(
            time_step=FRAME_STEP_S, pitch_floor=floor_hz, pitch_ceiling=ceiling_hz,
        ))
        times = np.asarray(pitch.xs(), dtype=np.float64)
        f0 = np.array(pitch.selected_array["frequency"], dtype=np.float64)
        strength = np.array(pitch.selected_array["strength"], dtype=np.float64)
        f0[f0 <= 0] = np.nan
        return times, f0, strength

    @cached_property
    def _first_pass(self) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
        """The wide first pitch pass (see :meth:`_track_pitch`)."""
        return self._track_pitch(_FIRST_PASS_FLOOR_HZ, _FIRST_PASS_CEILING_HZ)

    @cached_property
    def pitch_range(self) -> tuple[float, float]:
        """Pitch floor and ceiling (Hz) for the voice or voices in the recording.

        A first pass over 40-800 Hz tracks the F0 of the frames that are
        voiced and within 25 dB of the speech level. The range runs from
        one octave below their median to 1.5 octaves above it, widened to
        take in two thirds of their 10th percentile and 1.5 times their
        90th (so a second voice with a tenth of the voicing is covered),
        within 40-1200 Hz. Without enough voicing it is Praat's default,
        75-600 Hz. Excursions outside it are recovered separately (see
        :attr:`_tracked_pitch`).
        """
        times, f0, _ = self._first_pass
        starts, levels, digital_silence = self._frames
        if times.size == 0 or digital_silence.all():
            return PITCH_FLOOR_HZ, PITCH_CEILING_HZ
        speech = float(np.percentile(levels[~digital_silence], _SPEECH_LEVEL_PERCENTILE))
        loud = ~digital_silence & (levels >= speech - DEFAULT_SILENCE_THRESHOLD_DB)
        index = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(starts) - 1)
        voiced = f0[~np.isnan(f0) & loud[index]]
        if voiced.size < _MIN_RANGE_FRAMES:
            return PITCH_FLOOR_HZ, PITCH_CEILING_HZ
        percentiles = [_RANGE_PERCENTILES[0], 50.0, _RANGE_PERCENTILES[1]]
        low, median, high = (float(q) for q in np.percentile(voiced, percentiles))
        floor = min(median * 2.0**-_RANGE_OCTAVES_BELOW, low / _RANGE_QUANTILE_MARGIN)
        ceiling = max(median * 2.0**_RANGE_OCTAVES_ABOVE, high * _RANGE_QUANTILE_MARGIN)
        return max(_MIN_PITCH_FLOOR_HZ, floor), min(_MAX_PITCH_CEILING_HZ, ceiling)

    @cached_property
    def _tracked_pitch(self) -> tuple[_FloatArray, _FloatArray, _BoolArray]:
        """Frame times (s), F0 (Hz, NaN where unvoiced) and which frames are
        excursions.

        F0 is tracked within the speaker's :attr:`pitch_range`. Where a
        sustained, strongly voiced stretch of the first pass goes outside
        that range (see :data:`_MIN_EXCURSION_STRENGTH`), its frames outside
        the range, and those where the second pass is voiced at a different
        pitch (typically half or double), take the first pass's F0: the
        second pass can only find a submultiple of a shout above its
        ceiling, and nothing, or a multiple, in creak below its floor.
        """
        times, f0, _ = self._track_pitch(*self.pitch_range)
        excursion = np.zeros(times.size, dtype=np.bool_)
        first_times, first_f0, strength = self._first_pass
        if times.size == 0 or first_times.size == 0:
            return times, f0, excursion
        floor, ceiling = self.pitch_range
        for i0, i1 in _smooth_runs(first_f0):
            run_times, run_f0 = first_times[i0:i1], first_f0[i0:i1]
            if (
                i1 - i0 < _MIN_EXCURSION_FRAMES
                or not np.any((run_f0 < floor) | (run_f0 > ceiling))
                or float(np.median(strength[i0:i1])) < _MIN_EXCURSION_STRENGTH
            ):
                continue
            # The passes share a frame step, but their windows differ in
            # length, so their frames may be centred half a step apart: the
            # second pass's frames within the run take the run's F0 there.
            j0, j1 = np.searchsorted(times, [run_times[0] - 1e-9, run_times[-1] + 1e-9])
            values = np.interp(times[j0:j1], run_times, run_f0)
            with np.errstate(invalid="ignore"):
                ratio = f0[j0:j1] / values
            take = (
                (values < floor) | (values > ceiling)
                | (ratio > _EXCURSION_MAX_STEP) | (ratio < 1.0 / _EXCURSION_MAX_STEP)
            )
            f0[j0:j1][take] = values[take]
            excursion[j0:j1][take] = True
        return times, f0, excursion

    @cached_property
    def _raw_pitch(self) -> tuple[_FloatArray, _FloatArray]:
        """Frame times (s) and F0 (Hz, NaN where unvoiced) before octave
        cleaning (see :attr:`_tracked_pitch`)."""
        times, f0, _ = self._tracked_pitch
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
        times, f0, excursion = self._tracked_pitch
        f0 = f0.copy()
        # "Voicing" inside silence is a tracking artefact or distant background.
        f0[self._silent_at(times)] = np.nan
        return times, _remove_octave_jumps(f0, keep=excursion)

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

    def _intensity_contour(
        self, minimum_pitch_hz: float
    ) -> tuple[_FloatArray, _FloatArray, _BoolArray]:
        """Frame times (s), intensity (dB) and which frames are not silent,
        with Praat's window of 3.2 / *minimum_pitch_hz* seconds."""
        intensity = _run_praat("intensity", lambda: self.sound.to_intensity(
            minimum_pitch=minimum_pitch_hz, time_step=FRAME_STEP_S, subtract_mean=True,
        ))
        times = np.asarray(intensity.xs(), dtype=np.float64)
        values = np.asarray(intensity.values[0], dtype=np.float64)
        sounding = ~self._silent_at(times) & np.isfinite(values) & (values > _DIGITAL_SILENCE_DB)
        return times, values, sounding

    @cached_property
    def _intensity(self) -> tuple[_FloatArray, _FloatArray, _BoolArray]:
        """The intensity contour that spans are measured on."""
        return self._intensity_contour(INTENSITY_MIN_PITCH_HZ)

    @cached_property
    def _nuclei(self) -> _FloatArray:
        """Times (s) of syllable nuclei.

        Nuclei are the peaks of a smooth intensity contour (see
        :data:`_NUCLEUS_INTENSITY_MIN_PITCH_HZ`) that are separated from
        their neighbours by dips of :data:`_NUCLEUS_MIN_DIP_DB`, lie within
        :data:`_NUCLEUS_MAX_DROP_DB` of the loudest sound around them, and
        have voicing within :data:`_NUCLEUS_VOICING_TOLERANCE_FRAMES` frames.
        """
        times, values, sounding = self._intensity_contour(_NUCLEUS_INTENSITY_MIN_PITCH_HZ)
        pitch_times, f0 = self._raw_pitch
        if times.size == 0 or pitch_times.size == 0:
            return np.zeros(0)
        # Peaks are found on the whole contour, silence included: masking
        # silent frames would turn every flicker of the silence decision
        # inside a quiet word into a dip, and so into an extra syllable.
        contour = np.where(np.isfinite(values), values, _DIGITAL_SILENCE_DB)
        # The loudest sound within the context of each frame.
        context = int(round(_NUCLEUS_CONTEXT_S / FRAME_STEP_S))
        padded = np.pad(contour, context, constant_values=_DIGITAL_SILENCE_DB)
        loudest = np.lib.stride_tricks.sliding_window_view(padded, 2 * context + 1).max(axis=1)
        # Voicing as tracked, before octave-jump cleaning (a frame whose F0
        # is dropped as an octave error is still voiced), widened by the
        # tolerance on each side.
        voiced = ~np.isnan(f0) & ~self._silent_at(pitch_times)
        width = 2 * _NUCLEUS_VOICING_TOLERANCE_FRAMES + 1
        near_voicing = np.convolve(voiced.astype(np.int64), np.ones(width, np.int64), "same") > 0
        nuclei: list[float] = []
        for i in _peak_indices(contour, _NUCLEUS_MIN_DIP_DB):
            if not sounding[i] or contour[i] < loudest[i] - _NUCLEUS_MAX_DROP_DB:
                continue
            j = int(np.clip(np.searchsorted(pitch_times, times[i]), 0, pitch_times.size - 1))
            if j > 0 and abs(pitch_times[j - 1] - times[i]) < abs(pitch_times[j] - times[i]):
                j -= 1
            if abs(pitch_times[j] - times[i]) <= FRAME_STEP_S and near_voicing[j]:
                nuclei.append(float(times[i]))
        return np.array(nuclei, dtype=np.float64)

    def _speech_rate(self, start_s: float, end_s: float) -> float | None:
        """Syllables per second of speaking time around a span.

        Nuclei are counted in a window centred on the span that is widened
        (up to :data:`_MAX_RATE_WINDOW_S`) until it holds
        :data:`_RATE_WINDOW_S` of speech, and divided by the window's
        duration minus its pauses. ``None`` when the span has no voicing,
        the window holds too little speech to estimate a rate, or no
        syllable nucleus is found in it (voiced speech has at least one).
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
        return count / speaking if count else None

    # -- Voice quality ------------------------------------------------------------

    @cached_property
    def _excursions(self) -> list[tuple[float, float, float, float]]:
        """Each run of excursion frames (see :attr:`_tracked_pitch`): its
        start and end (s), and a pitch floor and ceiling (Hz) that take it
        in -- the fitted range, widened by :data:`_RANGE_QUANTILE_MARGIN`
        beyond the excursion's lowest or highest F0."""
        times, f0, excursion = self._tracked_pitch
        floor, ceiling = self.pitch_range
        found: list[tuple[float, float, float, float]] = []
        for a, b in _runs(excursion):
            low, high = float(np.min(f0[a:b])), float(np.max(f0[a:b]))
            found.append((
                float(times[a]) - FRAME_STEP_S / 2,
                float(times[b - 1]) + FRAME_STEP_S / 2,
                max(_MIN_PITCH_FLOOR_HZ, min(floor, low / _RANGE_QUANTILE_MARGIN)),
                min(_MAX_PITCH_CEILING_HZ, max(ceiling, high * _RANGE_QUANTILE_MARGIN)),
            ))
        return found

    def _around(self, start_s: float, end_s: float) -> parselmouth.Sound:
        """The audio between the times and :data:`_EXCURSION_PADDING_S`
        around them, at its times in the recording."""
        return self.sound.extract_part(
            from_time=max(0.0, start_s - _EXCURSION_PADDING_S),
            to_time=min(self.duration_s, end_s + _EXCURSION_PADDING_S),
            preserve_times=True,
        )

    @cached_property
    def _point_process(self) -> object:
        """Glottal pulses, found within :attr:`pitch_range`, and within each
        excursion in the range that takes it in (the fitted range would find
        every other pulse of a shout, and none in creak below it)."""
        floor, ceiling = self.pitch_range
        points = _run_praat("glottal pulse", lambda: _pulses(self.sound, floor, ceiling))
        for start, end, low, high in self._excursions:
            try:
                part = self._around(start, end)
                local = _pulses(part, low, high)
                call(local, "Remove points between", part.xmin, start)
                call(local, "Remove points between", end, part.xmax)
                call(points, "Remove points between", start, end)
                points = call([points, local], "Union")
            except parselmouth.PraatError:  # pragma: no cover - keep the fitted pulses
                continue
        return points

    @cached_property
    def _hnr_frames(self) -> tuple[_FloatArray, _FloatArray]:
        """Frame times (s) and harmonics-to-noise ratio (dB, NaN where Praat
        finds the audio silent), measured with :attr:`pitch_range`'s floor,
        and within each excursion below it with a floor below the excursion
        (the analysis finds no period longer than its floor's)."""
        floor = self.pitch_range[0]
        times, hnr = _run_praat("harmonicity", lambda: _hnr_track(self.sound, floor))
        for start, end, low, _ in self._excursions:
            if low >= floor:
                continue
            try:
                local_times, local_hnr = _hnr_track(self._around(start, end), low)
            except parselmouth.PraatError:  # pragma: no cover - keep the fitted values
                continue
            inside = np.flatnonzero((times >= start) & (times <= end))
            if local_times.size:
                hnr[inside] = local_hnr[_nearest(local_times, times[inside])]
        return times, hnr

    def _hnr(self, start_s: float, end_s: float) -> float | None:
        """Mean HNR (dB) of the frames between the times that are not silent,
        as Praat's "Get mean" of a Harmonicity; ``None`` without any."""
        times, hnr = self._hnr_frames
        i0 = int(np.searchsorted(times, start_s, side="left"))
        i1 = int(np.searchsorted(times, end_s, side="right"))
        values = hnr[i0:i1]
        values = values[~np.isnan(values)]
        return float(values.mean()) if values.size else None

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
        return (
            None if jitter is None else jitter * 100.0,
            None if shimmer is None else shimmer * 100.0,
            self._hnr(start_s, end_s),
        )

    def _voiced_f0(self, start_s: float, end_s: float) -> _FloatArray:
        """The cleaned F0 samples (Hz) of the voiced frames between the times."""
        times, f0 = self._pitch
        i0, i1 = np.searchsorted(times, [start_s, end_s])
        span = f0[i0:i1]
        voiced: _FloatArray = span[~np.isnan(span)]
        return voiced

    @cached_property
    def _quality_chunks(self) -> _FloatArray:
        """Voice measures of the recording's speech in word-sized chunks.

        One row per chunk of about :data:`_QUALITY_CHUNK_S` of each voiced
        region that has :data:`_MIN_QUALITY_VOICED_FRAMES` voiced frames:
        start and end (s), jitter (%), shimmer (%), HNR (dB), median F0 (Hz).
        """
        rows: list[tuple[float, ...]] = []
        for start_ms, end_ms in self.voiced_regions():
            start_s, end_s = start_ms / 1000, end_ms / 1000
            count = max(1, round((end_s - start_s) / _QUALITY_CHUNK_S))
            edges = np.linspace(start_s, end_s, count + 1)
            for a, b in zip(edges[:-1], edges[1:], strict=True):
                voiced = self._voiced_f0(float(a), float(b))
                if voiced.size < _MIN_QUALITY_VOICED_FRAMES:
                    continue
                jitter, shimmer, hnr = self._voice_quality(float(a), float(b))
                if jitter is not None and shimmer is not None and hnr is not None:
                    rows.append((a, b, jitter, shimmer, hnr, float(np.median(voiced))))
        return np.array(rows, dtype=np.float64).reshape(-1, 6)

    def _usual_voice(
        self, start_s: float, end_s: float
    ) -> tuple[float, float, float, float] | None:
        """The speaker's usual jitter, shimmer, HNR and F0 outside a span:
        medians over the chunks that do not overlap it, or ``None`` when
        fewer than :data:`_MIN_QUALITY_REFERENCE_CHUNKS` do."""
        chunks = self._quality_chunks
        outside = chunks[(chunks[:, 1] <= start_s) | (chunks[:, 0] >= end_s)]
        if len(outside) < _MIN_QUALITY_REFERENCE_CHUNKS:
            return None
        jitter, shimmer, hnr, f0 = np.median(outside[:, 2:], axis=0)
        return float(jitter), float(shimmer), float(hnr), float(f0)

    @cached_property
    def _noise_floor_db(self) -> float:
        """The recording's noise floor: the 5th percentile of frame levels (dB)."""
        return float(np.percentile(self._frames[1], _NOISE_FLOOR_PERCENTILE))

    def _snr_db(self, start_s: float, end_s: float) -> float:
        """How far the span's sounding frames lie above the noise floor (dB)."""
        starts, levels, _ = self._frames
        i0, i1 = np.searchsorted(starts, [start_s, end_s])
        sounding = levels[i0:i1][~self._silent[i0:i1]]
        if sounding.size == 0:
            return 0.0
        return _mean_level(sounding) - self._noise_floor_db

    def _quality(
        self,
        start_s: float,
        end_s: float,
        voiced: _FloatArray,
        measures: tuple[float | None, float | None, float | None],
    ) -> str | None:
        """Voice quality label of a span (see :func:`_classify_quality`), or
        ``None`` without enough voicing or a reference to compare with."""
        jitter, shimmer, hnr = measures
        if voiced.size < _MIN_QUALITY_VOICED_FRAMES or None in measures:
            return None
        usual = self._usual_voice(start_s, end_s)
        if usual is None or jitter is None or shimmer is None or hnr is None:
            return None
        span = (jitter, shimmer, hnr, float(np.median(voiced)))
        return _classify_quality(span, usual, self._snr_db(start_s, end_s))

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
        voiced = self._voiced_f0(start_s, end_s)
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
        quality = self._quality(start_s, end_s, voiced, (jitter, shimmer, hnr))
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
            quality=quality,
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
    Frames more than *silence_threshold_db* below the speech around them
    are silent (see :meth:`ProsodyAnalyzer.detect_pauses`). Passing
    *rms_threshold_db* uses that absolute level (dB on Praat's intensity
    scale) as the threshold instead. Like audio read from a file, *sound*
    is analysed as mono at no more than :data:`MAX_SAMPLE_RATE_HZ`.
    """
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    if sound.sampling_frequency > MAX_SAMPLE_RATE_HZ and sound.duration >= MIN_AUDIO_DURATION_S:
        given = sound
        sound = _run_praat("resampling", lambda: given.resample(MAX_SAMPLE_RATE_HZ))
    return _AudioAnalysis(sound).pauses(min_pause_ms, silence_threshold_db, rms_threshold_db)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ProsodyAnalyzer:
    """Analyze acoustic prosody from audio given word-level alignments.

    Audio is read with Praat (WAV, AIFF, FLAC, MP3); other formats such as
    OGG/Opus, WebM and M4A are decoded with ffmpeg when it is on ``PATH``
    (as is MP3 then, which ffmpeg reads more robustly). It is analysed as
    mono at no more than 16 kHz (:data:`MAX_SAMPLE_RATE_HZ`): channels are
    averaged, and faster audio is resampled as it is read, so the memory
    and time an analysis takes depend on the length of the audio, not on
    its sample rate or channels. Every failure to
    read or analyse the audio raises
    :class:`~prosody_protocol.exceptions.AudioProcessingError`, including
    audio that cannot be analysed: sampled below 4 kHz, shorter than
    100 ms, or holding NaN or infinite samples, or samples beyond
    :data:`MAX_SAMPLE_VALUE` (full scale is 1.0).

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
            and with the same ``start_ms``/``end_ms``; spans are clipped to
            the audio. Units: ``f0_mean``, ``f0_range`` and ``f0_contour``
            in Hz (the contour lists the voiced F0 samples of the span every
            10 ms, tracked within the speaker's range -- one octave below
            the recording's median F0 to 1.5 octaves above it -- or, for
            sustained and strongly voiced stretches outside it such as a
            shout or creak, over 40-800 Hz, with octave jumps removed);
            ``intensity_mean`` and ``intensity_range`` in dB
            over the non-silent part of the span; ``speech_rate`` in
            syllables per second of speaking time in a window of at least
            one second around the span (``None`` without voicing);
            ``jitter`` and ``shimmer`` in percent; ``hnr`` in dB.

            ``quality`` compares the span with the speaker's usual voice in
            the rest of the same recording (the medians over word-sized
            chunks that do not overlap the span): ``creaky`` (jitter at
            least doubled and F0 at least 3 semitones lower), ``harsh``
            (jitter and shimmer at least doubled), ``breathy`` (HNR at least
            6 dB lower, well above the noise floor) or ``modal`` (none of
            these; the speaker's usual phonation, whatever it is). It is
            ``None`` when unsure: with less than 100 ms of voicing in the
            span, fewer than three chunks of voiced speech outside it (so
            always for a span covering most of the recording), or
            deviations that fit none of the labels. ``tense`` and
            ``whispery`` are never measured.

            Features that cannot be measured (e.g. F0 of an unvoiced span,
            or a span outside the audio) are ``None``.
        """
        return _AudioAnalysis.from_path(audio_path, self.max_duration_s).features(alignments)

    def analyze_recording(self, audio_path: str | Path, text: str = "") -> SpanFeatures:
        """Measure the whole recording as a single span.

        The same as :meth:`analyze` with one span from 0 to the end of the
        audio, which the result's ``end_ms`` holds; *text* becomes its
        ``text``. F0 and voice quality measures come from the voiced
        frames and intensity from the non-silent ones, so silence before,
        between and after the speech does not change them.
        ``speech_rate`` is syllables per second of speaking time (pauses
        excluded). ``quality`` is ``None``: nothing is left to compare the
        voice with.
        """
        analysis = _AudioAnalysis.from_path(audio_path, self.max_duration_s)
        return analysis.span_features(WordAlignment(text, 0, analysis.duration_ms))

    def detect_pauses(
        self,
        audio_path: str | Path,
        min_pause_ms: int = DEFAULT_MIN_PAUSE_MS,
        silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
    ) -> list[PauseInterval]:
        """Detect silent pauses in the audio.

        Silence is relative to the speech around it: a 10 ms frame is
        silent when its level is more than *silence_threshold_db* below the
        loudest voiced sound within 0.25 s of it -- counting voiced sound no
        more than *silence_threshold_db* + 5 dB below the recording's speech
        level (99th percentile of frame levels) -- or, with no such sound
        that near, below the recording's speech level; or, in noisy
        recordings, when it is unvoiced and within 6 dB of the noise floor.
        So quieter speech is judged against itself, not against a shout
        elsewhere in the file. Exact digital silence is always silent.
        Clicks shorter than 30 ms do not interrupt a pause.

        Returns
        -------
        list[PauseInterval]
            Silent stretches of at least *min_pause_ms* milliseconds,
            sorted and non-overlapping, including silence at the start and
            end of the file.
        """
        analysis = _AudioAnalysis.from_path(audio_path, self.max_duration_s)
        return analysis.pauses(min_pause_ms, silence_threshold_db)
