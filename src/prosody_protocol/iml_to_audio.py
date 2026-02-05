"""IMLToAudio -- synthesize speech from IML markup.

Walks the parsed IML document tree and generates audio with prosodic
modifications applied.  Uses a simple waveform-based synthesis engine
by default (suitable for testing and demonstration).  The architecture
is designed so that a full TTS engine (Coqui, Piper, ElevenLabs) can
be swapped in via the ``engine`` parameter in the future.

Pipeline:
  1. Parse IML to IMLDocument
  2. Walk the document tree depth-first
  3. For each text span, generate a base tone
  4. Apply prosodic modifications (pitch shift, volume adjustment)
  5. Insert silence for <pause> elements
  6. Apply emphasis (louder + higher pitch)
  7. Concatenate segments and write WAV

Spec reference: Section 3 (IML tags drive synthesis parameters).
"""

from __future__ import annotations

import io
import re
import struct
import wave
from pathlib import Path
from typing import Literal

import numpy as np

from .exceptions import ConversionError
from .models import (
    ChildNode,
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)
from .parser import IMLParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22_050  # Hz -- common TTS sample rate
BASE_FREQ = 180.0  # Hz -- default fundamental frequency
BASE_AMPLITUDE = 0.5  # Relative amplitude [0.0, 1.0]
WORD_DURATION = 0.25  # Seconds per word (synthetic speech cadence)
INTER_WORD_GAP = 0.05  # 50 ms gap between words

_PITCH_PCT_RE = re.compile(r"^([+\-]\d+(?:\.\d+)?)%$")
_PITCH_ST_RE = re.compile(r"^([+\-]\d+(?:\.\d+)?)st$")
_PITCH_HZ_RE = re.compile(r"^(\d+(?:\.\d+)?)Hz$")
_VOLUME_DB_RE = re.compile(r"^([+\-]\d+(?:\.\d+)?)dB$")


# ---------------------------------------------------------------------------
# Pitch / volume parsing
# ---------------------------------------------------------------------------


def _parse_pitch(pitch: str | None, base_freq: float = BASE_FREQ) -> float:
    """Resolve an IML pitch value to an absolute frequency in Hz."""
    if pitch is None:
        return base_freq

    m = _PITCH_PCT_RE.match(pitch)
    if m:
        pct = float(m.group(1))
        return base_freq * (1.0 + pct / 100.0)

    m = _PITCH_ST_RE.match(pitch)
    if m:
        semitones = float(m.group(1))
        return base_freq * (2.0 ** (semitones / 12.0))

    m = _PITCH_HZ_RE.match(pitch)
    if m:
        return float(m.group(1))

    return base_freq


def _parse_volume(volume: str | None, base_amp: float = BASE_AMPLITUDE) -> float:
    """Resolve an IML volume value to an amplitude multiplier."""
    if volume is None:
        return base_amp

    m = _VOLUME_DB_RE.match(volume)
    if m:
        db = float(m.group(1))
        return base_amp * (10.0 ** (db / 20.0))

    return base_amp


# ---------------------------------------------------------------------------
# Waveform synthesis helpers
# ---------------------------------------------------------------------------


def _sine_wave(freq: float, duration_s: float, amplitude: float) -> np.ndarray:
    """Generate a sine-wave tone with smooth fade-in/fade-out."""
    n_samples = int(SAMPLE_RATE * duration_s)
    if n_samples == 0:
        return np.array([], dtype=np.float64)

    t = np.arange(n_samples) / SAMPLE_RATE
    wave_data = amplitude * np.sin(2 * np.pi * freq * t)

    # Apply 5ms fade-in and fade-out to avoid clicks.
    fade_samples = min(int(SAMPLE_RATE * 0.005), n_samples // 2)
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        wave_data[:fade_samples] *= fade_in
        wave_data[-fade_samples:] *= fade_out

    return wave_data


def _silence(duration_s: float) -> np.ndarray:
    """Generate silence of the given duration."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float64)


def _synthesize_text(
    text: str, freq: float, amplitude: float
) -> np.ndarray:
    """Synthesize a text span as a series of tone bursts (one per word)."""
    words = text.split()
    if not words:
        return np.array([], dtype=np.float64)

    segments: list[np.ndarray] = []
    for i, _word in enumerate(words):
        segments.append(_sine_wave(freq, WORD_DURATION, amplitude))
        if i < len(words) - 1:
            segments.append(_silence(INTER_WORD_GAP))

    return np.concatenate(segments) if segments else np.array([], dtype=np.float64)


# ---------------------------------------------------------------------------
# IML tree walker
# ---------------------------------------------------------------------------


def _render_children(
    children: tuple[ChildNode, ...],
    freq: float,
    amplitude: float,
) -> np.ndarray:
    """Recursively render mixed-content children to audio samples."""
    segments: list[np.ndarray] = []

    for child in children:
        if isinstance(child, str):
            if child.strip():
                segments.append(_synthesize_text(child, freq, amplitude))
        elif isinstance(child, Pause):
            segments.append(_silence(child.duration / 1000.0))
        elif isinstance(child, Prosody):
            child_freq = _parse_pitch(child.pitch, freq)
            child_amp = _parse_volume(child.volume, amplitude)
            segments.append(
                _render_children(child.children, child_freq, child_amp)
            )
        elif isinstance(child, Emphasis):
            # Emphasis: increase pitch by 10% and amplitude by 30%.
            emph_freq = freq * 1.10
            emph_amp = min(1.0, amplitude * 1.30)
            if child.level == "strong":
                emph_freq = freq * 1.15
                emph_amp = min(1.0, amplitude * 1.50)
            elif child.level == "reduced":
                emph_freq = freq * 1.05
                emph_amp = amplitude * 1.10
            segments.append(
                _render_children(child.children, emph_freq, emph_amp)
            )
        elif isinstance(child, Segment):
            # Segment has no acoustic effect beyond its children.
            segments.append(
                _render_children(child.children, freq, amplitude)
            )
        elif isinstance(child, Utterance):
            segments.append(
                _render_children(child.children, freq, amplitude)
            )

    if not segments:
        return np.array([], dtype=np.float64)
    return np.concatenate(segments)


def _render_utterance(utt: Utterance) -> np.ndarray:
    """Render a single utterance to audio samples."""
    return _render_children(utt.children, BASE_FREQ, BASE_AMPLITUDE)


def _render_document(doc: IMLDocument) -> np.ndarray:
    """Render a full IML document to audio samples."""
    segments: list[np.ndarray] = []
    for i, utt in enumerate(doc.utterances):
        samples = _render_utterance(utt)
        if samples.size > 0:
            segments.append(samples)
        # Add 300 ms gap between utterances.
        if i < len(doc.utterances) - 1:
            segments.append(_silence(0.3))

    if not segments:
        return _silence(0.1)  # Minimal silence for empty documents.
    return np.concatenate(segments)


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------


def _to_wav_bytes(samples: np.ndarray) -> bytes:
    """Encode float64 samples to 16-bit PCM WAV bytes."""
    # Clip to [-1, 1] and convert to 16-bit integers.
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class IMLToAudio:
    """Synthesize speech audio from IML markup.

    Parameters
    ----------
    voice:
        Voice identifier (reserved for TTS engine integration).
    engine:
        Synthesis engine.  Currently only ``"builtin"`` (waveform-based)
        is implemented.  ``"coqui"``, ``"piper"``, and ``"elevenlabs"``
        are accepted but fall back to the built-in engine.
    """

    def __init__(
        self,
        voice: str = "en_US-female-medium",
        engine: Literal["builtin", "coqui", "piper", "elevenlabs"] = "builtin",
    ) -> None:
        self.voice = voice
        self.engine = engine
        self._parser = IMLParser()

    def synthesize(self, iml_string: str) -> bytes:
        """Synthesize IML to raw audio bytes (WAV format).

        Returns a complete WAV file as ``bytes``.

        Raises :class:`~prosody_protocol.exceptions.ConversionError`
        if the IML cannot be parsed.
        """
        doc = self._parse(iml_string)
        samples = _render_document(doc)
        return _to_wav_bytes(samples)

    def synthesize_to_file(self, iml_string: str, output_path: str | Path) -> None:
        """Synthesize IML and write to a WAV file.

        Raises :class:`~prosody_protocol.exceptions.ConversionError`
        if the IML cannot be parsed.
        """
        wav_bytes = self.synthesize(iml_string)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(wav_bytes)

    def synthesize_doc(self, doc: IMLDocument) -> bytes:
        """Synthesize an :class:`IMLDocument` to WAV bytes."""
        samples = _render_document(doc)
        return _to_wav_bytes(samples)

    def _parse(self, iml_string: str) -> IMLDocument:
        try:
            return self._parser.parse(iml_string)
        except Exception as exc:
            raise ConversionError(
                f"Cannot parse IML for synthesis: {exc}"
            ) from exc
