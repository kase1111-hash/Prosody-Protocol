"""IMLToAudio -- synthesize audio from IML markup.

Two engines are available:

``"espeak"``
    Real speech from the espeak-ng synthesizer (an external program; install
    it with your system package manager, e.g. ``apt install espeak-ng``).
    The document is converted with :class:`~prosody_protocol.iml_to_ssml.
    IMLToSSML` in its espeak-ng mode, which adapts pitch, volume, rate,
    pauses, emphasis and pitch contours to what espeak-ng actually renders,
    and is run through ``espeak-ng -m`` (no shell, with a timeout).

``"tones"``
    A prosody *preview*, not speech: one sine tone per word, whose pitch,
    loudness, duration and pitch glide follow the markup (pitch, volume,
    rate, pitch_contour, emphasis, segment tempo, pauses). It needs only
    numpy and is useful for checking the prosody of a document by ear or in
    tests. Words are 250 ms at the baseline rate; tokens without letters or
    digits (punctuation, emoji) are silent. Volumes more than 30 dB above or
    below plain speech, pitches outside 40-2000 Hz and speeds outside
    x0.1-x10 are clamped, with a warning.

``engine="auto"`` (the default) uses espeak-ng when the ``espeak-ng`` program
is on PATH and otherwise falls back to the tone preview with a warning.

Utterance ``emotion`` is not rendered directly: like SSML, the engines render
the prosody that carries it. Segment ``rhythm`` and prosody ``quality`` are
not rendered.

Output is a mono 16-bit WAV. Its duration is capped by ``max_duration_s``:
a document that would exceed the cap raises
:class:`~prosody_protocol.exceptions.ConversionError`. The tone preview
raises before any audio is allocated; espeak-ng is not started for a
document whose pauses alone exceed the cap, and is stopped as soon as its
output does.

Spec reference: Section 3 (IML tags drive synthesis parameters).
"""

from __future__ import annotations

import functools
import io
import math
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import warnings
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ._install import install_hint

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without numpy
    raise ImportError(
        "IMLToAudio requires numpy. Install with: " + install_hint("audio")
    ) from exc

from .exceptions import ConversionError, IMLParseError
from .iml_to_ssml import (
    _CONTOUR_TARGETS,
    _EMPHASIS_EFFECTS,
    _ESPEAK_NOMINAL_F0_HZ,
    _MAX_GAIN_DB,
    _PITCH_HZ_RE,
    _TEMPO_RATES,
    _check_pause,
    _clamp_gain,
    _contour_value,
    _gain_db,
    _pitch_ratio,
    _rate_factor,
    _render,
    _volume_gain,
)
from .models import (
    ChildNode,
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
)
from .parser import IMLParser
from .validator import IMLValidator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22_050  # Hz -- tone preview sample rate (espeak-ng uses the same)
BASE_FREQ = 180.0  # Hz -- tone preview fundamental for the default voice
# Relative amplitude of a plain word: -14 dBFS, leaving headroom for louder
# spans. A document that would still clip is scaled down as a whole.
BASE_AMPLITUDE = 0.2
WORD_DURATION = 0.25  # Seconds per word at the baseline rate (tone preview)
INTER_WORD_GAP = 0.05  # 50 ms gap between whitespace-separated words
UTTERANCE_GAP = 0.3  # Silence between utterances (tone preview)
DEFAULT_MAX_DURATION_S = 300.0  # Default cap on the output duration
ESPEAK_TIMEOUT_S = 120.0  # Seconds espeak-ng may run before it is killed

Engine = Literal["auto", "espeak", "tones", "builtin"]
_UNIMPLEMENTED_ENGINES = ("coqui", "piper", "elevenlabs")
_ENGINE_CHOICES = "'auto', 'espeak' (espeak-ng speech) or 'tones' (prosody preview, not speech)"

# Tone preview: speed factors are clamped to this range.
_MIN_SPEED, _MAX_SPEED = 0.1, 10.0
# Tone preview: frequencies are clamped to this range (Hz).
_MIN_FREQ, _MAX_FREQ = 40.0, 2000.0
# Tone preview: points per word at which a pitch contour is sampled.
_GLIDE_POINTS = 16
_FADE_S = 0.005


def _parse_pitch(pitch: str | None, base_freq: float = BASE_FREQ) -> float:
    """Resolve an IML pitch value to an absolute frequency in Hz."""
    ratio = _pitch_ratio(pitch, base_freq)
    return base_freq if ratio is None else base_freq * ratio


def _parse_volume(volume: str | None, base_amp: float = BASE_AMPLITUDE) -> float:
    """Resolve an IML volume value to an amplitude multiplier."""
    gain = _volume_gain(volume)
    return base_amp if gain is None else base_amp * gain


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

_GENDERS = ("female", "male")
_LEVELS = ("low", "medium", "high")
_LANGUAGE_RE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})*")
_VARIANT_RE = re.compile(r"[A-Za-z0-9_]+")

# (gender, level) -> espeak-ng voice variant, chosen by measured mean F0.
_ESPEAK_VARIANTS: dict[tuple[str, str], str] = {
    ("female", "low"): "f1",  # ~175 Hz
    ("female", "medium"): "f2",  # ~189 Hz
    ("female", "high"): "f3",  # ~204 Hz
    ("male", "low"): "m1",  # ~91 Hz
    ("male", "medium"): "m2",  # ~99 Hz
    ("male", "high"): "m7",  # ~103 Hz
}
# (gender, level) -> tone preview fundamental (Hz).
_TONE_BASE_FREQS: dict[tuple[str, str], float] = {
    ("female", "low"): 160.0,
    ("female", "medium"): BASE_FREQ,
    ("female", "high"): 205.0,
    ("male", "low"): 95.0,
    ("male", "medium"): 110.0,
    ("male", "high"): 125.0,
}

_VOICE_FORMS = (
    'a language tag ("en-US", "de"), optionally followed by "-female" or "-male" '
    'and "-low", "-medium" or "-high" ("en_US-female-medium", "fr-male", "female"), '
    'or an espeak-ng voice with a variant ("en-us+f3")'
)


@dataclass(frozen=True)
class _Voice:
    """A parsed ``voice`` argument."""

    language: str | None = None  # BCP 47 tag; None = the document's language
    gender: str | None = None  # "female" or "male"
    level: str | None = None  # "low", "medium" or "high"
    variant: str | None = None  # explicit espeak-ng variant, e.g. "f3"

    def key(self) -> tuple[str, str]:
        return (self.gender or "female", self.level or "medium")


def _parse_voice(voice: str | None) -> _Voice:
    """Parse a ``voice`` argument; raises ConversionError when it cannot be used."""
    if voice is None:
        return _Voice()
    name, _, variant = voice.strip().partition("+")
    if variant and not _VARIANT_RE.fullmatch(variant):
        raise ConversionError(f"Voice {voice!r}: {variant!r} is not an espeak-ng variant name")
    parts = [p for p in re.split(r"[-_]", name) if p] if name else []
    level = parts.pop().lower() if parts and parts[-1].lower() in _LEVELS else None
    gender = parts.pop().lower() if parts and parts[-1].lower() in _GENDERS else None
    language = "-".join(parts) or None
    leftover = [p for p in parts if p.lower() in _GENDERS + _LEVELS]
    if (
        (not name and not variant)
        or leftover
        or (language is not None and not _LANGUAGE_RE.fullmatch(language))
        or (variant and (gender or level))
    ):
        raise ConversionError(f"Voice {voice!r} is not recognised. Use {_VOICE_FORMS}.")
    if variant:
        # espeak-ng's numbered variants are female (f1-f5) or male (m1-m8).
        gender = {"f": "female", "m": "male"}.get(variant[0]) if variant[1:].isdigit() else None
    return _Voice(language, gender, level, variant or None)


# ---------------------------------------------------------------------------
# espeak-ng
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _espeak_voice_table(binary: str) -> dict[str, str]:
    """Map each language code espeak-ng knows (lower case) to its voice file."""
    listing = _run_listing(binary, "--voices")
    best: dict[str, tuple[int, str]] = {}
    for line in listing.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 5 or not fields[0].isdigit():
            continue
        path = fields[4]
        codes = [(fields[1], int(fields[0]))]
        codes += [(m.group(1), int(m.group(2))) for m in re.finditer(r"\((\S+) (\d+)\)", line)]
        for code, priority in codes:
            key = code.lower()
            if key not in best or priority < best[key][0]:
                best[key] = (priority, path)
    return {code: path for code, (_, path) in best.items()}


@functools.lru_cache(maxsize=4)
def _espeak_variant_names(binary: str) -> frozenset[str]:
    listing = _run_listing(binary, "--voices=variant")
    names = set()
    for line in listing.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 5 and fields[4].startswith("!v/"):
            names.add(fields[4][3:])
    return frozenset(names)


def _run_listing(binary: str, option: str) -> str:
    try:
        result = subprocess.run(
            [binary, option], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConversionError(f"Cannot run espeak-ng ({binary}): {exc}") from exc
    return result.stdout


def _resolve_espeak_voice(binary: str, voice: _Voice, doc_language: str | None) -> tuple[str, str]:
    """Return the espeak-ng ``-v`` argument and the language tag it speaks."""
    tag = voice.language or doc_language or "en-US"
    table = _espeak_voice_table(binary)
    code = tag.lower().replace("_", "-")
    while code not in table and "-" in code:
        code = code.rsplit("-", 1)[0]
    if code not in table:
        source = "voice" if voice.language else "document language"
        raise ConversionError(
            f"espeak-ng has no voice for the {source} {tag!r} "
            "(see `espeak-ng --voices` for the available languages)"
        )
    variant = voice.variant or _ESPEAK_VARIANTS[voice.key()]
    if variant not in _espeak_variant_names(binary):
        raise ConversionError(
            f"espeak-ng has no voice variant {variant!r} "
            "(see `espeak-ng --voices=variant`)"
        )
    return f"{table[code]}+{variant}", code


def _estimate_f0(samples: np.ndarray, sample_rate: int) -> float | None:
    """Estimate the median F0 of voiced speech by autocorrelation."""
    frame = int(0.04 * sample_rate)
    hop = frame // 2
    min_lag, max_lag = int(sample_rate / 400), int(sample_rate / 60)
    if samples.size < frame:
        return None
    starts = range(0, samples.size - frame, hop)
    energies = np.array([float(np.sqrt(np.mean(samples[s : s + frame] ** 2))) for s in starts])
    threshold = 0.3 * float(energies.max()) if energies.size else 0.0
    estimates: list[float] = []
    for start, energy in zip(starts, energies, strict=True):
        if energy <= threshold or energy == 0.0:
            continue
        x = samples[start : start + frame] - samples[start : start + frame].mean()
        ac = np.correlate(x, x, mode="full")[frame - 1 :]
        lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
        if ac[0] > 0 and ac[lag] / ac[0] > 0.5:
            estimates.append(sample_rate / lag)
    return float(np.median(estimates)) if estimates else None


@functools.lru_cache(maxsize=16)
def _espeak_base_f0(binary: str, voice_id: str) -> float:
    """Measure the baseline F0 of an espeak-ng voice (for absolute Hz pitch)."""
    ssml = (
        '<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis">'
        "<s>This is a short sentence for measuring the voice.</s></speak>"
    )
    pcm, rate = _run_espeak(binary, voice_id, ssml, max_duration_s=30.0)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    return _estimate_f0(samples, rate) or _ESPEAK_NOMINAL_F0_HZ


def _parse_wav_header(data: bytes) -> tuple[int, int] | None:
    """Return (sample rate, offset of the PCM data) of espeak-ng's streamed WAV.

    espeak-ng writes placeholder chunk sizes when streaming, so the data
    chunk is taken to run to the end of the stream. Returns ``None`` if
    ``data`` does not yet hold the whole header.
    """
    if len(data) >= 12 and (data[:4] != b"RIFF" or data[8:12] != b"WAVE"):
        raise ConversionError("espeak-ng did not produce WAV audio")
    pos, rate = 12, None
    while len(data) >= pos + 8:
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body = pos + 8
        if chunk_id == b"data":
            if rate is None:
                raise ConversionError("espeak-ng WAV output has no format chunk")
            return rate, body
        if chunk_id == b"fmt ":
            if len(data) < body + 16:
                return None
            fmt, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[body : body + 16])
            if fmt != 1 or channels != 1 or bits != 16:
                raise ConversionError("espeak-ng produced audio that is not 16-bit mono PCM")
        pos = body + size + (size & 1)
    return None


def _run_espeak(
    binary: str, voice_id: str, ssml: str, *, max_duration_s: float
) -> tuple[bytes, int]:
    """Run ``espeak-ng -m`` on SSML; return 16-bit mono PCM and its sample rate.

    The SSML goes through a temporary file (no shell). Output is streamed
    and the process is killed as soon as it exceeds ``max_duration_s`` of
    audio or runs longer than :data:`ESPEAK_TIMEOUT_S`.
    """
    with tempfile.TemporaryDirectory(prefix="prosody-espeak-") as tmp:
        source = Path(tmp) / "input.ssml"
        source.write_text(ssml, encoding="utf-8")
        errors_path = Path(tmp) / "stderr.txt"
        command = [binary, "-m", "-b", "1", "-v", voice_id, "-f", str(source), "--stdout"]
        with errors_path.open("wb") as errors:
            try:
                proc = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors
                )
            except OSError as exc:
                raise ConversionError(f"Cannot run espeak-ng ({binary}): {exc}") from exc
            timed_out = threading.Event()

            def _kill() -> None:
                timed_out.set()
                proc.kill()

            timer = threading.Timer(ESPEAK_TIMEOUT_S, _kill)
            timer.start()
            buffer = bytearray()
            header: tuple[int, int] | None = None
            try:
                assert proc.stdout is not None
                while chunk := proc.stdout.read(65_536):
                    buffer += chunk
                    if header is None:
                        header = _parse_wav_header(bytes(buffer[:4096]))
                    if header is not None:
                        rate, offset = header
                        if len(buffer) - offset > max_duration_s * rate * 2:
                            raise ConversionError(
                                f"The synthesized audio exceeds max_duration_s={max_duration_s:g} s"
                            )
                proc.wait()
            finally:
                timer.cancel()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if proc.stdout is not None:
                    proc.stdout.close()
        stderr = errors_path.read_text(encoding="utf-8", errors="replace").strip()
    if timed_out.is_set():
        raise ConversionError(f"espeak-ng did not finish within {ESPEAK_TIMEOUT_S:g} s")
    if proc.returncode != 0:
        raise ConversionError(f"espeak-ng failed (exit code {proc.returncode}): {stderr}")
    header = _parse_wav_header(bytes(buffer[:4096]))
    if header is None:
        raise ConversionError(f"espeak-ng produced no audio: {stderr or 'empty output'}")
    rate, offset = header
    pcm = bytes(buffer[offset:])
    return pcm[: len(pcm) - len(pcm) % 2], rate


def _uses_absolute_pitch(children: tuple[ChildNode, ...]) -> bool:
    for child in children:
        if isinstance(child, Prosody) and child.pitch and _PITCH_HZ_RE.fullmatch(child.pitch):
            return True
        if isinstance(child, (Prosody, Emphasis, Segment)) and _uses_absolute_pitch(
            child.children
        ):
            return True
    return False


def _pause_seconds(children: tuple[ChildNode, ...]) -> float:
    """Total duration of the pauses in ``children``: a lower bound on the audio.

    Words are not counted: espeak-ng can speak a short word in about a
    millisecond at its fastest rate, so the length of the spoken text is
    enforced by the byte cap on espeak-ng's output instead.
    """
    total = 0.0
    for child in children:
        if isinstance(child, Pause):
            total += _check_pause(child) / 1000.0
        elif isinstance(child, (Prosody, Emphasis, Segment)):
            total += _pause_seconds(child.children)
    return total


# ---------------------------------------------------------------------------
# Tone preview
# ---------------------------------------------------------------------------


@dataclass
class _Tone:
    freq: float
    amplitude: float
    samples: int
    glide: list[float] | None = None  # F0 multipliers across the word


@dataclass
class _Silence:
    samples: int


@dataclass
class _ToneScore:
    """Flatten an IML document into timed tones and silences."""

    max_samples: int
    max_duration_s: float
    events: list[_Tone | _Silence] = field(default_factory=list)
    total: int = 0
    notes: list[str] = field(default_factory=list)
    _space_pending: bool = False

    def _add(self, event: _Tone | _Silence) -> None:
        self.total += event.samples
        if self.total > self.max_samples:
            raise ConversionError(
                f"The document would synthesize to more than max_duration_s="
                f"{self.max_duration_s:g} s of audio"
            )
        self.events.append(event)

    def _samples(self, seconds: float) -> int:
        return int(round(seconds * SAMPLE_RATE))

    def _last_is_tone(self) -> bool:
        return bool(self.events) and isinstance(self.events[-1], _Tone)

    def silence(self, seconds: float) -> None:
        self._add(_Silence(self._samples(seconds)))
        self._space_pending = False

    def children(
        self, children: tuple[ChildNode, ...], freq: float, amp: float, speed: float
    ) -> None:
        for child in children:
            if isinstance(child, str):
                self.text(child, freq, amp, speed)
            elif isinstance(child, Pause):
                self.silence(_check_pause(child) / 1000.0)
            elif isinstance(child, Prosody):
                self.prosody(child, freq, amp, speed)
            elif isinstance(child, Emphasis):
                ratio, gain = _EMPHASIS_EFFECTS.get(child.level, _EMPHASIS_EFFECTS["moderate"])
                self.children(
                    child.children, self._freq(freq * ratio), self._amp(amp * gain), speed
                )
            elif isinstance(child, Segment):
                factor = _rate_factor(_TEMPO_RATES.get(child.tempo or "")) or 1.0
                self.children(child.children, freq, amp, self._speed(speed * factor))

    def text(self, text: str, freq: float, amp: float, speed: float) -> None:
        for token in re.finditer(r"\s+|\S+", text):
            word = token.group()
            if word.isspace():
                self._space_pending = self._last_is_tone()
            elif any(ch.isalnum() for ch in word):
                if self._space_pending and self._last_is_tone():
                    self._add(_Silence(self._samples(INTER_WORD_GAP / speed)))
                self._space_pending = False
                self._add(_Tone(freq, amp, self._samples(WORD_DURATION / speed)))

    def prosody(self, p: Prosody, freq: float, amp: float, speed: float) -> None:
        start = len(self.events)
        ratio = _pitch_ratio(p.pitch, freq) or 1.0
        gain = _volume_gain(p.volume) or 1.0
        rate = _rate_factor(p.rate) or 1.0
        self.children(
            p.children, self._freq(freq * ratio), self._amp(amp * gain), self._speed(speed * rate)
        )
        targets = _CONTOUR_TARGETS.get(p.pitch_contour or "")
        if targets is not None:
            self._apply_contour(self.events[start:], targets)

    def _speed(self, speed: float) -> float:
        if not _MIN_SPEED <= speed <= _MAX_SPEED:
            self.notes.append(
                f"speech rate x{speed:g} is outside the renderable range "
                f"x{_MIN_SPEED:g}-x{_MAX_SPEED:g}; clamped"
            )
        return min(_MAX_SPEED, max(_MIN_SPEED, speed))

    def _freq(self, freq: float) -> float:
        if not _MIN_FREQ <= freq <= _MAX_FREQ:
            self.notes.append(
                f"pitch {freq:.0f} Hz is outside the tone preview's range "
                f"{_MIN_FREQ:g}-{_MAX_FREQ:g} Hz; clamped"
            )
        return min(_MAX_FREQ, max(_MIN_FREQ, freq))

    def _amp(self, amp: float) -> float:
        """Clamp an amplitude to +/-``_MAX_GAIN_DB`` around plain speech."""
        gain = amp / BASE_AMPLITUDE
        clamped = _clamp_gain(gain)
        if abs(math.log(clamped / gain)) > 1e-6:
            self.notes.append(
                f"a volume more than {_MAX_GAIN_DB:g} dB above or below plain speech was "
                f"clamped to {_gain_db(clamped):+.0f} dB"
            )
        return BASE_AMPLITUDE * clamped

    @staticmethod
    def _apply_contour(
        events: list[_Tone | _Silence], targets: tuple[tuple[float, float], ...]
    ) -> None:
        """Glide the pitch of the tones in ``events`` along a contour over their span."""
        span = sum(e.samples for e in events)
        if span == 0:
            return
        elapsed = 0
        for event in events:
            if isinstance(event, _Tone):
                glide = event.glide or [1.0] * _GLIDE_POINTS
                for k in range(_GLIDE_POINTS):
                    at = elapsed + event.samples * k / (_GLIDE_POINTS - 1)
                    glide[k] *= _contour_value(targets, at / span)
                event.glide = glide
            elapsed += event.samples

    def render(self) -> np.ndarray:
        """Allocate and fill the audio for the flattened events."""
        out = np.zeros(self.total, dtype=np.float32)
        pos = 0
        fade = int(SAMPLE_RATE * _FADE_S)
        for event in self.events:
            n = event.samples
            if isinstance(event, _Tone) and n > 0:
                if event.glide is None:
                    phase = 2.0 * np.pi * event.freq * np.arange(n) / SAMPLE_RATE
                else:
                    curve = np.interp(
                        np.linspace(0.0, 1.0, n),
                        np.linspace(0.0, 1.0, _GLIDE_POINTS),
                        np.clip(np.array(event.glide) * event.freq, _MIN_FREQ, _MAX_FREQ),
                    )
                    phase = 2.0 * np.pi * np.cumsum(curve) / SAMPLE_RATE
                tone = event.amplitude * np.sin(phase)
                edge = min(fade, n // 2)
                if edge > 0:
                    tone[:edge] *= np.linspace(0.0, 1.0, edge)
                    tone[-edge:] *= np.linspace(1.0, 0.0, edge)
                out[pos : pos + n] = tone
            pos += n
        return out


def _render_tones(
    doc: IMLDocument, base_freq: float, max_duration_s: float
) -> tuple[np.ndarray, list[str]]:
    """Render ``doc`` as a tone preview; returns float samples and notes."""
    score = _ToneScore(int(max_duration_s * SAMPLE_RATE), max_duration_s)
    for i, utt in enumerate(doc.utterances):
        if i > 0:
            score.silence(UTTERANCE_GAP)
        score.children(utt.children, base_freq, BASE_AMPLITUDE, 1.0)
    if score.total == 0:
        score.silence(0.1)  # Minimal silence for empty documents.
    samples = score.render()
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples *= np.float32(0.99 / peak)
        score.notes.append(
            f"the tone preview was scaled down by {20 * math.log10(peak / 0.99):.1f} dB "
            "to avoid clipping (relative levels are kept)"
        )
    return samples, score.notes


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap 16-bit mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _to_wav_bytes(samples: np.ndarray) -> bytes:
    """Encode float samples in [-1, 1] to 16-bit PCM WAV bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    return _pcm_to_wav(pcm.tobytes(), SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class IMLToAudio:
    """Synthesize audio from IML markup.

    Parameters
    ----------
    voice:
        Voice to use. ``None`` (default) picks a female voice for the
        document's ``language`` (``en-US`` when it has none). Otherwise a
        language tag (``"en-US"``, ``"de"``), optionally followed by a gender
        and a pitch level (``"en_US-female-medium"``, ``"fr-male-low"``,
        ``"male"``), or an espeak-ng voice with a variant (``"en-us+f3"``).
        A language in the voice overrides the document's language. With the
        ``"espeak"`` engine the voice selects the espeak-ng language and
        variant; with the ``"tones"`` preview only gender and level matter
        (they set the tone's base pitch). A voice that cannot be used raises
        :class:`~prosody_protocol.exceptions.ConversionError`.
    engine:
        ``"auto"`` (default): espeak-ng if the ``espeak-ng`` program is on
        PATH, otherwise the tone preview with a warning. ``"espeak"``: real
        speech via espeak-ng (raises ConversionError if it is not
        installed). ``"tones"``: the prosody preview (one tone per word, not
        speech). ``"builtin"`` is a deprecated alias of ``"tones"``.
        ``"coqui"``, ``"piper"`` and ``"elevenlabs"`` are not implemented
        and raise ConversionError.
    max_duration_s:
        Maximum duration of the synthesized audio in seconds (default 300).
        A longer document raises ConversionError, without allocating more
        than the cap's worth of audio (see the module docstring).
    strict:
        When true (default), a document with validation errors (see
        :class:`~prosody_protocol.validator.IMLValidator`) is rejected with
        :class:`~prosody_protocol.exceptions.IMLValidationError`. When
        false, invalid attribute values are ignored (spec 6.2), except
        values that cannot be rendered at all, such as a negative pause.

    Attributes
    ----------
    backend:
        The engine actually used: ``"espeak"`` or ``"tones"``.
    """

    def __init__(
        self,
        voice: str | None = None,
        engine: Engine = "auto",
        *,
        max_duration_s: float = DEFAULT_MAX_DURATION_S,
        strict: bool = True,
    ) -> None:
        if not (0 < max_duration_s < math.inf):
            raise ValueError(f"max_duration_s must be a positive number, got {max_duration_s!r}")
        self.voice = voice
        self.engine = engine
        self.max_duration_s = float(max_duration_s)
        self.strict = strict
        self._voice = _parse_voice(voice)
        self._parser = IMLParser()
        self._validator = IMLValidator()
        self._espeak: str | None = None

        if engine == "auto":
            self._espeak = shutil.which("espeak-ng")
            if self._espeak is None:
                warnings.warn(
                    "espeak-ng was not found on PATH, so IMLToAudio renders a tone preview "
                    "of the prosody (one tone per word), not speech. Install espeak-ng for "
                    "speech, or pass engine='tones' to choose the preview explicitly.",
                    stacklevel=2,
                )
        elif engine == "espeak":
            self._espeak = shutil.which("espeak-ng")
            if self._espeak is None:
                raise ConversionError(
                    "engine='espeak' needs the espeak-ng program on PATH "
                    "(e.g. `apt install espeak-ng` or `brew install espeak-ng`)"
                )
        elif engine == "builtin":
            warnings.warn(
                "engine='builtin' is deprecated; it is now called 'tones' and renders a "
                "tone preview of the prosody, not speech.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif engine in _UNIMPLEMENTED_ENGINES:
            raise ConversionError(
                f"Engine {engine!r} is not implemented. Available engines: {_ENGINE_CHOICES}."
            )
        elif engine != "tones":
            raise ConversionError(f"Unknown engine {engine!r}. Use {_ENGINE_CHOICES}.")

        self.backend: Literal["espeak", "tones"] = "espeak" if self._espeak else "tones"
        if self._espeak is not None and (self._voice.language or self._voice.variant):
            # Fail fast on a voice espeak-ng cannot speak.
            _resolve_espeak_voice(self._espeak, self._voice, None)
        if self.backend == "tones" and voice is not None and self._voice.gender is None:
            warnings.warn(
                f"The tone preview ignores the language of voice {voice!r}; only a gender "
                "and pitch level (e.g. 'male-low') change it.",
                stacklevel=2,
            )

    def synthesize(self, iml_string: str) -> bytes:
        """Synthesize IML to raw audio bytes (WAV format).

        Returns a complete WAV file as ``bytes``.

        Raises :class:`~prosody_protocol.exceptions.ConversionError` if the
        IML cannot be parsed or rendered or the audio would exceed
        ``max_duration_s``, and
        :class:`~prosody_protocol.exceptions.IMLValidationError` if it has
        validation errors (``strict`` mode).
        """
        try:
            doc = self._parser.parse(iml_string)
        except IMLParseError as exc:
            raise ConversionError(f"Cannot parse IML for synthesis: {exc}") from exc
        if self.strict:
            self._validator.validate(iml_string).raise_for_errors()
        return self._synthesize(doc)

    def synthesize_to_file(self, iml_string: str, output_path: str | Path) -> None:
        """Synthesize IML and write to a WAV file.

        Raises the same errors as :meth:`synthesize`.
        """
        wav_bytes = self.synthesize(iml_string)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(wav_bytes)

    def synthesize_doc(self, doc: IMLDocument) -> bytes:
        """Synthesize an :class:`IMLDocument` to WAV bytes.

        Raises the same errors as :meth:`synthesize`.
        """
        if self.strict:
            self._validator.validate(self._parser.to_iml_string(doc)).raise_for_errors()
        return self._synthesize(doc)

    def _synthesize(self, doc: IMLDocument) -> bytes:
        if self._espeak is not None:
            wav, notes = self._synthesize_espeak(self._espeak, doc)
        else:
            base_freq = _TONE_BASE_FREQS[self._voice.key()]
            samples, notes = _render_tones(doc, base_freq, self.max_duration_s)
            wav = _to_wav_bytes(samples)
        for note in dict.fromkeys(notes):
            warnings.warn(note, stacklevel=3)
        return wav

    def _synthesize_espeak(self, binary: str, doc: IMLDocument) -> tuple[bytes, list[str]]:
        pauses = sum(_pause_seconds(u.children) for u in doc.utterances)
        if pauses > self.max_duration_s:
            raise ConversionError(
                f"The document's pauses alone last {pauses:.0f} s, more than "
                f"max_duration_s={self.max_duration_s:g} s"
            )
        voice_id, language = _resolve_espeak_voice(binary, self._voice, doc.language)
        base_f0 = _ESPEAK_NOMINAL_F0_HZ
        if any(_uses_absolute_pitch(u.children) for u in doc.utterances):
            base_f0 = _espeak_base_f0(binary, voice_id)
        ssml, notes = _render(
            doc,
            vendor="espeak-ng",
            language=language,
            speaker_voices={},
            base_f0_hz=base_f0,
        )
        pcm, rate = _run_espeak(binary, voice_id, ssml, max_duration_s=self.max_duration_s)
        return _pcm_to_wav(pcm, rate), notes
