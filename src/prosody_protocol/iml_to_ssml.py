"""IMLToSSML -- convert IML markup to SSML 1.1 for TTS engines.

Maps IML tags to their SSML equivalents:

  IML              SSML
  ─────────────    ──────────────────────────────────────────────────────
  <iml>            <speak version="1.1" xmlns="..." xml:lang="...">
  <utterance>      <s> (inside <voice name="..."> when ``speaker_voices``
                   maps the utterance's speaker_id)
  <prosody>        <prosody> with pitch, volume and rate; pitch_contour
                   becomes an SSML ``contour`` ("flat" becomes range="x-low")
  <pause>          <break time="...ms"/>
  <emphasis>       <emphasis level="...">
  <segment>        <prosody rate="fast"> for tempo="rushed", rate="slow" for
                   tempo="drawn-out"; otherwise unwrapped (children promoted)

``<speak>`` always carries ``xml:lang``: the document's ``language``, or the
converter's ``default_language`` when the document has none. Text keeps its
order; comments and processing instructions are never spoken (the parser
drops them), and characters XML cannot carry are replaced by spaces.

Not mapped, because SSML has no vendor-neutral equivalent: utterance
``emotion`` and ``confidence``, ``quality``, segment ``rhythm``,
``speaker_id`` (unless ``speaker_voices`` maps it), and the extended
attributes (f0_mean, jitter, shimmer, ...). Emotion therefore reaches the
listener only through the prosody that carries it: two documents that differ
only in their emotion labels give the same SSML, while documents that differ
in pitch_contour, pitch, volume or rate give different SSML.

When a ``<prosody>`` has both ``pitch`` and ``pitch_contour``, the contour is
written on a nested ``<prosody>`` so that its targets are unambiguously
relative to the shifted pitch.

``vendor="espeak-ng"`` adapts the output to what espeak-ng 1.51 actually
renders (measured by analyzing its audio): pitch values are rescaled so the
realized F0 shift matches the IML value (espeak-ng applies relative pitch to
an internal 0-100 parameter, which roughly halves it, and treats "185Hz" as a
relative value; absolute values are resolved against the voice's measured
baseline, not the enclosing pitch); volume is written as a percentage
(espeak-ng misreads dB) and cumulative volume changes are clamped to
+/-30 dB; pitch contours are approximated by per-word pitch steps (espeak-ng
ignores ``contour``); emphasis becomes a pitch and volume change
(espeak-ng's own emphasis ignores the surrounding volume and can clip);
every sentence is wrapped in a volume that leaves headroom for louder spans
(40% of espeak-ng's default, lowered so that the loudest span stays at or
below 70%, which does not clip for any voice measured); and sentence-final
punctuation is placed where espeak-ng does not insert a spurious pause.
That output is meant for ``espeak-ng -m`` and is not portable SSML.

Values the validator accepts but no renderer can use ("+7000dB",
"+20000st") are limited before any arithmetic, so they are clamped (with a
warning where the output approximates) rather than raising.

Spec reference: Section 1.5 (relationship to SSML), Appendix A.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace

from ._types import normalize_language_tag
from .exceptions import ConversionError, IMLParseError
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
from .validator import IMLValidator

SSML_NAMESPACE = "http://www.w3.org/2001/10/synthesis"
DEFAULT_LANGUAGE = "en-US"

_ESPEAK_VENDORS = frozenset({"espeak-ng", "espeak"})

# Characters XML 1.0 cannot carry (C0 controls other than tab/LF/CR, lone
# surrogates, U+FFFE and U+FFFF).
_XML_INVALID_CHARS_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _escape_xml(text: str) -> str:
    return (
        _XML_INVALID_CHARS_RE.sub(" ", text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _ssml_attr(name: str, value: str | None) -> str:
    if value is None:
        return ""
    return f' {name}="{_escape_xml(value)}"'


# ---------------------------------------------------------------------------
# IML value parsing (shared with iml_to_audio)
# ---------------------------------------------------------------------------

_PITCH_PCT_RE = re.compile(r"([+-][0-9]+(?:\.[0-9]+)?)%")
_PITCH_ST_RE = re.compile(r"([+-][0-9]+(?:\.[0-9]+)?)st")
_PITCH_HZ_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)Hz")
_DB_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)dB$", re.IGNORECASE)
_RATE_PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")

# Smallest F0 ratio a pitch value resolves to ("-100%" would be silence).
_MIN_PITCH_RATIO = 0.05
# Semitone and dB values are limited to this magnitude before they are
# exponentiated, so that absurd values ("+7000dB", "+20000st"), which the
# validator accepts, cannot overflow. The renderers clamp far below this.
_MAX_ABS_SEMITONES = 120.0
_MAX_ABS_DB = 120.0

# Cumulative volume changes are clamped to +/-30 dB wherever volume is
# rendered as a gain (the espeak-ng adaptation and the tone preview): a wider
# spread would leave the quieter speech inaudible in 16-bit audio.
_MAX_GAIN_DB = 30.0
_MAX_GAIN = float(10.0 ** (_MAX_GAIN_DB / 20.0))

# Speed factors for named rates (spec 3.2; SSML's x-slow/x-fast are accepted
# leniently) and segment tempo (spec 3.5).
_NAMED_RATES: dict[str, float] = {
    "x-slow": 0.6,
    "slow": 0.8,
    "medium": 1.0,
    "fast": 1.25,
    "x-fast": 1.6,
}
_TEMPO_RATES: dict[str, str] = {"rushed": "fast", "drawn-out": "slow"}

# pitch_contour -> contour targets as (position in % of the span, pitch
# change in %), following the spec 3.2 descriptions. "flat" has no targets:
# it maps to a narrow pitch range instead.
_CONTOUR_TARGETS: dict[str, tuple[tuple[float, float], ...]] = {
    "rise": ((0, 0), (100, 20)),
    "fall": ((0, 0), (100, -20)),
    "rise-fall": ((0, 0), (50, 20), (100, -10)),
    "fall-rise": ((0, 0), (50, -15), (100, 15)),
    "rise-sharp": ((0, 0), (70, 5), (100, 40)),
    "fall-sharp": ((0, 10), (70, 5), (100, -30)),
}
_FLAT_CONTOUR = "flat"

# Emphasis level -> (F0 ratio, amplitude gain), used wherever emphasis is
# rendered as explicit pitch and volume changes (the espeak-ng adaptation
# and the tone preview). "reduced" is de-emphasis (spec 3.4), so it sits
# below the baseline; unknown or missing levels are treated as "moderate".
_EMPHASIS_EFFECTS: dict[str, tuple[float, float]] = {
    "strong": (1.15, 1.5),
    "moderate": (1.10, 1.3),
    "reduced": (0.97, 0.8),
}


def _pitch_ratio(pitch: str | None, base_hz: float) -> float | None:
    """Return the F0 ratio an IML pitch value asks for.

    ``base_hz`` is the baseline F0 that absolute values ("185Hz") are
    relative to. Returns ``None`` for a missing or malformed value.
    """
    if pitch is None:
        return None
    m = _PITCH_PCT_RE.fullmatch(pitch)
    if m:
        return max(_MIN_PITCH_RATIO, 1.0 + float(m.group(1)) / 100.0)
    m = _PITCH_ST_RE.fullmatch(pitch)
    if m:
        semitones = min(_MAX_ABS_SEMITONES, max(-_MAX_ABS_SEMITONES, float(m.group(1))))
        return max(_MIN_PITCH_RATIO, float(2.0 ** (semitones / 12.0)))
    m = _PITCH_HZ_RE.fullmatch(pitch)
    if m:
        return max(_MIN_PITCH_RATIO, float(m.group(1)) / base_hz)
    return None


def _volume_gain(volume: str | None) -> float | None:
    """Return the amplitude gain an IML volume ("+6dB") asks for, or ``None``.

    Values beyond +/-120 dB are treated as +/-120 dB.
    """
    if volume is None:
        return None
    m = _DB_RE.match(volume)
    if m is None:
        return None
    db = min(_MAX_ABS_DB, max(-_MAX_ABS_DB, float(m.group(1))))
    return float(10.0 ** (db / 20.0))


def _clamp_gain(gain: float) -> float:
    """Clamp a cumulative amplitude gain to +/-``_MAX_GAIN_DB``."""
    return min(_MAX_GAIN, max(1.0 / _MAX_GAIN, gain))


def _gain_db(gain: float) -> float:
    return 20.0 * math.log10(gain) if gain > 0 else -math.inf


def _rate_factor(rate: str | None) -> float | None:
    """Return the speed factor an IML rate asks for (1.0 = baseline), or ``None``."""
    if rate is None:
        return None
    if rate in _NAMED_RATES:
        return _NAMED_RATES[rate]
    m = _RATE_PCT_RE.fullmatch(rate)
    if m:
        return float(m.group(1)) / 100.0
    return None


def _contour_value(targets: tuple[tuple[float, float], ...], position: float) -> float:
    """Return the F0 ratio of a contour at ``position`` (0.0-1.0 across the span)."""
    pct = min(1.0, max(0.0, position)) * 100.0
    for (p0, c0), (p1, c1) in zip(targets, targets[1:], strict=False):
        if pct <= p1:
            frac = 0.0 if p1 == p0 else (pct - p0) / (p1 - p0)
            return 1.0 + (c0 + (c1 - c0) * frac) / 100.0
    return 1.0 + targets[-1][1] / 100.0


def _contour_attr(targets: tuple[tuple[float, float], ...]) -> str:
    return " ".join(f"({pos:g}%,{change:+g}%)" for pos, change in targets)


def _check_pause(pause: Pause) -> int:
    """Return a pause duration that can be rendered, or raise ConversionError."""
    duration = pause.duration
    if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
        raise ConversionError(
            f"<pause> duration={duration!r} cannot be rendered: it must be a whole "
            "number of milliseconds greater than zero (spec 3.3)"
        )
    return duration


def _iml_volume_to_ssml(volume: str) -> str:
    """Convert IML volume (relative dB) to an SSML-compatible volume string.

    SSML prosody volume accepts ``+XdB`` / ``-XdB`` per the W3C spec, so
    in most cases the IML value passes through directly.  However, some
    engines only support named values, so map extreme dB values to names
    as a fallback.
    """
    m = _DB_RE.match(volume)
    if m is None:
        return volume  # unrecognized format -- pass through unchanged
    # Rounding keeps tiny values out of exponent notation ("1e-07"), and
    # adding 0.0 turns -0.0 ("-0dB", valid IML) into 0.0, so that it is
    # written "+0dB" rather than "+-0dB".
    db = round(float(m.group(1)), 4) + 0.0
    # The W3C SSML spec defines these named levels:
    #   silent, x-soft, soft, medium, loud, x-loud
    # Map extreme values so engines without dB support still behave sensibly.
    if db <= -20:
        return "x-soft"
    if db >= 20:
        return "x-loud"
    # Standard +/-NdB is valid SSML; normalize the suffix to "dB".
    sign = "+" if db >= 0 else ""
    return f"{sign}{db:g}dB"


# ---------------------------------------------------------------------------
# SSML writers
# ---------------------------------------------------------------------------


@dataclass
class _ContourSteps:
    """Per-word pitch steps approximating a contour (espeak-ng only)."""

    targets: tuple[tuple[float, float], ...]
    total_words: int
    next_word: int = 0

    def next_ratio(self) -> float:
        if self.total_words <= 1:
            # A single word cannot carry a glide: use the contour's most
            # salient target, so the word still moves the right way.
            return max(
                (1.0 + change / 100.0 for _, change in self.targets),
                key=lambda ratio: abs(ratio - 1.0),
            )
        position = (self.next_word + 0.5) / self.total_words
        self.next_word += 1
        return _contour_value(self.targets, position)


@dataclass(frozen=True)
class _State:
    """Rendering state inherited from enclosing elements."""

    # F0 ratio to the voice baseline that the IML asks for (espeak-ng only).
    pitch_ratio: float = 1.0
    # espeak-ng pitch parameter in effect (50 = voice default).
    pitch_param: float = 50.0
    # Amplitude gain relative to the sentence volume (espeak-ng only).
    gain: float = 1.0
    contour: _ContourSteps | None = None


class _SSMLWriter:
    """Serialize an :class:`IMLDocument` as standard SSML 1.1."""

    def __init__(self, language: str, speaker_voices: Mapping[str, str]) -> None:
        self.language = language
        self.speaker_voices = speaker_voices
        self.notes: list[str] = []

    def document(self, doc: IMLDocument) -> str:
        body = "".join(self.utterance(u) for u in doc.utterances)
        return (
            f'<speak version="1.1" xmlns="{SSML_NAMESPACE}"'
            f"{_ssml_attr('xml:lang', self.language)}>{body}</speak>"
        )

    def utterance(self, utt: Utterance) -> str:
        sentence = f"<s>{self.children(utt.children, _State())}</s>"
        voice = self.speaker_voices.get(utt.speaker_id) if utt.speaker_id else None
        if voice:
            sentence = f"<voice{_ssml_attr('name', voice)}>{sentence}</voice>"
        return sentence

    def children(self, children: tuple[ChildNode, ...], state: _State) -> str:
        parts: list[str] = []
        for child in children:
            if isinstance(child, str):
                parts.append(self.text(child, state))
            elif isinstance(child, Pause):
                duration = _check_pause(child)
                if duration > 0:
                    parts.append(f'<break time="{duration}ms"/>')
            elif isinstance(child, Prosody):
                parts.append(self.prosody(child, state))
            elif isinstance(child, Emphasis):
                parts.append(self.emphasis(child, state))
            elif isinstance(child, Segment):
                parts.append(self.segment(child, state))
        return "".join(parts)

    def text(self, text: str, state: _State) -> str:
        return _escape_xml(text)

    def prosody(self, p: Prosody, state: _State) -> str:
        attrs = ""
        has_pitch = _pitch_ratio(p.pitch, 100.0) is not None
        if has_pitch:
            attrs += _ssml_attr("pitch", p.pitch)
        if p.volume is not None and _volume_gain(p.volume) is not None:
            attrs += _ssml_attr("volume", _iml_volume_to_ssml(p.volume))
        if p.rate is not None and _rate_factor(p.rate) is not None:
            attrs += _ssml_attr("rate", p.rate)

        contour = ""
        if p.pitch_contour == _FLAT_CONTOUR:
            contour = _ssml_attr("range", "x-low")
        elif p.pitch_contour in _CONTOUR_TARGETS:
            contour = _ssml_attr("contour", _contour_attr(_CONTOUR_TARGETS[p.pitch_contour]))

        inner = self.children(p.children, state)
        if contour and has_pitch:
            # Nest the contour so its targets are relative to the shifted pitch.
            inner = f"<prosody{contour}>{inner}</prosody>"
        else:
            attrs += contour
        if not attrs:
            return inner  # No mappable attrs -- emit content without wrapper.
        return f"<prosody{attrs}>{inner}</prosody>"

    def emphasis(self, e: Emphasis, state: _State) -> str:
        level = _ssml_attr("level", e.level) if e.level in _EMPHASIS_EFFECTS else ""
        return f"<emphasis{level}>{self.children(e.children, state)}</emphasis>"

    def segment(self, s: Segment, state: _State) -> str:
        inner = self.children(s.children, state)
        rate = _TEMPO_RATES.get(s.tempo or "")
        if rate is None:
            return inner  # No SSML equivalent -- unwrap children.
        return f"<prosody{_ssml_attr('rate', rate)}>{inner}</prosody>"


# ---------------------------------------------------------------------------
# espeak-ng adaptation
# ---------------------------------------------------------------------------

# espeak-ng pitch parameter (50 = the voice's default) -> realized F0 ratio,
# measured on espeak-ng 1.51 (mean over its default male voice and the +f3
# variant). espeak-ng applies a relative SSML pitch to this parameter:
# pitch="+20%" moves it from 50 to 60, which raises F0 by only ~10%.
_ESPEAK_PITCH_CURVE: tuple[tuple[float, float], ...] = (
    (10, 0.729), (20, 0.777), (30, 0.8405), (40, 0.912), (50, 1.0),
    (60, 1.098), (70, 1.213), (80, 1.3425), (90, 1.4925), (100, 1.654),
)
# Nominal baseline F0 of espeak-ng's default voices, for absolute Hz values.
_ESPEAK_NOMINAL_F0_HZ = 100.0
# Sentence volume in % of espeak-ng's default. At 100% espeak-ng speech
# peaks at 0.6-0.8 of full scale for most voices but up to ~1.16 (so it
# clips) for some, e.g. Spanish, measured across 16 languages and 6
# variants. Spans up to _ESPEAK_MAX_VOLUME (peak <= ~0.81) therefore do not
# clip; the sentence volume is lowered only for documents with louder spans.
_ESPEAK_BASE_VOLUME = 40.0
_ESPEAK_MAX_VOLUME = 70.0

# Characters of the sentence-final punctuation run (moved by
# _split_final_punctuation).
_FINAL_PUNCTUATION = ".,;:!?\u2026\"'\u201d\u2019)]}\u00bb"
_XML_WHITESPACE = " \t\r\n"
_WORD_TOKEN_RE = re.compile(r"\S+\s*")


def _espeak_param_for_ratio(ratio: float) -> tuple[float, bool]:
    """Return the espeak-ng pitch parameter giving ``ratio``, and whether it was clamped."""
    curve = _ESPEAK_PITCH_CURVE
    if not ratio > 0.0:
        return curve[0][0], True
    target = math.log(ratio)
    if target <= math.log(curve[0][1]):
        return curve[0][0], ratio < curve[0][1] * 0.99
    if target >= math.log(curve[-1][1]):
        return curve[-1][0], ratio > curve[-1][1] * 1.01
    for (p0, r0), (p1, r1) in zip(curve, curve[1:], strict=False):
        l0, l1 = math.log(r0), math.log(r1)
        if target <= l1:
            return p0 + (p1 - p0) * (target - l0) / (l1 - l0), False
    return curve[-1][0], False  # pragma: no cover - loop always returns


def _count_words(children: tuple[ChildNode, ...]) -> int:
    count = 0
    for child in children:
        if isinstance(child, str):
            count += len(child.split())
        elif isinstance(child, (Prosody, Emphasis, Segment)):
            count += _count_words(child.children)
    return count


def _peak_gain(children: tuple[ChildNode, ...], gain: float = 1.0) -> float:
    """Return the largest cumulative amplitude gain applied to any spoken text.

    Gains are combined and clamped exactly as :meth:`_EspeakWriter._shifted`
    combines and clamps them. Any non-space text counts, since espeak-ng
    speaks symbols too ("!" alone, "&", "%", emoji).
    """
    peak = 0.0
    for child in children:
        if isinstance(child, str):
            if child.strip(_XML_WHITESPACE):
                peak = max(peak, gain)
        elif isinstance(child, Prosody):
            inner = _clamp_gain(gain * (_volume_gain(child.volume) or 1.0))
            peak = max(peak, _peak_gain(child.children, inner))
        elif isinstance(child, Emphasis):
            _, emph_gain = _EMPHASIS_EFFECTS.get(child.level, _EMPHASIS_EFFECTS["moderate"])
            peak = max(peak, _peak_gain(child.children, _clamp_gain(gain * emph_gain)))
        elif isinstance(child, Segment):
            peak = max(peak, _peak_gain(child.children, gain))
    return peak


def _strip_final_text(node: ChildNode) -> tuple[ChildNode, str]:
    """Remove the trailing punctuation run from the last text inside ``node``."""
    if isinstance(node, str):
        body = node.rstrip(_XML_WHITESPACE)
        stripped = body.rstrip(_FINAL_PUNCTUATION)
        if len(stripped) == len(body):
            return node, ""
        return stripped, body[len(stripped) :]
    if isinstance(node, Pause):
        return node, ""
    kids = list(node.children)
    while kids and isinstance(kids[-1], str) and not kids[-1].strip(_XML_WHITESPACE):
        kids.pop()
    if not kids:
        return node, ""
    last, punct = _strip_final_text(kids[-1])
    if not punct:
        return node, ""
    kids[-1] = last
    return replace(node, children=tuple(kids)), punct


def _append_to_last_text(children: tuple[ChildNode, ...], suffix: str) -> tuple[ChildNode, ...]:
    """Append ``suffix`` to the innermost last text of ``children``."""
    kids = list(children)
    while kids and isinstance(kids[-1], str) and not kids[-1].strip(_XML_WHITESPACE):
        kids.pop()
    if kids and isinstance(kids[-1], (Prosody, Emphasis, Segment)):
        kids[-1] = replace(kids[-1], children=_append_to_last_text(kids[-1].children, suffix))
    elif kids and isinstance(kids[-1], str):
        kids[-1] = kids[-1].rstrip(_XML_WHITESPACE) + suffix
    else:
        kids.append(suffix)
    return tuple(kids)


def _split_final_punctuation(
    children: tuple[ChildNode, ...],
) -> tuple[tuple[ChildNode, ...], str, tuple[Pause, ...]]:
    """Place sentence-final punctuation where espeak-ng pauses correctly.

    espeak-ng 1.51 adds ~0.3 s of silence at the end of a sentence when a
    final "." follows a closing tag, and when any other final punctuation
    ("?", "!", ",", "...") precedes one. Returns the children with a final
    "." moved into the innermost last text, plus any other final
    punctuation and trailing pauses, which the caller writes after all
    elements.
    """
    kids = list(children)
    pauses: list[Pause] = []
    while kids and (
        isinstance(kids[-1], Pause)
        or (isinstance(kids[-1], str) and not kids[-1].strip(_XML_WHITESPACE))
    ):
        last = kids.pop()
        if isinstance(last, Pause):
            pauses.insert(0, last)
    if not kids:
        return children, "", ()
    stripped, punct = _strip_final_text(kids[-1])
    if not punct:
        return children, "", ()
    kids[-1] = stripped
    if punct == ".":
        return _append_to_last_text(tuple(kids), "."), "", tuple(pauses)
    return tuple(kids), punct, tuple(pauses)


class _EspeakWriter(_SSMLWriter):
    """SSML adapted to what espeak-ng 1.51 renders (see module docstring)."""

    def __init__(
        self,
        language: str,
        speaker_voices: Mapping[str, str],
        base_f0_hz: float = _ESPEAK_NOMINAL_F0_HZ,
    ) -> None:
        super().__init__(language, speaker_voices)
        self.base_f0_hz = base_f0_hz
        self.sentence_volume = _ESPEAK_BASE_VOLUME

    def document(self, doc: IMLDocument) -> str:
        peak = max(
            (_peak_gain(_split_final_punctuation(u.children)[0]) for u in doc.utterances),
            default=1.0,
        )
        self.sentence_volume = min(_ESPEAK_BASE_VOLUME, _ESPEAK_MAX_VOLUME / max(peak, 1e-9))
        return super().document(doc)

    def utterance(self, utt: Utterance) -> str:
        children, trailing, pauses = _split_final_punctuation(utt.children)
        inner = self.children(children, _State())
        volume = max(1, math.floor(self.sentence_volume))
        tail = _escape_xml(trailing) + self.children(pauses, _State())
        sentence = f'<s><prosody volume="{volume}%">{inner}</prosody>{tail}</s>'
        voice = self.speaker_voices.get(utt.speaker_id) if utt.speaker_id else None
        if voice:
            sentence = f"<voice{_ssml_attr('name', voice)}>{sentence}</voice>"
        return sentence

    def _pitch_step(self, ratio: float, state: _State) -> tuple[str, float]:
        """Return the espeak-ng pitch attribute value reaching ``ratio``, and its parameter."""
        param, clamped = _espeak_param_for_ratio(ratio)
        if clamped:
            semitones = 12.0 * math.log2(ratio) if ratio > 0 else -math.inf
            self.notes.append(
                f"pitch {semitones:+.1f} semitones from the voice baseline is beyond what "
                "espeak-ng can render (about -5.5 to +8.7 semitones); clamped"
            )
        pct = round((param / state.pitch_param - 1.0) * 100.0)
        return f"{pct:+d}%", state.pitch_param * (1.0 + pct / 100.0)

    def _shifted(
        self,
        state: _State,
        ratio: float | None,
        gain: float | None,
        rate: str | None,
        contour: _ContourSteps | None = None,
        *,
        absolute_pitch: bool = False,
    ) -> tuple[str, _State]:
        """Return prosody attributes for a change, and the state inside it.

        ``ratio`` is relative to the enclosing pitch, or to the voice
        baseline when ``absolute_pitch`` is true (an IML "185Hz" value).
        ``gain`` is relative to the enclosing volume; the resulting
        cumulative gain is clamped as in :func:`_peak_gain`.
        """
        attrs = ""
        new_state = state if contour is None else replace(state, contour=contour)
        if ratio is not None:
            target = ratio if absolute_pitch else state.pitch_ratio * ratio
            if abs(target - state.pitch_ratio) > 1e-9:
                value, param = self._pitch_step(target, state)
                if value != "+0%":
                    attrs += _ssml_attr("pitch", value)
                new_state = replace(new_state, pitch_ratio=target, pitch_param=param)
        if gain is not None and abs(gain - 1.0) > 1e-9:
            total = _clamp_gain(state.gain * gain)
            if abs(math.log(total / (state.gain * gain))) > 1e-6:
                self.notes.append(
                    f"a volume more than {_MAX_GAIN_DB:g} dB above or below the sentence "
                    f"volume was clamped to {_gain_db(total):+.0f} dB"
                )
            percent = max(1, round(total / state.gain * 100.0))
            if percent != 100:
                attrs += _ssml_attr("volume", f"{percent}%")
            new_state = replace(new_state, gain=state.gain * percent / 100.0)
        if rate is not None:
            attrs += _ssml_attr("rate", rate)
        return attrs, new_state

    def text(self, text: str, state: _State) -> str:
        steps = state.contour
        if steps is None:
            return _escape_xml(text)
        parts: list[str] = []
        stripped = text.lstrip()
        parts.append(_escape_xml(text[: len(text) - len(stripped)]))
        for m in _WORD_TOKEN_RE.finditer(stripped):
            target = state.pitch_ratio * steps.next_ratio()
            value, _ = self._pitch_step(target, state)
            token = _escape_xml(m.group())
            parts.append(token if value == "+0%" else f'<prosody pitch="{value}">{token}</prosody>')
        return "".join(parts)

    def prosody(self, p: Prosody, state: _State) -> str:
        rate = p.rate if _rate_factor(p.rate) is not None else None
        contour = None
        if p.pitch_contour in _CONTOUR_TARGETS:
            contour = _ContourSteps(
                _CONTOUR_TARGETS[p.pitch_contour], _count_words(p.children)
            )
        attrs, inner_state = self._shifted(
            state,
            _pitch_ratio(p.pitch, self.base_f0_hz),
            _volume_gain(p.volume),
            rate,
            contour,
            absolute_pitch=p.pitch is not None and _PITCH_HZ_RE.fullmatch(p.pitch) is not None,
        )
        if p.pitch_contour == _FLAT_CONTOUR:
            attrs += _ssml_attr("range", "x-low")
        inner = self.children(p.children, inner_state)
        return f"<prosody{attrs}>{inner}</prosody>" if attrs else inner

    def emphasis(self, e: Emphasis, state: _State) -> str:
        ratio, gain = _EMPHASIS_EFFECTS.get(e.level, _EMPHASIS_EFFECTS["moderate"])
        attrs, inner_state = self._shifted(state, ratio, gain, None)
        inner = self.children(e.children, inner_state)
        return f"<prosody{attrs}>{inner}</prosody>" if attrs else inner


def _render(
    doc: IMLDocument,
    *,
    vendor: str | None,
    language: str,
    speaker_voices: Mapping[str, str],
    base_f0_hz: float = _ESPEAK_NOMINAL_F0_HZ,
) -> tuple[str, list[str]]:
    """Render ``doc`` as SSML; returns the SSML and notes about approximations."""
    writer: _SSMLWriter
    if vendor in _ESPEAK_VENDORS:
        writer = _EspeakWriter(language, speaker_voices, base_f0_hz)
    else:
        writer = _SSMLWriter(language, speaker_voices)
    try:
        ssml = writer.document(doc)
    except ArithmeticError as exc:  # pragma: no cover - values are bounded above
        raise ConversionError(f"An attribute value is out of the renderable range: {exc}") from exc
    return ssml, list(dict.fromkeys(writer.notes))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class IMLToSSML:
    """Convert IML documents to SSML 1.1.

    Parameters
    ----------
    vendor:
        ``None`` (default) for standard SSML 1.1, or ``"espeak-ng"`` (alias
        ``"espeak"``) for SSML adapted to espeak-ng (see the module
        docstring). Other values are reserved for vendor-specific quirks
        (e.g. Google, Amazon, Microsoft); they warn and produce standard SSML.
    default_language:
        BCP 47 tag written as ``xml:lang`` when the document has no
        ``language`` attribute. Defaults to ``"en-US"``.
    speaker_voices:
        Optional mapping from IML ``speaker_id`` to a TTS voice name.
        Utterances by a mapped speaker are wrapped in ``<voice name="...">``;
        other speaker ids are not written (they are identifiers, not voices).
    strict:
        When true (default), a document with validation errors (see
        :class:`~prosody_protocol.validator.IMLValidator`) is rejected with
        :class:`~prosody_protocol.exceptions.IMLValidationError` instead of
        being converted. When false, invalid attribute values are ignored
        (spec 6.2), except values that cannot be rendered at all, such as a
        negative pause, which raise
        :class:`~prosody_protocol.exceptions.ConversionError`.
    """

    def __init__(
        self,
        vendor: str | None = None,
        *,
        default_language: str = DEFAULT_LANGUAGE,
        speaker_voices: Mapping[str, str] | None = None,
        strict: bool = True,
    ) -> None:
        self.vendor = vendor
        self.default_language = normalize_language_tag(default_language, "default_language")
        self.speaker_voices: dict[str, str] = dict(speaker_voices or {})
        self.strict = strict
        self._parser = IMLParser()
        self._validator = IMLValidator()
        if vendor is not None and vendor not in _ESPEAK_VENDORS:
            warnings.warn(
                f"Vendor-specific SSML adaptation for {vendor!r} is not yet "
                f"implemented. Output will use standard SSML 1.1 without "
                f"vendor extensions.",
                stacklevel=2,
            )

    def convert(self, iml_string: str) -> str:
        """Convert an IML XML string to an SSML XML string.

        Raises :class:`~prosody_protocol.exceptions.ConversionError` if the
        IML cannot be parsed or rendered, and
        :class:`~prosody_protocol.exceptions.IMLValidationError` if it has
        validation errors (``strict`` mode).
        """
        try:
            doc = self._parser.parse(iml_string)
        except IMLParseError as exc:
            raise ConversionError(f"Cannot parse IML for SSML conversion: {exc}") from exc
        if self.strict:
            self._validator.validate(iml_string).raise_for_errors()
        return self._convert(doc)

    def convert_doc(self, doc: IMLDocument) -> str:
        """Convert an :class:`IMLDocument` to an SSML XML string.

        Raises the same errors as :meth:`convert`.
        """
        if self.strict:
            self._validator.validate(self._parser.to_iml_string(doc)).raise_for_errors()
        return self._convert(doc)

    def _convert(self, doc: IMLDocument) -> str:
        ssml, notes = _render(
            doc,
            vendor=self.vendor,
            language=doc.language or self.default_language,
            speaker_voices=self.speaker_voices,
        )
        for note in notes:
            warnings.warn(note, stacklevel=3)
        return ssml
