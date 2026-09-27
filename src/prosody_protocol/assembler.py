"""IML Assembler -- constructs IMLDocument objects from pipeline outputs.

Takes STT word alignments, prosody features, emotion classification,
and pause information, and assembles a structured IML document.

Steps:
  1. Resolve pauses: each detected silence between two words (or, where
     none was detected, the gap between their timings) of at least 200 ms
  2. Group words into utterances at sentence ends -- or, in text without
     sentence punctuation, at pauses of 1 s or more. The pause that ends an
     utterance is kept at the start of the next one.
  3. Measure the speaker baseline: from calibration speech if given, else
     from the recording's utterances, each counting once
  4. Classify emotion per utterance relative to that baseline -- when there
     is one to compare with (see :meth:`IMLAssembler.assemble`)
  5. Wrap an utterance that is higher, louder, faster or slower as a whole
     (or has an unusual voice quality throughout) in one <prosody>
  6. Mark words that stand out: <emphasis> for words louder or higher than
     their neighbours, <prosody> for other offsets, pitch contours and
     unusual voice quality
  7. Insert <pause> elements

Pitch and volume offsets are relative to the speaker baseline; inside an
utterance-level ``<prosody>`` they are relative to the utterance's level,
and in a recording that is a single utterance (without calibration speech)
to that utterance's typical level. Markup is at most two elements deep
(spec Section 5.2).
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
import warnings
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, replace
from itertools import accumulate, groupby, pairwise
from typing import Any

from ._types import PauseInterval, SpanFeatures, WordAlignment
from .emotion_classifier import (
    BaselineAwareEmotionClassifier,
    EmotionClassifier,
    RuleBasedEmotionClassifier,
    SpeakerBaseline,
    _finite,
    _mean,
    _median,
    _positive,
    _semitones,
    _span_f0,
    _span_intensity,
    _span_rate,
)
from .models import (
    ChildNode,
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Utterance,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Utterances whose emotion is classified with lower confidence carry no
#: ``emotion`` or ``confidence`` attribute (spec 6.2.3 treats confidence
#: below 0.5 as low).
DEFAULT_MIN_EMOTION_CONFIDENCE = 0.5

# Silences at least this long (ms) become <pause> elements; shorter ones are
# ordinary speech rhythm (spec 3.3).
MIN_PAUSE_MS = 200

# In text without sentence punctuation, pauses at least this long (ms) also
# end an utterance. Punctuated text is split at sentence ends only.
UTTERANCE_SPLIT_PAUSE_MS = 1000

# Utterance-level <prosody>: the utterance's median pitch (semitones) or
# loudness (dB) differs from the speaker baseline by at least this much ...
UTTERANCE_PITCH_ST = 2.0  # about 12 %
UTTERANCE_VOLUME_DB = 4.0
# ... or its speech rate is at least this factor faster or slower. (Measured
# rates of speech at the same tempo can differ by 25 %, so smaller changes
# are not marked.)
UTTERANCE_RATE_RATIO = 1.4

# Without calibration speech, emotion is only classified when the recording
# has at least this many utterances and most of them lie within
# UTTERANCE_PITCH_ST and UTTERANCE_VOLUME_DB of the baseline: otherwise it
# does not show which level is the speaker's usual one.
MIN_BASELINE_UTTERANCES = 3

# Word-level <prosody>: the word's pitch or loudness differs from the
# utterance's level by at least this much.
WORD_PITCH_ST = 2.5  # about 15 %
WORD_VOLUME_DB = 5.0

# <emphasis>: a word's prominence is how much louder it is than its
# neighbours in units of EMPHASIS_VOLUME_DB plus how much higher in units of
# EMPHASIS_PITCH_ST (quieter or lower counts as zero). Prominence 1 is
# "moderate" emphasis, 2 "strong".
EMPHASIS_VOLUME_DB = 6.0
EMPHASIS_PITCH_ST = 3.5  # about 22 %
# Neighbours on each side that a word is compared with.
EMPHASIS_CONTEXT_WORDS = 4

# pitch_contour: pitch movement of at least CONTOUR_MIN_ST semitones within
# a word; "-sharp" when it covers SHARP_CONTOUR_ST semitones or more at
# SHARP_CONTOUR_ST_PER_S or faster.
CONTOUR_MIN_ST = 2.0
SHARP_CONTOUR_ST = 5.0
SHARP_CONTOUR_ST_PER_S = 20.0
# Contours are only classified from at least this many voiced F0 samples,
# and only over spans up to this long (a whole sentence has no one contour).
MIN_CONTOUR_SAMPLES = 8
MAX_CONTOUR_SPAN_MS = 2000
# F0 samples this far (semitones) from the span's median are pitch-tracking
# errors (octave jumps) and are ignored.
_OCTAVE_ERROR_ST = 10.0

# quality: emitted only when the word's voice is far from the speaker's
# usual -- HNR this many dB lower (breathy, whispery), or jitter (creaky)
# or shimmer (tense, harsh) this many times higher.
QUALITY_HNR_DROP_DB = 6.0
QUALITY_PERTURBATION_RATIO = 2.0

# Larger offsets are measurement errors (such as pitch-tracking octave
# jumps), not speech: they are neither emitted nor counted as emphasis.
MAX_PITCH_ST = 12.0
MAX_VOLUME_DB = 40.0
# Likewise, speech rates more than this factor faster or slower than the
# speaker's.
MAX_RATE_RATIO = 2.0

# Extended f0_contour attributes carry at most this many values.
EXTENDED_CONTOUR_POINTS = 10

# Thresholds of the previous assembler, which compared each word with the
# mean of its own utterance. They no longer affect the output.
_DEPRECATED_CONSTANTS = {
    "F0_DEVIATION_PCT": 15.0,
    "INTENSITY_DEVIATION_DB": 5.0,
    "EMPHASIS_INTENSITY_DB": 6.0,
    "EMPHASIS_F0_PCT": 20.0,
}


def __getattr__(name: str) -> float:
    if name in _DEPRECATED_CONSTANTS:
        warnings.warn(
            f"prosody_protocol.assembler.{name} is deprecated and no longer used; see"
            " UTTERANCE_PITCH_ST, WORD_PITCH_ST, EMPHASIS_PITCH_ST and related constants",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_CONSTANTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Sentence-final punctuation; closing quotes and brackets may follow it.
_SENTENCE_END = frozenset(".!?。！？")
_CLOSERS = "\"'”’)]}»"
_OPENERS = "\"'“‘([{«¿¡"
# Abbreviations that introduce what follows and never end a sentence.
_TITLES = frozenset({"mr.", "mrs.", "ms.", "dr.", "prof.", "vs.", "e.g.", "i.e."})
# Abbreviations and initials that may end a sentence ("Take vitamin C.") or
# not ("John F. Kennedy", "the U.S. Army"). The pronoun "I." is not one.
_ABBREVIATIONS = frozenset(
    {"st.", "mt.", "ave.", "jr.", "sr.", "ph.d.", "inc.", "ltd.", "co.", "corp.", "etc."}
)
_INITIALS = re.compile(r"[A-Z]\.|(?:[A-Za-z]\.){2,}")
# After such an abbreviation, a sentence is taken to end only when one of
# these common sentence openers follows. This is a heuristic: "J. Doe" and
# "the U.S. Army" continue the sentence, but so, wrongly, does "... in the
# U.S. Americans are ...".
_SENTENCE_OPENERS = frozenset({
    "i", "i'm", "i've", "i'll", "i'd", "you", "you're", "we", "we're", "they", "they're",
    "he", "he's", "she", "she's", "it", "it's", "this", "that", "that's", "these", "those",
    "there", "there's", "here", "here's", "let's", "the", "a", "an", "my", "our", "your",
    "his", "her", "their", "and", "but", "so", "or", "then", "now", "well", "oh", "yes",
    "yeah", "no", "okay", "ok", "please", "thanks", "what", "what's", "who", "why", "how",
    "when", "where", "which", "if", "because", "also", "after", "before", "do", "does",
    "did", "don't", "is", "are", "was", "were", "can", "could", "will", "would", "should",
})
# Punctuation that closes or ends a phrase and follows the previous token
# without a space (besides closing brackets and quotes); other punctuation,
# such as dashes, "&" and "/", is spaced like a word.
_ATTACHING = frozenset(",.;:!?…%‰。，、！？；：")


# ---------------------------------------------------------------------------
# Words and pauses
# ---------------------------------------------------------------------------


@dataclass
class _Word:
    text: str  # the token without surrounding whitespace
    alignment: WordAlignment
    features: SpanFeatures | None
    attaches: bool = False  # follows the previous token without a space
    opens: bool = False  # the next token follows it without a space


def _pair_features(
    alignments: Sequence[WordAlignment], features: Sequence[SpanFeatures]
) -> list[SpanFeatures | None]:
    """The features of each alignment.

    Features that match the alignments one to one (as ProsodyAnalyzer
    returns them) are paired by position; otherwise by ``(start_ms, end_ms)``.
    """
    if len(features) == len(alignments) and all(
        (f.start_ms, f.end_ms) == (a.start_ms, a.end_ms)
        for a, f in zip(alignments, features, strict=True)
    ):
        return list(features)
    by_span = {(f.start_ms, f.end_ms): f for f in features}
    return [by_span.get((a.start_ms, a.end_ms)) for a in alignments]


def _closes(text: str) -> bool:
    """Whether *text* is only closing or terminal punctuation (``,``, ``?!``, ``)``, ``”``)."""
    return bool(text) and all(
        c in _ATTACHING or unicodedata.category(c) in ("Pe", "Pf") for c in text
    )


def _opens(text: str) -> bool:
    """Whether *text* is only opening punctuation (``(``, ``“``, ``¿``)."""
    return bool(text) and all(c in "¿¡" or unicodedata.category(c) in ("Ps", "Pi") for c in text)


def _mark_spacing(words: list[_Word]) -> None:
    """Decide which tokens are written without a space next to their neighbour.

    Closing and terminal punctuation attaches to the token before it and
    opening punctuation to the token after it. A standalone straight quote
    ``"`` opens when an even number of them came before it, and closes
    otherwise.
    """
    straight_quotes = 0
    for word in words:
        if word.text == '"':
            word.opens = straight_quotes % 2 == 0
            word.attaches = not word.opens
        else:
            word.attaches, word.opens = _closes(word.text), _opens(word.text)
        straight_quotes += word.text.count('"')


def _ends_sentence(text: str, next_text: str | None, *, cased: bool = True) -> bool:
    """Whether *text* ends a sentence, given the token after it.

    Ellipses and titles such as ``Dr.`` do not end a sentence; other
    abbreviations and initials (``etc.``, ``B.``, ``U.S.``) do only when a
    common sentence opener follows (``Plan B. We``, not ``the U.S. Army``).
    In *cased* text (text that uses capitals at all), punctuation followed
    by a lower-case word does not end a sentence either (``5 p.m. in``).
    """
    core = text.rstrip(_CLOSERS)
    if not core or core[-1] not in _SENTENCE_END or core.endswith(".."):
        return False
    following = None if next_text is None else next_text.lstrip(_OPENERS)
    if cased and following is not None and following[:1].islower():
        return False
    if core[-1] != ".":
        return True
    lower = core.lower()
    if lower in _TITLES:
        return False
    if lower in _ABBREVIATIONS or (core != "I." and _INITIALS.fullmatch(core)):
        opener = "" if following is None else following.rstrip(_CLOSERS + ",.;:!?")
        return opener.lower().replace("’", "'") in _SENTENCE_OPENERS
    return True


def _resolve_pauses(words: list[_Word], pauses: Sequence[PauseInterval]) -> list[int]:
    """The pause (ms) at each boundary between consecutive words, 0 for none.

    A detected silence belongs to the boundary its midpoint falls in (between
    the centres of the two words), clipped to the outer edges of those words;
    silence before the first or after the last word is not a pause. Where no
    silence was detected, the gap between the word timings is used.
    Durations are rounded to whole milliseconds, pauses shorter than
    :data:`MIN_PAUSE_MS` are dropped, and a pause before punctuation moves
    after it -- unless the punctuation ends the transcript, in which case
    the silence is after the last word and dropped.
    """
    if len(words) < 2:
        return []
    centres = list(
        accumulate(((w.alignment.start_ms + w.alignment.end_ms) / 2 for w in words), max)
    )
    detected: dict[int, tuple[int, int]] = {}
    for pause in pauses:
        boundary = bisect_right(centres, (pause.start_ms + pause.end_ms) / 2) - 1
        if not 0 <= boundary < len(words) - 1:
            continue
        start = max(pause.start_ms, words[boundary].alignment.start_ms)
        end = min(pause.end_ms, words[boundary + 1].alignment.end_ms)
        if end > start:
            known = detected.get(boundary)
            detected[boundary] = (start, end) if known is None else (
                min(known[0], start), max(known[1], end)
            )

    result: list[int] = []
    for boundary in range(len(words) - 1):
        if boundary in detected:
            start, end = detected[boundary]
        else:
            start = words[boundary].alignment.end_ms
            end = words[boundary + 1].alignment.start_ms
        duration = round(end - start)
        result.append(duration if duration >= MIN_PAUSE_MS else 0)

    for boundary, duration in enumerate(result):
        if duration and words[boundary + 1].attaches:
            result[boundary] = 0
            if boundary + 1 < len(result):
                result[boundary + 1] += duration
    return result


def _group_into_utterances(
    words: list[_Word], boundary_pauses: list[int]
) -> list[tuple[int, list[int]]]:
    """Split word indices into utterances, each with the pause (ms) before it."""
    punctuated = any(_ends_sentence(w.text, None) for w in words)
    cased = any(c.isupper() for w in words for c in w.text)
    groups: list[tuple[int, list[int]]] = []
    leading = 0
    current: list[int] = []
    for i, word in enumerate(words):
        current.append(i)
        if i == len(words) - 1:
            break
        pause = boundary_pauses[i]
        if _ends_sentence(word.text, words[i + 1].text, cased=cased) or (
            not punctuated and pause >= UTTERANCE_SPLIT_PAUSE_MS
        ):
            groups.append((leading, current))
            leading, current = pause, []
    groups.append((leading, current))
    return groups


# ---------------------------------------------------------------------------
# Prosodic descriptions
# ---------------------------------------------------------------------------


def _format_pitch(st: float) -> str:
    """A pitch offset in semitones as a relative percentage (``+15%``)."""
    return f"{(2.0 ** (st / 12.0) - 1.0) * 100.0:+.0f}%"


def _format_volume(db: float) -> str:
    return f"{db:+.0f}dB"


def _running_median(values: list[float], half_window: int = 2) -> list[float]:
    return [
        statistics.median(values[max(0, i - half_window): i + half_window + 1])
        for i in range(len(values))
    ]


def _pitch_contour(features: SpanFeatures) -> str | None:
    """Classify the pitch movement within a span (spec 3.2 vocabulary).

    Returns ``None`` when the span is too long or has too few voiced F0
    samples for a reliable contour. The contour is converted to semitones,
    cleared of octave errors and median-smoothed; the first and last thirds
    are then compared with each other and with the middle.
    """
    duration_ms = features.end_ms - features.start_ms
    if not 0 < duration_ms <= MAX_CONTOUR_SPAN_MS:
        return None
    samples = [v for v in features.f0_contour or () if math.isfinite(v) and v > 0.0]
    if len(samples) < MIN_CONTOUR_SAMPLES:
        return None
    reference = statistics.median(samples)
    st = [_semitones(v, reference) for v in samples]
    st = _running_median([v for v in st if abs(v) <= _OCTAVE_ERROR_ST])
    if len(st) < MIN_CONTOUR_SAMPLES:
        return None

    third = len(st) // 3
    start = statistics.median(st[:third])
    end = statistics.median(st[-third:])
    middle = st[third:-third]
    if max(middle) - max(start, end) >= CONTOUR_MIN_ST:
        return "rise-fall"
    if min(start, end) - min(middle) >= CONTOUR_MIN_ST:
        return "fall-rise"
    movement = end - start
    if abs(movement) < CONTOUR_MIN_ST:
        return "flat"
    sharp = (
        abs(movement) >= SHARP_CONTOUR_ST
        and abs(movement) / (duration_ms / 1000.0) >= SHARP_CONTOUR_ST_PER_S
    )
    if movement > 0:
        return "rise-sharp" if sharp else "rise"
    return "fall-sharp" if sharp else "fall"


def _notable_quality(features: SpanFeatures, baseline: SpeakerBaseline) -> str | None:
    """The span's voice quality label, if its voice is far from the speaker's usual."""
    label = features.quality
    if label in ("breathy", "whispery"):
        hnr, usual = _finite(features.hnr), baseline.hnr
        notable = hnr is not None and usual is not None and hnr <= usual - QUALITY_HNR_DROP_DB
    elif label in ("creaky", "tense", "harsh"):
        if label == "creaky":
            value, usual = _finite(features.jitter), baseline.jitter
        else:
            value, usual = _finite(features.shimmer), baseline.shimmer
        notable = (
            value is not None
            and usual is not None
            and usual > 0.0
            and value >= QUALITY_PERTURBATION_RATIO * usual
        )
    else:  # modal, unmeasured or unknown
        notable = False
    return label if notable else None


def _downsample(values: list[float], points: int) -> list[float]:
    """At most *points* values: the medians of equal consecutive chunks."""
    if len(values) <= points:
        return values
    edges = [round(i * len(values) / points) for i in range(points + 1)]
    return [statistics.median(values[a:b]) for a, b in pairwise(edges)]


def _extended_attrs(features: SpanFeatures) -> dict[str, Any]:
    """Extended attributes (spec Section 4) of one span, as far as they were measured.

    The analyzer's values are already in spec units (Hz, dB, syllables/s,
    percent); they are rounded, and invalid ones (negative ranges or
    perturbations, intensities of digital silence) are left out.
    """
    attrs: dict[str, Any] = {}
    f0 = _span_f0(features)
    if f0 is not None:
        attrs["f0_mean"] = round(f0, 1)
        if features.f0_range is not None:
            low, high = (_positive(v) for v in features.f0_range)
            if low is not None and high is not None and high >= low:
                attrs["f0_range"] = f"{low:.0f}-{high:.0f}"
        contour = [v for v in features.f0_contour or () if math.isfinite(v) and v > 0.0]
        if contour:
            attrs["f0_contour"] = ",".join(
                f"{v:.0f}" for v in _downsample(contour, EXTENDED_CONTOUR_POINTS)
            )
    intensity = _span_intensity(features)
    if intensity is not None:
        attrs["intensity_mean"] = round(intensity, 1)
        intensity_range = _finite(features.intensity_range)
        if intensity_range is not None and intensity_range >= 0.0:
            attrs["intensity_range"] = round(intensity_range, 1)
    rate = _finite(features.speech_rate)
    if rate is not None and rate >= 0.0:
        attrs["speech_rate"] = round(rate, 1)
    duration_ms = round(features.end_ms - features.start_ms)
    if duration_ms > 0:
        attrs["duration_ms"] = duration_ms
    for name in ("jitter", "shimmer"):
        value = _finite(getattr(features, name))
        if value is not None and value >= 0.0:
            attrs[name] = round(value, 2)
    hnr = _finite(features.hnr)
    if hnr is not None:
        attrs["hnr"] = round(hnr, 1)
    return attrs


# ---------------------------------------------------------------------------
# Utterance building
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Level:
    """What an utterance's words are compared with.

    ``f0`` (Hz) and ``intensity`` (dB) are the utterance's level. The shifts
    (semitones, dB) are how far that level lies from the speaker baseline;
    they are non-zero only when the utterance-level ``<prosody>`` states them.
    """

    f0: float | None
    intensity: float | None
    pitch_shift_st: float = 0.0
    volume_shift_db: float = 0.0


def _utterance_shift(
    spans: list[SpanFeatures], baseline: SpeakerBaseline
) -> tuple[dict[str, str], _Level]:
    """Utterance-level <prosody> attributes, and the level its words are measured from.

    The utterance's median pitch and loudness become the level when they
    differ notably from the baseline (and are then emitted); otherwise the
    baseline is the level. Without a baseline, the utterance's own median is.
    """
    attrs: dict[str, str] = {}
    f0 = _median(_span_f0(f) for f in spans)
    level_f0 = baseline.f0_mean if baseline.f0_mean is not None else f0
    pitch_shift = 0.0
    if f0 is not None and baseline.f0_mean is not None:
        shift = _semitones(f0, baseline.f0_mean)
        if UTTERANCE_PITCH_ST <= abs(shift) <= MAX_PITCH_ST:
            attrs["pitch"] = _format_pitch(shift)
            level_f0, pitch_shift = f0, shift

    intensity = _median(_span_intensity(f) for f in spans)
    level_intensity = (
        baseline.intensity_mean if baseline.intensity_mean is not None else intensity
    )
    volume_shift = 0.0
    if intensity is not None and baseline.intensity_mean is not None:
        shift = intensity - baseline.intensity_mean
        if UTTERANCE_VOLUME_DB <= abs(shift) <= MAX_VOLUME_DB:
            attrs["volume"] = _format_volume(shift)
            level_intensity, volume_shift = intensity, shift

    rate = _mean(_span_rate(f) for f in spans)
    if rate is not None and baseline.speech_rate is not None:
        ratio = rate / baseline.speech_rate
        if UTTERANCE_RATE_RATIO <= max(ratio, 1.0 / ratio) <= MAX_RATE_RATIO:
            attrs["rate"] = f"{round(ratio * 20.0) * 5}%"  # nearest 5 %
    return attrs, _Level(level_f0, level_intensity, pitch_shift, volume_shift)


def _offset_attrs(pitch_st: float, volume_db: float) -> dict[str, str]:
    """``pitch`` and ``volume`` attributes for these offsets, except those that round to 0."""
    attrs = {"pitch": _format_pitch(pitch_st), "volume": _format_volume(volume_db)}
    return {name: value for name, value in attrs.items() if value[1:] not in ("0%", "0dB")}


def _neighbour_median(values: list[float | None], index: int) -> float | None:
    """Median of the measured values near *index*, excluding it."""
    lo = max(0, index - EMPHASIS_CONTEXT_WORDS)
    hi = index + EMPHASIS_CONTEXT_WORDS + 1
    return _median(values[lo:index] + values[index + 1: hi])


def _prosody(children: tuple[ChildNode, ...], attrs: dict[str, Any]) -> Prosody:
    return Prosody(children=children, **attrs)


def _join(nodes: Iterable[ChildNode]) -> list[ChildNode]:
    """*nodes* with adjacent strings joined."""
    joined: list[ChildNode] = []
    for node in nodes:
        if isinstance(node, str) and joined and isinstance(joined[-1], str):
            joined[-1] += node
        else:
            joined.append(node)
    return joined


def _is_gap(node: ChildNode) -> bool:
    """Whether *node* is a pause or a space rather than a word."""
    return isinstance(node, Pause) or (isinstance(node, str) and not node.strip())


def _wrapped(nodes: list[ChildNode], attrs: dict[str, str]) -> Prosody:
    """*nodes* inside a <prosody> with the utterance-level *attrs*."""
    if len(nodes) == 1 and isinstance(nodes[0], Prosody):
        # A single marked-up word: one element suffices, unless the word's
        # own attributes would clash with the utterance's.
        word = nodes[0]
        own = {
            f.name: getattr(word, f.name)
            for f in fields(Prosody)
            if f.name not in ("children", "extra_attributes") and getattr(word, f.name) is not None
        }
        if not own.keys() & attrs.keys():
            return _prosody(word.children, {**own, **attrs})
    return _prosody(tuple(nodes), attrs)


def _wrap(items: list[tuple[ChildNode, bool]], attrs: dict[str, str]) -> list[ChildNode]:
    """Put an utterance's words inside its utterance-level <prosody>.

    *items* pairs each node with whether it stands alone (see
    :meth:`_UtteranceBuilder._word`). Standalone words stay outside and
    split the <prosody> into several; the spaces and pauses at either end of
    each one stay outside too.
    """
    out: list[ChildNode] = []
    for alone, group in groupby(items, key=lambda item: item[1]):
        nodes = [node for node, _ in group]
        if alone:
            out.extend(nodes)
            continue
        lo, hi = 0, len(nodes)
        while lo < hi and _is_gap(nodes[lo]):
            lo += 1
        while hi > lo and _is_gap(nodes[hi - 1]):
            hi -= 1
        out.extend(nodes[:lo])
        if lo < hi:
            out.append(_wrapped(_join(nodes[lo:hi]), attrs))
        out.extend(nodes[hi:])
    return _join(out)


class _UtteranceBuilder:
    """Builds the children of one utterance."""

    def __init__(
        self,
        words: list[_Word],
        pauses_after: list[int],
        baseline: SpeakerBaseline,
        include_extended: bool,
    ) -> None:
        self.words = words
        self.pauses_after = pauses_after  # pause after each word but the last
        self.baseline = baseline
        self.include_extended = include_extended
        self.spans = [w.features for w in words if w.features is not None]
        # Voice quality is judged against the baseline, or where that has no
        # value, against the utterance's own typical voice.
        own = SpeakerBaseline.from_features(self.spans)
        self.voice = replace(baseline, **{
            f.name: getattr(own, f.name)
            for f in fields(SpeakerBaseline)
            if getattr(baseline, f.name) is None
        })

    def build(self, leading_pause: int) -> tuple[ChildNode, ...]:
        wrapper, level = _utterance_shift(self.spans, self.baseline)
        qualities = [
            None if w.features is None else _notable_quality(w.features, self.voice)
            for w in self.words
        ]
        measured = [q for w, q in zip(self.words, qualities, strict=True) if w.features]
        if len(measured) >= 2 and measured[0] is not None and len(set(measured)) == 1:
            # The whole utterance has this voice quality: say so once.
            wrapper["quality"] = measured[0]
            qualities = [None] * len(qualities)
        items = self._inline(level, qualities, wrapper)

        children: list[ChildNode] = [Pause(duration=leading_pause)] if leading_pause else []
        if wrapper:
            children.extend(_wrap(items, wrapper))
        else:
            children.extend(_join(node for node, _ in items))
        return tuple(children)

    def _inline(
        self, level: _Level, qualities: list[str | None], wrapper: dict[str, str]
    ) -> list[tuple[ChildNode, bool]]:
        """Words (marked up where notable), spaces and pauses.

        Each node comes with whether it stands alone, outside the
        utterance-level ``<prosody>`` (see :meth:`_word`).
        """
        f0s = [None if w.features is None else _span_f0(w.features) for w in self.words]
        levels = [
            None if w.features is None else _span_intensity(w.features) for w in self.words
        ]
        last_voiced = max((i for i, f0 in enumerate(f0s) if f0 is not None), default=-1)

        out: list[tuple[ChildNode, bool]] = []
        for i, word in enumerate(self.words):
            if i > 0 and not word.attaches and not self.words[i - 1].opens:
                out.append((" ", False))
            out.append(self._word(
                i, word, f0s, levels, level, qualities[i], i == last_voiced, wrapper
            ))
            if i < len(self.pauses_after) and self.pauses_after[i]:
                out.append((Pause(duration=self.pauses_after[i]), False))
        return out

    def _word(
        self,
        index: int,
        word: _Word,
        f0s: list[float | None],
        levels: list[float | None],
        level: _Level,
        quality: str | None,
        is_last_voiced: bool,
        wrapper: dict[str, str],
    ) -> tuple[ChildNode, bool]:
        """The word, marked up where notable, and whether it stands alone.

        An emphasized word with attributes of its own cannot stay inside the
        utterance-level ``<prosody>`` (*wrapper*): ``<emphasis><prosody>``
        inside it would be three elements deep. It stands alone instead,
        carrying the utterance's attributes itself.
        """
        feat = word.features
        if feat is None:
            return word.text, False
        f0, intensity = f0s[index], levels[index]

        # Emphasis: louder and/or higher than the neighbouring words.
        prominence = 0.0
        context_f0 = _neighbour_median(f0s, index)
        if f0 is not None and context_f0 is not None:
            rise = _semitones(f0, context_f0)
            if rise <= MAX_PITCH_ST:
                prominence += max(0.0, rise) / EMPHASIS_PITCH_ST
        context_intensity = _neighbour_median(levels, index)
        if intensity is not None and context_intensity is not None:
            rise = intensity - context_intensity
            if rise <= MAX_VOLUME_DB:
                prominence += max(0.0, rise) / EMPHASIS_VOLUME_DB
        emphasis = "strong" if prominence >= 2.0 else "moderate" if prominence >= 1.0 else None

        attrs: dict[str, Any] = {}
        pitch_st = volume_db = 0.0
        if f0 is not None and level.f0 is not None:
            offset = _semitones(f0, level.f0)
            if WORD_PITCH_ST <= abs(offset) <= MAX_PITCH_ST:
                attrs["pitch"] = _format_pitch(offset)
                pitch_st = offset
        if intensity is not None and level.intensity is not None:
            offset = intensity - level.intensity
            if WORD_VOLUME_DB <= abs(offset) <= MAX_VOLUME_DB:
                attrs["volume"] = _format_volume(offset)
                volume_db = offset
        if attrs or emphasis or is_last_voiced:
            contour = _pitch_contour(feat)
            if contour is not None and contour != "flat":
                attrs["pitch_contour"] = contour
        if quality is not None:
            attrs["quality"] = quality
        if self.include_extended:
            attrs.update(_extended_attrs(feat))

        if emphasis is None:
            return (_prosody((word.text,), attrs) if attrs else word.text), False
        if wrapper and attrs:
            # Standing alone, the word's pitch and volume add up the
            # utterance's shift and the word's own offset from it.
            alone: dict[str, Any] = {**wrapper, **attrs}
            alone.pop("pitch", None)
            alone.pop("volume", None)
            alone.update(_offset_attrs(
                level.pitch_shift_st + pitch_st, level.volume_shift_db + volume_db
            ))
            return Emphasis(level=emphasis, children=(_prosody((word.text,), alone),)), True
        inner: ChildNode = _prosody((word.text,), attrs) if attrs else word.text
        return Emphasis(level=emphasis, children=(inner,)), False


def _baseline_is_established(
    utterances: list[list[SpanFeatures]], baseline: SpeakerBaseline
) -> bool:
    """Whether a recording shows its speaker's typical prosody.

    That takes at least :data:`MIN_BASELINE_UTTERANCES` measured utterances,
    most of which lie within :data:`UTTERANCE_PITCH_ST` and
    :data:`UTTERANCE_VOLUME_DB` of the baseline -- a level that the other
    utterances can be said to deviate from. Two utterances that differ, or
    a recording that swings between levels, do not show which level is the
    speaker's usual one.
    """
    measured = typical = 0
    for spans in utterances:
        f0 = _median(_span_f0(f) for f in spans)
        intensity = _median(_span_intensity(f) for f in spans)
        if f0 is None and intensity is None:
            continue
        measured += 1
        pitch_typical = (
            f0 is None
            or baseline.f0_mean is None
            or abs(_semitones(f0, baseline.f0_mean)) < UTTERANCE_PITCH_ST
        )
        volume_typical = (
            intensity is None
            or baseline.intensity_mean is None
            or abs(intensity - baseline.intensity_mean) < UTTERANCE_VOLUME_DB
        )
        typical += pitch_typical and volume_typical
    return measured >= MIN_BASELINE_UTTERANCES and 2 * typical > measured


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class IMLAssembler:
    """Assemble an IMLDocument from pipeline components.

    Parameters
    ----------
    emotion_classifier:
        Classifier for each utterance's emotion. Defaults to
        :class:`~prosody_protocol.emotion_classifier.RuleBasedEmotionClassifier`.
        Classifiers with a ``classify_relative(features, baseline)`` method
        (:class:`~prosody_protocol.emotion_classifier.BaselineAwareEmotionClassifier`)
        get the speaker baseline; others get ``classify(features)``.
    include_extended:
        When ``True``, every word with acoustic features is wrapped in a
        ``<prosody>`` carrying its measurements (spec Section 4:
        ``f0_mean``, ``f0_range``, ``f0_contour``, ``intensity_mean``,
        ``intensity_range``, ``speech_rate``, ``duration_ms``, ``jitter``,
        ``shimmer``, ``hnr``). To keep nesting within two levels, an
        emphasized word in an utterance-level ``<prosody>`` steps out of
        it and carries the utterance's attributes itself.
    min_emotion_confidence:
        Utterances whose emotion is classified with lower confidence carry no
        ``emotion`` or ``confidence`` attribute. ``0.0`` keeps every label.
    """

    def __init__(
        self,
        emotion_classifier: EmotionClassifier | None = None,
        include_extended: bool = False,
        min_emotion_confidence: float = DEFAULT_MIN_EMOTION_CONFIDENCE,
    ) -> None:
        if not 0.0 <= min_emotion_confidence <= 1.0:
            raise ValueError(
                f"min_emotion_confidence must be between 0.0 and 1.0; got {min_emotion_confidence}"
            )
        self._classifier: EmotionClassifier = (
            RuleBasedEmotionClassifier() if emotion_classifier is None else emotion_classifier
        )
        self._include_extended = include_extended
        self._min_emotion_confidence = min_emotion_confidence

    def assemble(
        self,
        alignments: Sequence[WordAlignment],
        features: Sequence[SpanFeatures],
        pauses: Sequence[PauseInterval],
        language: str | None = None,
        *,
        reference_features: Sequence[SpanFeatures] | None = None,
    ) -> IMLDocument:
        """Build an IMLDocument from pipeline outputs.

        Parameters
        ----------
        alignments:
            Word-level time boundaries from STT, in time order. Each ``word``
            is a token that may carry surrounding whitespace; the output
            separates tokens with one space. Closing and terminal
            punctuation tokens (``,``, ``?``, ``)``) attach to the token
            before them, opening ones (``(``, ``“``) to the token after.
        features:
            Prosodic features per word span (as returned by
            :meth:`ProsodyAnalyzer.analyze` for *alignments*).
        pauses:
            Detected silence intervals. Each one between two words that is
            at least 200 ms long becomes a ``<pause>``; where none was
            detected, a gap of 200 ms or more between word timings does.
        language:
            Optional BCP-47 language tag.
        reference_features:
            Features of the same speaker talking neutrally (e.g. calibration
            speech, recorded with the same setup). They define the speaker
            baseline that pitch, volume, rate and emotion are measured
            against. Without them, the baseline is the median of the
            recording's utterances, each counting once
            (:meth:`SpeakerBaseline.from_utterances`), and emotion is only
            classified when at least :data:`MIN_BASELINE_UTTERANCES`
            utterances were measured and most of them lie near that
            baseline; otherwise the classifier gets an empty
            ``SpeakerBaseline()`` (the rule-based one then returns
            ``("neutral", 0.0)``). A recording that is a single utterance
            thus gets no utterance-level offsets and no emotion, and two
            utterances that differ are each marked relative to the midpoint
            between them, with no emotion.

        Returns
        -------
        IMLDocument
        """
        paired = _pair_features(alignments, features)
        words = [
            _Word(text=a.word.strip(), alignment=a, features=f)
            for a, f in zip(alignments, paired, strict=True)
            if a.word.strip()
        ]
        if not words:
            return IMLDocument(utterances=(Utterance(),), version="0.1.0", language=language)

        _mark_spacing(words)
        boundary_pauses = _resolve_pauses(words, pauses)
        groups = _group_into_utterances(words, boundary_pauses)
        group_spans = [
            [f for i in indices if (f := words[i].features) is not None] for _, indices in groups
        ]
        if reference_features:
            baseline = SpeakerBaseline.from_features(reference_features)
            emotion_baseline = baseline
        else:
            baseline = SpeakerBaseline.from_utterances(group_spans)
            established = _baseline_is_established(group_spans, baseline)
            emotion_baseline = baseline if established else SpeakerBaseline()

        utterances: list[Utterance] = []
        for (leading_pause, indices), spans in zip(groups, group_spans, strict=True):
            builder = _UtteranceBuilder(
                [words[i] for i in indices],
                boundary_pauses[indices[0]: indices[-1]],
                baseline,
                self._include_extended,
            )
            emotion, confidence = self._classify(spans, emotion_baseline)
            utterances.append(Utterance(
                children=builder.build(leading_pause),
                emotion=emotion,
                confidence=confidence,
            ))

        return IMLDocument(
            utterances=tuple(utterances),
            version="0.1.0",
            language=language,
        )

    def _classify(
        self, spans: list[SpanFeatures], baseline: SpeakerBaseline
    ) -> tuple[str | None, float | None]:
        """The utterance's emotion and confidence, or ``(None, None)`` when not confident."""
        if isinstance(self._classifier, BaselineAwareEmotionClassifier):
            emotion, confidence = self._classifier.classify_relative(spans, baseline)
        else:
            emotion, confidence = self._classifier.classify(spans)
        if not math.isfinite(confidence) or confidence < self._min_emotion_confidence:
            return None, None
        return emotion, min(1.0, max(0.0, confidence))

