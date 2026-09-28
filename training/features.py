"""Feature extraction shared by data preparation and inference.

A model only works if it sees the same kind of input when it is used as
when it was trained, so every conversion from data to model input lives
here and is used by both sides:

- **SER** (speech emotion recognition): :func:`recording_features` measures
  a recording with :class:`~prosody_protocol.ProsodyAnalyzer`, and
  :func:`utterance_features` / :func:`feature_vector` summarise the
  :class:`~prosody_protocol.SpanFeatures` of an utterance -- one span for
  the whole recording, or one per word as
  :class:`~prosody_protocol.AudioToIML` produces them -- into the named
  features of :data:`SER_FEATURES`.
- **text_to_prosody**: :func:`extract_text_features` describes each token of
  plain text; :func:`token_prosody_labels` reads the prosody label of each
  token from the ``<prosody>``, ``<emphasis>`` and ``<segment>`` markup of an
  IML document.
- **pitch_contour**: :func:`contour_features` turns a measured F0 track into
  a fixed-length curve in semitones; :func:`utterance_contour_label` reads
  the ``pitch_contour`` annotation that covers a whole utterance.

Labels always come from annotations (the dataset entry or its IML), and
features always from the audio or the text: no feature is ever derived from
a label.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from prosody_protocol import IMLParser, SpanFeatures, WordAlignment
from prosody_protocol.models import ChildNode, Emphasis, Pause, Prosody, Segment

if TYPE_CHECKING:
    from prosody_protocol.prosody_analyzer import ProsodyAnalyzer

FloatArray = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# SER: acoustic features of an utterance
# ---------------------------------------------------------------------------

#: Utterance-level acoustic features a SER model can use, with their units
#: (those of :class:`~prosody_protocol.SpanFeatures`, spec Section 4).
SER_FEATURES: dict[str, str] = {
    "f0_mean": "mean F0 of the voiced frames (Hz)",
    "f0_range": "highest minus lowest voiced F0 (Hz)",
    "intensity_mean": "mean level of the non-silent frames (dB)",
    "speech_rate": "syllables per second of speaking time",
    "jitter": "local jitter (percent)",
    "shimmer": "local shimmer (percent)",
    "hnr": "harmonics-to-noise ratio (dB)",
}

#: The SER features in their canonical order.
DEFAULT_SER_FEATURES: tuple[str, ...] = tuple(SER_FEATURES)

# End of the span that measures a whole recording. It lies past the end of
# any real recording; the analyzer clips spans to the audio.
_WHOLE_RECORDING_END_MS = 2**31 - 1


def recording_features(
    audio_path: str | Path,
    text: str = "",
    analyzer: ProsodyAnalyzer | None = None,
) -> SpanFeatures:
    """Measure a whole recording as a single span.

    Uses :class:`~prosody_protocol.ProsodyAnalyzer` (which needs the
    ``audio`` extra). Its measurements ignore silence -- F0 comes from the
    voiced frames and intensity from the non-silent ones -- so leading and
    trailing silence do not change them. The span's ``end_ms`` is a
    placeholder beyond the end of the audio, not the recording's length.

    Raises :class:`~prosody_protocol.exceptions.AudioProcessingError` when
    the audio cannot be read or analysed.
    """
    if analyzer is None:
        from prosody_protocol.prosody_analyzer import ProsodyAnalyzer

        analyzer = ProsodyAnalyzer()
    span = WordAlignment(word=text, start_ms=0, end_ms=_WHOLE_RECORDING_END_MS)
    return analyzer.analyze(audio_path, [span])[0]


def has_voiced_speech(spans: Sequence[SpanFeatures]) -> bool:
    """Whether any span has a pitch measurement, i.e. contains voiced speech."""
    return any(_positive(s.f0_mean) is not None for s in spans)


def utterance_features(spans: Sequence[SpanFeatures]) -> dict[str, float | None]:
    """Summarise the spans of one utterance into the features of :data:`SER_FEATURES`.

    The spans may be a single span covering the utterance (as in training,
    see :func:`recording_features`) or one span per word (as
    :class:`~prosody_protocol.assembler.IMLAssembler` passes them to an
    emotion classifier); both give nearly the same values:

    - ``f0_mean``, ``jitter``, ``shimmer`` and ``hnr`` are averaged over the
      spans weighted by their number of voiced F0 samples, which makes the
      F0 mean exactly that of all voiced frames;
    - ``f0_range`` is the highest minus the lowest F0 of any span;
    - ``intensity_mean`` is averaged as power, and ``speech_rate`` as a
      rate, both weighted by span duration.

    A feature is ``None`` when no span has a measurement for it.
    """
    voiced_weights = [
        float(len(s.f0_contour)) if s.f0_contour else 1.0 for s in spans
    ]
    durations = [float(max(1, s.end_ms - s.start_ms)) for s in spans]

    f0_mean = _weighted_mean([_positive(s.f0_mean) for s in spans], voiced_weights)
    lows = [low for s in spans if s.f0_range and (low := _positive(s.f0_range[0]))]
    highs = [high for s in spans if s.f0_range and (high := _positive(s.f0_range[1]))]
    f0_range = max(highs) - min(lows) if lows and highs else None

    powers = [
        None if (level := _finite(s.intensity_mean)) is None else 10.0 ** (level / 10.0)
        for s in spans
    ]
    power = _weighted_mean(powers, durations)
    intensity_mean = 10.0 * math.log10(power) if power else None

    return {
        "f0_mean": f0_mean,
        "f0_range": f0_range,
        "intensity_mean": intensity_mean,
        "speech_rate": _weighted_mean([_finite(s.speech_rate) for s in spans], durations),
        "jitter": _weighted_mean([_finite(s.jitter) for s in spans], voiced_weights),
        "shimmer": _weighted_mean([_finite(s.shimmer) for s in spans], voiced_weights),
        "hnr": _weighted_mean([_finite(s.hnr) for s in spans], voiced_weights),
    }


def feature_vector(spans: Sequence[SpanFeatures], names: Sequence[str]) -> FloatArray:
    """The features *names* of an utterance as a vector; ``NaN`` where unmeasured.

    Raises ``ValueError`` for a name that is not in :data:`SER_FEATURES`.
    """
    unknown = [n for n in names if n not in SER_FEATURES]
    if unknown:
        raise ValueError(f"Unknown SER feature(s) {unknown}; available: {list(SER_FEATURES)}")
    summary = utterance_features(spans)
    return np.array(
        [math.nan if (v := summary[n]) is None else v for n in names], dtype=np.float64
    )


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _positive(value: float | None) -> float | None:
    value = _finite(value)
    return value if value is not None and value > 0.0 else None


def _weighted_mean(values: Sequence[float | None], weights: Sequence[float]) -> float | None:
    pairs = [(v, w) for v, w in zip(values, weights, strict=True) if v is not None]
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0.0:
        return None
    return sum(v * w for v, w in pairs) / total


# ---------------------------------------------------------------------------
# text_to_prosody: text features and labels read from IML markup
# ---------------------------------------------------------------------------

#: Per-token text features, in canonical order.
TEXT_FEATURES: tuple[str, ...] = (
    "word_length",
    "position_ratio",
    "is_capitalized",
    "has_punctuation",
    "sentence_position",
    "prev_word_length",
    "next_word_length",
)

_PUNCTUATION = ".,!?;:\"'()-"


def extract_text_features(
    words: Sequence[str], names: Sequence[str] | None = None
) -> FloatArray:
    """Extract feature vectors from a list of word tokens.

    Parameters
    ----------
    words:
        Word tokens (punctuation attached, as in ``text.split()``).
    names:
        The features to return, in this order; default all of
        :data:`TEXT_FEATURES`.

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(len(words), len(names))``.
    """
    columns = [TEXT_FEATURES.index(n) for n in (names or TEXT_FEATURES)]
    n = len(words)
    lengths = [len(w.strip(_PUNCTUATION)) for w in words]
    features = []
    for i, word in enumerate(words):
        clean = word.strip(_PUNCTUATION)
        features.append([
            lengths[i],                                    # word_length
            i / max(n - 1, 1),                             # position_ratio
            1.0 if clean and clean[0].isupper() else 0.0,  # is_capitalized
            1.0 if word != clean else 0.0,                 # has_punctuation
            float(i),                                      # sentence_position
            lengths[i - 1] if i > 0 else 0.0,              # prev_word_length
            lengths[i + 1] if i < n - 1 else 0.0,          # next_word_length
        ])
    matrix = np.array(features, dtype=np.float64).reshape(n, len(TEXT_FEATURES))
    return matrix[:, columns]


#: Label dimensions for text_to_prosody and the values each can take.
PROSODY_LABELS: dict[str, tuple[str, ...]] = {
    "pitch_level": ("high", "mid", "low"),
    "volume_level": ("loud", "normal", "quiet"),
    "rate_level": ("fast", "normal", "slow"),
    "emphasis_level": ("strong", "moderate", "reduced", "none"),
}

# A token is "high"/"low" when its <prosody pitch> is at least this far
# above/below the baseline: PITCH_LEVEL_PERCENT for "+15%" values,
# PITCH_LEVEL_ST for "+3st" values. Absolute values ("185Hz") have no
# baseline to compare with and count as "mid".
PITCH_LEVEL_PERCENT = 5.0
PITCH_LEVEL_ST = 1.0
# "loud"/"quiet": <prosody volume> at least this many dB above/below.
VOLUME_LEVEL_DB = 2.0
# "fast"/"slow": <prosody rate> of at least 110 % / at most 90 %, or the
# named rates; <segment tempo> "rushed"/"drawn-out" when no rate is given.
RATE_LEVEL_PERCENT = 10.0

_PITCH_RE = re.compile(r"^([+-]\d+(?:\.\d+)?)(%|st)$")
_VOLUME_RE = re.compile(r"^([+-]\d+(?:\.\d+)?)dB$")
_RATE_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")
_NAMED_RATES = {"fast": "fast", "slow": "slow", "medium": "normal"}
_TEMPO_RATES = {"rushed": "fast", "drawn-out": "slow", "steady": "normal"}

_parser = IMLParser()


def token_prosody_labels(iml: str, dimensions: Sequence[str]) -> list[tuple[str, str]]:
    """The tokens of an IML document with the prosody label of each.

    Tokens are the whitespace-separated words of the document's text
    (punctuation stays attached, as in ``text.split()``); a ``<pause>`` also
    ends a token. A token's label joins its value on each of *dimensions*
    (keys of :data:`PROSODY_LABELS`) with ``_``, e.g. ``"high_loud_normal"``,
    taken from the innermost markup around its first letter or digit:

    - ``pitch_level``: ``<prosody pitch>`` -- ``high`` at +5 % / +1 st or
      more, ``low`` at -5 % / -1 st or less, otherwise (and for absolute Hz
      values) ``mid``;
    - ``volume_level``: ``<prosody volume>`` -- ``loud`` at +2 dB or more,
      ``quiet`` at -2 dB or less, otherwise ``normal``;
    - ``rate_level``: ``<prosody rate>`` -- ``fast``/``slow`` for the named
      rates or 110 % / 90 % and beyond; without a rate, ``<segment tempo>``
      ``rushed``/``drawn-out``; otherwise ``normal``;
    - ``emphasis_level``: the ``level`` of the enclosing ``<emphasis>``, or
      ``none``.

    Raises :class:`~prosody_protocol.exceptions.IMLParseError` for markup
    that cannot be parsed, and ``ValueError`` for an unknown dimension.
    """
    unknown = [d for d in dimensions if d not in PROSODY_LABELS]
    if unknown:
        raise ValueError(
            f"Unknown prosody label dimension(s) {unknown}; available: {list(PROSODY_LABELS)}"
        )
    return [
        (token, "_".join(_LEVELS[d](context) for d in dimensions))
        for token, context in _tokens(iml)
    ]


def _pitch_level(context: Mapping[str, str]) -> str:
    match = _PITCH_RE.match(context.get("pitch", ""))
    if match is None:
        return "mid"
    value = float(match.group(1))
    threshold = PITCH_LEVEL_PERCENT if match.group(2) == "%" else PITCH_LEVEL_ST
    if value >= threshold:
        return "high"
    return "low" if value <= -threshold else "mid"


def _volume_level(context: Mapping[str, str]) -> str:
    match = _VOLUME_RE.match(context.get("volume", ""))
    if match is None:
        return "normal"
    value = float(match.group(1))
    if value >= VOLUME_LEVEL_DB:
        return "loud"
    return "quiet" if value <= -VOLUME_LEVEL_DB else "normal"


def _rate_level(context: Mapping[str, str]) -> str:
    rate = context.get("rate")
    if rate is None:
        return _TEMPO_RATES.get(context.get("tempo", ""), "normal")
    if rate in _NAMED_RATES:
        return _NAMED_RATES[rate]
    match = _RATE_RE.match(rate)
    if match is None:
        return "normal"
    value = float(match.group(1))
    if value >= 100.0 + RATE_LEVEL_PERCENT:
        return "fast"
    return "slow" if value <= 100.0 - RATE_LEVEL_PERCENT else "normal"


def _emphasis_level(context: Mapping[str, str]) -> str:
    level = context.get("emphasis", "")
    return level if level in PROSODY_LABELS["emphasis_level"] else "none"


_LEVELS = {
    "pitch_level": _pitch_level,
    "volume_level": _volume_level,
    "rate_level": _rate_level,
    "emphasis_level": _emphasis_level,
}


def _tokens(iml: str) -> list[tuple[str, dict[str, str]]]:
    """Whitespace-separated tokens of *iml* with the markup context of each.

    The context maps ``pitch``, ``volume``, ``rate``, ``pitch_contour``,
    ``emphasis`` and ``tempo`` to the innermost value around the token's
    first letter or digit (its first character when it has none).
    """
    tokens: list[tuple[str, dict[str, str]]] = []
    chars: list[str] = []
    context: dict[str, str] = {}
    anchored = False

    def flush() -> None:
        nonlocal anchored
        if chars:
            tokens.append(("".join(chars), context))
        chars.clear()
        anchored = False

    for utterance in _parser.parse(iml).utterances:
        for text, ctx in _leaves(utterance.children, {}):
            if text is None:  # a <pause>
                flush()
                continue
            for char in text:
                if char.isspace():
                    flush()
                    continue
                if not chars or (not anchored and char.isalnum()):
                    context = ctx
                    anchored = char.isalnum()
                chars.append(char)
        flush()
    return tokens


def _leaves(
    nodes: Sequence[ChildNode], context: dict[str, str]
) -> Iterator[tuple[str | None, dict[str, str]]]:
    """Text pieces of *nodes* in order with their markup context (``None`` for a pause)."""
    for node in nodes:
        if isinstance(node, str):
            yield node, context
        elif isinstance(node, Pause):
            yield None, context
        elif isinstance(node, Prosody):
            attrs = {
                "pitch": node.pitch,
                "volume": node.volume,
                "rate": node.rate,
                "pitch_contour": node.pitch_contour,
            }
            inner = {**context, **{k: v for k, v in attrs.items() if v is not None}}
            yield from _leaves(node.children, inner)
        elif isinstance(node, Emphasis):
            yield from _leaves(node.children, {**context, "emphasis": node.level})
        elif isinstance(node, Segment):
            inner = dict(context)
            if node.tempo is not None:
                inner["tempo"] = node.tempo
            yield from _leaves(node.children, inner)


# ---------------------------------------------------------------------------
# pitch_contour: F0 curves and utterance contour labels
# ---------------------------------------------------------------------------

#: The spec's ``pitch_contour`` vocabulary (Section 3.2).
CONTOUR_VALUES: tuple[str, ...] = (
    "rise", "fall", "rise-fall", "fall-rise", "rise-sharp", "fall-sharp", "flat",
)

#: A contour is only measured from at least this many voiced F0 samples
#: (80 ms at the analyzer's 10 ms frame step).
MIN_CONTOUR_SAMPLES = 8


def resample_f0(
    f0_sequence: Sequence[float] | FloatArray, target_length: int = 20
) -> FloatArray:
    """Resample an F0 sequence to a fixed length via linear interpolation.

    Parameters
    ----------
    f0_sequence:
        Variable-length F0 values.
    target_length:
        Fixed output length.

    Returns
    -------
    np.ndarray
        Resampled F0 array of shape (target_length,).
    """
    seq = np.array(f0_sequence, dtype=np.float64)
    if len(seq) == 0:
        return np.zeros(target_length)
    if len(seq) == 1:
        return np.full(target_length, seq[0])

    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, seq)


def contour_features(f0_hz: Sequence[float], length: int) -> FloatArray | None:
    """An F0 track as *length* points in semitones relative to its median.

    Only the shape of the curve remains: a rise from 100 to 120 Hz and one
    from 200 to 240 Hz give the same features. Samples that are not
    positive finite frequencies are dropped first. Returns ``None`` when
    fewer than :data:`MIN_CONTOUR_SAMPLES` samples remain.
    """
    voiced = np.array([v for v in f0_hz if math.isfinite(v) and v > 0.0], dtype=np.float64)
    if voiced.size < MIN_CONTOUR_SAMPLES:
        return None
    semitones = 12.0 * np.log2(voiced / np.median(voiced))
    return resample_f0(semitones, length)


def utterance_contour_label(iml: str) -> str | None:
    """The ``pitch_contour`` annotation that covers all of an IML document's text.

    Returns the value when every token lies inside a ``<prosody>`` with the
    same ``pitch_contour``, and ``None`` otherwise (no annotation, or one
    that covers only part of the text: without word timings, the F0 of
    that part cannot be cut out of the recording).
    """
    values = {context.get("pitch_contour") for _, context in _tokens(iml)}
    if len(values) != 1:
        return None
    return values.pop()
