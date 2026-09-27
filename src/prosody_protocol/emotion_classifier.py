"""Emotion classification from prosodic features.

Provides a protocol for emotion classifiers and a rule-based baseline
implementation that labels an utterance by how its prosody deviates from
the speaker's own baseline.

Absolute pitch and loudness say little about emotion: a deep voice is not
sad, and a recording made with the gain turned down is not calm. The
rule-based classifier therefore only looks at *differences* from a
:class:`SpeakerBaseline` -- pitch in semitones, loudness in dB, speech rate
and pitch movement relative to the speaker's typical values.

Spec reference: Section 3.1 (core emotion vocabulary).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Protocol, runtime_checkable

from ._types import SpanFeatures

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class EmotionClassifier(Protocol):
    """Protocol for emotion classifiers."""

    def classify(self, features: list[SpanFeatures]) -> tuple[str, float]:
        """Classify the emotion of a span of speech.

        Parameters
        ----------
        features:
            Prosodic features for the words in the utterance.

        Returns
        -------
        tuple[str, float]
            ``(emotion_label, confidence)`` where confidence is in [0.0, 1.0].
        """
        ...


@runtime_checkable
class BaselineAwareEmotionClassifier(Protocol):
    """Protocol for classifiers that judge an utterance against a speaker baseline.

    :class:`~prosody_protocol.assembler.IMLAssembler` calls
    :meth:`classify_relative` on classifiers that have it, passing the
    baseline it measured (from calibration speech or from the whole
    recording); other classifiers get :meth:`EmotionClassifier.classify`.
    When the recording does not show the speaker's typical prosody (for
    example, it holds only one or two utterances and no calibration speech
    was given), the baseline passed has no values (``SpeakerBaseline()``).
    """

    def classify_relative(
        self, features: Sequence[SpanFeatures], baseline: SpeakerBaseline
    ) -> tuple[str, float]:
        """Classify *features* relative to *baseline*; same result as ``classify``."""
        ...


# ---------------------------------------------------------------------------
# Speaker baseline
# ---------------------------------------------------------------------------

# A span needs this many voiced F0 samples for its pitch spread to be measured
# from the contour (otherwise ``f0_range`` is used).
_MIN_SPREAD_SAMPLES = 5


def _semitones(f0_hz: float, reference_hz: float) -> float:
    """Pitch interval from *reference_hz* to *f0_hz* in semitones."""
    return 12.0 * math.log2(f0_hz / reference_hz)


def _positive(value: float | None) -> float | None:
    """*value* if it is a finite number above zero, else ``None``."""
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _finite(value: float | None) -> float | None:
    """*value* if it is a finite number, else ``None``."""
    if value is None or not math.isfinite(value):
        return None
    return value


def _span_f0(features: SpanFeatures) -> float | None:
    """Mean F0 of a span in Hz, or ``None`` when unvoiced or invalid."""
    return _positive(features.f0_mean)


def _span_intensity(features: SpanFeatures) -> float | None:
    """Mean intensity of a span in dB, or ``None``.

    Values at or below 0 dB are not speech levels but artefacts such as
    Praat's -300 dB floor for digital silence, so they are ignored.
    """
    return _positive(features.intensity_mean)


def _span_rate(features: SpanFeatures) -> float | None:
    """Speech rate of a span in syllables/second, or ``None`` (also for 0)."""
    return _positive(features.speech_rate)


def _span_f0_spread(features: SpanFeatures) -> float | None:
    """How far pitch moves within a span, in semitones.

    The distance between the 10th and 90th percentiles of the F0 contour,
    which ignores a stray sample at either end; spans with a short contour
    fall back to the full ``f0_range``.
    """
    contour = [v for v in features.f0_contour or () if math.isfinite(v) and v > 0.0]
    if len(contour) >= _MIN_SPREAD_SAMPLES:
        deciles = statistics.quantiles(contour, n=10)
        return _semitones(deciles[-1], deciles[0])
    if features.f0_range is not None:
        low, high = (_positive(v) for v in features.f0_range)
        if low is not None and high is not None and high >= low:
            return _semitones(high, low)
    return None


def _median(values: Iterable[float | None]) -> float | None:
    """Median of the values that are not ``None``, or ``None`` if there are none."""
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _mean(values: Iterable[float | None]) -> float | None:
    """Mean of the values that are not ``None``, or ``None`` if there are none."""
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


@dataclass(frozen=True)
class SpeakerBaseline:
    """A speaker's typical prosody, the reference that deviations are measured from.

    Each field is the median over word-sized spans of the speaker's speech
    (:meth:`from_features`), or over utterances (:meth:`from_utterances`),
    in the units of :class:`~prosody_protocol.SpanFeatures`, or ``None``
    when it was not measured. Medians keep a few shouted or whispered words
    from moving the baseline. Within a stretch of speech, speech rate is the
    exception: it is the mean, because per-span rates are coarse (syllable
    counts over about a second) and a median would snap to one of a few
    values.

    Attributes
    ----------
    f0_mean:
        Typical pitch (Hz).
    intensity_mean:
        Typical loudness (dB). Only differences from it are meaningful:
        it moves with the recording gain.
    speech_rate:
        Typical speech rate (syllables/second).
    f0_spread:
        Typical pitch movement within a span (semitones; see
        :meth:`from_features`).
    jitter, shimmer:
        Typical perturbation (percent).
    hnr:
        Typical harmonics-to-noise ratio (dB).
    """

    f0_mean: float | None = None
    intensity_mean: float | None = None
    speech_rate: float | None = None
    f0_spread: float | None = None
    jitter: float | None = None
    shimmer: float | None = None
    hnr: float | None = None

    @classmethod
    def from_features(cls, features: Iterable[SpanFeatures]) -> SpeakerBaseline:
        """Measure a baseline from spans of the speaker's speech.

        Spans without a measurement are skipped for that field, as are
        intensities at or below 0 dB (digital-silence artefacts). A span's
        pitch movement is the 10th-90th percentile distance of its F0
        contour in semitones (``f0_range`` when the contour is short).
        """
        spans = list(features)
        return cls(
            f0_mean=_median(_span_f0(f) for f in spans),
            intensity_mean=_median(_span_intensity(f) for f in spans),
            speech_rate=_mean(_span_rate(f) for f in spans),
            f0_spread=_median(_span_f0_spread(f) for f in spans),
            jitter=_median(_finite(f.jitter) for f in spans),
            shimmer=_median(_finite(f.shimmer) for f in spans),
            hnr=_median(_finite(f.hnr) for f in spans),
        )

    @classmethod
    def from_utterances(cls, utterances: Iterable[Iterable[SpanFeatures]]) -> SpeakerBaseline:
        """Measure a baseline from several utterances, each counting once.

        Each field is the median of the utterances' own baselines
        (:meth:`from_features`). Weighting utterances rather than words
        keeps one long emotional utterance from outvoting several ordinary
        ones: in a recording of four calm sentences and one long shouted
        one, the calm sentences define the baseline.
        """
        own = [cls.from_features(spans) for spans in utterances]
        return cls(**{
            f.name: _median(getattr(baseline, f.name) for baseline in own) for f in fields(cls)
        })


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------

# The rule-based classifier describes an utterance by four cues relative to
# the speaker baseline, each measured in "steps" -- the size of a change
# that is clearly audible:
#   pitch:    median F0 of the words, 2 semitones per step (about 12 %)
#   loudness: median intensity of the words, 3 dB per step
#   rate:     mean speech rate, 15 % faster (or 1/1.15 slower) per step
#   spread:   median pitch movement within words, 1.5 semitones per step
_PITCH_STEP_ST = 2.0
_LOUDNESS_STEP_DB = 3.0
_RATE_STEP = math.log(1.15)
_SPREAD_STEP_ST = 1.5

# A cue further out than this many steps counts as this many, so one wild
# measurement cannot outvote the others.
_MAX_STEPS = 6.0

# Short utterances give noisy medians, so their cues are shrunk towards the
# baseline by T / (T + _SHRINK_SECONDS), T being the measured speech time:
# a 0.3 s word keeps half of its deviation, a 3 s sentence 91 %.
_SHRINK_SECONDS = 0.3

# How each emotion typically shifts the cues (pitch, loudness, rate,
# spread), in steps, after the reviews of vocal emotion cues by Juslin &
# Laukka (2003) and Banse & Scherer (1996). The vector is where the emotion
# becomes recognisable; larger deviations in the same direction fit it at
# least as well.
_SIGNATURES: dict[str, tuple[float, float, float, float]] = {
    # Much louder, higher, faster.
    "angry": (1.0, 2.5, 1.0, 0.5),
    # Higher, with much livelier pitch movement.
    "joyful": (1.0, 1.0, 0.5, 2.5),
    # Much higher and faster, but not louder, with constrained movement.
    "fearful": (2.5, 0.0, 1.5, -0.5),
    # Lower, quieter, slower, flatter -- clearly so.
    "sad": (-1.25, -1.75, -1.75, -1.25),
}
# Calm is a mild version of the same low-arousal pattern. It is a single
# point: a stronger deviation is sadness, not more calm.
_CALM: tuple[float, float, float, float] = (-0.75, -1.25, -0.75, -0.75)

# Each hypothesis is scored by how far the cues lie from what it predicts,
# with this spread (in steps) of measurement noise and natural variation ...
_NOISE_STEPS = 1.5
# ... which grows with the size of the deviation: squared errors are divided
# by 1 + (size / _NOISE_GROWTH_STEPS)^2. A strong deviation is thus judged by
# its shape (which cues move, and how much relative to each other) rather
# than by its size; as it grows, neutral is ruled out while each emotion's
# error levels off at a value set by how well the shapes match.
_NOISE_GROWTH_STEPS = 6.0

# Cues can also deviate in a way that fits no emotion (say, lower pitch but
# much faster speech). That possibility is scored as a fixed squared error,
# as if a pattern were about 3.5 steps away, so when nothing fits well every
# label gets a low confidence -- "neutral" included.
_UNEXPLAINED_ERROR = 12.0

# A rule-based system never claims certainty.
_MAX_CONFIDENCE = 0.9

# Order in which ties are broken: neutral first.
_LABELS = ("neutral", "calm", "sad", "angry", "joyful", "fearful")


def _utterance_cues(
    features: Sequence[SpanFeatures], baseline: SpeakerBaseline
) -> dict[int, float]:
    """The utterance's cue values in steps, keyed by cue index; unmeasured cues are absent."""
    raw: dict[int, float] = {}
    f0 = _median(_span_f0(f) for f in features)
    if f0 is not None and baseline.f0_mean is not None and baseline.f0_mean > 0:
        raw[0] = _semitones(f0, baseline.f0_mean) / _PITCH_STEP_ST
    intensity = _median(_span_intensity(f) for f in features)
    if intensity is not None and baseline.intensity_mean is not None:
        raw[1] = (intensity - baseline.intensity_mean) / _LOUDNESS_STEP_DB
    rate = _mean(_span_rate(f) for f in features)
    if rate is not None and baseline.speech_rate is not None and baseline.speech_rate > 0:
        raw[2] = math.log(rate / baseline.speech_rate) / _RATE_STEP
    spread = _median(_span_f0_spread(f) for f in features)
    if spread is not None and baseline.f0_spread is not None:
        raw[3] = (spread - baseline.f0_spread) / _SPREAD_STEP_ST
    if not raw:
        return raw

    measured_s = sum(
        max(0, f.end_ms - f.start_ms) / 1000.0
        for f in features
        if _span_f0(f) is not None or _span_intensity(f) is not None
    )
    shrink = measured_s / (measured_s + _SHRINK_SECONDS)
    return {
        cue: shrink * max(-_MAX_STEPS, min(_MAX_STEPS, value)) for cue, value in raw.items()
    }


def _squared_error(
    cues: dict[int, float], signature: tuple[float, ...], *, ray: bool
) -> float | None:
    """Squared distance from the cues to what a hypothesis predicts.

    With ``ray=True`` the hypothesis predicts the signature or any stronger
    deviation in the same direction; otherwise exactly the signature. Returns
    ``None`` when the measured cues cannot tell the hypothesis from neutral.
    """
    target = {cue: signature[cue] for cue in cues}
    norm_sq = sum(v * v for v in target.values())
    if norm_sq == 0.0:
        return None
    scale = 1.0
    if ray:
        scale = max(1.0, sum(cues[c] * target[c] for c in cues) / norm_sq)
    return sum((cues[c] - scale * target[c]) ** 2 for c in cues)


class RuleBasedEmotionClassifier:
    """Heuristic classifier that labels an utterance by its deviation from a speaker baseline.

    Four cues are compared with the baseline: pitch (semitones), loudness
    (dB), speech rate and pitch movement within words. Each emotion shifts
    them in a typical direction (anger: much louder, higher, faster; joy:
    higher with lively pitch movement; fear: much higher and faster but
    not louder; sadness: lower, quieter, slower, flatter; calm: a mild
    version of sadness) and *neutral* means no shift. The label is the
    emotion that best explains the cues, and the confidence is its
    probability under a simple noise model -- which also allows for a
    deviation that fits no emotion -- capped at 0.9. It therefore:

    - grows as the cues deviate further in a consistent direction and as
      more cues are measured;
    - stays low when the cues are weak, few, fit several emotions (e.g.
      loud and high-pitched, but with no pitch-movement evidence to
      separate anger from joy) or fit none (e.g. lower but faster);
    - stays below 0.5 for ``"neutral"``, which means only "no evidence of
      an emotion": with :class:`~prosody_protocol.assembler.IMLAssembler`'s
      default threshold, such utterances carry no emotion attribute.

    Labels are ``neutral``, ``calm``, ``sad``, ``angry``, ``joyful`` and
    ``fearful`` (spec Section 3.1). Results are deterministic.

    The preferred way to supply the baseline is :meth:`classify_relative`
    with a :class:`SpeakerBaseline` measured from the same speaker (which
    is what :class:`~prosody_protocol.assembler.IMLAssembler` does).

    Parameters
    ----------
    baseline_f0, baseline_intensity, baseline_rate, baseline_f0_spread:
        Optional fixed speaker baseline: pitch (Hz), loudness (dB, on the
        same recording chain), speech rate (syllables/second) and pitch
        movement within words (semitones, see
        :meth:`SpeakerBaseline.from_features`). Values given here take
        precedence over the baseline passed to :meth:`classify_relative`.
        When none is given and no baseline is passed, there is nothing to
        compare with and :meth:`classify` returns ``("neutral", 0.0)``.
        Without ``baseline_f0_spread``, :meth:`classify` cannot use pitch
        movement, which is what separates anger (loud, higher, but flat)
        from joy (higher with lively movement); loud, high-pitched speech
        then gets a low-confidence label.
    """

    def __init__(
        self,
        baseline_f0: float | None = None,
        baseline_intensity: float | None = None,
        baseline_rate: float | None = None,
        baseline_f0_spread: float | None = None,
    ) -> None:
        self.baseline_f0 = baseline_f0
        self.baseline_intensity = baseline_intensity
        self.baseline_rate = baseline_rate
        self.baseline_f0_spread = baseline_f0_spread

    def classify(self, features: list[SpanFeatures]) -> tuple[str, float]:
        """Classify emotion relative to the baseline given to the constructor."""
        return self.classify_relative(features, SpeakerBaseline())

    def classify_relative(
        self, features: Sequence[SpanFeatures], baseline: SpeakerBaseline
    ) -> tuple[str, float]:
        """Classify emotion from the deviation of *features* from *baseline*.

        Returns ``("neutral", 0.0)`` when no cue can be compared with the
        baseline.
        """
        fixed = {
            "f0_mean": self.baseline_f0,
            "intensity_mean": self.baseline_intensity,
            "speech_rate": self.baseline_rate,
            "f0_spread": self.baseline_f0_spread,
        }
        baseline = replace(baseline, **{k: v for k, v in fixed.items() if v is not None})

        cues = _utterance_cues(features, baseline)
        if not cues:
            return ("neutral", 0.0)

        errors: dict[str, float] = {"neutral": sum(v * v for v in cues.values())}
        for label, signature in _SIGNATURES.items():
            error = _squared_error(cues, signature, ray=True)
            if error is not None:
                errors[label] = error
        calm = _squared_error(cues, _CALM, ray=False)
        if calm is not None:
            errors["calm"] = calm

        # Likelihood of each hypothesis under Gaussian noise; the smallest
        # error is subtracted first so that exp() cannot underflow to 0.
        growth = 1.0 + sum(v * v for v in cues.values()) / _NOISE_GROWTH_STEPS**2
        errors = {label: error / growth for label, error in errors.items()}
        least = min(_UNEXPLAINED_ERROR, *errors.values())
        weights = {
            label: math.exp(-(error - least) / (2.0 * _NOISE_STEPS**2))
            for label, error in errors.items()
        }
        unexplained = math.exp(-(_UNEXPLAINED_ERROR - least) / (2.0 * _NOISE_STEPS**2))
        total = sum(weights.values()) + unexplained
        best = max((label for label in _LABELS if label in weights), key=lambda k: weights[k])
        confidence = min(_MAX_CONFIDENCE, weights[best] / total)
        return (best, round(confidence, 2))
