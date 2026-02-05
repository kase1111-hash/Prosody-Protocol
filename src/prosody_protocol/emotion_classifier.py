"""Emotion classification from prosodic features.

Provides a protocol for emotion classifiers and a rule-based baseline
implementation that maps prosodic feature combinations to emotions
using heuristic thresholds.

Spec reference: Section 3.1 (core emotion vocabulary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .prosody_analyzer import SpanFeatures


# ---------------------------------------------------------------------------
# Protocol
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


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------


@dataclass
class _FeatureAverages:
    """Aggregated prosodic features across an utterance."""

    f0_mean: float | None = None
    f0_range_span: float | None = None
    intensity_mean: float | None = None
    speech_rate: float | None = None
    jitter: float | None = None
    shimmer: float | None = None
    hnr: float | None = None


def _aggregate(features: list[SpanFeatures]) -> _FeatureAverages:
    """Compute averages of prosodic features across spans."""
    f0s: list[float] = []
    f0_spans: list[float] = []
    intensities: list[float] = []
    rates: list[float] = []
    jitters: list[float] = []
    shimmers: list[float] = []
    hnrs: list[float] = []

    for f in features:
        if f.f0_mean is not None:
            f0s.append(f.f0_mean)
        if f.f0_range is not None:
            f0_spans.append(f.f0_range[1] - f.f0_range[0])
        if f.intensity_mean is not None:
            intensities.append(f.intensity_mean)
        if f.speech_rate is not None:
            rates.append(f.speech_rate)
        if f.jitter is not None:
            jitters.append(f.jitter)
        if f.shimmer is not None:
            shimmers.append(f.shimmer)
        if f.hnr is not None:
            hnrs.append(f.hnr)

    return _FeatureAverages(
        f0_mean=sum(f0s) / len(f0s) if f0s else None,
        f0_range_span=sum(f0_spans) / len(f0_spans) if f0_spans else None,
        intensity_mean=sum(intensities) / len(intensities) if intensities else None,
        speech_rate=sum(rates) / len(rates) if rates else None,
        jitter=sum(jitters) / len(jitters) if jitters else None,
        shimmer=sum(shimmers) / len(shimmers) if shimmers else None,
        hnr=sum(hnrs) / len(hnrs) if hnrs else None,
    )


class RuleBasedEmotionClassifier:
    """Heuristic classifier that maps prosodic feature ranges to emotions.

    Uses thresholds derived from emotional speech literature to produce
    a best-guess emotion label and a confidence score based on how
    strongly the features match.

    The baseline speaker model can be tuned via constructor parameters.
    """

    def __init__(
        self,
        baseline_f0: float = 180.0,
        baseline_intensity: float = 65.0,
        baseline_rate: float = 4.0,
    ) -> None:
        self.baseline_f0 = baseline_f0
        self.baseline_intensity = baseline_intensity
        self.baseline_rate = baseline_rate

    def classify(self, features: list[SpanFeatures]) -> tuple[str, float]:
        """Classify emotion from aggregated prosodic features.

        Returns ``("neutral", 0.5)`` when features are insufficient.
        """
        if not features:
            return ("neutral", 0.5)

        avg = _aggregate(features)

        # If we have no measurable features, return neutral.
        if avg.f0_mean is None and avg.intensity_mean is None:
            return ("neutral", 0.5)

        scores: dict[str, float] = {}

        scores["angry"] = self._score_angry(avg)
        scores["frustrated"] = self._score_frustrated(avg)
        scores["joyful"] = self._score_joyful(avg)
        scores["sad"] = self._score_sad(avg)
        scores["fearful"] = self._score_fearful(avg)
        scores["sarcastic"] = self._score_sarcastic(avg)
        scores["calm"] = self._score_calm(avg)
        scores["neutral"] = self._score_neutral(avg)

        best_emotion = max(scores, key=lambda k: scores[k])
        best_score = scores[best_emotion]

        # Confidence = how strongly the best matches relative to runner-up.
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            margin = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, 0.5 + margin * 2)
        else:
            confidence = 0.5

        # Floor at 0.5, cap at 0.95 for rule-based system.
        confidence = max(0.5, min(0.95, confidence))

        return (best_emotion, round(confidence, 2))

    # -- Per-emotion scoring heuristics -------------------------------------

    def _score_angry(self, avg: _FeatureAverages) -> float:
        """High F0, high intensity, fast rate."""
        score = 0.0
        if avg.f0_mean is not None and avg.f0_mean > self.baseline_f0 * 1.3:
            score += 0.3
        if avg.intensity_mean is not None and avg.intensity_mean > self.baseline_intensity + 6:
            score += 0.3
        if avg.speech_rate is not None and avg.speech_rate > self.baseline_rate * 1.3:
            score += 0.2
        if avg.jitter is not None and avg.jitter > 0.015:
            score += 0.1
        return score

    def _score_frustrated(self, avg: _FeatureAverages) -> float:
        """Moderately high F0, high intensity, irregular voice."""
        score = 0.0
        if avg.f0_mean is not None and avg.f0_mean > self.baseline_f0 * 1.1:
            score += 0.2
        if avg.intensity_mean is not None and avg.intensity_mean > self.baseline_intensity + 3:
            score += 0.2
        if avg.jitter is not None and avg.jitter > 0.01:
            score += 0.2
        if avg.shimmer is not None and avg.shimmer > 0.06:
            score += 0.1
        return score

    def _score_joyful(self, avg: _FeatureAverages) -> float:
        """High F0, wide range, moderate-high intensity, fast rate."""
        score = 0.0
        if avg.f0_mean is not None and avg.f0_mean > self.baseline_f0 * 1.2:
            score += 0.2
        if avg.f0_range_span is not None and avg.f0_range_span > 80:
            score += 0.2
        if avg.intensity_mean is not None and avg.intensity_mean > self.baseline_intensity + 2:
            score += 0.1
        if avg.speech_rate is not None and avg.speech_rate > self.baseline_rate * 1.1:
            score += 0.1
        if avg.hnr is not None and avg.hnr > 15.0:
            score += 0.1
        return score

    def _score_sad(self, avg: _FeatureAverages) -> float:
        """Low F0, narrow range, low intensity, slow rate."""
        score = 0.0
        if avg.f0_mean is not None and avg.f0_mean < self.baseline_f0 * 0.85:
            score += 0.3
        if avg.f0_range_span is not None and avg.f0_range_span < 30:
            score += 0.2
        if avg.intensity_mean is not None and avg.intensity_mean < self.baseline_intensity - 3:
            score += 0.2
        if avg.speech_rate is not None and avg.speech_rate < self.baseline_rate * 0.8:
            score += 0.2
        return score

    def _score_fearful(self, avg: _FeatureAverages) -> float:
        """High F0, high jitter/shimmer, fast rate."""
        score = 0.0
        if avg.f0_mean is not None and avg.f0_mean > self.baseline_f0 * 1.25:
            score += 0.2
        if avg.jitter is not None and avg.jitter > 0.015:
            score += 0.2
        if avg.shimmer is not None and avg.shimmer > 0.08:
            score += 0.2
        if avg.speech_rate is not None and avg.speech_rate > self.baseline_rate * 1.2:
            score += 0.2
        return score

    def _score_sarcastic(self, avg: _FeatureAverages) -> float:
        """Wide F0 range (exaggerated contour), moderate rate."""
        score = 0.0
        if avg.f0_range_span is not None and avg.f0_range_span > 100:
            score += 0.3
        if avg.f0_mean is not None and self.baseline_f0 * 0.9 < avg.f0_mean < self.baseline_f0 * 1.2:
            score += 0.1
        if avg.speech_rate is not None and self.baseline_rate * 0.8 < avg.speech_rate < self.baseline_rate * 1.1:
            score += 0.1
        return score

    def _score_calm(self, avg: _FeatureAverages) -> float:
        """Near-baseline F0, low intensity, slow-moderate rate, good HNR."""
        score = 0.0
        if avg.f0_mean is not None and self.baseline_f0 * 0.9 < avg.f0_mean < self.baseline_f0 * 1.1:
            score += 0.2
        if avg.intensity_mean is not None and avg.intensity_mean < self.baseline_intensity:
            score += 0.2
        if avg.speech_rate is not None and avg.speech_rate < self.baseline_rate * 0.95:
            score += 0.1
        if avg.hnr is not None and avg.hnr > 15.0:
            score += 0.1
        return score

    def _score_neutral(self, avg: _FeatureAverages) -> float:
        """Features near baseline across all dimensions."""
        score = 0.0
        if avg.f0_mean is not None:
            ratio = avg.f0_mean / self.baseline_f0
            if 0.9 < ratio < 1.1:
                score += 0.2
        if avg.intensity_mean is not None:
            diff = abs(avg.intensity_mean - self.baseline_intensity)
            if diff < 3:
                score += 0.2
        if avg.speech_rate is not None:
            ratio = avg.speech_rate / self.baseline_rate
            if 0.85 < ratio < 1.15:
                score += 0.15
        return score
