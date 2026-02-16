"""Tests for prosody_protocol.emotion_classifier.

Feeds known feature vectors to the RuleBasedEmotionClassifier and
asserts expected emotion labels.
"""

from __future__ import annotations

import pytest

from prosody_protocol.emotion_classifier import RuleBasedEmotionClassifier
from prosody_protocol.prosody_analyzer import SpanFeatures


@pytest.fixture()
def classifier() -> RuleBasedEmotionClassifier:
    return RuleBasedEmotionClassifier(
        baseline_f0=180.0,
        baseline_intensity=65.0,
        baseline_rate=4.0,
    )


def _make_features(
    f0_mean: float | None = None,
    f0_range: tuple[float, float] | None = None,
    intensity_mean: float | None = None,
    speech_rate: float | None = None,
    jitter: float | None = None,
    shimmer: float | None = None,
    hnr: float | None = None,
) -> list[SpanFeatures]:
    return [
        SpanFeatures(
            start_ms=0,
            end_ms=1000,
            text="test",
            f0_mean=f0_mean,
            f0_range=f0_range,
            intensity_mean=intensity_mean,
            speech_rate=speech_rate,
            jitter=jitter,
            shimmer=shimmer,
            hnr=hnr,
        )
    ]


# ---------------------------------------------------------------------------
# Core emotion tests
# ---------------------------------------------------------------------------


class TestAngry:
    def test_high_f0_high_intensity_fast(self, classifier: RuleBasedEmotionClassifier) -> None:
        emotion, confidence = classifier.classify(
            _make_features(f0_mean=260, intensity_mean=75, speech_rate=6.0, jitter=0.02)
        )
        assert emotion == "angry"
        assert confidence >= 0.3

    def test_angry_has_reasonable_confidence(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        _, confidence = classifier.classify(
            _make_features(f0_mean=280, intensity_mean=80, speech_rate=7.0, jitter=0.025)
        )
        assert 0.3 <= confidence <= 1.0


class TestSad:
    def test_low_f0_low_intensity_slow(self, classifier: RuleBasedEmotionClassifier) -> None:
        emotion, confidence = classifier.classify(
            _make_features(f0_mean=140, intensity_mean=58, speech_rate=2.5, f0_range=(130, 150))
        )
        assert emotion == "sad"
        assert confidence >= 0.3


class TestJoyful:
    def test_high_f0_wide_range_fast(self, classifier: RuleBasedEmotionClassifier) -> None:
        emotion, confidence = classifier.classify(
            _make_features(
                f0_mean=240,
                f0_range=(150, 340),
                intensity_mean=70,
                speech_rate=5.0,
                hnr=18.0,
            )
        )
        assert emotion == "joyful"
        assert confidence >= 0.3


class TestFrustrated:
    def test_moderately_high_f0_irregular(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        emotion, confidence = classifier.classify(
            _make_features(
                f0_mean=210,
                intensity_mean=70,
                jitter=0.015,
                shimmer=0.08,
            )
        )
        assert emotion == "frustrated"
        assert confidence >= 0.3


class TestNeutral:
    def test_baseline_features(self, classifier: RuleBasedEmotionClassifier) -> None:
        emotion, confidence = classifier.classify(
            _make_features(f0_mean=180, intensity_mean=65, speech_rate=4.0)
        )
        assert emotion == "neutral"
        assert confidence >= 0.3


class TestCalm:
    def test_near_baseline_low_intensity(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        emotion, confidence = classifier.classify(
            _make_features(
                f0_mean=175,
                intensity_mean=60,
                speech_rate=3.5,
                hnr=18.0,
            )
        )
        assert emotion == "calm"
        assert confidence >= 0.3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_features_returns_neutral(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        emotion, confidence = classifier.classify([])
        assert emotion == "neutral"
        assert confidence == 0.5

    def test_no_measurable_features_returns_neutral(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        emotion, confidence = classifier.classify(
            _make_features()  # all None
        )
        assert emotion == "neutral"
        assert confidence == 0.5

    def test_confidence_in_valid_range(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        """Confidence should always be between 0.3 and 0.95."""
        for f0 in [100, 150, 200, 250, 300]:
            _, confidence = classifier.classify(
                _make_features(f0_mean=float(f0), intensity_mean=65.0)
            )
            assert 0.3 <= confidence <= 0.95

    def test_returns_tuple(self, classifier: RuleBasedEmotionClassifier) -> None:
        result = classifier.classify(_make_features(f0_mean=200, intensity_mean=70))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

    def test_custom_baseline(self) -> None:
        """Custom baseline should shift what counts as 'neutral'."""
        low_baseline = RuleBasedEmotionClassifier(
            baseline_f0=100.0, baseline_intensity=50.0, baseline_rate=2.0
        )
        emotion, _ = low_baseline.classify(
            _make_features(f0_mean=100, intensity_mean=50, speech_rate=2.0)
        )
        assert emotion == "neutral"
