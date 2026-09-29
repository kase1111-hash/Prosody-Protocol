"""Tests for prosody_protocol.emotion_classifier.

Feeds utterances whose prosody deviates from a speaker baseline in known
ways to the RuleBasedEmotionClassifier, and checks the label, that the
confidence follows the strength of the evidence, and that the result does
not depend on recording gain or on the speaker's natural pitch.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from prosody_protocol import SpanFeatures
from prosody_protocol.assembler import DEFAULT_MIN_EMOTION_CONFIDENCE
from prosody_protocol.emotion_classifier import (
    BaselineAwareEmotionClassifier,
    EmotionClassifier,
    RuleBasedEmotionClassifier,
    SpeakerBaseline,
)

# A typical adult speaker: 120 Hz, 65 dB, 4.5 syllables/s, and pitch moving
# about 3 semitones within a word.
SPEAKER = SpeakerBaseline(f0_mean=120.0, intensity_mean=65.0, speech_rate=4.5, f0_spread=3.0)

SPEC_EMOTIONS = {
    "neutral", "sincere", "sarcastic", "frustrated", "joyful", "uncertain", "angry",
    "sad", "fearful", "surprised", "disgusted", "calm", "empathetic",
}


@pytest.fixture()
def classifier() -> RuleBasedEmotionClassifier:
    return RuleBasedEmotionClassifier()


def _glide(f0: float, spread_st: float, samples: int = 20) -> list[float]:
    """An F0 contour around *f0* whose 10th-90th percentile span is *spread_st*."""
    half = spread_st / 2 / 0.8  # the deciles of a linear glide cover 80 % of it
    return [
        f0 * 2 ** ((-half + 2 * half * i / (samples - 1)) / 12) for i in range(samples)
    ]


def _utterance(
    pitch_st: float = 0.0,
    loudness_db: float = 0.0,
    rate: float = 1.0,
    spread_st: float = 3.0,
    *,
    base: SpeakerBaseline = SPEAKER,
    words: int = 6,
    cues: tuple[str, ...] = ("pitch", "loudness", "rate", "spread"),
) -> list[SpanFeatures]:
    """Word features deviating from *base* by the given amounts."""
    assert base.f0_mean and base.intensity_mean and base.speech_rate
    f0 = base.f0_mean * 2 ** (pitch_st / 12)
    return [
        SpanFeatures(
            start_ms=i * 300,
            end_ms=i * 300 + 280,
            text="word",
            f0_mean=f0 if "pitch" in cues else None,
            f0_contour=_glide(f0, spread_st) if "spread" in cues else None,
            intensity_mean=base.intensity_mean + loudness_db if "loudness" in cues else None,
            speech_rate=base.speech_rate * rate if "rate" in cues else None,
        )
        for i in range(words)
    ]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_angry_much_louder_higher_faster(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        emotion, confidence = classifier.classify_relative(
            _utterance(pitch_st=4, loudness_db=10, rate=1.25, spread_st=3.5), SPEAKER
        )
        assert emotion == "angry"
        assert confidence >= DEFAULT_MIN_EMOTION_CONFIDENCE

    def test_joyful_higher_with_lively_pitch(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        emotion, confidence = classifier.classify_relative(
            _utterance(pitch_st=4, loudness_db=4, rate=1.1, spread_st=7.0), SPEAKER
        )
        assert emotion == "joyful"
        assert confidence >= DEFAULT_MIN_EMOTION_CONFIDENCE

    def test_fearful_much_higher_and_faster_not_louder(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        emotion, confidence = classifier.classify_relative(
            _utterance(pitch_st=6, loudness_db=1, rate=1.35, spread_st=2.2), SPEAKER
        )
        assert emotion == "fearful"
        assert confidence >= DEFAULT_MIN_EMOTION_CONFIDENCE

    def test_sad_lower_quieter_slower_flatter(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        emotion, confidence = classifier.classify_relative(
            _utterance(pitch_st=-4, loudness_db=-8, rate=0.65, spread_st=1.0), SPEAKER
        )
        assert emotion == "sad"
        assert confidence >= DEFAULT_MIN_EMOTION_CONFIDENCE

    def test_calm_is_a_mild_lowering(self, classifier: RuleBasedEmotionClassifier) -> None:
        emotion, confidence = classifier.classify_relative(
            _utterance(pitch_st=-1.5, loudness_db=-4, rate=0.87, spread_st=1.8), SPEAKER
        )
        assert emotion == "calm"
        # Calm lies between neutral and sad, so the heuristic is never sure of it.
        assert confidence < DEFAULT_MIN_EMOTION_CONFIDENCE

    def test_at_baseline_is_neutral(self, classifier: RuleBasedEmotionClassifier) -> None:
        emotion, confidence = classifier.classify_relative(_utterance(), SPEAKER)
        assert emotion == "neutral"
        # "No evidence of an emotion" is not asserted at the default threshold.
        assert confidence < DEFAULT_MIN_EMOTION_CONFIDENCE

    def test_only_spec_vocabulary(self, classifier: RuleBasedEmotionClassifier) -> None:
        grid = itertools.product((-6, -2, 0, 2, 6), (-10, -3, 0, 3, 10), (0.6, 1.0, 1.5), (1, 3, 8))
        labels = {
            classifier.classify_relative(_utterance(p, db, r, s), SPEAKER)[0]
            for p, db, r, s in grid
        }
        assert labels <= SPEC_EMOTIONS
        # The grid reaches every label the heuristic knows.
        assert labels == {"neutral", "calm", "sad", "angry", "joyful", "fearful"}


# ---------------------------------------------------------------------------
# Confidence follows the evidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_grows_with_deviation(self, classifier: RuleBasedEmotionClassifier) -> None:
        results = [
            classifier.classify_relative(
                _utterance(pitch_st=0.4 * k, loudness_db=k, rate=1 + 0.03 * k), SPEAKER
            )
            for k in range(4, 15, 2)
        ]
        assert [emotion for emotion, _ in results] == ["angry"] * len(results)
        confidences = [confidence for _, confidence in results]
        assert confidences == sorted(confidences)
        assert confidences[0] < DEFAULT_MIN_EMOTION_CONFIDENCE <= confidences[-1]

    def test_grows_with_agreeing_cues(self, classifier: RuleBasedEmotionClassifier) -> None:
        # Every cue points to sadness about as strongly as the others.
        deviation: dict[str, Any] = {
            "pitch_st": -4, "loudness_db": -8, "rate": 0.65, "spread_st": 0.5
        }
        cue_sets = [("pitch", "loudness"), ("pitch", "loudness", "rate"),
                    ("pitch", "loudness", "rate", "spread")]
        results = [
            classifier.classify_relative(_utterance(**deviation, cues=cues), SPEAKER)
            for cues in cue_sets
        ]
        assert {emotion for emotion, _ in results} == {"sad"}
        confidences = [confidence for _, confidence in results]
        assert confidences[0] < confidences[1] < confidences[2]

    def test_one_weak_cue_is_not_confident(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        # The old heuristic gave ("neutral", 0.95) for one weak cue.
        _, confidence = classifier.classify_relative(
            _utterance(pitch_st=-1.8, loudness_db=2, cues=("pitch", "loudness")), SPEAKER
        )
        assert confidence < DEFAULT_MIN_EMOTION_CONFIDENCE

    @pytest.mark.parametrize(
        "cues", [("loudness",), ("pitch", "loudness", "rate")], ids=["loud-only", "no-spread"]
    )
    def test_ambiguous_evidence_is_not_confident(
        self, classifier: RuleBasedEmotionClassifier, cues: tuple[str, ...]
    ) -> None:
        """Loud, high and fast fits anger and joy alike without pitch-movement evidence."""
        emotion, confidence = classifier.classify_relative(
            _utterance(pitch_st=4, loudness_db=10, rate=1.25, cues=cues), SPEAKER
        )
        assert emotion in {"angry", "joyful"}
        assert confidence < DEFAULT_MIN_EMOTION_CONFIDENCE

    def test_short_utterances_count_for_less(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        deviation: dict[str, Any] = {
            "pitch_st": 4, "loudness_db": 10, "rate": 1.25, "spread_st": 3.5
        }
        _, one_word = classifier.classify_relative(_utterance(**deviation, words=1), SPEAKER)
        _, sentence = classifier.classify_relative(_utterance(**deviation, words=8), SPEAKER)
        assert one_word < sentence

    def test_strong_consistent_evidence(self, classifier: RuleBasedEmotionClassifier) -> None:
        emotion, confidence = classifier.classify_relative(
            _utterance(pitch_st=4, loudness_db=15, rate=1.32, spread_st=4.5, words=20), SPEAKER
        )
        assert emotion == "angry"
        assert confidence >= 0.75

    def test_never_certain(self, classifier: RuleBasedEmotionClassifier) -> None:
        grid = itertools.product(
            (-12, -4, 0, 4, 12), (-30, -9, 0, 9, 30), (0.4, 1.0, 2.5), (0, 3, 12)
        )
        confidences = [
            classifier.classify_relative(_utterance(p, db, r, s, words=20), SPEAKER)[1]
            for p, db, r, s in grid
        ]
        assert min(confidences) >= 0.0
        assert max(confidences) <= 0.9


# ---------------------------------------------------------------------------
# Invariance: gain and voice
# ---------------------------------------------------------------------------


class TestInvariance:
    @pytest.mark.parametrize("gain_db", [-24.0, -12.0, 6.0])
    def test_recording_gain_does_not_change_the_result(
        self, classifier: RuleBasedEmotionClassifier, gain_db: float
    ) -> None:
        """Turning the gain down moves every intensity, the baseline's included."""
        quieter = SpeakerBaseline(
            f0_mean=120.0, intensity_mean=65.0 + gain_db, speech_rate=4.5, f0_spread=3.0
        )
        for deviation in [(4, 10, 1.25, 3.5), (-4, -8, 0.65, 1.0), (0, 0, 1.0, 3.0)]:
            assert classifier.classify_relative(
                _utterance(*deviation, base=quieter), quieter
            ) == classifier.classify_relative(_utterance(*deviation), SPEAKER)

    def test_a_deep_voice_is_not_sad(self, classifier: RuleBasedEmotionClassifier) -> None:
        """The old heuristic compared with 180 Hz and called every 110 Hz voice sad."""
        deep = SpeakerBaseline(f0_mean=105.0, intensity_mean=65.0, speech_rate=4.5, f0_spread=3.0)
        high = SpeakerBaseline(f0_mean=220.0, intensity_mean=65.0, speech_rate=4.5, f0_spread=3.0)
        for voice in (deep, high):
            emotion, confidence = classifier.classify_relative(_utterance(base=voice), voice)
            assert (emotion, confidence) == classifier.classify_relative(_utterance(), SPEAKER)
            assert emotion == "neutral"

    def test_deterministic(self, classifier: RuleBasedEmotionClassifier) -> None:
        features = _utterance(pitch_st=3, loudness_db=5, rate=1.1, spread_st=4.0)
        results = {classifier.classify_relative(features, SPEAKER) for _ in range(5)}
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Baselines and abstention
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_no_baseline_means_no_evidence(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        loud = _utterance(pitch_st=4, loudness_db=10, rate=1.25, spread_st=3.5)
        assert classifier.classify(loud) == ("neutral", 0.0)
        assert classifier.classify_relative(loud, SpeakerBaseline()) == ("neutral", 0.0)

    def test_empty_features(self, classifier: RuleBasedEmotionClassifier) -> None:
        assert classifier.classify_relative([], SPEAKER) == ("neutral", 0.0)

    def test_unmeasured_features(self, classifier: RuleBasedEmotionClassifier) -> None:
        silent = [SpanFeatures(start_ms=0, end_ms=1000, text="x")]
        assert classifier.classify_relative(silent, SPEAKER) == ("neutral", 0.0)

    def test_digital_silence_intensity_is_ignored(
        self, classifier: RuleBasedEmotionClassifier
    ) -> None:
        """Praat's -300 dB floor is not a quiet voice (it used to read as sad)."""
        silence = [
            SpanFeatures(start_ms=0, end_ms=1000, text="x", intensity_mean=-300.0, speech_rate=0.0)
        ]
        assert classifier.classify_relative(silence, SPEAKER) == ("neutral", 0.0)

    def test_constructor_baseline(self) -> None:
        """Baselines given to the constructor still work with plain classify()."""
        classifier = RuleBasedEmotionClassifier(
            baseline_f0=120.0, baseline_intensity=65.0, baseline_rate=4.5
        )
        loud = _utterance(pitch_st=4, loudness_db=10, rate=1.25, cues=("pitch", "loudness", "rate"))
        assert classifier.classify(loud) == RuleBasedEmotionClassifier().classify_relative(
            loud, SpeakerBaseline(f0_mean=120.0, intensity_mean=65.0, speech_rate=4.5)
        )

    @pytest.mark.parametrize(("spread_st", "expected"), [(3.5, "angry"), (8.0, "joyful")])
    def test_constructor_spread_separates_anger_from_joy(
        self, spread_st: float, expected: str
    ) -> None:
        """Without a spread baseline, classify() read this as ('joyful', 0.45) either way."""
        features = [
            SpanFeatures(
                start_ms=i * 300, end_ms=i * 300 + 280, text="x", f0_mean=260.0,
                f0_contour=_glide(260.0, spread_st), intensity_mean=75.0, speech_rate=5.5,
            )
            for i in range(6)
        ]
        with_spread = RuleBasedEmotionClassifier(
            baseline_f0=180.0, baseline_intensity=65.0, baseline_rate=4.0, baseline_f0_spread=3.0
        )
        emotion, confidence = with_spread.classify(features)
        assert emotion == expected and confidence >= DEFAULT_MIN_EMOTION_CONFIDENCE
        without = RuleBasedEmotionClassifier(
            baseline_f0=180.0, baseline_intensity=65.0, baseline_rate=4.0
        )
        assert without.classify(features)[1] < DEFAULT_MIN_EMOTION_CONFIDENCE

    def test_constructor_baseline_takes_precedence(self) -> None:
        fixed = RuleBasedEmotionClassifier(baseline_f0=120.0, baseline_intensity=65.0)
        misleading = SpeakerBaseline(f0_mean=300.0, intensity_mean=90.0, speech_rate=4.5)
        features = _utterance(cues=("pitch", "loudness"))
        assert fixed.classify_relative(features, misleading) == fixed.classify_relative(
            features, SpeakerBaseline()
        )

    def test_protocols(self, classifier: RuleBasedEmotionClassifier) -> None:
        assert isinstance(classifier, BaselineAwareEmotionClassifier)

        class LabelOnly:
            def classify(self, features: list[SpanFeatures]) -> tuple[str, float]:
                return ("calm", 0.8)

        plain: EmotionClassifier = LabelOnly()
        assert not isinstance(plain, BaselineAwareEmotionClassifier)
        assert plain.classify([]) == ("calm", 0.8)


class TestSpeakerBaseline:
    def test_medians_resist_outliers(self) -> None:
        spans = [
            SpanFeatures(start_ms=0, end_ms=300, text="a", f0_mean=110.0, intensity_mean=64.0),
            SpanFeatures(start_ms=300, end_ms=600, text="b", f0_mean=120.0, intensity_mean=65.0),
            SpanFeatures(start_ms=600, end_ms=900, text="c", f0_mean=125.0, intensity_mean=66.0),
            # A shouted word and a stretch of digital silence.
            SpanFeatures(start_ms=900, end_ms=1200, text="d", f0_mean=260.0, intensity_mean=85.0),
            SpanFeatures(start_ms=1200, end_ms=1500, text="e", intensity_mean=-300.0),
        ]
        baseline = SpeakerBaseline.from_features(spans)
        assert baseline.f0_mean == 122.5
        assert baseline.intensity_mean == 65.5
        assert baseline.speech_rate is None

    def test_rate_is_a_mean(self) -> None:
        """Per-span rates are coarse counts, so a median would snap to one of them."""
        rates = [2.0, 3.0, 3.0, 2.0, 2.0]
        spans = [
            SpanFeatures(start_ms=i * 300, end_ms=i * 300 + 280, text="w", speech_rate=r)
            for i, r in enumerate(rates)
        ]
        assert SpeakerBaseline.from_features(spans).speech_rate == pytest.approx(2.4)

    def test_pitch_spread_from_contour_deciles(self) -> None:
        # 120 to 240 Hz with one stray sample at 60 Hz: the deciles ignore it.
        contour = [60.0] + [120.0 * 2 ** (i / 18) for i in range(19)]
        span = SpanFeatures(start_ms=0, end_ms=300, text="w", f0_mean=160.0, f0_contour=contour)
        spread = SpeakerBaseline.from_features([span]).f0_spread
        assert spread is not None and 8.0 < spread < 12.0

    def test_pitch_spread_does_not_grow_with_span_length(self) -> None:
        """A sentence measured as one span (a transcript without timings) used
        to seem livelier than the same speech measured in word-sized spans,
        as calibration speech is: its pitch drifts from word to word. Neutral
        speech then came out 'joyful' against its own calibration."""
        contours = [_glide(150.0 * 2 ** (-i / 24), 1.5, samples=30) for i in range(8)]
        words = [
            SpanFeatures(start_ms=i * 300, end_ms=i * 300 + 300, text="w",
                         f0_mean=sum(c) / len(c), f0_contour=c)
            for i, c in enumerate(contours)
        ]
        sentence = SpanFeatures(
            start_ms=0, end_ms=2400, text="s", f0_mean=words[4].f0_mean,
            f0_contour=[v for c in contours for v in c],
        )
        by_word = SpeakerBaseline.from_features(words).f0_spread
        whole = SpeakerBaseline.from_features([sentence]).f0_spread
        assert by_word == pytest.approx(1.5, abs=0.2)
        assert whole == pytest.approx(by_word, abs=0.5)
        label, _ = RuleBasedEmotionClassifier().classify_relative(
            [sentence], SpeakerBaseline.from_features(words)
        )
        assert label == "neutral"

    def test_from_utterances_counts_each_utterance_once(self) -> None:
        """Two short calm sentences outvote one long shouted one; word by word they
        would not."""
        def utterance(words: int, f0: float, db: float, rate: float) -> list[SpanFeatures]:
            return [
                SpanFeatures(
                    start_ms=i * 300, end_ms=i * 300 + 280, text="w", f0_mean=f0,
                    intensity_mean=db, speech_rate=rate, jitter=1.0,
                )
                for i in range(words)
            ]

        calm = [utterance(3, 120.0, 65.0, 4.0), utterance(3, 124.0, 66.0, 5.0)]
        shouted = utterance(8, 180.0, 80.0, 6.0)
        assert SpeakerBaseline.from_features([*calm[0], *calm[1], *shouted]).f0_mean == 180.0
        assert SpeakerBaseline.from_utterances([*calm, shouted]) == SpeakerBaseline(
            f0_mean=124.0, intensity_mean=66.0, speech_rate=5.0, jitter=1.0
        )

    def test_from_no_utterances(self) -> None:
        assert SpeakerBaseline.from_utterances([]) == SpeakerBaseline()
        assert SpeakerBaseline.from_utterances([[], []]) == SpeakerBaseline()

    def test_pitch_spread_falls_back_to_range(self) -> None:
        span = SpanFeatures(
            start_ms=0, end_ms=300, text="w", f0_mean=150.0, f0_range=(100.0, 200.0)
        )
        assert SpeakerBaseline.from_features([span]).f0_spread == pytest.approx(12.0)
