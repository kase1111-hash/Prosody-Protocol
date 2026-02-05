"""Tests for prosody_protocol.prosody_analyzer.

Uses synthetic audio fixtures (pure tones, silence, gaps) with
known acoustic properties to make deterministic assertions about
F0, intensity, pause detection, and voice quality classification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prosody_protocol.exceptions import AudioProcessingError
from prosody_protocol.prosody_analyzer import (
    PauseInterval,
    ProsodyAnalyzer,
    SpanFeatures,
    WordAlignment,
)

AUDIO_DIR = Path(__file__).parent / "fixtures" / "audio"


@pytest.fixture()
def analyzer() -> ProsodyAnalyzer:
    return ProsodyAnalyzer()


# ---------------------------------------------------------------------------
# Data model tests (preserved from Phase 2 stubs)
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_word_alignment(self) -> None:
        wa = WordAlignment(word="hello", start_ms=0, end_ms=500)
        assert wa.word == "hello"
        assert wa.start_ms == 0

    def test_span_features_defaults(self) -> None:
        sf = SpanFeatures(start_ms=0, end_ms=500, text="hello")
        assert sf.f0_mean is None
        assert sf.quality is None

    def test_pause_interval(self) -> None:
        pi = PauseInterval(start_ms=500, end_ms=1300)
        assert pi.duration_ms == 800


# ---------------------------------------------------------------------------
# F0 extraction
# ---------------------------------------------------------------------------


class TestF0Extraction:
    def test_220hz_tone_f0(self, analyzer: ProsodyAnalyzer) -> None:
        """A pure 220 Hz sine wave should yield F0 close to 220 Hz."""
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        assert len(results) == 1
        f = results[0]
        assert f.f0_mean is not None
        assert 200 < f.f0_mean < 240, f"Expected ~220 Hz, got {f.f0_mean}"

    def test_440hz_tone_f0(self, analyzer: ProsodyAnalyzer) -> None:
        """A pure 440 Hz sine wave should yield F0 close to 440 Hz."""
        alignments = [WordAlignment(word="tone", start_ms=50, end_ms=450)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_440hz.wav"), alignments)
        assert len(results) == 1
        f = results[0]
        assert f.f0_mean is not None
        assert 420 < f.f0_mean < 460, f"Expected ~440 Hz, got {f.f0_mean}"

    def test_f0_range_is_tuple(self, analyzer: ProsodyAnalyzer) -> None:
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        f = results[0]
        assert f.f0_range is not None
        assert isinstance(f.f0_range, tuple)
        assert len(f.f0_range) == 2
        assert f.f0_range[0] <= f.f0_range[1]

    def test_f0_contour_is_list(self, analyzer: ProsodyAnalyzer) -> None:
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        f = results[0]
        assert f.f0_contour is not None
        assert isinstance(f.f0_contour, list)
        assert len(f.f0_contour) > 0


# ---------------------------------------------------------------------------
# Intensity extraction
# ---------------------------------------------------------------------------


class TestIntensityExtraction:
    def test_tone_has_intensity(self, analyzer: ProsodyAnalyzer) -> None:
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        f = results[0]
        assert f.intensity_mean is not None
        assert f.intensity_mean > 0

    def test_loud_louder_than_quiet(self, analyzer: ProsodyAnalyzer) -> None:
        """The first half of loud_quiet.wav is louder than the second half."""
        loud_align = [WordAlignment(word="loud", start_ms=50, end_ms=450)]
        quiet_align = [WordAlignment(word="quiet", start_ms=550, end_ms=950)]

        loud_res = analyzer.analyze(str(AUDIO_DIR / "loud_quiet.wav"), loud_align)
        quiet_res = analyzer.analyze(str(AUDIO_DIR / "loud_quiet.wav"), quiet_align)

        loud_i = loud_res[0].intensity_mean
        quiet_i = quiet_res[0].intensity_mean
        assert loud_i is not None
        assert quiet_i is not None
        assert loud_i > quiet_i


# ---------------------------------------------------------------------------
# Voice quality (jitter/shimmer/HNR)
# ---------------------------------------------------------------------------


class TestVoiceQuality:
    def test_tone_produces_quality(self, analyzer: ProsodyAnalyzer) -> None:
        """A clean sine tone should produce a quality label."""
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        f = results[0]
        assert f.quality in ("modal", "breathy", "tense", "creaky", None)

    def test_jitter_shimmer_hnr_present(self, analyzer: ProsodyAnalyzer) -> None:
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        f = results[0]
        has_any = f.jitter is not None or f.shimmer is not None or f.hnr is not None
        assert has_any


# ---------------------------------------------------------------------------
# Pause detection
# ---------------------------------------------------------------------------


class TestPauseDetection:
    def test_silence_file_is_one_big_pause(self, analyzer: ProsodyAnalyzer) -> None:
        pauses = analyzer.detect_pauses(str(AUDIO_DIR / "silence_1s.wav"))
        assert len(pauses) >= 1
        total_ms = sum(p.duration_ms for p in pauses)
        assert total_ms >= 800

    def test_tone_gap_tone_detects_pause(self, analyzer: ProsodyAnalyzer) -> None:
        """0.5s tone + 0.8s silence + 0.5s tone should detect a pause."""
        pauses = analyzer.detect_pauses(str(AUDIO_DIR / "tone_gap_tone.wav"))
        assert len(pauses) >= 1
        gap = pauses[0]
        assert gap.duration_ms >= 500

    def test_continuous_tone_no_pause(self, analyzer: ProsodyAnalyzer) -> None:
        """A continuous tone should not have any pauses."""
        pauses = analyzer.detect_pauses(str(AUDIO_DIR / "tone_220hz.wav"))
        assert len(pauses) == 0

    def test_min_pause_ms_filter(self, analyzer: ProsodyAnalyzer) -> None:
        """Short gaps below threshold should not be detected."""
        pauses = analyzer.detect_pauses(
            str(AUDIO_DIR / "tone_gap_tone.wav"), min_pause_ms=1000
        )
        short_pauses = [p for p in pauses if p.duration_ms < 1000]
        assert len(short_pauses) == 0


# ---------------------------------------------------------------------------
# Speech rate
# ---------------------------------------------------------------------------


class TestSpeechRate:
    def test_speech_rate_not_none(self, analyzer: ProsodyAnalyzer) -> None:
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        f = results[0]
        assert f.speech_rate is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nonexistent_file_raises(self, analyzer: ProsodyAnalyzer) -> None:
        with pytest.raises(AudioProcessingError, match="not found"):
            analyzer.analyze("/nonexistent.wav", [])

    def test_empty_alignments(self, analyzer: ProsodyAnalyzer) -> None:
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), [])
        assert results == []

    def test_alignment_beyond_audio_clipped(self, analyzer: ProsodyAnalyzer) -> None:
        """Alignment end beyond audio duration should be clamped."""
        alignments = [WordAlignment(word="long", start_ms=0, end_ms=99999)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        assert len(results) == 1
        assert results[0].f0_mean is not None

    def test_multiple_alignments(self, analyzer: ProsodyAnalyzer) -> None:
        alignments = [
            WordAlignment(word="a", start_ms=100, end_ms=400),
            WordAlignment(word="b", start_ms=500, end_ms=900),
        ]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        assert len(results) == 2

    def test_detect_pauses_nonexistent_file(self, analyzer: ProsodyAnalyzer) -> None:
        with pytest.raises(AudioProcessingError, match="not found"):
            analyzer.detect_pauses("/nonexistent.wav")
