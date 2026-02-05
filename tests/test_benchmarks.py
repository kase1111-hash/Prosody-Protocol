"""Tests for Phase 12: Evaluation & Benchmarks.

Covers acceptance criteria:
- Benchmark runs against the emotional-speech dataset
- Report is saved as JSON for tracking over time
- CI can run benchmarks on a subset and fail if metrics regress
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from prosody_protocol import Benchmark, BenchmarkReport
from prosody_protocol.benchmarks import (
    compute_ece,
    compute_f1_from_counts,
    _compute_pause_f1,
    _extract_pauses,
    _extract_pitch_contours,
    _per_class_f1,
)
from prosody_protocol.datasets import Dataset, DatasetEntry, DatasetLoader
from prosody_protocol.parser import IMLParser

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC_DATASET = FIXTURES / "datasets" / "training_synthetic"


# ---------------------------------------------------------------------------
# Mock converter for testing
# ---------------------------------------------------------------------------


class MockConverter:
    """A fake AudioToIML converter that returns predictable IML."""

    def __init__(self, emotion: str = "neutral", confidence: float = 0.8):
        self.emotion = emotion
        self.confidence = confidence
        self.call_count = 0

    def convert(self, audio_path: str | Path) -> str:
        self.call_count += 1
        return (
            f'<utterance emotion="{self.emotion}" confidence="{self.confidence}">'
            f"Predicted text.</utterance>"
        )


class EmotionMappingConverter:
    """Returns IML with emotion matching the filename pattern."""

    _emotion_map = {
        "synth_001": "neutral",
        "synth_002": "angry",
        "synth_003": "joyful",
        "synth_004": "sad",
        "synth_005": "fearful",
        "synth_006": "sarcastic",
        "synth_007": "calm",
        "synth_008": "frustrated",
        "synth_009": "neutral",
        "synth_010": "angry",
    }

    def convert(self, audio_path: str | Path) -> str:
        stem = Path(audio_path).stem
        emotion = self._emotion_map.get(stem, "neutral")
        return (
            f'<utterance emotion="{emotion}" confidence="0.85">'
            f"Some text.</utterance>"
        )


class ConverterWithPauses:
    """Returns IML with pause elements for testing pause detection."""

    def convert(self, audio_path: str | Path) -> str:
        return (
            '<utterance emotion="neutral" confidence="0.7">'
            'Hello <pause duration="500"/> world.'
            "</utterance>"
        )


class ConverterWithProsody:
    """Returns IML with prosody elements for testing pitch contour metrics."""

    def convert(self, audio_path: str | Path) -> str:
        return (
            '<utterance emotion="neutral" confidence="0.7">'
            '<prosody pitch_contour="rise">Hello world</prosody>'
            "</utterance>"
        )


class FailingConverter:
    """Converter that always raises an exception."""

    def convert(self, audio_path: str | Path) -> str:
        raise RuntimeError("Conversion failed")


# ---------------------------------------------------------------------------
# Acceptance Criteria Tests
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """Test the three acceptance criteria from the execution guide."""

    def test_benchmark_runs_against_dataset(self):
        """AC1: Benchmark runs against the emotional-speech dataset."""
        loader = DatasetLoader()
        dataset = loader.load(SYNTHETIC_DATASET)
        converter = MockConverter()

        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()

        assert isinstance(report, BenchmarkReport)
        assert report.num_samples == 10
        assert 0.0 <= report.emotion_accuracy <= 1.0
        assert 0.0 <= report.validity_rate <= 1.0
        assert 0.0 <= report.confidence_ece <= 1.0
        assert report.duration_seconds >= 0

    def test_report_saved_as_json(self, tmp_path):
        """AC2: Report is saved as JSON for tracking over time."""
        loader = DatasetLoader()
        dataset = loader.load(SYNTHETIC_DATASET)
        converter = MockConverter()

        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()

        # Save
        json_path = tmp_path / "benchmark_report.json"
        report.save(json_path)
        assert json_path.exists()

        # Verify JSON contents
        with open(json_path) as f:
            data = json.load(f)
        assert "emotion_accuracy" in data
        assert "emotion_f1" in data
        assert "confidence_ece" in data
        assert "validity_rate" in data
        assert "num_samples" in data

        # Load back
        loaded = BenchmarkReport.load(json_path)
        assert loaded.num_samples == report.num_samples
        assert abs(loaded.emotion_accuracy - report.emotion_accuracy) < 1e-4
        assert abs(loaded.validity_rate - report.validity_rate) < 1e-4

    def test_ci_regression_check(self, tmp_path):
        """AC3: CI can run benchmarks on a subset and fail if metrics regress."""
        loader = DatasetLoader()
        dataset = loader.load(SYNTHETIC_DATASET)

        # Run baseline with perfect converter
        perfect = EmotionMappingConverter()
        benchmark = Benchmark(dataset, perfect, dataset_dir=SYNTHETIC_DATASET)
        baseline_report = benchmark.run()

        # Save baseline
        baseline_path = tmp_path / "baseline.json"
        baseline_report.save(baseline_path)

        # Run current with worse converter
        worse = MockConverter(emotion="angry", confidence=0.3)
        benchmark2 = Benchmark(dataset, worse, dataset_dir=SYNTHETIC_DATASET)
        current_report = benchmark2.run()

        # Check regression against baseline
        baseline_loaded = BenchmarkReport.load(baseline_path)
        failures = current_report.check_regression(baseline=baseline_loaded)

        # Should detect that emotion_accuracy regressed
        assert len(failures) > 0
        assert any("emotion_accuracy" in f for f in failures)

    def test_ci_threshold_check(self):
        """CI can check against minimum thresholds."""
        report = BenchmarkReport(
            emotion_accuracy=0.60,
            emotion_f1={"neutral": 0.7},
            confidence_ece=0.15,
            pitch_accuracy=0.0,
            pause_f1=0.0,
            validity_rate=1.0,
            num_samples=10,
            duration_seconds=1.0,
        )

        # Should fail: accuracy below threshold
        failures = report.check_regression(thresholds={"emotion_accuracy": 0.75})
        assert len(failures) > 0
        assert any("emotion_accuracy" in f for f in failures)

        # Should pass: accuracy above threshold
        failures = report.check_regression(thresholds={"emotion_accuracy": 0.50})
        assert len(failures) == 0

    def test_benchmark_subset(self):
        """CI can run on a subset using max_samples."""
        loader = DatasetLoader()
        dataset = loader.load(SYNTHETIC_DATASET)
        converter = MockConverter()

        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run(max_samples=3)

        assert report.num_samples == 3


# ---------------------------------------------------------------------------
# BenchmarkReport Tests
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    """Test BenchmarkReport serialization and methods."""

    @pytest.fixture()
    def sample_report(self):
        return BenchmarkReport(
            emotion_accuracy=0.85,
            emotion_f1={"neutral": 0.9, "angry": 0.8, "sad": 0.75},
            confidence_ece=0.05,
            pitch_accuracy=0.7,
            pause_f1=0.88,
            validity_rate=1.0,
            num_samples=100,
            duration_seconds=12.5,
        )

    def test_to_dict(self, sample_report):
        d = sample_report.to_dict()
        assert d["emotion_accuracy"] == 0.85
        assert d["emotion_f1"]["neutral"] == 0.9
        assert d["num_samples"] == 100
        # Values should be rounded
        assert isinstance(d["confidence_ece"], float)

    def test_to_dict_is_json_serializable(self, sample_report):
        d = sample_report.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        # Round-trip
        parsed = json.loads(serialized)
        assert parsed["emotion_accuracy"] == 0.85

    def test_save_and_load(self, sample_report, tmp_path):
        path = tmp_path / "report.json"
        sample_report.save(path)

        loaded = BenchmarkReport.load(path)
        assert loaded.emotion_accuracy == sample_report.emotion_accuracy
        assert loaded.emotion_f1 == sample_report.emotion_f1
        assert loaded.confidence_ece == sample_report.confidence_ece
        assert loaded.num_samples == sample_report.num_samples

    def test_save_creates_parent_dirs(self, sample_report, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "report.json"
        sample_report.save(path)
        assert path.exists()

    def test_check_regression_no_baseline(self, sample_report):
        failures = sample_report.check_regression()
        assert failures == []

    def test_check_regression_passes(self, sample_report):
        """No regression when current >= baseline."""
        baseline = BenchmarkReport(
            emotion_accuracy=0.80,
            emotion_f1={},
            confidence_ece=0.08,
            pitch_accuracy=0.65,
            pause_f1=0.85,
            validity_rate=1.0,
            num_samples=100,
            duration_seconds=10.0,
        )
        failures = sample_report.check_regression(baseline=baseline)
        assert failures == []

    def test_check_regression_detects_drop(self):
        current = BenchmarkReport(
            emotion_accuracy=0.60,
            emotion_f1={},
            confidence_ece=0.20,
            pitch_accuracy=0.50,
            pause_f1=0.70,
            validity_rate=0.90,
            num_samples=100,
            duration_seconds=10.0,
        )
        baseline = BenchmarkReport(
            emotion_accuracy=0.85,
            emotion_f1={},
            confidence_ece=0.05,
            pitch_accuracy=0.80,
            pause_f1=0.90,
            validity_rate=1.0,
            num_samples=100,
            duration_seconds=10.0,
        )
        failures = current.check_regression(baseline=baseline)
        assert len(failures) >= 3  # accuracy, pitch, pause, validity, ece all regressed

    def test_check_threshold_ece(self):
        """ECE is lower-is-better; exceeding threshold should fail."""
        report = BenchmarkReport(
            emotion_accuracy=0.9,
            emotion_f1={},
            confidence_ece=0.15,
            pitch_accuracy=0.8,
            pause_f1=0.9,
            validity_rate=1.0,
            num_samples=50,
            duration_seconds=5.0,
        )
        failures = report.check_regression(thresholds={"confidence_ece": 0.10})
        assert len(failures) == 1
        assert "confidence_ece" in failures[0]


# ---------------------------------------------------------------------------
# Metric Function Tests
# ---------------------------------------------------------------------------


class TestECE:
    """Test Expected Calibration Error computation."""

    def test_perfectly_calibrated(self):
        """If confidence exactly matches accuracy, ECE should be near 0."""
        confidences = [0.9] * 9 + [0.1]
        correct = [True] * 9 + [False]
        ece = compute_ece(confidences, correct)
        assert ece < 0.15  # Not exactly 0 due to binning

    def test_overconfident(self):
        """High confidence but low accuracy → high ECE."""
        confidences = [0.95] * 10
        correct = [False] * 10
        ece = compute_ece(confidences, correct)
        assert ece > 0.8

    def test_empty_inputs(self):
        assert compute_ece([], []) == 0.0

    def test_single_prediction(self):
        ece = compute_ece([0.8], [True])
        assert 0.0 <= ece <= 1.0


class TestF1FromCounts:
    """Test raw F1 computation."""

    def test_perfect_f1(self):
        assert compute_f1_from_counts(10, 0, 0) == 1.0

    def test_zero_f1(self):
        assert compute_f1_from_counts(0, 5, 5) == 0.0

    def test_partial_f1(self):
        f1 = compute_f1_from_counts(5, 5, 5)
        assert 0.0 < f1 < 1.0


class TestPerClassF1:
    """Test per-class F1 computation."""

    def test_perfect_predictions(self):
        y_true = ["a", "b", "c", "a"]
        y_pred = ["a", "b", "c", "a"]
        f1 = _per_class_f1(y_true, y_pred)
        assert f1["a"] == 1.0
        assert f1["b"] == 1.0
        assert f1["c"] == 1.0

    def test_all_wrong(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["b", "b", "a", "a"]
        f1 = _per_class_f1(y_true, y_pred)
        assert f1["a"] == 0.0
        assert f1["b"] == 0.0


class TestPauseF1:
    """Test pause detection F1."""

    def test_no_pauses_both_sides(self):
        assert _compute_pause_f1([], []) == 1.0

    def test_predicted_but_no_truth(self):
        assert _compute_pause_f1([500], []) == 0.0

    def test_truth_but_no_predicted(self):
        assert _compute_pause_f1([], [500]) == 0.0

    def test_exact_match(self):
        assert _compute_pause_f1([500, 300], [500, 300]) == 1.0

    def test_within_tolerance(self):
        f1 = _compute_pause_f1([500], [550], tolerance_ms=200)
        assert f1 == 1.0

    def test_outside_tolerance(self):
        f1 = _compute_pause_f1([500], [800], tolerance_ms=200)
        assert f1 == 0.0


# ---------------------------------------------------------------------------
# IML Extraction Tests
# ---------------------------------------------------------------------------


class TestIMLExtraction:
    """Test extraction of pauses and pitch contours from IML."""

    def test_extract_pauses_simple(self):
        parser = IMLParser()
        doc = parser.parse(
            '<utterance>Hello <pause duration="500"/> world.</utterance>'
        )
        pauses = _extract_pauses(doc)
        assert pauses == [500]

    def test_extract_pauses_multiple(self):
        parser = IMLParser()
        doc = parser.parse(
            '<utterance>A <pause duration="200"/> B <pause duration="800"/> C</utterance>'
        )
        pauses = _extract_pauses(doc)
        assert pauses == [200, 800]

    def test_extract_pauses_none(self):
        parser = IMLParser()
        doc = parser.parse("<utterance>No pauses here.</utterance>")
        pauses = _extract_pauses(doc)
        assert pauses == []

    def test_extract_pitch_contours(self):
        parser = IMLParser()
        doc = parser.parse(
            '<utterance>'
            '<prosody pitch_contour="rise">Going up</prosody>'
            '</utterance>'
        )
        contours = _extract_pitch_contours(doc)
        assert contours == ["rise"]

    def test_extract_no_contours(self):
        parser = IMLParser()
        doc = parser.parse("<utterance>Plain text.</utterance>")
        contours = _extract_pitch_contours(doc)
        assert contours == []


# ---------------------------------------------------------------------------
# Benchmark Execution Tests
# ---------------------------------------------------------------------------


class TestBenchmarkExecution:
    """Test Benchmark class execution details."""

    @pytest.fixture()
    def dataset(self):
        loader = DatasetLoader()
        return loader.load(SYNTHETIC_DATASET)

    def test_perfect_converter_high_accuracy(self, dataset):
        """A converter that returns matching emotions should score well."""
        converter = EmotionMappingConverter()
        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()

        assert report.emotion_accuracy == 1.0
        assert report.validity_rate == 1.0
        assert all(v == 1.0 for v in report.emotion_f1.values())

    def test_wrong_emotion_low_accuracy(self, dataset):
        """A converter always returning 'angry' should have low accuracy."""
        converter = MockConverter(emotion="angry")
        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()

        # Only 2 out of 10 entries are "angry"
        assert report.emotion_accuracy < 0.5

    def test_validity_rate_always_valid(self, dataset):
        """Mock converter always produces valid IML."""
        converter = MockConverter()
        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()
        assert report.validity_rate == 1.0

    def test_pause_detection_with_pauses(self, dataset):
        converter = ConverterWithPauses()
        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()
        # Ground truth has no pauses, so predicted pauses are all false positives
        assert report.pause_f1 == 0.0

    def test_failing_converter_handles_gracefully(self, dataset):
        """Failing converter should not crash, just skip entries."""
        converter = FailingConverter()
        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()
        assert report.num_samples == 0

    def test_no_dataset_dir_uses_entry_iml(self, dataset):
        """Without dataset_dir, benchmark evaluates ground-truth IML itself."""
        converter = MockConverter()  # Won't be called
        benchmark = Benchmark(dataset, converter, dataset_dir=None)
        report = benchmark.run()

        assert report.num_samples == 10
        assert report.validity_rate == 1.0
        # Emotion accuracy is 1.0 since ground truth IML matches ground truth label
        assert report.emotion_accuracy == 1.0

    def test_per_class_f1_in_report(self, dataset):
        converter = EmotionMappingConverter()
        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()

        assert isinstance(report.emotion_f1, dict)
        assert len(report.emotion_f1) > 0
        for label, f1_val in report.emotion_f1.items():
            assert isinstance(label, str)
            assert 0.0 <= f1_val <= 1.0

    def test_ece_in_range(self, dataset):
        converter = MockConverter(confidence=0.99)
        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()
        assert 0.0 <= report.confidence_ece <= 1.0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataset(self):
        dataset = Dataset(name="empty", entries=[], metadata={})
        converter = MockConverter()
        benchmark = Benchmark(dataset, converter)
        report = benchmark.run()

        assert report.num_samples == 0
        assert report.emotion_accuracy == 0.0
        assert report.validity_rate == 0.0

    def test_single_entry_dataset(self):
        entry = DatasetEntry(
            id="e1",
            timestamp="2025-01-01T00:00:00Z",
            source="synthetic",
            language="en-US",
            audio_file="audio/test.wav",
            transcript="Hello.",
            iml='<utterance emotion="joyful" confidence="0.9">Hello.</utterance>',
            emotion_label="joyful",
            annotator="human",
            consent=True,
        )
        dataset = Dataset(name="single", entries=[entry], metadata={})
        converter = MockConverter(emotion="joyful", confidence=0.9)

        benchmark = Benchmark(dataset, converter, dataset_dir=None)
        report = benchmark.run()

        assert report.num_samples == 1
        assert report.emotion_accuracy == 1.0

    def test_report_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BenchmarkReport.load(tmp_path / "nonexistent.json")

    def test_max_samples_zero(self):
        dataset = Dataset(name="test", entries=[], metadata={})
        converter = MockConverter()
        benchmark = Benchmark(dataset, converter)
        report = benchmark.run(max_samples=0)
        assert report.num_samples == 0
