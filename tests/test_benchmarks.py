"""Tests for Phase 12: Evaluation & Benchmarks.

Covers acceptance criteria:
- Benchmark runs against the emotional-speech dataset
- Report is saved as JSON for tracking over time
- CI can run benchmarks on a subset and fail if metrics regress

and that every entry counts: failed conversions lower every rate, pauses
and contours are matched per entry by position, unmeasured metrics are
None, and metric values agree with scikit-learn.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from prosody_protocol import Benchmark, BenchmarkReport
from prosody_protocol.benchmarks import (
    _compute_pause_f1,
    _extract_pauses,
    _extract_pitch_contours,
    _per_class_f1,
    compute_ece,
    compute_f1_from_counts,
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


class ScriptedConverter:
    """Returns the IML given for each audio file name; raises for others."""

    def __init__(self, outputs: dict[str, object]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []

    def convert(self, audio_path: str | Path) -> str:
        name = Path(audio_path).name
        self.calls.append(name)
        if name not in self.outputs:
            raise RuntimeError(f"cannot convert {name}")
        return self.outputs[name]  # type: ignore[return-value]


def _entry(entry_id: str, emotion: str, iml: str | None = None) -> DatasetEntry:
    return DatasetEntry(
        id=entry_id,
        timestamp="2025-01-01T00:00:00Z",
        source="synthetic",
        language="en-US",
        audio_file=f"audio/{entry_id}.wav",
        transcript="x",
        iml=iml if iml is not None else "<utterance>x</utterance>",
        emotion_label=emotion,
        annotator="human",
        consent=True,
    )


def _run(entries: list[DatasetEntry], outputs: dict[str, object], **kwargs) -> BenchmarkReport:
    dataset = Dataset(name="scripted", entries=entries)
    return Benchmark(dataset, ScriptedConverter(outputs), dataset_dir="/data", **kwargs).run()


def _report(**changes) -> BenchmarkReport:
    values = dict(
        emotion_accuracy=0.9, emotion_f1={"angry": 0.9, "neutral": 0.95}, confidence_ece=0.05,
        pitch_accuracy=0.8, pause_f1=0.9, validity_rate=1.0, num_samples=100, num_failures=0,
        duration_seconds=1.0, pitch_coverage=0.9,
    )
    values.update(changes)
    return BenchmarkReport(**values)


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
            num_failures=0,
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
            num_failures=0,
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
            num_failures=0,
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
            num_failures=0,
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
            num_failures=0,
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
            num_failures=0,
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
        assert ece == pytest.approx(0.2)

    def test_hand_computed(self):
        # Bin [0.9, 1.0]: conf 0.95, acc 0.5 -> 0.45; bin [0.5, 0.6): conf 0.55, acc 1 -> 0.45.
        assert compute_ece([0.95, 0.95, 0.55, 0.55], [True, False, True, True]) == (
            pytest.approx(0.45)
        )

    def test_out_of_range_confidences_are_clipped(self):
        """A wrong prediction at confidence 1.5 is maximally miscalibrated."""
        assert compute_ece([1.2], [False]) == pytest.approx(1.0)
        assert compute_ece([1.5, 1.5], [False, False]) == pytest.approx(1.0)
        assert compute_ece([-0.1], [True]) == pytest.approx(1.0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="finite"):
            compute_ece([float("nan")], [True])
        with pytest.raises(ValueError, match="2 confidences but 1"):
            compute_ece([0.5, 0.5], [True])
        with pytest.raises(ValueError, match="n_bins"):
            compute_ece([0.5], [True], n_bins=0)


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

    def test_matching_is_optimal(self):
        """Greedy nearest-first pairing would match 200 with 250 and miss 0."""
        assert _compute_pause_f1([200, 400], [0, 250]) == 1.0

    def test_matching_is_one_to_one(self):
        assert _compute_pause_f1([500, 500], [500]) == pytest.approx(2 / 3)


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

    def test_extract_nested(self):
        doc = IMLParser().parse(
            '<utterance><emphasis level="strong"><prosody pitch_contour="fall">A '
            '<pause duration="300"/><prosody pitch_contour="rise">B</prosody></prosody>'
            "</emphasis></utterance>"
        )
        assert _extract_pitch_contours(doc) == ["fall", "rise"]
        assert _extract_pauses(doc) == [300]


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
        """A failing converter does not crash the run; every entry counts as wrong."""
        converter = FailingConverter()
        benchmark = Benchmark(dataset, converter, dataset_dir=SYNTHETIC_DATASET)
        report = benchmark.run()
        assert report.num_samples == 0
        assert report.num_failures == 10
        assert report.failure_rate == 1.0
        assert report.emotion_accuracy == 0.0
        assert report.validity_rate == 0.0
        assert report.confidence_ece is None
        assert report.pause_f1 is None

    def test_loaded_dataset_supplies_its_directory(self, dataset):
        """Without dataset_dir, the converter still runs, on the dataset's own audio."""
        converter = MockConverter()
        report = Benchmark(dataset, converter).run()
        assert converter.call_count == 10
        assert report.num_samples == 10

    def test_in_memory_dataset_needs_a_directory(self):
        dataset = Dataset(name="mem", entries=[_entry("e1", "neutral")])
        with pytest.raises(ValueError, match="dataset_dir"):
            Benchmark(dataset, MockConverter())

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

    def test_empty_dataset(self, tmp_path):
        dataset = Dataset(name="empty", entries=[], metadata={})
        converter = MockConverter()
        benchmark = Benchmark(dataset, converter, dataset_dir=tmp_path)
        report = benchmark.run()

        assert report.num_samples == 0
        assert report.emotion_accuracy == 0.0
        assert report.validity_rate == 0.0
        assert report.check_regression() == ["no dataset entries were evaluated"]

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

        benchmark = Benchmark(dataset, converter, dataset_dir=FIXTURES)
        report = benchmark.run()

        assert report.num_samples == 1
        assert report.emotion_accuracy == 1.0

    def test_report_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BenchmarkReport.load(tmp_path / "nonexistent.json")

    def test_max_samples_zero(self, tmp_path):
        dataset = Dataset(name="test", entries=[_entry("e1", "neutral")], metadata={})
        converter = MockConverter()
        benchmark = Benchmark(dataset, converter, dataset_dir=tmp_path)
        report = benchmark.run(max_samples=0)
        assert report.num_samples == 0
        assert converter.call_count == 0
        with pytest.raises(ValueError, match="max_samples"):
            benchmark.run(max_samples=-1)


# ---------------------------------------------------------------------------
# Every entry counts
# ---------------------------------------------------------------------------

_JOYFUL = '<utterance emotion="joyful" confidence="0.9">x</utterance>'


class TestFailuresCount:
    def test_crashes_count_against_every_rate(self):
        entries = [_entry("ok", "joyful")] + [_entry(f"f{i}", "sad") for i in range(9)]
        report = _run(entries, {"ok.wav": _JOYFUL})
        assert report.num_samples == 1
        assert report.num_failures == 9
        assert report.failure_rate == pytest.approx(0.9)
        assert report.emotion_accuracy == pytest.approx(0.1)
        assert report.validity_rate == pytest.approx(0.1)
        assert report.emotion_f1 == {"joyful": 1.0, "sad": 0.0}

    def test_default_regression_check_fails_on_failures(self):
        entries = [_entry("ok", "joyful")] + [_entry(f"f{i}", "sad") for i in range(9)]
        report = _run(entries, {"ok.wav": _JOYFUL})
        failures = report.check_regression(
            thresholds={"emotion_accuracy": 0.05, "confidence_ece": 0.2}
        )
        assert failures == ["failure_rate = 0.9000 > threshold 0.0000"]
        assert report.check_regression(thresholds={"failure_rate": 0.9}) == []

    def test_unparseable_output_is_a_failure(self):
        entries = [_entry("ok", "joyful"), _entry("bad", "sad")]
        report = _run(entries, {"ok.wav": _JOYFUL, "bad.wav": "<utterance emotion='angry'"})
        assert report.num_samples == 1
        assert report.num_failures == 1
        assert report.emotion_accuracy == 0.5
        assert report.validity_rate == 0.5
        assert report.emotion_f1 == {"joyful": 1.0, "sad": 0.0}

    @pytest.mark.parametrize("output", ["", "   ", "Hello world", "<result>ok</result>"])
    def test_non_iml_output_fails_the_default_gate(self, output):
        """A converter that swallows its errors and returns "" used to pass
        check_regression() with failure_rate 0."""
        entries = [_entry("e1", "sad"), _entry("e2", "joyful")]
        report = _run(entries, {"e1.wav": output, "e2.wav": output})
        assert report.num_failures == 2
        assert report.failure_rate == 1.0
        assert report.check_regression() == ["failure_rate = 1.0000 > threshold 0.0000"]

    def test_parseable_but_invalid_output_is_a_sample(self):
        """Output that parses but breaks an IML rule is scored, as invalid."""
        no_confidence = '<utterance emotion="sad">x</utterance>'
        report = _run([_entry("e1", "sad")], {"e1.wav": no_confidence})
        assert (report.num_samples, report.num_failures) == (1, 0)
        assert report.validity_rate == 0.0
        assert report.emotion_accuracy == 1.0

    def test_non_string_output_is_a_failure(self):
        report = _run([_entry("e1", "sad")], {"e1.wav": b"<utterance>x</utterance>"})
        assert report.num_failures == 1

    @pytest.mark.parametrize("audio_file", ["../../etc/hostname", "audio/a\x00b.wav"])
    def test_audio_outside_the_dataset_is_not_converted(self, audio_file):
        """A NUL byte in the path used to abort the whole run with ValueError."""
        entries = [replace(_entry("e1", "sad"), audio_file=audio_file), _entry("ok", "joyful")]
        converter = ScriptedConverter({"hostname": _JOYFUL, "ok.wav": _JOYFUL})
        report = Benchmark(Dataset("t", entries), converter, dataset_dir="/data").run()
        assert converter.calls == ["ok.wav"]
        assert (report.num_samples, report.num_failures) == (1, 1)

    def test_failed_entries_miss_their_pauses_and_contours(self):
        truth = (
            '<utterance><prosody pitch_contour="rise">so</prosody> '
            'what <pause duration="400"/> now</utterance>'
        )
        report = _run([_entry("e1", "neutral", truth)], {})
        assert report.pause_f1 == 0.0
        assert report.pitch_coverage == 0.0
        assert report.pitch_accuracy is None


# ---------------------------------------------------------------------------
# Emotion metrics
# ---------------------------------------------------------------------------


class TestEmotionMetrics:
    def test_matches_sklearn(self):
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        labels = ["angry", "angry", "sad", "sad", "sad", "joyful", "calm", "calm", "angry", "sad"]
        predicted = ["angry", "sad", "sad", None, "joyful", "joyful", "calm", "angry", "angry",
                     "sad"]
        entries = [_entry(f"e{i}", label) for i, label in enumerate(labels)]
        outputs = {
            f"e{i}.wav": f'<utterance emotion="{p}" confidence="0.7">x</utterance>'
            for i, p in enumerate(predicted)
            if p is not None
        }
        report = _run(entries, outputs)

        y_pred = [p if p is not None else "<failed>" for p in predicted]
        classes = sorted(set(labels) | set(predicted) - {None})
        expected = sklearn_metrics.f1_score(
            labels, y_pred, labels=classes, average=None, zero_division=0
        )
        assert report.emotion_f1 == pytest.approx(dict(zip(classes, expected, strict=True)))
        assert report.emotion_f1_macro == pytest.approx(
            sklearn_metrics.f1_score(
                labels, y_pred, labels=classes, average="macro", zero_division=0
            )
        )
        assert report.emotion_accuracy == pytest.approx(
            sklearn_metrics.accuracy_score(labels, y_pred)
        )

    def test_one_prediction_per_entry(self):
        """A multi-utterance output is one prediction: its most confident emotion."""
        output = (
            "<iml><utterance>Well.</utterance>"
            '<utterance emotion="sad" confidence="0.6">I see.</utterance>'
            '<utterance emotion="sarcastic" confidence="0.8">Great.</utterance></iml>'
        )
        report = _run([_entry("e1", "sarcastic")], {"e1.wav": output})
        assert report.emotion_accuracy == 1.0
        assert report.confidence_ece == pytest.approx(0.2)

    def test_no_emotion_means_neutral_without_confidence(self):
        report = _run([_entry("e1", "neutral")], {"e1.wav": "<utterance>x</utterance>"})
        assert report.emotion_accuracy == 1.0
        assert report.confidence_ece is None

    def test_invalid_confidence_does_not_score_as_calibrated(self):
        """confidence="1.5" is invalid IML: it lowers validity and is left out
        of the ECE instead of scoring as perfect calibration."""
        invalid = '<utterance emotion="angry" confidence="1.5">x</utterance>'
        report = _run(
            [_entry("a", "sad"), _entry("b", "sad")], {"a.wav": invalid, "b.wav": invalid}
        )
        assert report.confidence_ece is None
        assert report.validity_rate == 0.0
        assert report.check_regression(
            thresholds={"confidence_ece": 0.1, "validity_rate": 1.0}
        ) == [
            "confidence_ece was not measured (threshold 0.1000)",
            "validity_rate = 0.0000 < threshold 1.0000",
        ]

        valid_wrong = '<utterance emotion="angry" confidence="0.9">x</utterance>'
        report = _run(
            [_entry("a", "sad"), _entry("b", "sad")], {"a.wav": invalid, "b.wav": valid_wrong}
        )
        assert report.confidence_ece == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Pause F1: per entry, by position and duration
# ---------------------------------------------------------------------------


class TestPauseMatching:
    def test_pauses_do_not_match_across_entries(self):
        entries = [
            _entry("a", "neutral", '<utterance>hello <pause duration="500"/> world</utterance>'),
            _entry("b", "neutral", "<utterance>good morning</utterance>"),
        ]
        outputs = {
            "a.wav": "<utterance>hello world</utterance>",
            "b.wav": '<utterance>good <pause duration="500"/> morning</utterance>',
        }
        assert _run(entries, outputs).pause_f1 == 0.0

    def test_pause_in_the_wrong_place_does_not_match(self):
        truth = '<utterance>one <pause duration="500"/> two three four five</utterance>'
        wrong = '<utterance>one two three four <pause duration="500"/> five</utterance>'
        right = '<utterance>one <pause duration="480"/> two three four five</utterance>'
        assert _run([_entry("e", "neutral", truth)], {"e.wav": wrong}).pause_f1 == 0.0
        assert _run([_entry("e", "neutral", truth)], {"e.wav": right}).pause_f1 == 1.0

    def test_position_tolerance(self):
        truth = '<utterance>one <pause duration="500"/> two three four five</utterance>'
        near = '<utterance>one two <pause duration="500"/> three four five</utterance>'
        entries = [_entry("e", "neutral", truth)]
        assert _run(entries, {"e.wav": near}).pause_f1 == 1.0
        strict = _run(entries, {"e.wav": near}, pause_position_tolerance=0)
        assert strict.pause_f1 == 0.0

    def test_duration_tolerance(self):
        truth = '<utterance>one <pause duration="500"/> two</utterance>'
        longer = '<utterance>one <pause duration="900"/> two</utterance>'
        entries = [_entry("e", "neutral", truth)]
        assert _run(entries, {"e.wav": longer}).pause_f1 == 0.0
        assert _run(entries, {"e.wav": longer}, pause_tolerance_ms=500).pause_f1 == 1.0

    def test_positions_follow_the_transcript_alignment(self):
        """The prediction's transcript lost words; the pause still lines up."""
        truth = (
            "<utterance>well you know I really do not think so "
            '<pause duration="600"/> honestly</utterance>'
        )
        pred = '<utterance>I don\'t think so <pause duration="600"/> honestly</utterance>'
        assert _run([_entry("e", "neutral", truth)], {"e.wav": pred}).pause_f1 == 1.0

    def test_counts_are_summed_over_entries(self):
        truth = '<utterance>a <pause duration="300"/> b <pause duration="300"/> c</utterance>'
        entries = [_entry("e1", "neutral", truth), _entry("e2", "neutral", truth)]
        outputs = {
            "e1.wav": truth,
            "e2.wav": '<utterance>a b <pause duration="300"/> c</utterance>',
        }
        # tp 3, fp 0, fn 1
        assert _run(entries, outputs).pause_f1 == pytest.approx(6 / 7)

    def test_no_pauses_anywhere_is_not_measured(self):
        report = _run([_entry("e", "neutral")], {"e.wav": "<utterance>x</utterance>"})
        assert report.pause_f1 is None


# ---------------------------------------------------------------------------
# Pitch contours: by word position, with coverage
# ---------------------------------------------------------------------------

_CONTOUR_TRUTH = (
    '<utterance><prosody pitch_contour="rise">a</prosody> '
    '<prosody pitch_contour="fall">b</prosody> '
    '<prosody pitch_contour="fall">c</prosody></utterance>'
)


class TestPitchContours:
    @pytest.mark.parametrize(
        ("prediction", "accuracy", "coverage"),
        [
            ('<utterance><prosody pitch_contour="rise">a</prosody> b c</utterance>', 1.0, 1 / 3),
            ('<utterance>a b <prosody pitch_contour="fall">c</prosody></utterance>', 1.0, 1 / 3),
            ('<utterance>a <prosody pitch_contour="rise">b</prosody> c</utterance>', 0.0, 1 / 3),
            ("<utterance>a b c</utterance>", None, 0.0),
            (_CONTOUR_TRUTH, 1.0, 1.0),
        ],
        ids=["first-only", "last-only", "wrong", "none", "all"],
    )
    def test_contours_compared_on_the_same_words(self, prediction, accuracy, coverage):
        report = _run([_entry("e", "neutral", _CONTOUR_TRUTH)], {"e.wav": prediction})
        assert report.pitch_accuracy == (None if accuracy is None else pytest.approx(accuracy))
        assert report.pitch_coverage == pytest.approx(coverage)

    def test_word_level_prediction_of_a_phrase_contour(self):
        truth = '<utterance><prosody pitch_contour="rise">going up now</prosody></utterance>'
        pred = (
            '<utterance><prosody pitch_contour="flat">going</prosody> '
            '<prosody pitch_contour="rise">up</prosody> '
            '<prosody pitch_contour="rise">now</prosody></utterance>'
        )
        report = _run([_entry("e", "neutral", truth)], {"e.wav": pred})
        assert report.pitch_accuracy == 1.0

    def test_ground_truth_without_contours_is_not_measured(self):
        pred = '<utterance><prosody pitch_contour="rise">x</prosody></utterance>'
        report = _run([_entry("e", "neutral")], {"e.wav": pred})
        assert report.pitch_accuracy is None
        assert report.pitch_coverage is None


# ---------------------------------------------------------------------------
# Regression checks
# ---------------------------------------------------------------------------


class TestRegressionChecks:
    def test_unknown_threshold_key_rejected(self):
        unknown = r"Unknown threshold metric\(s\) \['emotion_acuracy'\]"
        with pytest.raises(ValueError, match=unknown):
            _report().check_regression(thresholds={"emotion_acuracy": 0.99})
        with pytest.raises(ValueError, match="emotion_f1"):
            _report().check_regression(thresholds={"emotion_f1": 0.7})

    @pytest.mark.parametrize("limit", [float("nan"), True, "0.5"])
    def test_threshold_must_be_a_number(self, limit):
        with pytest.raises(ValueError, match="finite number"):
            _report().check_regression(thresholds={"emotion_accuracy": limit})

    def test_threshold_directions(self):
        report = _report(emotion_accuracy=0.7, confidence_ece=0.2, num_failures=5)
        failures = report.check_regression(thresholds={
            "emotion_accuracy": 0.75,
            "emotion_f1_macro": 0.99,
            "confidence_ece": 0.1,
            "failure_rate": 0.01,
            "pitch_coverage": 0.5,
        })
        assert failures == [
            "emotion_accuracy = 0.7000 < threshold 0.7500",
            "emotion_f1_macro = 0.9250 < threshold 0.9900",
            "confidence_ece = 0.2000 > threshold 0.1000",
            "failure_rate = 0.0476 > threshold 0.0100",
        ]

    def test_unmeasured_metric_with_threshold_fails(self):
        report = _report(pitch_accuracy=None, pause_f1=None)
        assert report.check_regression(thresholds={"pause_f1": 0.85}) == [
            "pause_f1 was not measured (threshold 0.8500)"
        ]

    def test_per_class_f1_collapse_is_a_regression(self):
        current = _report(emotion_f1={"angry": 0.05, "neutral": 0.95})
        failures = current.check_regression(baseline=_report())
        assert "emotion_f1[angry] regressed: 0.0500 < baseline 0.9000" in failures
        assert any(f.startswith("emotion_f1_macro regressed") for f in failures)

    def test_failure_rate_rise_is_a_regression(self):
        failures = _report(num_failures=10).check_regression(
            baseline=_report(), thresholds={"failure_rate": 0.5}
        )
        assert failures == ["failure_rate regressed: 0.0909 > baseline 0.0000"]

    def test_unmeasured_metrics_are_not_compared_with_baseline(self):
        current = _report(pitch_accuracy=None, pitch_coverage=None, pause_f1=None)
        assert current.check_regression(baseline=_report()) == []

    def test_tolerance(self):
        current = _report(emotion_accuracy=0.87)
        assert current.check_regression(baseline=_report()) == [
            "emotion_accuracy regressed: 0.8700 < baseline 0.9000"
        ]
        assert current.check_regression(baseline=_report(), tolerance=0.05) == []


class TestReportSerialization:
    def test_unmeasured_metrics_round_trip_as_null(self, tmp_path):
        report = _report(confidence_ece=None, pitch_accuracy=None, pitch_coverage=None,
                         pause_f1=None, num_failures=1)
        path = tmp_path / "report.json"
        report.save(path)
        data = json.loads(path.read_text())
        assert data["pitch_accuracy"] is None and data["pause_f1"] is None
        assert data["failure_rate"] == pytest.approx(1 / 101, abs=1e-4)
        assert data["emotion_f1_macro"] == pytest.approx(0.925)
        loaded = BenchmarkReport.load(path)
        assert loaded == report

    def test_reports_saved_before_coverage_existed_load(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({
            "emotion_accuracy": 0.8, "emotion_f1": {"a": 0.8}, "confidence_ece": 0.1,
            "pitch_accuracy": 0.0, "pause_f1": 1.0, "validity_rate": 1.0,
            "num_samples": 10, "duration_seconds": 1.0,
        }))
        loaded = BenchmarkReport.load(path)
        assert loaded.pitch_coverage is None
        assert loaded.num_failures == 0


# ---------------------------------------------------------------------------
# End to end with the real converter
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")  # no transcript: placeholder words
def test_audio_to_iml_benchmark_runs_on_fixture_dataset():
    pytest.importorskip("parselmouth")
    from prosody_protocol import AudioToIML

    dataset = DatasetLoader().load(SYNTHETIC_DATASET)
    report = Benchmark(dataset, AudioToIML(stt="none")).run()
    assert report.num_failures == 0
    assert report.num_samples == 10
    assert report.validity_rate == 1.0
    # The fixture ground truth has no contours or pauses to compare against.
    assert report.pitch_accuracy is None
    assert report.pitch_coverage is None


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng is not installed")
@pytest.mark.filterwarnings("ignore::UserWarning")  # no transcript: placeholder words
def test_pause_detection_benchmark_on_real_speech(tmp_path):
    """Speech with known breaks, converted by AudioToIML, scored against IML
    that marks those breaks, marks none, or gives them the wrong length.

    Without speech recognition the converter writes one placeholder per
    stretch of speech, so pauses are matched to within a stretch of speech.
    """
    pytest.importorskip("parselmouth")
    from prosody_protocol import AudioToIML

    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "while",
             "the", "children", "watch"]
    breaks = {1, 3, 5, 8}  # a 600 ms break after these word indices
    ssml = " ".join(
        w + (' <break time="600ms"/>' if i in breaks else "") for i, w in enumerate(words)
    )
    root = tmp_path / "ds"
    (root / "audio").mkdir(parents=True)
    (root / "entries").mkdir()
    subprocess.run(
        ["espeak-ng", "-m", "-w", str(root / "audio" / "e1.wav"), f"<speak>{ssml}</speak>"],
        check=True, capture_output=True,
    )

    def score(pause_ms: int | None) -> float | None:
        text = " ".join(
            w + (f' <pause duration="{pause_ms}"/>' if pause_ms and i in breaks else "")
            for i, w in enumerate(words)
        )
        entry = {
            "id": "e1", "timestamp": "2025-01-01T00:00:00Z", "source": "synthetic",
            "language": "en-US", "audio_file": "audio/e1.wav", "transcript": " ".join(words),
            "iml": f"<utterance>{text}</utterance>", "emotion_label": "neutral",
            "annotator": "human", "consent": True,
        }
        (root / "entries" / "e1.json").write_text(json.dumps(entry))
        dataset = DatasetLoader().load(root, check_audio=True)
        report = Benchmark(dataset, AudioToIML(stt="none")).run()
        assert report.num_failures == 0
        return report.pause_f1

    assert score(600) == 1.0
    assert score(None) == 0.0  # every detected pause is a false positive
    assert score(1500) == 0.0  # right places, wrong length
