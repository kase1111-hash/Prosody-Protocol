"""Evaluation harness and benchmarking for prosody models.

Provides a Benchmark class that runs an AudioToIML converter against a
labelled dataset, computing emotion accuracy, confidence calibration,
pitch/pause metrics, and IML validity.

Phase 12 deliverable from EXECUTION_GUIDE.md.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

from .datasets import Dataset
from .models import IMLDocument, Pause, Prosody, Utterance
from .parser import IMLParser
from .validator import IMLValidator


# ---------------------------------------------------------------------------
# Converter protocol -- accepts AudioToIML or any compatible object
# ---------------------------------------------------------------------------


class _Converter(Protocol):
    def convert(self, audio_path: str | Path) -> str: ...


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """Results of a benchmark run against a dataset.

    Attributes
    ----------
    emotion_accuracy:
        Fraction of utterances with correctly predicted emotion label.
    emotion_f1:
        Per-class F1 scores keyed by emotion label.
    confidence_ece:
        Expected Calibration Error of the emotion confidence values.
    pitch_accuracy:
        Fraction of utterances with correctly predicted pitch contour.
    pause_f1:
        F1 score for pause detection (predicted vs ground truth pauses).
    validity_rate:
        Fraction of generated IML strings that pass validation.
    num_samples:
        Total number of dataset entries processed.
    duration_seconds:
        Wall-clock time taken for the benchmark run.
    """

    emotion_accuracy: float
    emotion_f1: dict[str, float]
    confidence_ece: float
    pitch_accuracy: float
    pause_f1: float
    validity_rate: float
    num_samples: int
    num_failures: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "emotion_accuracy": round(self.emotion_accuracy, 4),
            "emotion_f1": {k: round(v, 4) for k, v in self.emotion_f1.items()},
            "confidence_ece": round(self.confidence_ece, 4),
            "pitch_accuracy": round(self.pitch_accuracy, 4),
            "pause_f1": round(self.pause_f1, 4),
            "validity_rate": round(self.validity_rate, 4),
            "num_samples": self.num_samples,
            "num_failures": self.num_failures,
            "duration_seconds": round(self.duration_seconds, 3),
        }

    def save(self, path: str | Path) -> None:
        """Save report as JSON for tracking over time."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkReport:
        """Load a previously saved benchmark report from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(
            emotion_accuracy=data["emotion_accuracy"],
            emotion_f1=data["emotion_f1"],
            confidence_ece=data["confidence_ece"],
            pitch_accuracy=data["pitch_accuracy"],
            pause_f1=data["pause_f1"],
            validity_rate=data["validity_rate"],
            num_samples=data["num_samples"],
            num_failures=data.get("num_failures", 0),
            duration_seconds=data["duration_seconds"],
        )

    def check_regression(
        self,
        baseline: BenchmarkReport | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> list[str]:
        """Check whether metrics regress below baseline or minimum thresholds.

        Parameters
        ----------
        baseline:
            A previous report to compare against. If any metric in the
            current report is worse, a failure message is produced.
        thresholds:
            Minimum acceptable values keyed by metric name.  For example:
            ``{"emotion_accuracy": 0.75, "validity_rate": 1.0}``

        Returns
        -------
        list[str]
            List of failure messages.  Empty means all checks passed.
        """
        failures: list[str] = []

        # Higher-is-better metrics
        _higher = {
            "emotion_accuracy": self.emotion_accuracy,
            "pitch_accuracy": self.pitch_accuracy,
            "pause_f1": self.pause_f1,
            "validity_rate": self.validity_rate,
        }
        # Lower-is-better metrics
        _lower = {
            "confidence_ece": self.confidence_ece,
        }

        if thresholds:
            for metric, minimum in thresholds.items():
                if metric in _higher and _higher[metric] < minimum:
                    failures.append(
                        f"{metric} = {_higher[metric]:.4f} < threshold {minimum:.4f}"
                    )
                if metric == "confidence_ece" and self.confidence_ece > minimum:
                    failures.append(
                        f"confidence_ece = {self.confidence_ece:.4f} > threshold {minimum:.4f}"
                    )

        if baseline is not None:
            for metric, current in _higher.items():
                baseline_val = getattr(baseline, metric)
                if current < baseline_val - 0.01:  # allow 1% tolerance
                    failures.append(
                        f"{metric} regressed: {current:.4f} < baseline {baseline_val:.4f}"
                    )
            if self.confidence_ece > baseline.confidence_ece + 0.01:
                failures.append(
                    f"confidence_ece regressed: {self.confidence_ece:.4f} > "
                    f"baseline {baseline.confidence_ece:.4f}"
                )

        return failures


# ---------------------------------------------------------------------------
# Metric helper functions
# ---------------------------------------------------------------------------


def compute_ece(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Bins predictions by confidence and measures the gap between average
    confidence and actual accuracy in each bin, weighted by bin size.

    Parameters
    ----------
    confidences:
        Model confidence for each prediction.
    correct:
        Whether each prediction was correct.
    n_bins:
        Number of equal-width bins between 0 and 1.

    Returns
    -------
    float
        ECE value in [0, 1].  Lower is better.
    """
    if not confidences:
        return 0.0

    conf_arr = np.array(confidences, dtype=np.float64)
    corr_arr = np.array(correct, dtype=np.float64)
    n = len(conf_arr)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr < hi)
        count = mask.sum()
        if count == 0:
            continue
        bin_accuracy = corr_arr[mask].mean()
        bin_confidence = conf_arr[mask].mean()
        ece += (count / n) * abs(float(bin_accuracy) - float(bin_confidence))

    return float(ece)


def compute_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_pauses(doc: IMLDocument) -> list[int]:
    """Extract pause durations from an IML document, in order of appearance."""
    pauses: list[int] = []
    for utt in doc.utterances:
        _collect_pauses(utt.children, pauses)
    return pauses


def _collect_pauses(children: tuple, pauses: list[int]) -> None:
    """Recursively collect pause durations from children."""
    for child in children:
        if isinstance(child, Pause):
            pauses.append(child.duration)
        elif isinstance(child, (Prosody,)):
            _collect_pauses(child.children, pauses)
        elif hasattr(child, "children"):
            _collect_pauses(child.children, pauses)


def _extract_pitch_contours(doc: IMLDocument) -> list[str]:
    """Extract pitch_contour labels from prosody tags in an IML document."""
    contours: list[str] = []
    for utt in doc.utterances:
        _collect_contours(utt.children, contours)
    return contours


def _collect_contours(children: tuple, contours: list[str]) -> None:
    """Recursively collect pitch_contour labels."""
    for child in children:
        if isinstance(child, Prosody) and child.pitch_contour:
            contours.append(child.pitch_contour)
            _collect_contours(child.children, contours)
        elif hasattr(child, "children"):
            _collect_contours(child.children, contours)


def _compute_pause_f1(
    predicted_pauses: list[int],
    truth_pauses: list[int],
    tolerance_ms: int = 200,
) -> float:
    """Compute F1 for pause detection using duration-based matching.

    Each predicted pause is matched to the nearest unmatched ground-truth
    pause within ``tolerance_ms``.  Unmatched predictions are false
    positives; unmatched truths are false negatives.
    """
    if not predicted_pauses and not truth_pauses:
        return 1.0  # No pauses expected, none predicted
    if not predicted_pauses or not truth_pauses:
        return 0.0  # One side has pauses, the other doesn't

    truth_remaining = list(truth_pauses)
    tp = 0
    for pred in predicted_pauses:
        best_idx = -1
        best_diff = float("inf")
        for i, truth in enumerate(truth_remaining):
            diff = abs(pred - truth)
            if diff <= tolerance_ms and diff < best_diff:
                best_diff = diff
                best_idx = i
        if best_idx >= 0:
            tp += 1
            truth_remaining.pop(best_idx)

    fp = len(predicted_pauses) - tp
    fn = len(truth_pauses) - tp
    return compute_f1_from_counts(tp, fp, fn)


def _per_class_f1(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, float]:
    """Compute per-class F1 scores."""
    labels = sorted(set(y_true) | set(y_pred))
    result: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        result[label] = compute_f1_from_counts(tp, fp, fn)
    return result


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class Benchmark:
    """Evaluation harness that benchmarks an AudioToIML converter.

    Parameters
    ----------
    dataset:
        Labelled dataset with ground-truth IML and emotion labels.
    converter:
        An AudioToIML instance (or any object with a ``convert()`` method).
    dataset_dir:
        Root directory of the dataset, used to resolve relative audio paths.
        If ``None``, audio-based conversion is skipped and only the
        ground-truth IML from dataset entries is evaluated.
    """

    def __init__(
        self,
        dataset: Dataset,
        converter: _Converter,
        dataset_dir: str | Path | None = None,
    ) -> None:
        self.dataset = dataset
        self.converter = converter
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self._parser = IMLParser()
        self._validator = IMLValidator()

    def run(self, max_samples: int | None = None) -> BenchmarkReport:
        """Run the benchmark and return a report.

        Parameters
        ----------
        max_samples:
            Limit evaluation to the first *max_samples* entries.
            Useful for quick CI checks on a subset.

        Returns
        -------
        BenchmarkReport
            Aggregated metrics across all processed entries.
        """
        start = time.time()

        entries = self.dataset.entries
        if max_samples is not None:
            entries = entries[:max_samples]

        true_emotions: list[str] = []
        pred_emotions: list[str] = []
        pred_confidences: list[float] = []
        correct_flags: list[bool] = []

        true_pitch_labels: list[str] = []
        pred_pitch_labels: list[str] = []

        all_pred_pauses: list[int] = []
        all_true_pauses: list[int] = []

        valid_count = 0
        processed = 0
        num_failures = 0

        for entry in entries:
            # Get predicted IML
            predicted_iml = self._get_predicted_iml(entry)
            if predicted_iml is None:
                num_failures += 1
                continue
            processed += 1

            # Validate generated IML
            result = self._validator.validate(predicted_iml)
            if result.valid:
                valid_count += 1

            # Parse predicted IML
            try:
                pred_doc = self._parser.parse(predicted_iml)
            except Exception:
                continue

            # Extract predicted emotion / confidence
            for utt in pred_doc.utterances:
                pred_emotion = utt.emotion or "neutral"
                pred_conf = utt.confidence if utt.confidence is not None else 0.5

                pred_emotions.append(pred_emotion)
                pred_confidences.append(pred_conf)
                true_emotions.append(entry.emotion_label)
                correct_flags.append(pred_emotion == entry.emotion_label)

            # Parse ground-truth IML once for both contour and pause comparison.
            try:
                true_doc = self._parser.parse(entry.iml)
            except Exception:
                true_doc = None

            # Pitch contour comparison
            pred_contours = _extract_pitch_contours(pred_doc)
            true_contours = _extract_pitch_contours(true_doc) if true_doc else []

            # Align contour lists (truncate to shorter)
            n_contours = min(len(pred_contours), len(true_contours))
            for i in range(n_contours):
                pred_pitch_labels.append(pred_contours[i])
                true_pitch_labels.append(true_contours[i])

            # Pause comparison
            pred_pauses = _extract_pauses(pred_doc)
            true_pauses = _extract_pauses(true_doc) if true_doc else []

            all_pred_pauses.extend(pred_pauses)
            all_true_pauses.extend(true_pauses)

        elapsed = time.time() - start

        # Compute aggregate metrics
        if true_emotions:
            emotion_accuracy = sum(correct_flags) / len(correct_flags)
            emotion_f1 = _per_class_f1(true_emotions, pred_emotions)
        else:
            emotion_accuracy = 0.0
            emotion_f1 = {}

        confidence_ece = compute_ece(pred_confidences, correct_flags)

        if true_pitch_labels:
            pitch_correct = sum(
                1 for t, p in zip(true_pitch_labels, pred_pitch_labels) if t == p
            )
            pitch_accuracy = pitch_correct / len(true_pitch_labels)
        else:
            pitch_accuracy = 0.0

        pause_f1 = _compute_pause_f1(all_pred_pauses, all_true_pauses)

        validity_rate = valid_count / processed if processed > 0 else 0.0

        return BenchmarkReport(
            emotion_accuracy=emotion_accuracy,
            emotion_f1=emotion_f1,
            confidence_ece=confidence_ece,
            pitch_accuracy=pitch_accuracy,
            pause_f1=pause_f1,
            validity_rate=validity_rate,
            num_samples=processed,
            num_failures=num_failures,
            duration_seconds=elapsed,
        )

    def _get_predicted_iml(self, entry: Any) -> str | None:
        """Run the converter on an entry's audio, or fall back to entry IML."""
        if self.dataset_dir is not None:
            audio_path = self.dataset_dir / entry.audio_file
            try:
                return self.converter.convert(audio_path)
            except Exception:
                logger.warning(
                    "Conversion failed for entry %s (%s): skipping",
                    getattr(entry, "id", "?"),
                    audio_path,
                    exc_info=True,
                )
                return None
        # If no dataset_dir, use the ground-truth IML as "predicted"
        # (useful for validating IML quality in the dataset itself)
        return entry.iml
