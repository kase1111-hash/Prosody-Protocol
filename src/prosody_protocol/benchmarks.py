"""Evaluation harness and benchmarking for prosody models.

Provides a Benchmark class that runs an AudioToIML converter against a
labelled dataset, computing emotion accuracy and F1, confidence calibration,
pitch contour and pause metrics, and IML validity.

Every dataset entry counts. An entry whose conversion fails -- the converter
raises, or returns something that is not parseable IML (an empty string, say)
-- is scored as a wrong emotion prediction, as missing every pause and pitch
contour of its ground truth, and as invalid IML. Metrics that had nothing to
compare (no ground-truth contours, no pauses on either side, no stated
confidences) are reported as ``None`` rather than as a score.
:meth:`BenchmarkReport.check_regression` fails on any conversion failure
unless a ``failure_rate`` threshold allows it.

Phase 12 deliverable from EXECUTION_GUIDE.md.
"""

from __future__ import annotations

import bisect
import difflib
import json
import logging
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without numpy
    raise ImportError(
        "Benchmark requires numpy. Install with: pip install 'prosody-protocol[audio]'"
    ) from exc

from .datasets import Dataset, DatasetEntry, resolve_audio_path
from .exceptions import DatasetError
from .models import ChildNode, IMLDocument, Pause, Prosody
from .parser import IMLParser
from .validator import IMLValidator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Converter protocol -- accepts AudioToIML or any compatible object
# ---------------------------------------------------------------------------


class _Converter(Protocol):
    def convert(self, audio_path: str | Path) -> str: ...


# Predicted label of an entry whose conversion failed or whose output could
# not be parsed. It never equals a ground-truth label and is not a class of
# its own in the per-class F1 scores.
_FAILED = "<failed>"

# Metrics check_regression() knows, by direction.
_HIGHER_IS_BETTER = (
    "emotion_accuracy",
    "emotion_f1_macro",
    "pitch_accuracy",
    "pitch_coverage",
    "pause_f1",
    "validity_rate",
)
_LOWER_IS_BETTER = ("confidence_ece", "failure_rate")
_METRICS = _HIGHER_IS_BETTER + _LOWER_IS_BETTER


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """Results of a benchmark run against a dataset.

    Rates are over all evaluated entries, failed conversions included.

    Attributes
    ----------
    emotion_accuracy:
        Fraction of entries whose predicted emotion equals the label.
    emotion_f1:
        Per-class F1 scores keyed by emotion label (labels that occur in the
        ground truth or the predictions).
    confidence_ece:
        Expected Calibration Error of the stated emotion confidences, or
        ``None`` when no prediction stated one.
    pitch_accuracy:
        Fraction of compared ground-truth ``pitch_contour`` spans whose
        predicted contour on the same words matches, or ``None`` when no
        span could be compared. Read it together with ``pitch_coverage``.
    pause_f1:
        F1 of predicted against ground-truth pauses, matched one-to-one
        within each entry by position and duration; ``None`` when neither
        side has any pause.
    validity_rate:
        Fraction of entries whose converter output passes IML validation.
    num_samples:
        Number of entries the converter produced parseable IML for (it may
        still fail validation; see ``validity_rate``).
    num_failures:
        Number of entries without such output: the conversion raised,
        returned something other than a string or text that does not parse
        as IML, or the entry's audio path was outside the dataset.
    duration_seconds:
        Wall-clock time taken for the benchmark run.
    pitch_coverage:
        Fraction of ground-truth ``pitch_contour`` spans for which the
        prediction had a contour on the same words, or ``None`` when the
        ground truth has no contours.
    """

    emotion_accuracy: float
    emotion_f1: dict[str, float]
    confidence_ece: float | None
    pitch_accuracy: float | None
    pause_f1: float | None
    validity_rate: float
    num_samples: int
    num_failures: int
    duration_seconds: float
    pitch_coverage: float | None = None

    @property
    def num_entries(self) -> int:
        """Number of dataset entries evaluated."""
        return self.num_samples + self.num_failures

    @property
    def failure_rate(self) -> float:
        """Fraction of evaluated entries whose conversion failed."""
        return self.num_failures / self.num_entries if self.num_entries else 0.0

    @property
    def emotion_f1_macro(self) -> float:
        """Unweighted mean of the per-class F1 scores (0.0 when there are none)."""
        return statistics.fmean(self.emotion_f1.values()) if self.emotion_f1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary.

        Unmeasured metrics are ``None`` (JSON ``null``).
        """
        return {
            "emotion_accuracy": round(self.emotion_accuracy, 4),
            "emotion_f1": {k: round(v, 4) for k, v in self.emotion_f1.items()},
            "emotion_f1_macro": round(self.emotion_f1_macro, 4),
            "confidence_ece": _round_optional(self.confidence_ece),
            "pitch_accuracy": _round_optional(self.pitch_accuracy),
            "pitch_coverage": _round_optional(self.pitch_coverage),
            "pause_f1": _round_optional(self.pause_f1),
            "validity_rate": round(self.validity_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
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
        """Load a previously saved benchmark report from JSON.

        Derived values (``emotion_f1_macro``, ``failure_rate``) are
        recomputed from the stored fields.
        """
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
            pitch_coverage=data.get("pitch_coverage"),
        )

    def check_regression(
        self,
        baseline: BenchmarkReport | None = None,
        thresholds: dict[str, float] | None = None,
        *,
        tolerance: float = 0.01,
    ) -> list[str]:
        """Check whether metrics regress against a baseline or thresholds.

        A run that evaluated no entries always fails.

        Parameters
        ----------
        baseline:
            A previous report to compare against. A failure is produced for
            every metric (including each per-class F1) that is worse than
            the baseline by more than *tolerance*. Metrics that either run
            could not measure are not compared; ``pitch_coverage`` shows a
            converter that stopped predicting contours.
        thresholds:
            Limits keyed by metric name: minimums for ``emotion_accuracy``,
            ``emotion_f1_macro``, ``pitch_accuracy``, ``pitch_coverage``,
            ``pause_f1`` and ``validity_rate``; maximums for
            ``confidence_ece`` and ``failure_rate``. For example:
            ``{"emotion_accuracy": 0.75, "confidence_ece": 0.1}``. A metric
            with a threshold that this run could not measure fails.
            ``failure_rate`` defaults to 0.0, so any failed conversion fails
            the check unless a higher limit is given.
        tolerance:
            Allowed slack when comparing against *baseline*.

        Returns
        -------
        list[str]
            List of failure messages.  Empty means all checks passed.

        Raises
        ------
        ValueError
            If a threshold names an unknown metric or is not a finite number.
        """
        limits = dict(thresholds or {})
        unknown = sorted(set(limits) - set(_METRICS))
        if unknown:
            raise ValueError(
                f"Unknown threshold metric(s) {unknown}; expected any of {sorted(_METRICS)}"
            )
        for metric, limit in limits.items():
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(
                limit
            ):
                raise ValueError(f"Threshold for {metric} must be a finite number, got {limit!r}")
        limits.setdefault("failure_rate", 0.0)

        failures: list[str] = []
        if self.num_entries == 0:
            failures.append("no dataset entries were evaluated")

        for metric, limit in limits.items():
            value = self._metric(metric)
            if value is None:
                failures.append(f"{metric} was not measured (threshold {limit:.4f})")
            elif metric in _LOWER_IS_BETTER and value > limit:
                failures.append(f"{metric} = {value:.4f} > threshold {limit:.4f}")
            elif metric in _HIGHER_IS_BETTER and value < limit:
                failures.append(f"{metric} = {value:.4f} < threshold {limit:.4f}")

        if baseline is not None:
            for metric in _METRICS:
                current, previous = self._metric(metric), baseline._metric(metric)
                if current is None or previous is None:
                    continue
                if metric in _HIGHER_IS_BETTER and current < previous - tolerance:
                    failures.append(
                        f"{metric} regressed: {current:.4f} < baseline {previous:.4f}"
                    )
                elif metric in _LOWER_IS_BETTER and current > previous + tolerance:
                    failures.append(
                        f"{metric} regressed: {current:.4f} > baseline {previous:.4f}"
                    )
            for label, previous_f1 in sorted(baseline.emotion_f1.items()):
                current_f1 = self.emotion_f1.get(label)
                if current_f1 is not None and current_f1 < previous_f1 - tolerance:
                    failures.append(
                        f"emotion_f1[{label}] regressed: {current_f1:.4f} < "
                        f"baseline {previous_f1:.4f}"
                    )

        return failures

    def _metric(self, name: str) -> float | None:
        value: float | None = getattr(self, name)
        return value


def _round_optional(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


# ---------------------------------------------------------------------------
# Metric helper functions
# ---------------------------------------------------------------------------


def compute_ece(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Bins predictions by confidence and measures the gap between average
    confidence and actual accuracy in each bin, weighted by bin size.

    Parameters
    ----------
    confidences:
        Model confidence for each prediction. Values outside [0, 1] are
        clipped to it (1.5 counts as fully confident).
    correct:
        Whether each prediction was correct.
    n_bins:
        Number of equal-width bins between 0 and 1.

    Returns
    -------
    float
        ECE value in [0, 1].  Lower is better.

    Raises
    ------
    ValueError
        If the inputs differ in length, a confidence is not finite, or
        *n_bins* is less than 1.
    """
    if len(confidences) != len(correct):
        raise ValueError(
            f"Got {len(confidences)} confidences but {len(correct)} correctness flags"
        )
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if not confidences:
        return 0.0

    conf_arr = np.array(confidences, dtype=np.float64)
    if not np.all(np.isfinite(conf_arr)):
        raise ValueError("Confidences must be finite numbers")
    conf_arr = np.clip(conf_arr, 0.0, 1.0)
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
    return [duration for _, duration in _annotate(doc).pauses]


def _extract_pitch_contours(doc: IMLDocument) -> list[str]:
    """Extract pitch_contour labels from prosody tags in an IML document."""
    contours: list[str] = []
    for utt in doc.utterances:
        _collect_contours(utt.children, contours)
    return contours


def _collect_contours(children: tuple[ChildNode, ...], contours: list[str]) -> None:
    """Recursively collect pitch_contour labels, outermost first."""
    for child in children:
        if isinstance(child, Prosody) and child.pitch_contour:
            contours.append(child.pitch_contour)
        if not isinstance(child, (str, Pause)):
            _collect_contours(child.children, contours)


def _max_matching(candidates: list[list[int]], n_right: int) -> int:
    """Size of a maximum one-to-one matching (Kuhn's augmenting paths).

    ``candidates[i]`` lists the right-hand items left item *i* may pair with.
    """
    owner = [-1] * n_right

    def augment(left: int, visited: set[int]) -> bool:
        for right in candidates[left]:
            if right in visited:
                continue
            visited.add(right)
            if owner[right] == -1 or augment(owner[right], visited):
                owner[right] = left
                return True
        return False

    return sum(1 for left in range(len(candidates)) if augment(left, set()))


def _compute_pause_f1(
    predicted_pauses: list[int],
    truth_pauses: list[int],
    tolerance_ms: int = 200,
) -> float:
    """Compute F1 for pause detection using duration-based matching.

    Predicted and ground-truth pauses are paired one-to-one when their
    durations differ by at most ``tolerance_ms``, maximising the number of
    pairs. Unmatched predictions are false positives; unmatched truths are
    false negatives. Returns 1.0 when neither side has a pause.
    """
    if not predicted_pauses and not truth_pauses:
        return 1.0  # No pauses expected, none predicted
    candidates = [
        [j for j, truth in enumerate(truth_pauses) if abs(pred - truth) <= tolerance_ms]
        for pred in predicted_pauses
    ]
    tp = _max_matching(candidates, len(truth_pauses))
    return compute_f1_from_counts(tp, len(predicted_pauses) - tp, len(truth_pauses) - tp)


def _per_class_f1(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, float]:
    """Compute per-class F1 scores.

    Classes are the labels in *y_true* and *y_pred*, except the marker for
    failed predictions (which still counts as a miss for the true class).
    """
    labels = sorted((set(y_true) | set(y_pred)) - {_FAILED})
    result: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)
        result[label] = compute_f1_from_counts(tp, fp, fn)
    return result


# ---------------------------------------------------------------------------
# Aligning a prediction with its ground truth
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+(?:['’]\w+)*")


@dataclass
class _Annotation:
    """The words of an IML document and where its pauses and contours sit."""

    words: list[str] = field(default_factory=list)
    # (number of words before the pause, duration in ms)
    pauses: list[tuple[int, int]] = field(default_factory=list)
    # (index of the first word, index after the last word, label)
    contours: list[tuple[int, int, str]] = field(default_factory=list)


def _annotate(doc: IMLDocument | None) -> _Annotation:
    annotation = _Annotation()
    if doc is not None:
        for utt in doc.utterances:
            _walk(utt.children, annotation)
    return annotation


def _walk(children: tuple[ChildNode, ...], annotation: _Annotation) -> None:
    for child in children:
        if isinstance(child, str):
            annotation.words.extend(_WORD_RE.findall(child.lower()))
        elif isinstance(child, Pause):
            annotation.pauses.append((len(annotation.words), child.duration))
        else:
            start = len(annotation.words)
            _walk(child.children, annotation)
            if isinstance(child, Prosody) and child.pitch_contour:
                annotation.contours.append((start, len(annotation.words), child.pitch_contour))


def _boundary_map(truth_words: list[str], pred_words: list[str]) -> list[float]:
    """For each word boundary in the ground truth, the matching one in the prediction.

    Boundary *b* is the point after the first *b* words. Boundaries in runs of
    words both transcripts share map exactly; the others are interpolated
    between the nearest shared ones, so a pause or contour keeps its place
    when the predicted transcript differs in a few words.
    """
    n = len(truth_words)
    anchors: dict[int, float] = {0: 0.0, n: float(len(pred_words))}
    matcher = difflib.SequenceMatcher(a=truth_words, b=pred_words, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size + 1):
            anchors[block.a + k] = float(block.b + k)
    known = sorted(anchors)
    mapped: list[float] = []
    for b in range(n + 1):
        if b in anchors:
            mapped.append(anchors[b])
            continue
        i = bisect.bisect(known, b)
        before, after = known[i - 1], known[i]
        share = (b - before) / (after - before)
        mapped.append(anchors[before] + share * (anchors[after] - anchors[before]))
    return mapped


def _pause_counts(
    truth: _Annotation,
    pred: _Annotation,
    boundaries: list[float],
    tolerance_ms: int,
    position_tolerance: float,
) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) for one entry's pauses."""
    candidates = [
        [
            j
            for j, (pred_pos, pred_ms) in enumerate(pred.pauses)
            if abs(boundaries[truth_pos] - pred_pos) <= position_tolerance
            and abs(truth_ms - pred_ms) <= tolerance_ms
        ]
        for truth_pos, truth_ms in truth.pauses
    ]
    tp = _max_matching(candidates, len(pred.pauses))
    return tp, len(pred.pauses) - tp, len(truth.pauses) - tp


def _contour_counts(
    truth: _Annotation, pred: _Annotation, boundaries: list[float]
) -> tuple[int, int, int]:
    """(ground-truth contour spans, spans compared, spans correct) for one entry.

    A ground-truth span is compared when the prediction has a contour on at
    least one of its words; the predicted label is the most common one over
    those words (the innermost contour wins on each word).
    """
    labels: list[str | None] = [None] * len(pred.words)
    for start, end, label in sorted(pred.contours, key=lambda c: c[1] - c[0], reverse=True):
        labels[start:end] = [label] * (end - start)

    total = compared = correct = 0
    for start, end, label in truth.contours:
        if end <= start:
            continue
        total += 1
        lo = math.floor(boundaries[start] + 0.5)
        hi = math.floor(boundaries[end] + 0.5)
        observed = [x for x in labels[lo:hi] if x is not None]
        if not observed:
            continue
        compared += 1
        if Counter(observed).most_common(1)[0][0] == label:
            correct += 1
    return total, compared, correct


def _predicted_emotion(doc: IMLDocument) -> tuple[str, float | None]:
    """The entry-level emotion prediction of a converted document.

    One prediction per entry: the emotion of the utterance with the highest
    stated confidence among those that carry an emotion, or ``"neutral"``
    when none does (spec 3.1: no marked emotional state). The confidence is
    that utterance's, or ``None`` when it states no finite one.
    """
    best: tuple[str, float | None] | None = None
    best_rank = -math.inf
    for utt in doc.utterances:
        if not utt.emotion:
            continue
        confidence = utt.confidence
        if confidence is not None and not math.isfinite(confidence):
            confidence = None
        rank = -1.0 if confidence is None else confidence
        if best is None or rank > best_rank:
            best, best_rank = (utt.emotion, confidence), rank
    return best if best is not None else ("neutral", None)


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
        An AudioToIML instance (or any object with a ``convert()`` method
        that takes an audio path and returns an IML string).
    dataset_dir:
        Root directory of the dataset, used to resolve relative audio paths.
        Defaults to ``dataset.root``, which :meth:`DatasetLoader.load` sets;
        one of the two is required.
    pause_tolerance_ms:
        Largest duration difference (ms) at which a predicted pause matches
        a ground-truth one.
    pause_position_tolerance:
        Largest distance, in words, between a predicted pause and a
        ground-truth one that it matches. Positions are compared after
        aligning the predicted transcript with the ground truth.

    Raises
    ------
    ValueError
        If neither *dataset_dir* nor ``dataset.root`` is set.
    """

    def __init__(
        self,
        dataset: Dataset,
        converter: _Converter,
        dataset_dir: str | Path | None = None,
        *,
        pause_tolerance_ms: int = 200,
        pause_position_tolerance: float = 1.0,
    ) -> None:
        root = dataset_dir if dataset_dir is not None else dataset.root
        if root is None:
            raise ValueError(
                "Benchmark needs the dataset directory to find the audio: pass dataset_dir, "
                "or load the dataset with DatasetLoader.load(), which records it"
            )
        self.dataset = dataset
        self.converter = converter
        self.dataset_dir = Path(root)
        self.pause_tolerance_ms = pause_tolerance_ms
        self.pause_position_tolerance = pause_position_tolerance
        self._parser = IMLParser()
        self._validator = IMLValidator()

    def run(self, max_samples: int | None = None) -> BenchmarkReport:
        """Run the converter on every entry and return a report.

        Parameters
        ----------
        max_samples:
            Limit evaluation to the first *max_samples* entries.
            Useful for quick CI checks on a subset.

        Returns
        -------
        BenchmarkReport
            Aggregated metrics across all evaluated entries.
        """
        if max_samples is not None and max_samples < 0:
            raise ValueError(f"max_samples must not be negative, got {max_samples}")
        start = time.time()

        entries = self.dataset.entries
        if max_samples is not None:
            entries = entries[:max_samples]

        true_emotions: list[str] = []
        pred_emotions: list[str] = []
        confidences: list[float] = []
        confident_correct: list[bool] = []
        pause_tp = pause_fp = pause_fn = 0
        contours_total = contours_compared = contours_correct = 0
        valid_count = processed = num_failures = 0

        for entry in entries:
            predicted_iml = self._get_predicted_iml(entry)
            pred_doc: IMLDocument | None = None
            if predicted_iml is not None:
                if self._validator.validate(predicted_iml).valid:
                    valid_count += 1
                try:
                    pred_doc = self._parser.parse(predicted_iml)
                except Exception:
                    # A converter that swallows its errors and returns "" or
                    # other non-IML text has failed as much as one that raises.
                    logger.warning("Output for entry %s cannot be parsed as IML", entry.id)
            if pred_doc is None:
                num_failures += 1
            else:
                processed += 1

            emotion, confidence = (
                _predicted_emotion(pred_doc) if pred_doc is not None else (_FAILED, None)
            )
            true_emotions.append(entry.emotion_label)
            pred_emotions.append(emotion)
            if confidence is not None:
                confidences.append(confidence)
                confident_correct.append(emotion == entry.emotion_label)

            truth = _annotate(self._parse_truth(entry))
            pred = _annotate(pred_doc)
            boundaries = _boundary_map(truth.words, pred.words)
            tp, fp, fn = _pause_counts(
                truth, pred, boundaries, self.pause_tolerance_ms, self.pause_position_tolerance
            )
            pause_tp, pause_fp, pause_fn = pause_tp + tp, pause_fp + fp, pause_fn + fn
            total, compared, correct = _contour_counts(truth, pred, boundaries)
            contours_total += total
            contours_compared += compared
            contours_correct += correct

        elapsed = time.time() - start

        n = len(entries)
        correct_count = sum(1 for t, p in zip(true_emotions, pred_emotions, strict=True) if t == p)
        return BenchmarkReport(
            emotion_accuracy=correct_count / n if n else 0.0,
            emotion_f1=_per_class_f1(true_emotions, pred_emotions),
            confidence_ece=compute_ece(confidences, confident_correct) if confidences else None,
            pitch_accuracy=(
                contours_correct / contours_compared if contours_compared else None
            ),
            pause_f1=(
                compute_f1_from_counts(pause_tp, pause_fp, pause_fn)
                if pause_tp + pause_fp + pause_fn
                else None
            ),
            validity_rate=valid_count / n if n else 0.0,
            num_samples=processed,
            num_failures=num_failures,
            duration_seconds=elapsed,
            pitch_coverage=contours_compared / contours_total if contours_total else None,
        )

    def _parse_truth(self, entry: DatasetEntry) -> IMLDocument | None:
        try:
            return self._parser.parse(entry.iml)
        except Exception:
            logger.warning("Ground-truth IML of entry %s cannot be parsed", entry.id)
            return None

    def _get_predicted_iml(self, entry: DatasetEntry) -> str | None:
        """Run the converter on an entry's audio; ``None`` when that fails."""
        try:
            audio_path = resolve_audio_path(self.dataset_dir, entry.audio_file)
        except DatasetError as exc:
            logger.warning("Entry %s not converted: %s", entry.id, exc)
            return None
        try:
            result = self.converter.convert(audio_path)
        except Exception:
            logger.warning(
                "Conversion failed for entry %s (%s)",
                entry.id,
                audio_path,
                exc_info=True,
            )
            return None
        if not isinstance(result, str):
            logger.warning(
                "Converter returned %s instead of an IML string for entry %s",
                type(result).__name__,
                entry.id,
            )
            return None
        return result
