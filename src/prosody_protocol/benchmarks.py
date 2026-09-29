"""Evaluation harness and benchmarking for prosody models.

Provides a Benchmark class that runs an AudioToIML converter against a
labelled dataset and reports:

- emotion: accuracy over the entries whose output carries an emotion,
  coverage (the share of entries whose output carries one), per-class F1
  and the calibration (ECE) of the stated confidences;
- pitch contours: accuracy over the ground-truth contours the output has a
  contour for on the same words, and coverage;
- pauses: F1 of predicted against ground-truth pauses, matched per entry by
  position and duration;
- IML validity and the failure rate.

There is no round-trip (synthesis and re-analysis) fidelity metric.

The converter gets each entry's words when it accepts them: word timings
from the entry's ``metadata["word_timings"]`` (any format
:func:`~prosody_protocol.alignment.parse_word_timings` reads) as
``words=``, else its transcript as ``transcript=``; see ``words_from``.

Every dataset entry counts. An entry whose conversion fails -- the converter
raises, or returns something that is not parseable IML (an empty string, say)
-- carries no emotion (lowering emotion coverage and counting as a miss in
the per-class F1), misses every pause and pitch contour of its ground truth,
and is invalid IML. An output without an emotion is an abstention: it lowers
emotion coverage, not accuracy (unless ``abstention_label`` names the label
it stands for). An output whose text is only ``[speech]`` placeholders (no
speech recognition) has no words to align pauses and contours with, so it is
left out of those metrics. Metrics that had nothing to compare (no
ground-truth contours, no pauses on either side, no stated emotion or
confidence) are reported as ``None`` rather than as a score.
:meth:`BenchmarkReport.check_regression` fails on any conversion failure
unless a ``failure_rate`` threshold allows it.
"""

from __future__ import annotations

import bisect
import difflib
import importlib.util
import inspect
import json
import logging
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ._install import install_hint

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without numpy
    raise ImportError(
        "Benchmark requires numpy. Install with: " + install_hint("audio")
    ) from exc

from ._types import WordAlignment
from .alignment import parse_word_timings
from .datasets import Dataset, DatasetEntry, resolve_audio_path
from .exceptions import ConversionError, DatasetError
from .models import ChildNode, IMLDocument, Pause, Prosody
from .parser import IMLParser
from .validator import IMLValidator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Converter protocol -- accepts AudioToIML or any compatible object
# ---------------------------------------------------------------------------


class _Converter(Protocol):
    """``convert(audio_path)`` returning IML; it may also accept the keyword
    arguments ``words`` and/or ``transcript``, as AudioToIML does."""

    def convert(self, audio_path: str | Path) -> str: ...


WordsFrom = Literal["auto", "timings", "transcript", "stt"]
_WORDS_FROM: tuple[str, ...] = ("auto", "timings", "transcript", "stt")

#: Key of a dataset entry's ``metadata`` that holds its word timings, in any
#: format :func:`~prosody_protocol.alignment.parse_word_timings` reads.
WORD_TIMINGS_KEY = "word_timings"

# What AudioToIML writes for each stretch of speech when it has no words
# (audio_to_iml.PLACEHOLDER_TOKEN; not imported, as that needs parselmouth).
_PLACEHOLDER = "[speech]"

# Predicted labels of an entry whose conversion failed or whose output could
# not be parsed, and of an output without an emotion. Neither equals a
# ground-truth label or is a class of its own in the per-class F1 scores.
_FAILED = "<failed>"
_ABSTAINED = "<none>"

# Metrics check_regression() knows, by direction.
_HIGHER_IS_BETTER = (
    "emotion_accuracy",
    "emotion_coverage",
    "emotion_f1_macro",
    "pitch_accuracy",
    "pitch_coverage",
    "pause_f1",
    "validity_rate",
)
_LOWER_IS_BETTER = ("confidence_ece", "failure_rate")
_METRICS = _HIGHER_IS_BETTER + _LOWER_IS_BETTER
# Metrics whose value depends on how outputs without an emotion are scored.
_ABSTENTION_DEPENDENT = ("emotion_accuracy", "emotion_coverage", "emotion_f1_macro")

# An accuracy measured only where the output answered, its coverage metric,
# and how a threshold on it is checked when no coverage threshold is given.
_COVERAGE_OF = {
    "emotion_accuracy": (
        "emotion_coverage",
        "over all entries: those without an emotion count as wrong; add an "
        "emotion_coverage threshold to score only the entries with one",
    ),
    "pitch_accuracy": (
        "pitch_coverage",
        "over all ground-truth contours: those without a predicted contour count as "
        "wrong; add a pitch_coverage threshold to score only the compared ones",
    ),
}

# How each entry's words were given to the converter (BenchmarkReport.word_sources).
_WORD_SOURCES = ("timings", "transcript", "stt")

# Emotion scoring of reports saved before emotion coverage existed: an
# output without an emotion counted as "neutral".
_LEGACY_ABSTENTION_LABEL = "neutral"


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """Results of a benchmark run against a dataset.

    Rates are over all evaluated entries, failed conversions included,
    unless stated otherwise.

    Attributes
    ----------
    emotion_accuracy:
        Fraction of the entries whose output carries an emotion (see
        ``emotion_coverage``) where that emotion equals the label, or
        ``None`` when no output carries one.
    emotion_f1:
        Per-class F1 scores keyed by emotion label (labels that occur in the
        ground truth or the predictions), over all entries: an entry without
        a predicted emotion (an abstention or a failed conversion) is a
        miss for its label.
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
        as IML, the entry's audio path was outside the dataset, or its word
        timings were invalid.
    duration_seconds:
        Wall-clock time taken for the benchmark run.
    pitch_coverage:
        Fraction of ground-truth ``pitch_contour`` spans for which the
        prediction had a contour on the same words, or ``None`` when the
        ground truth has no contours.
    emotion_coverage:
        Fraction of entries whose output carries an emotion (abstentions and
        failed conversions lower it), or ``None`` when no entry was
        evaluated (and in reports made before it existed).
    num_unaligned:
        Number of outputs whose text was only ``[speech]`` placeholders, and
        so had no words to place pauses and contours by; they are left out
        of ``pause_f1``, ``pitch_accuracy`` and ``pitch_coverage``.
    word_sources:
        How many entries the converter was given word timings
        (``"timings"``), the transcript (``"transcript"``), or neither
        (``"stt"``: the converter's own speech recognition).
    abstention_label:
        The label an output without an emotion was scored as, or ``None``
        when it counted as an abstention (see :class:`Benchmark`).
    """

    emotion_accuracy: float | None
    emotion_f1: dict[str, float]
    confidence_ece: float | None
    pitch_accuracy: float | None
    pause_f1: float | None
    validity_rate: float
    num_samples: int
    num_failures: int
    duration_seconds: float
    pitch_coverage: float | None = None
    emotion_coverage: float | None = None
    num_unaligned: int = 0
    word_sources: dict[str, int] = field(default_factory=dict)
    abstention_label: str | None = None

    @property
    def num_entries(self) -> int:
        """Number of dataset entries evaluated."""
        return self.num_samples + self.num_failures

    @property
    def failure_rate(self) -> float:
        """Fraction of evaluated entries whose conversion failed."""
        return self.num_failures / self.num_entries if self.num_entries else 0.0

    @property
    def emotion_f1_macro(self) -> float | None:
        """Unweighted mean of the per-class F1 scores (``None`` when there are none)."""
        return statistics.fmean(self.emotion_f1.values()) if self.emotion_f1 else None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary.

        Unmeasured metrics are ``None`` (JSON ``null``).
        """
        return {
            "emotion_accuracy": _round_optional(self.emotion_accuracy),
            "emotion_coverage": _round_optional(self.emotion_coverage),
            "emotion_f1": {k: round(v, 4) for k, v in self.emotion_f1.items()},
            "emotion_f1_macro": _round_optional(self.emotion_f1_macro),
            "confidence_ece": _round_optional(self.confidence_ece),
            "pitch_accuracy": _round_optional(self.pitch_accuracy),
            "pitch_coverage": _round_optional(self.pitch_coverage),
            "pause_f1": _round_optional(self.pause_f1),
            "validity_rate": round(self.validity_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "num_samples": self.num_samples,
            "num_failures": self.num_failures,
            "num_unaligned": self.num_unaligned,
            "word_sources": dict(self.word_sources),
            "abstention_label": self.abstention_label,
            "duration_seconds": round(self.duration_seconds, 3),
        }

    def save(self, path: str | Path) -> None:
        """Save report as JSON for tracking over time."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkReport:
        """Load a previously saved benchmark report from JSON.

        Derived values (``emotion_f1_macro``, ``failure_rate``) are
        recomputed from the stored fields. A report saved before
        ``abstention_label`` existed scored outputs without an emotion as
        ``"neutral"``, and loads with that ``abstention_label``.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
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
            emotion_coverage=data.get("emotion_coverage"),
            num_unaligned=data.get("num_unaligned", 0),
            word_sources=dict(data.get("word_sources", {})),
            abstention_label=data.get("abstention_label", _LEGACY_ABSTENTION_LABEL),
        )

    def check_regression(
        self,
        baseline: BenchmarkReport | None = None,
        thresholds: dict[str, float] | None = None,
        *,
        tolerance: float = 0.01,
        class_tolerance: float | None = None,
    ) -> list[str]:
        """Check whether metrics regress against a baseline or thresholds.

        A run that evaluated no entries always fails.

        Parameters
        ----------
        baseline:
            A previous report to compare against. A failure is produced for
            every metric (including each per-class F1) that is worse than
            the baseline by more than *tolerance*, and for every
            higher-is-better metric the baseline measured (above
            *tolerance*) but this run could not, such as ``pause_f1`` once
            every output is placeholders. Metrics the baseline could not
            measure are not compared; the coverage metrics show a converter
            that stopped predicting emotions or contours. When the two runs
            scored outputs without an emotion differently (see
            ``abstention_label``; reports saved before it existed counted
            them as ``"neutral"``), that is a failure and the emotion
            accuracy, coverage and F1 scores are not compared.
        thresholds:
            Limits keyed by metric name: minimums for ``emotion_accuracy``,
            ``emotion_coverage``, ``emotion_f1_macro``, ``pitch_accuracy``,
            ``pitch_coverage``, ``pause_f1`` and ``validity_rate``;
            maximums for ``confidence_ece`` and ``failure_rate``. For
            example: ``{"emotion_accuracy": 0.75, "confidence_ece": 0.1}``.
            A metric with a threshold that this run could not measure
            fails. ``failure_rate`` defaults to 0.0, so any failed
            conversion fails the check unless a higher limit is given. An
            ``emotion_accuracy`` (``pitch_accuracy``) threshold without an
            ``emotion_coverage`` (``pitch_coverage``) threshold is checked
            against the accuracy over all entries (ground-truth contours),
            counting those without a prediction as wrong, so that a
            converter cannot pass it by abstaining; give a coverage
            threshold (0 allows any) to check the accuracy over the
            predictions alone.
        tolerance:
            Allowed slack when comparing against *baseline*.
        class_tolerance:
            Allowed slack for each per-class F1 against *baseline*
            (default: *tolerance*). On a small dataset one entry moves the
            F1 of a rare class far more than the overall metrics.

        Returns
        -------
        list[str]
            List of failure messages.  Empty means all checks passed.

        Raises
        ------
        ValueError
            If a threshold names an unknown metric or is not a finite number,
            or a tolerance is negative or not a finite number.
        """
        for name, slack in (("tolerance", tolerance), ("class_tolerance", class_tolerance)):
            if slack is not None and (
                isinstance(slack, bool)
                or not isinstance(slack, (int, float))
                or not math.isfinite(slack)
                or slack < 0
            ):
                raise ValueError(f"{name} must be a non-negative number, got {slack!r}")
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
            note = ""
            if metric in _COVERAGE_OF:
                coverage_metric, explanation = _COVERAGE_OF[metric]
                coverage = self._metric(coverage_metric)
                if coverage_metric not in limits and coverage is not None:
                    # accuracy * coverage: correct predictions over everything.
                    value = (value or 0.0) * coverage
                    note = f" ({explanation})"
            if value is None:
                failures.append(f"{metric} was not measured (threshold {limit:.4f})")
            elif metric in _LOWER_IS_BETTER and value > limit:
                failures.append(f"{metric} = {value:.4f} > threshold {limit:.4f}{note}")
            elif metric in _HIGHER_IS_BETTER and value < limit:
                failures.append(f"{metric} = {value:.4f} < threshold {limit:.4f}{note}")

        if baseline is not None:
            failures.extend(self._compare(
                baseline, tolerance, tolerance if class_tolerance is None else class_tolerance
            ))
        return failures

    def _compare(
        self, baseline: BenchmarkReport, tolerance: float, class_tolerance: float
    ) -> list[str]:
        """The failures of this report against *baseline*."""
        failures: list[str] = []
        comparable = baseline.abstention_label == self.abstention_label
        if not comparable:
            failures.append(
                "the baseline scored outputs without an emotion as "
                f"{_describe_abstention(baseline.abstention_label)} and this run as "
                f"{_describe_abstention(self.abstention_label)}, so their emotion accuracy, "
                "coverage and F1 are not compared; save a new baseline made the same way"
            )
        for metric in _METRICS:
            if not comparable and metric in _ABSTENTION_DEPENDENT:
                continue
            current, previous = self._metric(metric), baseline._metric(metric)
            if previous is None:
                continue
            if current is None:
                if metric in _HIGHER_IS_BETTER and previous > tolerance:
                    failures.append(f"{metric} was not measured (baseline {previous:.4f})")
            elif metric in _HIGHER_IS_BETTER and current < previous - tolerance:
                failures.append(f"{metric} regressed: {current:.4f} < baseline {previous:.4f}")
            elif metric in _LOWER_IS_BETTER and current > previous + tolerance:
                failures.append(f"{metric} regressed: {current:.4f} > baseline {previous:.4f}")
        if comparable:
            for label, previous_f1 in sorted(baseline.emotion_f1.items()):
                current_f1 = self.emotion_f1.get(label)
                if current_f1 is not None and current_f1 < previous_f1 - class_tolerance:
                    failures.append(
                        f"emotion_f1[{label}] regressed: {current_f1:.4f} < "
                        f"baseline {previous_f1:.4f}"
                    )
        return failures

    def _metric(self, name: str) -> float | None:
        value: float | None = getattr(self, name)
        return value


def _describe_abstention(label: str | None) -> str:
    return "abstentions" if label is None else repr(label)


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
    """Size of a maximum one-to-one matching (Hopcroft-Karp).

    ``candidates[i]`` lists the right-hand items left item *i* may pair
    with, preferred ones first. The search is iterative, so an entry with
    thousands of mutually matchable pauses (a long recording) cannot exhaust
    the interpreter's recursion limit; it takes O(E * sqrt(V)) steps.
    """
    n_left = len(candidates)
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    # Greedy start: most pairs are found here, each with its preferred item.
    for left, options in enumerate(candidates):
        for right in options:
            if match_right[right] == -1:
                match_left[left], match_right[right] = right, left
                break

    while True:
        # Breadth-first layers of left items, from the unmatched ones.
        layer = [-1] * n_left
        queue = [left for left in range(n_left) if match_left[left] == -1]
        for left in queue:
            layer[left] = 0
        found_free = False
        for left in queue:  # the queue grows while it is read
            for right in candidates[left]:
                owner = match_right[right]
                if owner == -1:
                    found_free = True
                elif layer[owner] == -1:
                    layer[owner] = layer[left] + 1
                    queue.append(owner)
        if not found_free:
            return sum(1 for right in match_left if right != -1)

        # Depth-first along the layers, with an explicit stack, for a
        # maximal set of vertex-disjoint shortest augmenting paths.
        next_option = [0] * n_left
        for root in range(n_left):
            if match_left[root] != -1:
                continue
            path = [root]  # left items on the path
            via: list[int] = []  # right items linking path[k] to path[k + 1]
            while path:
                left = path[-1]
                options = candidates[left]
                advanced = False
                while next_option[left] < len(options):
                    right = options[next_option[left]]
                    next_option[left] += 1
                    owner = match_right[right]
                    if owner == -1:
                        # Augment: each left item on the path takes the next right item.
                        for k, item in enumerate(path):
                            paired = via[k] if k < len(via) else right
                            match_left[item], match_right[paired] = paired, item
                        path = []
                        advanced = True
                        break
                    if layer[owner] == layer[left] + 1:
                        path.append(owner)
                        via.append(right)
                        advanced = True
                        break
                if not advanced:
                    layer[left] = -2  # a dead end for the rest of this phase
                    path.pop()
                    if via:
                        via.pop()


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
    # With one dimension, pairing the sorted durations in order is optimal.
    predicted, truth = sorted(predicted_pauses), sorted(truth_pauses)
    tp = i = j = 0
    while i < len(predicted) and j < len(truth):
        if abs(predicted[i] - truth[j]) <= tolerance_ms:
            tp, i, j = tp + 1, i + 1, j + 1
        elif predicted[i] < truth[j]:
            i += 1
        else:
            j += 1
    return compute_f1_from_counts(tp, len(predicted_pauses) - tp, len(truth_pauses) - tp)


def _per_class_f1(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, float]:
    """Compute per-class F1 scores.

    Classes are the labels in *y_true* and *y_pred*, except the markers for
    failed predictions and abstentions (which still count as a miss for the
    true class).
    """
    labels = sorted((set(y_true) | set(y_pred)) - {_FAILED, _ABSTAINED})
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
    # ``[speech]`` placeholders in the text (not counted as words)
    placeholders: int = 0

    @property
    def unaligned(self) -> bool:
        """Whether the text is only placeholders: there are no words to
        place pauses and contours by."""
        return self.placeholders > 0 and not self.words


def _annotate(doc: IMLDocument | None) -> _Annotation:
    annotation = _Annotation()
    if doc is not None:
        for utt in doc.utterances:
            _walk(utt.children, annotation)
    return annotation


def _walk(children: tuple[ChildNode, ...], annotation: _Annotation) -> None:
    for child in children:
        if isinstance(child, str):
            if _PLACEHOLDER in child:
                annotation.placeholders += child.count(_PLACEHOLDER)
                child = child.replace(_PLACEHOLDER, " ")
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
    # Pauses are in document order, so their word positions never decrease.
    positions = [position for position, _ in pred.pauses]
    slack = 1e-9  # the bisection window is only a bound; the test below is exact
    candidates: list[list[int]] = []
    for truth_pos, truth_ms in truth.pauses:
        at = boundaries[truth_pos]
        lo = bisect.bisect_left(positions, at - position_tolerance - slack)
        hi = bisect.bisect_right(positions, at + position_tolerance + slack)
        near = [
            j
            for j in range(lo, hi)
            if abs(at - pred.pauses[j][0]) <= position_tolerance
            and abs(truth_ms - pred.pauses[j][1]) <= tolerance_ms
        ]
        # The closest pause first: in position, then in duration.
        near.sort(key=lambda j: (abs(at - pred.pauses[j][0]), abs(truth_ms - pred.pauses[j][1])))
        candidates.append(near)
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


def _predicted_emotion(doc: IMLDocument) -> tuple[str | None, float | None]:
    """The entry-level emotion prediction of a converted document.

    One prediction per entry: the emotion of the utterance with the highest
    stated confidence among those that carry an emotion, or ``None`` when
    none does (the converter abstained). The confidence is that
    utterance's, or ``None`` when it states no finite one.
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
    return best if best is not None else (None, None)


def _accepted_keywords(convert: Callable[..., object]) -> frozenset[str]:
    """Which of ``words`` and ``transcript`` *convert* takes as keywords."""
    wanted = frozenset({"words", "transcript"})
    try:
        parameters = inspect.signature(convert).parameters.values()
    except (TypeError, ValueError):  # no signature to inspect (a builtin, say)
        return frozenset()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return wanted
    return frozenset(
        p.name
        for p in parameters
        if p.name in wanted
        and p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


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
        An AudioToIML instance, or any object with a ``convert()`` method
        that takes an audio path and returns an IML string. If ``convert``
        also takes a ``words`` or ``transcript`` keyword argument, it is
        given the entry's words as *words_from* says.
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
    words_from:
        What the converter is given besides the audio, for converters that
        accept it:

        - ``"auto"`` (default): the entry's word timings as ``words=``
          when its ``metadata["word_timings"]`` has them (any format
          :func:`~prosody_protocol.alignment.parse_word_timings` reads);
          else nothing when the converter can recognize speech itself (it
          has an ``stt`` mode other than ``"none"`` and openai-whisper is
          installed), since recognized words carry timings; else its
          transcript as ``transcript=``;
        - ``"timings"``: word timings when the entry has them, else nothing
          (the converter's own speech recognition);
        - ``"transcript"``: always the transcript;
        - ``"stt"``: nothing; the converter finds the words itself.

        AudioToIML places pauses and word-level prosody only when it has
        word timings (given, or from Whisper); with a transcript alone it
        measures each recording as one span and places no pauses.
    abstention_label:
        The label an output without an emotion stands for, such as
        ``"neutral"``. ``None`` (default): such an output abstains; it
        lowers ``emotion_coverage`` and counts as a miss in the per-class
        F1, but not in ``emotion_accuracy``.

    Raises
    ------
    ValueError
        If neither *dataset_dir* nor ``dataset.root`` is set, *words_from*
        is not one of the values above or asks for keyword arguments the
        converter does not take, or *abstention_label* is an empty string.
    """

    def __init__(
        self,
        dataset: Dataset,
        converter: _Converter,
        dataset_dir: str | Path | None = None,
        *,
        pause_tolerance_ms: int = 200,
        pause_position_tolerance: float = 1.0,
        words_from: WordsFrom = "auto",
        abstention_label: str | None = None,
    ) -> None:
        root = dataset_dir if dataset_dir is not None else dataset.root
        if root is None:
            raise ValueError(
                "Benchmark needs the dataset directory to find the audio: pass dataset_dir, "
                "or load the dataset with DatasetLoader.load(), which records it"
            )
        if words_from not in _WORDS_FROM:
            raise ValueError(
                f"words_from must be one of {', '.join(_WORDS_FROM)}; got {words_from!r}"
            )
        if abstention_label is not None and (
            not isinstance(abstention_label, str) or not abstention_label.strip()
        ):
            raise ValueError(
                f"abstention_label must be None or a label, got {abstention_label!r}"
            )
        accepted = _accepted_keywords(converter.convert)
        needed = {"timings": "words", "transcript": "transcript"}.get(words_from)
        if needed is not None and needed not in accepted:
            raise ValueError(
                f"words_from={words_from!r} needs a converter whose convert() takes "
                f"{needed}=; {type(converter).__name__}.convert does not"
            )
        self.dataset = dataset
        self.converter = converter
        self.dataset_dir = Path(root)
        self.pause_tolerance_ms = pause_tolerance_ms
        self.pause_position_tolerance = pause_position_tolerance
        self.words_from: WordsFrom = words_from
        self.abstention_label = abstention_label
        self._accepted = accepted
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
        predicted = correct = 0
        confidences: list[float] = []
        confident_correct: list[bool] = []
        pause_tp = pause_fp = pause_fn = 0
        contours_total = contours_compared = contours_correct = 0
        valid_count = processed = num_failures = num_unaligned = 0
        sources = dict.fromkeys(_WORD_SOURCES, 0)

        for entry in entries:
            predicted_iml = self._get_predicted_iml(entry, sources)
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

            if pred_doc is None:
                emotion: str | None = _FAILED
                confidence: float | None = None
            else:
                emotion, confidence = _predicted_emotion(pred_doc)
                if emotion is None:
                    emotion = self.abstention_label
                if emotion is not None:
                    predicted += 1
                    correct += emotion == entry.emotion_label
            true_emotions.append(entry.emotion_label)
            pred_emotions.append(_ABSTAINED if emotion is None else emotion)
            if confidence is not None:
                confidences.append(confidence)
                confident_correct.append(emotion == entry.emotion_label)

            pred = _annotate(pred_doc)
            if pred_doc is not None and pred.unaligned:
                num_unaligned += 1
                continue
            truth = _annotate(self._parse_truth(entry))
            boundaries = _boundary_map(truth.words, pred.words)
            tp, fp, fn = _pause_counts(
                truth, pred, boundaries, self.pause_tolerance_ms, self.pause_position_tolerance
            )
            pause_tp, pause_fp, pause_fn = pause_tp + tp, pause_fp + fp, pause_fn + fn
            total, compared, right = _contour_counts(truth, pred, boundaries)
            contours_total += total
            contours_compared += compared
            contours_correct += right

        elapsed = time.time() - start
        if num_unaligned:
            logger.warning(
                "%d of %d outputs have only '%s' placeholders instead of words, so their "
                "pauses and pitch contours are not scored. Give the entries word timings "
                "(metadata '%s'), or use a converter that recognizes the words.",
                num_unaligned,
                processed,
                _PLACEHOLDER,
                WORD_TIMINGS_KEY,
            )

        n = len(entries)
        return BenchmarkReport(
            emotion_accuracy=correct / predicted if predicted else None,
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
            emotion_coverage=predicted / n if n else None,
            num_unaligned=num_unaligned,
            word_sources=sources,
            abstention_label=self.abstention_label,
        )

    def _parse_truth(self, entry: DatasetEntry) -> IMLDocument | None:
        try:
            return self._parser.parse(entry.iml)
        except Exception:
            logger.warning("Ground-truth IML of entry %s cannot be parsed", entry.id)
            return None

    def _inputs(self, entry: DatasetEntry) -> tuple[dict[str, Any], str]:
        """The keyword arguments for converting *entry*, and their source.

        Raises ConversionError when the entry's word timings are invalid.
        """
        if self.words_from in ("auto", "timings") and "words" in self._accepted:
            raw = entry.metadata.get(WORD_TIMINGS_KEY)
            if raw is not None:
                if not isinstance(raw, (str, dict, list)):
                    raise ConversionError(
                        f"metadata '{WORD_TIMINGS_KEY}' must be word timings, "
                        f"got {type(raw).__name__}"
                    )
                words: list[WordAlignment] = parse_word_timings(raw)
                return {"words": words}, "timings"
        if self.words_from == "auto" and self._converter_recognizes_speech():
            return {}, "stt"
        if self.words_from in ("auto", "transcript") and "transcript" in self._accepted:
            return {"transcript": entry.transcript}, "transcript"
        return {}, "stt"

    def _converter_recognizes_speech(self) -> bool:
        """Whether the converter would find the words itself, with timings.

        True for an :class:`~prosody_protocol.AudioToIML` whose ``stt`` is
        not ``"none"`` when openai-whisper is installed.
        """
        if getattr(self.converter, "stt", "none") not in ("auto", "whisper"):
            return False
        try:
            return importlib.util.find_spec("whisper") is not None
        except (ImportError, ValueError):
            return False

    def _get_predicted_iml(self, entry: DatasetEntry, sources: dict[str, int]) -> str | None:
        """Run the converter on an entry's audio; ``None`` when that fails."""
        try:
            audio_path = resolve_audio_path(self.dataset_dir, entry.audio_file)
        except DatasetError as exc:
            logger.warning("Entry %s not converted: %s", entry.id, exc)
            return None
        try:
            kwargs, source = self._inputs(entry)
        except (ConversionError, TypeError, ValueError, RecursionError) as exc:
            logger.warning("Entry %s not converted: invalid word timings: %s", entry.id, exc)
            return None
        sources[source] += 1
        try:
            result = self.converter.convert(audio_path, **kwargs)
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
