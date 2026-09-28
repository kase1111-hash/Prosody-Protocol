"""Evaluation metrics for training pipelines.

Computes per-class and macro-averaged precision, recall, and F1 scores.

The macro average covers the classes that occur in the evaluated split --
in its true labels or its predictions -- as scikit-learn's does when it is
given no label list. Classes that are only listed (say, config labels
absent from a small test split) still get a row in the report, marked as
not evaluated, but do not pull the average down.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import precision_recall_fscore_support


@dataclass
class ClassMetrics:
    """Metrics for a single class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int
    #: False when the class occurs in neither the true labels nor the
    #: predictions: its scores are then undefined (reported as 0.0 here,
    #: ``None`` by :meth:`EvaluationReport.to_dict`, ``n/a`` in the table)
    #: and it is not part of the macro average.
    evaluated: bool = True


@dataclass
class EvaluationReport:
    """Full evaluation report with per-class and aggregate metrics."""

    per_class: list[ClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    accuracy: float
    total_samples: int

    def to_dict(self) -> dict:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "per_class": [
                {
                    "label": m.label,
                    "precision": round(m.precision, 4) if m.evaluated else None,
                    "recall": round(m.recall, 4) if m.evaluated else None,
                    "f1": round(m.f1, 4) if m.evaluated else None,
                    "support": m.support,
                }
                for m in self.per_class
            ],
            "macro": {
                "precision": round(self.macro_precision, 4),
                "recall": round(self.macro_recall, 4),
                "f1": round(self.macro_f1, 4),
            },
            "accuracy": round(self.accuracy, 4),
            "total_samples": self.total_samples,
        }

    def format_table(self) -> str:
        """Format report as a human-readable table."""
        lines = []
        header = f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for m in self.per_class:
            if m.evaluated:
                scores = f"{m.precision:>10.4f} {m.recall:>10.4f} {m.f1:>10.4f}"
            else:
                scores = f"{'n/a':>10} {'n/a':>10} {'n/a':>10}"
            lines.append(f"{m.label:<20} {scores} {m.support:>8d}")
        lines.append("-" * len(header))
        lines.append(
            f"{'macro avg':<20} {self.macro_precision:>10.4f} "
            f"{self.macro_recall:>10.4f} {self.macro_f1:>10.4f} "
            f"{self.total_samples:>8d}"
        )
        lines.append(f"{'accuracy':<20} {'':>10} {'':>10} {self.accuracy:>10.4f} "
                      f"{self.total_samples:>8d}")
        return "\n".join(lines)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> EvaluationReport:
    """Compute precision, recall, and F1 per class and macro-averaged.

    Parameters
    ----------
    y_true:
        Ground truth labels.
    y_pred:
        Predicted labels.
    labels:
        Optional explicit label order for the per-class rows. If None,
        derived from data. Labels that occur in the data but not in this
        list are appended, so no sample is left out of the report.

    Returns
    -------
    EvaluationReport
        Complete evaluation report. Its macro averages cover the labels
        that occur in *y_true* or *y_pred*.
    """
    present = set(y_true) | set(y_pred)
    listed = [] if labels is None else list(labels)
    labels = listed + sorted(present - set(listed))

    per_class: list[ClassMetrics] = []
    if labels:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0.0,
        )
        per_class = [
            ClassMetrics(
                label=label,
                precision=float(p),
                recall=float(r),
                f1=float(f),
                support=int(s),
                evaluated=label in present,
            )
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        ]

    macro_p = macro_r = macro_f = 0.0
    evaluated = [label for label in labels if label in present]
    if evaluated:
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=evaluated, average="macro", zero_division=0.0,
        )

    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    return EvaluationReport(
        per_class=per_class,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f),
        accuracy=accuracy,
        total_samples=len(y_true),
    )
