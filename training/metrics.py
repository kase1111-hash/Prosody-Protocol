"""Evaluation metrics for training pipelines.

Computes per-class and macro-averaged precision, recall, and F1 scores.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import classification_report, precision_recall_fscore_support


@dataclass
class ClassMetrics:
    """Metrics for a single class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int


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
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
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
            lines.append(
                f"{m.label:<20} {m.precision:>10.4f} {m.recall:>10.4f} "
                f"{m.f1:>10.4f} {m.support:>8d}"
            )
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
        Optional explicit label order. If None, derived from data.

    Returns
    -------
    EvaluationReport
        Complete evaluation report.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

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
        )
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    ]

    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0.0,
    )

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    return EvaluationReport(
        per_class=per_class,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f),
        accuracy=accuracy,
        total_samples=len(y_true),
    )
