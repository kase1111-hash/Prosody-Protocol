#!/usr/bin/env python3
"""Evaluation and metric reporting.

Usage:
    python training/scripts/evaluate.py \\
        --checkpoint training/checkpoints/ser_v1 \\
        --dataset datasets/emotional-speech --split test

    # Or with pre-prepared data:
    python training/scripts/evaluate.py \\
        --checkpoint training/checkpoints/ser_v1 \\
        --prepared-data /tmp/prepared_data --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from training.models.base import BaseModel
from training.metrics import compute_metrics, EvaluationReport


def evaluate(
    checkpoint_path: str | Path,
    dataset_dir: str | Path | None = None,
    prepared_data: str | Path | None = None,
    split: str = "test",
    config_path: str | Path | None = None,
) -> EvaluationReport:
    """Evaluate a trained model and produce a classification report.

    Parameters
    ----------
    checkpoint_path:
        Path to the model checkpoint directory.
    dataset_dir:
        Path to the raw dataset directory.
    prepared_data:
        Path to pre-prepared data directory.
    split:
        Which split to evaluate on ('train', 'val', or 'test').
    config_path:
        Path to the training config YAML (required when using dataset_dir).

    Returns
    -------
    EvaluationReport
        Full evaluation report with per-class and aggregate metrics.
    """
    # Load model
    model = BaseModel.load(checkpoint_path)

    # Load evaluation data
    if prepared_data is not None:
        data_dir = Path(prepared_data) / split
        if not data_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {data_dir}")
        X = np.load(data_dir / "X.npy", allow_pickle=True)
        y = np.load(data_dir / "y.npy", allow_pickle=True)
    elif dataset_dir is not None:
        if config_path is None:
            raise ValueError("--config is required when using --dataset")
        from training.config import load_config
        from training.scripts.data_prep import _TASK_PREPARERS
        import tempfile

        config = load_config(config_path)
        preparer = _TASK_PREPARERS.get(config.task)
        if preparer is None:
            raise ValueError(f"No data preparer for task '{config.task}'")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preparer(Path(dataset_dir), config.data, tmp_path)
            split_dir = tmp_path / split
            X = np.load(split_dir / "X.npy", allow_pickle=True)
            y = np.load(split_dir / "y.npy", allow_pickle=True)
    else:
        raise ValueError("Either --dataset or --prepared-data must be provided")

    if len(X) == 0:
        raise ValueError(f"No data found in '{split}' split")

    # Run predictions
    y_pred = model.predict_labels(X)
    y_true = list(y)

    # Get labels from model metadata if available
    metadata_path = Path(checkpoint_path) / "metadata.json"
    labels = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        classes = meta.get("params", {}).get("classes", [])
        if classes:
            labels = classes

    return compute_metrics(y_true, y_pred, labels=labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained prosody model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Path to dataset directory")
    group.add_argument("--prepared-data", help="Path to pre-prepared data directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--config", help="Path to training config (required with --dataset)")
    parser.add_argument("--output", help="Optional path to save evaluation report as JSON")
    args = parser.parse_args()

    report = evaluate(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset,
        prepared_data=args.prepared_data,
        split=args.split,
        config_path=args.config,
    )

    # Print human-readable report
    print("\n" + report.format_table() + "\n")

    # Optionally save JSON report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
