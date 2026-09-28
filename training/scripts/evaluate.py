#!/usr/bin/env python3
"""Evaluation and metric reporting.

Reports precision, recall and F1 per class, their macro average over the
classes present in the split, and accuracy. With ``--dataset`` the split is
prepared again from the dataset (same split as in training), using the
config saved in the checkpoint unless ``--config`` is given.

The checkpoint's ``model.joblib`` is a pickle, and loading it runs code
from the file: only evaluate checkpoints you trust.

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
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Allow imports from project root (the imports below must follow this)
# ruff: noqa: E402
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from prosody_protocol.exceptions import ProsodyProtocolError
from training.config import load_config
from training.metrics import EvaluationReport, compute_metrics
from training.models.base import BaseModel
from training.scripts.data_prep import _TASK_PREPARERS, read_prep_metadata
from training.scripts.train import CHECKPOINT_CONFIG


def _load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    X = np.load(split_dir / "X.npy", allow_pickle=False)
    y = np.load(split_dir / "y.npy", allow_pickle=False)
    return X, y, read_prep_metadata(data_dir)


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
        Path to the model checkpoint directory. Its ``model.joblib`` is
        unpickled, which runs code from the file: only use trusted
        checkpoints.
    dataset_dir:
        Path to the raw dataset directory.
    prepared_data:
        Path to pre-prepared data directory.
    split:
        Which split to evaluate on ('train', 'val', or 'test').
    config_path:
        Path to the training config YAML used to prepare *dataset_dir*;
        by default the config saved in the checkpoint.

    Returns
    -------
    EvaluationReport
        Full evaluation report with per-class and aggregate metrics.
    """
    checkpoint_path = Path(checkpoint_path)
    # Load model
    model = BaseModel.load(checkpoint_path)

    # Load evaluation data
    if prepared_data is not None:
        X, y, meta = _load_split(Path(prepared_data), split)
    elif dataset_dir is not None:
        if config_path is None:
            config_path = checkpoint_path / CHECKPOINT_CONFIG
            if not config_path.is_file():
                raise ValueError(
                    f"{checkpoint_path} has no saved {CHECKPOINT_CONFIG}; "
                    "pass the training config with --config"
                )
        config = load_config(config_path)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _TASK_PREPARERS[config.task](Path(dataset_dir), config.data, tmp_path)
            X, y, meta = _load_split(tmp_path, split)
    else:
        raise ValueError("Either --dataset or --prepared-data must be provided")

    if len(X) == 0:
        raise ValueError(f"No data found in '{split}' split")
    prepared_features = meta.get("feature_names")
    if prepared_features and prepared_features != model.feature_names:
        raise ValueError(
            f"The data has features {prepared_features}, "
            f"but the model was trained on {model.feature_names}"
        )

    # Run predictions
    y_pred = model.predict_labels(X)
    y_true = [str(label) for label in y]

    # Report every class the model knows, in its order
    classes = model.get_params().get("classes") or None
    return compute_metrics(y_true, y_pred, labels=classes)


# Errors reported as a one-line message instead of a traceback.
_USER_ERRORS = (ValueError, FileNotFoundError, ProsodyProtocolError)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained prosody model",
        epilog="The checkpoint's model.joblib is a pickle: loading it runs code from the file. "
        "Only evaluate checkpoints you trust.",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a model checkpoint directory (trusted files only)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Path to dataset directory")
    group.add_argument("--prepared-data", help="Path to pre-prepared data directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--config",
                        help="Training config for --dataset (default: the checkpoint's copy)")
    parser.add_argument("--output", help="Optional path to save evaluation report as JSON")
    args = parser.parse_args()

    try:
        report = evaluate(
            checkpoint_path=args.checkpoint,
            dataset_dir=args.dataset,
            prepared_data=args.prepared_data,
            split=args.split,
            config_path=args.config,
        )
    except _USER_ERRORS as exc:
        parser.exit(1, f"Error: {exc}\n")

    # Print human-readable report
    print("\n" + report.format_table() + "\n")

    # Optionally save JSON report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
