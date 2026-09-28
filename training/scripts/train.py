#!/usr/bin/env python3
"""Unified training entry point.

Trains the scikit-learn baseline a config names, on features prepared from
a dataset (see ``data_prep.py``) or on data prepared earlier, and writes a
checkpoint directory:

- ``model.joblib``: the trained model. It is a pickle -- loading it runs
  code from the file -- so only load checkpoints you trust; share models
  as JSON exports (``export.py``) instead.
- ``metadata.json``: model class and parameters
- ``config.yaml``: a copy of the training config (``evaluate.py`` and
  ``export.py`` read it)
- ``training_results.json``: training and validation metrics

Usage:
    python training/scripts/train.py \\
        --config training/configs/ser_logreg.yaml \\
        --dataset datasets/emotional-speech \\
        --output training/checkpoints/ser_v1

    # Or with pre-prepared data:
    python training/scripts/train.py \\
        --config training/configs/ser_logreg.yaml \\
        --prepared-data /tmp/prepared_data \\
        --output training/checkpoints/ser_v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Allow imports from project root (the imports below must follow this)
# ruff: noqa: E402
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from prosody_protocol.exceptions import ProsodyProtocolError, TrainingError
from training.config import TrainingConfig, load_config
from training.metrics import compute_metrics
from training.models import ModelRegistry
from training.scripts.data_prep import _TASK_PREPARERS, read_prep_metadata

#: Name of the config copy saved in a checkpoint.
CHECKPOINT_CONFIG = "config.yaml"


def load_prepared_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-prepared train and validation data."""
    X_train = np.load(data_dir / "train" / "X.npy", allow_pickle=False)
    y_train = np.load(data_dir / "train" / "y.npy", allow_pickle=False)

    val_dir = data_dir / "val"
    if val_dir.exists() and (val_dir / "X.npy").exists():
        X_val = np.load(val_dir / "X.npy", allow_pickle=False)
        y_val = np.load(val_dir / "y.npy", allow_pickle=False)
    else:
        X_val = np.zeros((0, X_train.shape[1] if X_train.ndim > 1 else 0))
        y_val = np.array([])

    return X_train, y_train, X_val, y_val


def _load_data(
    config: TrainingConfig,
    dataset_dir: str | Path | None,
    prepared_data: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Train/val arrays and preparation metadata, preparing the dataset if needed."""
    if prepared_data is not None:
        data_dir = Path(prepared_data)
        meta = read_prep_metadata(data_dir)
        if meta.get("task", config.task) != config.task:
            raise ValueError(
                f"{data_dir} was prepared for task {meta['task']!r}, "
                f"but the config trains {config.task!r}"
            )
        return (*load_prepared_data(data_dir), meta)
    if dataset_dir is None:
        raise ValueError("Either --dataset or --prepared-data must be provided")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _TASK_PREPARERS[config.task](Path(dataset_dir), config.data, tmp_path)
        return (*load_prepared_data(tmp_path), read_prep_metadata(tmp_path))


def train(
    config_path: str | Path,
    dataset_dir: str | Path | None = None,
    prepared_data: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full training pipeline.

    Parameters
    ----------
    config_path:
        Path to the YAML training config.
    dataset_dir:
        Path to the raw dataset directory. Mutually exclusive with prepared_data.
    prepared_data:
        Path to pre-prepared data directory. Mutually exclusive with dataset_dir.
    output_dir:
        Where to save the trained model checkpoint.

    Returns
    -------
    dict
        Training results including metrics and output path.

    Raises
    ------
    ValueError
        For an invalid config or arguments.
    TrainingError
        When the data cannot be used: an empty training set, or labels
        that the config's ``labels`` list does not name.
    """
    config = load_config(config_path)
    X_train, y_train, X_val, y_val, meta = _load_data(config, dataset_dir, prepared_data)

    skipped = {reason: len(ids) for reason, ids in (meta.get("skipped") or {}).items()}
    if len(X_train) == 0:
        raise TrainingError(
            f"Training set is empty: no usable entries in the train split "
            f"(entries skipped in all splits, by reason: {skipped or 'none'})"
        )
    if config.labels:
        unexpected = sorted({str(v) for v in (*y_train, *y_val)} - set(config.labels))
        if unexpected:
            raise TrainingError(
                f"The data has labels the config does not list: {unexpected} "
                f"(expected: {config.labels})"
            )

    feature_names = meta.get("feature_names")
    model = ModelRegistry.create({**config.model_params(), "feature_names": feature_names})

    # Train
    start_time = time.time()
    train_metrics = model.train(X_train, y_train)
    elapsed = time.time() - start_time

    # Validate
    val_metrics = {}
    if len(X_val) > 0 and len(y_val) > 0:
        val_pred = model.predict_labels(X_val)
        val_report = compute_metrics(
            [str(label) for label in y_val], val_pred, labels=config.labels or None
        )
        val_metrics = {
            "accuracy": val_report.accuracy,
            "macro_f1": val_report.macro_f1,
            "n_samples": len(y_val),
        }

    results: dict[str, Any] = {
        "task": config.task,
        "model_type": config.model_type,
        "feature_names": model.feature_names,
        "classes": model.get_params()["classes"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "training_time_seconds": round(elapsed, 3),
        "train_samples": len(X_train),
        "skipped_entries": skipped,
    }

    # Save checkpoint
    if output_dir is not None:
        output_path = Path(output_dir)
        model.save(output_path)
        shutil.copyfile(config_path, output_path / CHECKPOINT_CONFIG)
        results["checkpoint_path"] = str(output_path)

        # Save training results alongside checkpoint
        with open(output_path / "training_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    return results


# Errors reported as a one-line message instead of a traceback.
_USER_ERRORS = (ValueError, FileNotFoundError, ProsodyProtocolError)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a prosody model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Path to dataset directory")
    group.add_argument("--prepared-data", help="Path to pre-prepared data directory")
    parser.add_argument(
        "--output",
        help="Path to output checkpoint directory (default: output.checkpoint_dir of the config)",
    )
    args = parser.parse_args()

    try:
        output = args.output
        if output is None:
            with warnings.catch_warnings():  # train() reports them
                warnings.simplefilter("ignore")
                output = load_config(args.config).output.get("checkpoint_dir")
        if output is None:
            parser.error("--output is required when the config has no output.checkpoint_dir")
        results = train(
            config_path=args.config,
            dataset_dir=args.dataset,
            prepared_data=args.prepared_data,
            output_dir=output,
        )
    except _USER_ERRORS as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Training complete for task '{results['task']}'")
    print(f"  Model type: {results['model_type']}")
    print(f"  Features: {', '.join(results['feature_names'])}")
    print(f"  Classes: {', '.join(results['classes'])}")
    print(f"  Training samples: {results['train_samples']}")
    print(f"  Training time: {results['training_time_seconds']}s")
    print(f"  Train accuracy: {results['train_metrics'].get('accuracy', 'N/A')}")
    if results["val_metrics"]:
        val = results["val_metrics"]
        print(f"  Val accuracy: {val['accuracy']:.4f} ({val['n_samples']} samples)")
        print(f"  Val macro F1: {val['macro_f1']:.4f}")
    else:
        print("  Val: no validation samples")
    print(f"  Checkpoint saved to: {results.get('checkpoint_path', 'N/A')}")


if __name__ == "__main__":
    main()
