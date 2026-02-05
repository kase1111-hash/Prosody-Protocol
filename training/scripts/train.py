#!/usr/bin/env python3
"""Unified training entry point.

Usage:
    python training/scripts/train.py \\
        --config training/configs/ser_wav2vec2.yaml \\
        --dataset datasets/emotional-speech \\
        --output training/checkpoints/ser_v1

    # Or with pre-prepared data:
    python training/scripts/train.py \\
        --config training/configs/ser_wav2vec2.yaml \\
        --prepared-data /tmp/prepared_data \\
        --output training/checkpoints/ser_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from training.config import load_config
from training.models import ModelRegistry

# Ensure all model types are registered
import training.models.ser  # noqa: F401
import training.models.text_prosody  # noqa: F401
import training.models.pitch_contour  # noqa: F401


def load_prepared_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-prepared train and validation data."""
    X_train = np.load(data_dir / "train" / "X.npy", allow_pickle=True)
    y_train = np.load(data_dir / "train" / "y.npy", allow_pickle=True)

    val_dir = data_dir / "val"
    if val_dir.exists() and (val_dir / "X.npy").exists():
        X_val = np.load(val_dir / "X.npy", allow_pickle=True)
        y_val = np.load(val_dir / "y.npy", allow_pickle=True)
    else:
        X_val = np.zeros((0, X_train.shape[1] if X_train.ndim > 1 else 0))
        y_val = np.array([])

    return X_train, y_train, X_val, y_val


def prepare_and_load(
    dataset_dir: Path,
    config_obj,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data from a raw dataset directory and return arrays."""
    import tempfile
    from training.scripts.data_prep import _TASK_PREPARERS

    preparer = _TASK_PREPARERS.get(config_obj.task)
    if preparer is None:
        raise ValueError(f"No data preparer for task '{config_obj.task}'")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        preparer(dataset_dir, config_obj.data, tmp_path)
        return load_prepared_data(tmp_path)


def train(
    config_path: str | Path,
    dataset_dir: str | Path | None = None,
    prepared_data: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
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
    """
    config = load_config(config_path)

    # Load data
    if prepared_data is not None:
        X_train, y_train, X_val, y_val = load_prepared_data(Path(prepared_data))
    elif dataset_dir is not None:
        X_train, y_train, X_val, y_val = prepare_and_load(Path(dataset_dir), config)
    else:
        raise ValueError("Either --dataset or --prepared-data must be provided")

    if len(X_train) == 0:
        raise ValueError("Training set is empty")

    # Create model from config
    model_config = dict(config.model)
    # Pass training hyperparameters to model constructor
    for key in ("max_iter", "regularization", "min_samples_split", "min_samples_leaf"):
        if key in config.training and key not in model_config:
            model_config[key] = config.training[key]

    model = ModelRegistry.create(model_config)

    # Train
    start_time = time.time()
    train_metrics = model.train(X_train, y_train)
    elapsed = time.time() - start_time

    # Validate
    val_metrics = {}
    if len(X_val) > 0 and len(y_val) > 0:
        val_pred = model.predict_labels(X_val)
        from training.metrics import compute_metrics
        val_report = compute_metrics(list(y_val), val_pred, labels=config.labels or None)
        val_metrics = {
            "accuracy": val_report.accuracy,
            "macro_f1": val_report.macro_f1,
        }

    results = {
        "task": config.task,
        "model_type": config.model_type,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "training_time_seconds": round(elapsed, 3),
        "train_samples": len(X_train),
    }

    # Save checkpoint
    if output_dir is not None:
        output_path = Path(output_dir)
        model.save(output_path)
        results["checkpoint_path"] = str(output_path)

        # Save training results alongside checkpoint
        with open(output_path / "training_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a prosody model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Path to dataset directory")
    group.add_argument("--prepared-data", help="Path to pre-prepared data directory")
    parser.add_argument("--output", required=True, help="Path to output checkpoint directory")
    args = parser.parse_args()

    results = train(
        config_path=args.config,
        dataset_dir=args.dataset,
        prepared_data=args.prepared_data,
        output_dir=args.output,
    )

    print(f"Training complete for task '{results['task']}'")
    print(f"  Model type: {results['model_type']}")
    print(f"  Training samples: {results['train_samples']}")
    print(f"  Training time: {results['training_time_seconds']}s")
    print(f"  Train accuracy: {results['train_metrics'].get('accuracy', 'N/A')}")
    if results["val_metrics"]:
        print(f"  Val accuracy: {results['val_metrics'].get('accuracy', 'N/A')}")
        print(f"  Val macro F1: {results['val_metrics'].get('macro_f1', 'N/A')}")
    print(f"  Checkpoint saved to: {results.get('checkpoint_path', 'N/A')}")


if __name__ == "__main__":
    main()
