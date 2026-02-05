"""YAML configuration loading and validation for training pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Parsed training configuration from a YAML file."""

    task: str
    description: str
    model: dict[str, Any]
    data: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    output: dict[str, Any]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def model_type(self) -> str:
        return self.model.get("type", "unknown")

    @property
    def labels(self) -> list[str]:
        """Get the class labels for the task."""
        if "labels" in self.model:
            return self.model["labels"]
        if "contour_classes" in self.data:
            return self.data["contour_classes"]
        return []

    @property
    def metrics(self) -> list[str]:
        return self.evaluation.get("metrics", ["precision", "recall", "f1"])

    @property
    def average(self) -> str:
        return self.evaluation.get("average", "macro")


_REQUIRED_KEYS = {"task", "model", "data", "training", "evaluation", "output"}


def load_config(path: str | Path) -> TrainingConfig:
    """Load and validate a training configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    TrainingConfig
        Parsed configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the config is missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")

    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    return TrainingConfig(
        task=raw["task"],
        description=raw.get("description", ""),
        model=raw["model"],
        data=raw["data"],
        training=raw["training"],
        evaluation=raw["evaluation"],
        output=raw["output"],
        raw=raw,
    )
