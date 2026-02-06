"""Abstract base model and model registry for training pipelines."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all trainable models.

    Subclasses implement the specific model logic while this class
    provides serialization, loading, and the training interface contract.
    """

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train the model on feature matrix X and label array y.

        Returns a dict of training metrics (e.g. loss, accuracy).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices or labels for input X."""

    @abstractmethod
    def predict_labels(self, X: np.ndarray) -> list[str]:
        """Return predicted string labels for input X."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model parameters as a serializable dictionary."""

    def save(self, path: str | Path) -> None:
        """Save the model checkpoint to a directory.

        Creates:
        - model.joblib: serialized model object (via joblib, safer than pickle)
        - metadata.json: model metadata and config
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path / "model.joblib")

        metadata = {
            "model_class": type(self).__name__,
            "params": self.get_params(),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Load a model checkpoint from a directory."""
        path = Path(path)

        # Support both new (.joblib) and legacy (.pkl) checkpoints.
        model_file = path / "model.joblib"
        if not model_file.exists():
            model_file = path / "model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"No model checkpoint found at {path}")

        model = joblib.load(model_file)

        if not isinstance(model, BaseModel):
            raise TypeError(f"Loaded object is not a BaseModel: {type(model)}")
        return model

    def export(self, path: str | Path) -> None:
        """Export model in a portable format for SDK integration.

        Creates:
        - config.json: model configuration for reconstruction
        - weights.npz: numpy arrays of model weights (if applicable)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "model_class": type(self).__name__,
            "params": self.get_params(),
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Subclasses override _export_weights for custom weight formats
        self._export_weights(path)

    def _export_weights(self, path: Path) -> None:
        """Export model weights. Override in subclasses for custom formats."""
        # Default: save via joblib for non-neural models
        joblib.dump(self, path / "model.joblib")


class ModelRegistry:
    """Registry mapping model type strings to model classes."""

    _registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: type[BaseModel]) -> None:
        """Register a model class under a name."""
        cls._registry[name] = model_class

    @classmethod
    def create(cls, config: dict[str, Any]) -> BaseModel:
        """Create a model instance from a config dict.

        Parameters
        ----------
        config:
            Must contain a 'type' key matching a registered model name.
            All other keys are passed as constructor arguments.
        """
        model_type = config.get("type")
        if model_type not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. Available: {available}"
            )

        kwargs = {k: v for k, v in config.items() if k != "type"}
        return cls._registry[model_type](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return list of registered model type names."""
        return sorted(cls._registry.keys())
