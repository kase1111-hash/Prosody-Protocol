"""Abstract base model and model registry for training pipelines."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..portable import FORMAT, FORMAT_VERSION, write_model

#: Formats :meth:`BaseModel.export` can write.
EXPORT_FORMATS = ("json", "pickle")


class BaseModel(ABC):
    """Abstract base class for all trainable models.

    Subclasses implement the specific model logic while this class
    provides serialization, loading, and the training interface contract.

    ``feature_names`` names the columns of the feature matrices the model
    is trained on and applied to; it is set at construction or, when not
    given, to ``x0``, ``x1``, ... by :meth:`train`.
    """

    feature_names: list[str]

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

    def portable_params(self) -> dict[str, Any]:
        """The fitted model as :mod:`training.portable` JSON parameters.

        Raises ``NotImplementedError`` for model classes without a JSON
        export (export them with ``export_format="pickle"``).
        """
        raise NotImplementedError(f"{type(self).__name__} has no JSON export")

    def _portable_header(self, classes: list[str]) -> dict[str, Any]:
        """The fields every portable model file starts with."""
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "model_class": type(self).__name__,
            "model_type": self.get_params()["type"],
            "classes": classes,
            "feature_names": list(self.feature_names),
        }

    def _check_features(self, X: np.ndarray) -> None:
        """Name the columns of the training matrix *X*, or check they match the names."""
        n_features = X.shape[1] if X.ndim == 2 else 0
        if not self.feature_names:
            self.feature_names = [f"x{i}" for i in range(n_features)]
        elif len(self.feature_names) != n_features:
            raise ValueError(
                f"X has {n_features} columns but the model has "
                f"{len(self.feature_names)} feature names: {self.feature_names}"
            )

    def save(self, path: str | Path) -> None:
        """Save the model checkpoint to a directory.

        Creates:
        - model.joblib: the model object, pickled by joblib. Loading it
          (:meth:`load`) runs code from the file, like any pickle.
        - metadata.json: model class and parameters
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path / "model.joblib")

        metadata = {
            "model_class": type(self).__name__,
            "params": self.get_params(),
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """Load a model checkpoint (``model.joblib``) from a directory.

        .. warning::
           ``model.joblib`` is a pickle. Unpickling runs code stored in the
           file, before any check of what it contains can be made, so only
           load checkpoints you created yourself or otherwise trust. To
           share a model, export it with ``export_format="json"`` and load
           that with :class:`training.portable.PortableModel`, which never
           runs code from the file.
        """
        path = Path(path)
        model_file = path / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"No model checkpoint (model.joblib) found at {path}")

        model = joblib.load(model_file)

        if not isinstance(model, BaseModel):
            raise TypeError(f"Loaded object is not a BaseModel: {type(model)}")
        return model

    def export(self, path: str | Path, export_format: str = "json") -> list[str]:
        """Export the trained model for use outside the training pipeline.

        Creates ``config.json`` (model class and parameters) and:

        - with ``export_format="json"``: ``model.json``, the fitted
          parameters as plain JSON (see :mod:`training.portable`), which
          :class:`~training.portable.PortableModel` runs with numpy alone
          and which never runs code when loaded;
        - with ``"pickle"``: ``model.joblib``, a pickle for
          :meth:`BaseModel.load` -- only for trusted use, see there.

        Returns the names of the files written.
        """
        if export_format not in EXPORT_FORMATS:
            raise ValueError(
                f"Unknown export format {export_format!r}; expected one of {EXPORT_FORMATS}"
            )
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "model_class": type(self).__name__,
            "params": self.get_params(),
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        if export_format == "json":
            weights = write_model(self.portable_params(), path).name
        else:
            weights = "model.joblib"
            joblib.dump(self, path / weights)
        return ["config.json", weights]


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
            All other keys are passed as constructor arguments; a key the
            model does not accept raises ``ValueError``.
        """
        model_type = config.get("type")
        if model_type not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. Available: {available}"
            )

        kwargs = {k: v for k, v in config.items() if k != "type"}
        try:
            return cls._registry[model_type](**kwargs)
        except TypeError as exc:
            raise ValueError(f"Invalid parameters for model type '{model_type}': {exc}") from exc

    @classmethod
    def available(cls) -> list[str]:
        """Return list of registered model type names."""
        return sorted(cls._registry.keys())
