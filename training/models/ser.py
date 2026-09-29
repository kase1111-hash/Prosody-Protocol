"""Speech Emotion Recognition model using prosodic features.

Baseline implementation: logistic regression on utterance-level prosodic
features measured from the audio (see :mod:`training.features`). Features
the analyzer could not measure arrive as ``NaN`` and are replaced by the
training median of that feature; all features are then standardised.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..features import DEFAULT_SER_FEATURES
from .base import BaseModel, ModelRegistry

# Feature names in canonical order (matches config data.features)
DEFAULT_FEATURES = list(DEFAULT_SER_FEATURES)


class SERModel(BaseModel):
    """Speech Emotion Recognition classifier.

    Uses sklearn LogisticRegression as a lightweight baseline.
    Feature vectors are standardized before classification.

    Parameters
    ----------
    num_classes, labels, feature_dim:
        Descriptive only; the classes and the number of features are
        those of the training data.
    feature_names:
        Names of the feature columns, e.g. ``["f0_mean", "jitter"]``.
    C, max_iter, solver, class_weight, random_state:
        Passed to :class:`sklearn.linear_model.LogisticRegression`.
    """

    def __init__(
        self,
        num_classes: int = 8,
        labels: list[str] | None = None,
        feature_dim: int = 7,
        *,
        feature_names: Sequence[str] | None = None,
        C: float = 1.0,
        max_iter: int = 200,
        solver: str = "lbfgs",
        class_weight: str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.label_names = labels or []
        self.feature_names = list(feature_names or [])

        self._scaler = StandardScaler()
        self._encoder = LabelEncoder()
        self._classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=random_state,
        )
        self._fill_values = np.zeros(0)
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train on feature matrix X and string label array y.

        ``NaN`` entries of X are unmeasured features; they are replaced by
        the column's median (0 for a column with no measurement at all).
        """
        self._check_features(X)
        self._record_feature_stats(X)
        self._fill_values = _column_medians(X)
        y_encoded = self._encoder.fit_transform(y)
        X_scaled = self._scaler.fit_transform(self._fill(X))

        self._classifier.fit(X_scaled, y_encoded)
        self._trained = True
        self.num_classes = len(self._encoder.classes_)
        self.feature_dim = X.shape[1]

        # Compute training accuracy
        train_pred = self._classifier.predict(X_scaled)
        accuracy = float(np.mean(train_pred == y_encoded))

        return {"accuracy": accuracy, "n_samples": len(y)}

    def _fill(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=np.float64)
        return np.where(np.isnan(data), self._fill_values, data)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet")
        X_scaled = self._scaler.transform(self._fill(X))
        return self._classifier.predict(X_scaled)

    def predict_labels(self, X: np.ndarray) -> list[str]:
        """Return predicted string labels."""
        indices = self.predict(X)
        return [str(label) for label in self._encoder.inverse_transform(indices)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability distribution over classes."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet")
        X_scaled = self._scaler.transform(self._fill(X))
        return self._classifier.predict_proba(X_scaled)

    def get_params(self) -> dict[str, Any]:
        classifier = self._classifier.get_params()
        return {
            "type": "logistic_regression",
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "feature_names": self.feature_names,
            "labels": self.label_names,
            "C": classifier["C"],
            "max_iter": classifier["max_iter"],
            "solver": classifier["solver"],
            "class_weight": classifier["class_weight"],
            "random_state": classifier["random_state"],
            "trained": self._trained,
            "classes": [str(c) for c in self._encoder.classes_] if self._trained else [],
        }

    def portable_params(self) -> dict[str, Any]:
        if not self._trained:
            raise RuntimeError("Model has not been trained yet")
        return {
            **self._portable_header([str(c) for c in self._encoder.classes_]),
            "preprocessing": {
                "fill_values": self._fill_values.tolist(),
                "mean": self._scaler.mean_.tolist(),
                "scale": self._scaler.scale_.tolist(),
            },
            "estimator": {
                "kind": "linear",
                "coef": self._classifier.coef_.tolist(),
                "intercept": self._classifier.intercept_.tolist(),
            },
        }


def _column_medians(X: np.ndarray) -> np.ndarray:
    """Median of the measured (non-NaN) values of each column; 0 for a column with none."""
    medians = []
    for column in np.asarray(X, dtype=np.float64).T:
        measured = column[~np.isnan(column)]
        medians.append(float(np.median(measured)) if measured.size else 0.0)
    return np.array(medians, dtype=np.float64)


# Register with the model registry
ModelRegistry.register("logistic_regression", SERModel)
