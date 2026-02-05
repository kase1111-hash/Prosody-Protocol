"""Speech Emotion Recognition model using prosodic features.

Baseline implementation uses logistic regression on aggregated prosodic
feature vectors. Designed to be swappable with wav2vec2-based models
when the ``ml`` extras are installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base import BaseModel, ModelRegistry

# Feature names in canonical order (matches config data.features)
DEFAULT_FEATURES = [
    "f0_mean", "f0_range", "intensity_mean",
    "speech_rate", "jitter", "shimmer", "hnr",
]


class SERModel(BaseModel):
    """Speech Emotion Recognition classifier.

    Uses sklearn LogisticRegression as a lightweight baseline.
    Feature vectors are standardized before classification.
    """

    def __init__(
        self,
        num_classes: int = 8,
        labels: list[str] | None = None,
        feature_dim: int = 7,
        **kwargs: Any,
    ) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.label_names = labels or []

        max_iter = kwargs.get("max_iter", 200)
        C = kwargs.get("regularization", 1.0)

        self._scaler = StandardScaler()
        self._encoder = LabelEncoder()
        self._classifier = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver="lbfgs",
        )
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train on feature matrix X and string label array y."""
        y_encoded = self._encoder.fit_transform(y)
        X_scaled = self._scaler.fit_transform(X)

        self._classifier.fit(X_scaled, y_encoded)
        self._trained = True

        # Compute training accuracy
        train_pred = self._classifier.predict(X_scaled)
        accuracy = float(np.mean(train_pred == y_encoded))

        return {"accuracy": accuracy, "n_samples": len(y)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet")
        X_scaled = self._scaler.transform(X)
        return self._classifier.predict(X_scaled)

    def predict_labels(self, X: np.ndarray) -> list[str]:
        """Return predicted string labels."""
        indices = self.predict(X)
        return list(self._encoder.inverse_transform(indices))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability distribution over classes."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet")
        X_scaled = self._scaler.transform(X)
        return self._classifier.predict_proba(X_scaled)

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "logistic_regression",
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "labels": self.label_names,
            "trained": self._trained,
            "classes": list(self._encoder.classes_) if self._trained else [],
        }


# Register with the model registry
ModelRegistry.register("logistic_regression", SERModel)
