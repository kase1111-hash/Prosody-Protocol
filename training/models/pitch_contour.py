"""Pitch contour classification model.

Baseline implementation uses a random forest classifier on resampled
F0 sequences. Designed to be swappable with a 1D CNN or LSTM when the
``ml`` extras are installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel, ModelRegistry

CONTOUR_CLASSES = ["rise", "fall", "rise-fall", "fall-rise", "flat"]


def resample_f0(f0_sequence: list[float] | np.ndarray, target_length: int = 20) -> np.ndarray:
    """Resample an F0 sequence to a fixed length via linear interpolation.

    Parameters
    ----------
    f0_sequence:
        Variable-length F0 values.
    target_length:
        Fixed output length.

    Returns
    -------
    np.ndarray
        Resampled F0 array of shape (target_length,).
    """
    seq = np.array(f0_sequence, dtype=np.float64)
    if len(seq) == 0:
        return np.zeros(target_length)
    if len(seq) == 1:
        return np.full(target_length, seq[0])

    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, seq)


class PitchContourModel(BaseModel):
    """Pitch contour shape classifier.

    Classifies F0 sequences into contour categories:
    rise, fall, rise-fall, fall-rise, flat.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 8,
        sequence_length: int = 20,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sequence_length = sequence_length

        min_samples_split = kwargs.get("min_samples_split", 2)
        min_samples_leaf = kwargs.get("min_samples_leaf", 1)

        self._classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        self._encoder = LabelEncoder()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train on feature matrix X and string label array y.

        X should have shape (n_samples, sequence_length).
        """
        y_encoded = self._encoder.fit_transform(y)
        self._classifier.fit(X, y_encoded)
        self._trained = True

        train_pred = self._classifier.predict(X)
        accuracy = float(np.mean(train_pred == y_encoded))
        return {"accuracy": accuracy, "n_samples": len(y)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet")
        return self._classifier.predict(X)

    def predict_labels(self, X: np.ndarray) -> list[str]:
        """Return predicted string labels."""
        indices = self.predict(X)
        return list(self._encoder.inverse_transform(indices))

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "random_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "sequence_length": self.sequence_length,
            "trained": self._trained,
            "classes": list(self._encoder.classes_) if self._trained else [],
        }


ModelRegistry.register("random_forest", PitchContourModel)
