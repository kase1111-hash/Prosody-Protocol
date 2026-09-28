"""Pitch contour classification model.

Baseline implementation uses a random forest classifier on F0 tracks
measured from the audio, resampled to a fixed length and expressed in
semitones relative to their median (:func:`training.features.contour_features`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from ..features import resample_f0
from ..portable import tree_params
from .base import BaseModel, ModelRegistry

CONTOUR_CLASSES = ["rise", "fall", "rise-fall", "fall-rise", "flat"]

__all__ = ["CONTOUR_CLASSES", "PitchContourModel", "resample_f0"]


class PitchContourModel(BaseModel):
    """Pitch contour shape classifier.

    Classifies F0 sequences into contour categories such as
    rise, fall, rise-fall, fall-rise, flat.

    Parameters
    ----------
    n_estimators, max_depth, min_samples_split, min_samples_leaf, class_weight, random_state:
        Passed to :class:`sklearn.ensemble.RandomForestClassifier`.
    sequence_length:
        Length of the input sequences; set from the training data.
    feature_names:
        Names of the feature columns; ``f0_0``, ``f0_1``, ... by default.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int | None = 8,
        sequence_length: int = 20,
        *,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: str | None = None,
        random_state: int | None = 42,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sequence_length = sequence_length
        self.feature_names = list(feature_names or [])

        self._classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
        )
        self._encoder = LabelEncoder()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train on feature matrix X and string label array y.

        X should have shape (n_samples, sequence_length).
        """
        if not self.feature_names and X.ndim == 2:
            self.feature_names = [f"f0_{i}" for i in range(X.shape[1])]
        self._check_features(X)
        self.sequence_length = X.shape[1]
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
        return [str(label) for label in self._encoder.inverse_transform(indices)]

    def get_params(self) -> dict[str, Any]:
        classifier = self._classifier.get_params()
        return {
            "type": "random_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "sequence_length": self.sequence_length,
            "min_samples_split": classifier["min_samples_split"],
            "min_samples_leaf": classifier["min_samples_leaf"],
            "class_weight": classifier["class_weight"],
            "random_state": classifier["random_state"],
            "feature_names": self.feature_names,
            "trained": self._trained,
            "classes": [str(c) for c in self._encoder.classes_] if self._trained else [],
        }

    def portable_params(self) -> dict[str, Any]:
        if not self._trained:
            raise RuntimeError("Model has not been trained yet")
        return {
            **self._portable_header([str(c) for c in self._encoder.classes_]),
            "estimator": {
                "kind": "trees",
                "trees": [tree_params(tree) for tree in self._classifier.estimators_],
            },
        }


ModelRegistry.register("random_forest", PitchContourModel)
