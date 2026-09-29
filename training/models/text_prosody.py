"""Text-to-prosody prediction model.

Baseline implementation uses a decision tree classifier to predict
per-token prosodic labels (pitch level, volume level, rate level,
emphasis) from text features. The labels are read from the IML markup of
the training data (:func:`training.features.token_prosody_labels`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from ..features import TEXT_FEATURES, extract_text_features
from ..portable import tree_params
from .base import BaseModel, ModelRegistry

# Text feature names in canonical order
DEFAULT_TEXT_FEATURES = list(TEXT_FEATURES)

__all__ = ["DEFAULT_TEXT_FEATURES", "TextProsodyModel", "extract_text_features"]


class TextProsodyModel(BaseModel):
    """Per-token prosodic label prediction from text features.

    Predicts a combined label such as ``high_loud_normal`` (one value per
    configured label dimension) for each token.

    Parameters
    ----------
    max_depth, min_samples_split, min_samples_leaf, class_weight, random_state:
        Passed to :class:`sklearn.tree.DecisionTreeClassifier`.
    feature_names:
        Names of the feature columns (see :data:`training.features.TEXT_FEATURES`).
    """

    def __init__(
        self,
        max_depth: int | None = 10,
        *,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: str | None = None,
        random_state: int | None = 42,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.feature_names = list(feature_names or [])

        self._classifier = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
        )
        self._encoder = LabelEncoder()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train on feature matrix X and string label array y."""
        self._check_features(X)
        self._record_feature_stats(X)
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
            "type": "decision_tree",
            "max_depth": self.max_depth,
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
            "estimator": {"kind": "trees", "trees": [tree_params(self._classifier)]},
        }


ModelRegistry.register("decision_tree", TextProsodyModel)
