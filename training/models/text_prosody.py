"""Text-to-prosody prediction model.

Baseline implementation uses a decision tree classifier to predict
per-token prosodic labels (pitch level, volume level, rate level)
from text features. Designed to be swappable with BERT + sequence
labeling when the ``ml`` extras are installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel, ModelRegistry

# Text feature names in canonical order
DEFAULT_TEXT_FEATURES = [
    "word_length", "position_ratio", "is_capitalized",
    "has_punctuation", "sentence_position", "prev_word_length",
    "next_word_length",
]


def extract_text_features(words: list[str]) -> np.ndarray:
    """Extract feature vectors from a list of words.

    Parameters
    ----------
    words:
        List of word tokens.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_words, 7).
    """
    n = len(words)
    features = []
    for i, word in enumerate(words):
        clean = word.strip(".,!?;:\"'()-")
        feat = [
            len(clean),                                   # word_length
            i / max(n - 1, 1),                            # position_ratio
            1.0 if clean and clean[0].isupper() else 0.0, # is_capitalized
            1.0 if word != clean else 0.0,                # has_punctuation
            float(i),                                     # sentence_position
            len(words[i - 1].strip(".,!?;:\"'()-")) if i > 0 else 0.0,  # prev_word_length
            len(words[i + 1].strip(".,!?;:\"'()-")) if i < n - 1 else 0.0,  # next_word_length
        ]
        features.append(feat)
    return np.array(features, dtype=np.float64)


class TextProsodyModel(BaseModel):
    """Per-token prosodic label prediction from text features.

    Predicts a combined label encoding pitch_level, volume_level,
    and rate_level for each token.
    """

    def __init__(
        self,
        max_depth: int = 10,
        **kwargs: Any,
    ) -> None:
        self.max_depth = max_depth
        min_samples_split = kwargs.get("min_samples_split", 2)
        min_samples_leaf = kwargs.get("min_samples_leaf", 1)

        self._classifier = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        self._encoder = LabelEncoder()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Train on feature matrix X and string label array y."""
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
            "type": "decision_tree",
            "max_depth": self.max_depth,
            "trained": self._trained,
            "classes": list(self._encoder.classes_) if self._trained else [],
        }


ModelRegistry.register("decision_tree", TextProsodyModel)
