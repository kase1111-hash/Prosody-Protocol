"""Pickle-free model files: trained baselines as JSON, run with numpy.

A checkpoint's ``model.joblib`` is a pickle, and loading a pickle runs
whatever code the file contains -- it must only be loaded from a trusted
source. ``export.py --format json`` instead writes a model's fitted
parameters to ``model.json``: plain numbers and strings. :class:`PortableModel`
runs them with numpy alone (no scikit-learn, joblib or ``training.models``
classes needed), and loading one never executes code from the file.

Format (``format_version`` 1)::

    {
      "format": "prosody-protocol-baseline",
      "format_version": 1,
      "model_class": "SERModel",
      "model_type": "logistic_regression",
      "classes": ["angry", "calm", "sad"],      # output labels, column order
      "feature_names": ["f0_mean", ...],        # input columns, in order
      "preprocessing": {                        # optional, applied in order:
        "fill_values": [...],                   #   replaces NaN inputs
        "mean": [...], "scale": [...]           #   (x - mean) / scale
      },
      "feature_stats": {                        # optional: the training data's
        "mean": [...], "std": [...]             #   mean and standard deviation
      },                                        #   of each feature
      "estimator": {"kind": "linear", "coef": [[...]], "intercept": [...]}
    }

``feature_stats`` describes the measured (non-NaN) training values of each
feature; a ``std`` of 0 marks a feature that was constant or never measured
in training. :meth:`PortableModel.feature_z_scores` uses it to tell how far
an input lies outside the training data (models exported before it existed
fall back to ``preprocessing.mean`` and ``scale``). Predictions do not use it.

A linear estimator gives class probabilities as the softmax of
``coef @ x + intercept``; with a single row of coefficients (two classes)
the logistic sigmoid of that row gives the probability of the second
class. A tree estimator is ``{"kind": "trees", "trees": [...]}``, each tree
holding scikit-learn's node arrays ``children_left``, ``children_right``,
``feature``, ``threshold`` and ``value`` (the class distribution of each
node); the probabilities are the mean over the trees of the distribution
of the leaf the input reaches (going left when ``x[feature] <= threshold``,
with ``x`` rounded to float32 as scikit-learn does).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FORMAT = "prosody-protocol-baseline"
FORMAT_VERSION = 1

#: File name of a portable model inside an export directory.
MODEL_FILE = "model.json"

_FloatArray = npt.NDArray[np.float64]
_IntArray = npt.NDArray[np.int64]


def tree_params(tree: Any) -> dict[str, Any]:
    """The node arrays of a fitted scikit-learn decision tree classifier, as lists."""
    nodes = tree.tree_
    value = np.asarray(nodes.value, dtype=np.float64)[:, 0, :]
    totals = value.sum(axis=1, keepdims=True)
    distribution = np.divide(value, totals, out=np.zeros_like(value), where=totals > 0)
    return {
        "children_left": nodes.children_left.tolist(),
        "children_right": nodes.children_right.tolist(),
        "feature": nodes.feature.tolist(),
        "threshold": nodes.threshold.tolist(),
        "value": distribution.tolist(),
    }


def feature_stats(X: npt.ArrayLike) -> dict[str, list[float]]:
    """The ``feature_stats`` of training matrix *X*: mean and std of each column.

    Only measured (non-NaN, finite) values count. A column with no
    measurement, or a single distinct value, gets a ``std`` of 0 (not
    checked by :meth:`PortableModel.feature_z_scores`).
    """
    data = np.array(X, dtype=np.float64, ndmin=2)
    means, stds = [], []
    for column in data.T:
        measured = column[np.isfinite(column)]
        means.append(float(measured.mean()) if measured.size else 0.0)
        stds.append(float(measured.std()) if measured.size > 1 else 0.0)
    return {"mean": means, "std": stds}


def write_model(params: Mapping[str, Any], path: str | Path) -> Path:
    """Validate *params* and write them to ``path/model.json``; return the file path."""
    PortableModel(params)
    target = Path(path) / MODEL_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(params, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return target


class _Tree:
    """One decision tree of a tree estimator."""

    def __init__(self, raw: Any, n_features: int, n_classes: int, where: str) -> None:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where} must be an object")
        try:
            self.left = np.asarray(raw["children_left"], dtype=np.int64)
            self.right = np.asarray(raw["children_right"], dtype=np.int64)
            self.feature = np.asarray(raw["feature"], dtype=np.int64)
            self.threshold = np.asarray(raw["threshold"], dtype=np.float64)
            self.value = np.asarray(raw["value"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where}: missing or malformed node arrays ({exc})") from exc
        n = self.left.shape[0] if self.left.ndim == 1 else 0
        if n == 0 or any(
            a.shape != (n,) for a in (self.left, self.right, self.feature, self.threshold)
        ):
            raise ValueError(f"{where}: node arrays must be non-empty and of equal length")
        if self.value.shape != (n, n_classes):
            raise ValueError(f"{where}: value must have one row of {n_classes} per node")
        index = np.arange(n)
        leaf = self.left == -1
        # Children come after their parent (as scikit-learn numbers them),
        # which also guarantees that every walk reaches a leaf.
        if not (
            np.all(leaf == (self.right == -1))
            and np.all(leaf | ((self.left > index) & (self.left < n)))
            and np.all(leaf | ((self.right > index) & (self.right < n)))
            and np.all(leaf | ((self.feature >= 0) & (self.feature < n_features)))
        ):
            raise ValueError(f"{where}: inconsistent tree structure")

    def predict_proba(self, X: _FloatArray) -> _FloatArray:
        rows = np.arange(X.shape[0])
        node: _IntArray = np.zeros(X.shape[0], dtype=np.int64)
        while True:
            inner = self.left[node] != -1
            if not inner.any():
                return self.value[node]
            feature = np.maximum(self.feature[node], 0)
            go_left = X[rows, feature] <= self.threshold[node]
            step = np.where(go_left, self.left[node], self.right[node])
            node = np.where(inner, step, node)


class PortableModel:
    """A trained baseline loaded from its pickle-free JSON parameters.

    Parameters
    ----------
    params:
        The parsed contents of a ``model.json`` file (see the module
        docstring). Raises ``ValueError`` when they are malformed.
    """

    def __init__(self, params: Mapping[str, Any]) -> None:
        if not isinstance(params, Mapping):
            raise ValueError("Model parameters must be a JSON object")
        if params.get("format") != FORMAT or params.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Not a {FORMAT} version {FORMAT_VERSION} model "
                f"(format={params.get('format')!r}, "
                f"format_version={params.get('format_version')!r})"
            )
        self.model_class = str(params.get("model_class", ""))
        self.model_type = str(params.get("model_type", ""))
        self.classes: tuple[str, ...] = _names(params.get("classes"), "classes")
        self.feature_names: tuple[str, ...] = _names(params.get("feature_names"), "feature_names")
        n_features, n_classes = len(self.feature_names), len(self.classes)

        preprocessing = params.get("preprocessing") or {}
        if not isinstance(preprocessing, Mapping):
            raise ValueError("preprocessing must be an object")
        self._fill = _vector(preprocessing.get("fill_values"), n_features, "fill_values")
        self._mean = _vector(preprocessing.get("mean"), n_features, "mean")
        self._scale = _vector(preprocessing.get("scale"), n_features, "scale")
        if self._scale is not None and not np.all(self._scale != 0):
            raise ValueError("scale must not contain zeros")

        stats = params.get("feature_stats")
        self._stats_mean: _FloatArray | None = None
        self._stats_std: _FloatArray | None = None
        if stats is not None:
            if not isinstance(stats, Mapping):
                raise ValueError("feature_stats must be an object")
            self._stats_mean = _vector(stats.get("mean"), n_features, "feature_stats.mean")
            self._stats_std = _vector(stats.get("std"), n_features, "feature_stats.std")
            if self._stats_mean is None or self._stats_std is None:
                raise ValueError("feature_stats needs both mean and std")
            if np.any(self._stats_std < 0):
                raise ValueError("feature_stats.std must not be negative")

        estimator = params.get("estimator")
        if not isinstance(estimator, Mapping):
            raise ValueError("estimator must be an object")
        self._coef: _FloatArray | None = None
        self._intercept: _FloatArray | None = None
        self._trees: list[_Tree] = []
        kind = estimator.get("kind")
        if kind == "linear":
            try:
                coef = np.asarray(estimator["coef"], dtype=np.float64)
                intercept = np.asarray(estimator["intercept"], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed linear estimator ({exc})") from exc
            rows = 1 if n_classes == 2 else n_classes
            if coef.shape != (rows, n_features) or intercept.shape != (rows,):
                raise ValueError(
                    f"A linear estimator for {n_classes} classes and {n_features} features "
                    f"needs coef of shape ({rows}, {n_features}) and {rows} intercept(s)"
                )
            if not (np.all(np.isfinite(coef)) and np.all(np.isfinite(intercept))):
                raise ValueError("Linear estimator parameters must be finite")
            self._coef, self._intercept = coef, intercept
        elif kind == "trees":
            trees = estimator.get("trees")
            if not isinstance(trees, list) or not trees:
                raise ValueError("A tree estimator needs a non-empty list of trees")
            self._trees = [
                _Tree(t, n_features, n_classes, f"tree {i}") for i, t in enumerate(trees)
            ]
        else:
            raise ValueError(f"Unknown estimator kind {kind!r}; expected 'linear' or 'trees'")

    @classmethod
    def load(cls, path: str | Path) -> PortableModel:
        """Load a model from ``model.json`` or from an export directory holding one."""
        path = Path(path)
        if path.is_dir():
            path = path / MODEL_FILE
        if not path.is_file():
            raise FileNotFoundError(f"No portable model found at {path}")
        try:
            params = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read portable model {path}: {exc}") from exc
        try:
            return cls(params)
        except ValueError as exc:
            raise ValueError(f"Invalid portable model {path}: {exc}") from exc

    def predict_proba(self, X: npt.ArrayLike) -> _FloatArray:
        """Class probabilities for each row of *X*, columns in :attr:`classes` order."""
        data = np.array(X, dtype=np.float64, ndmin=2)
        if data.ndim != 2 or data.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected rows of {len(self.feature_names)} features "
                f"({', '.join(self.feature_names)}), got shape {data.shape}"
            )
        if self._fill is not None:
            data = np.where(np.isnan(data), self._fill, data)
        if np.isnan(data).any():
            raise ValueError("Input contains NaN and the model has no fill values")
        if self._mean is not None:
            data = data - self._mean
        if self._scale is not None:
            data = data / self._scale

        if self._coef is not None and self._intercept is not None:
            scores = data @ self._coef.T + self._intercept
            if self._coef.shape[0] == 1:
                positive = 1.0 / (1.0 + np.exp(-scores[:, 0]))
                return np.column_stack([1.0 - positive, positive])
            scores -= scores.max(axis=1, keepdims=True)
            exp = np.exp(scores)
            result: _FloatArray = exp / exp.sum(axis=1, keepdims=True)
            return result
        rounded = data.astype(np.float32).astype(np.float64)
        total = sum(tree.predict_proba(rounded) for tree in self._trees)
        return np.asarray(total, dtype=np.float64) / len(self._trees)

    def predict_labels(self, X: npt.ArrayLike) -> list[str]:
        """The most probable class of each row of *X*."""
        best = np.argmax(self.predict_proba(X), axis=1)
        return [self.classes[i] for i in best]

    @property
    def has_feature_stats(self) -> bool:
        """Whether :meth:`feature_z_scores` can tell how unusual an input is."""
        return self._stats_mean is not None or (
            self._mean is not None and self._scale is not None
        )

    def feature_z_scores(self, X: npt.ArrayLike) -> _FloatArray:
        """How many training standard deviations each value of *X* is from the training mean.

        Signed, one column per feature. ``NaN`` for unmeasured (``NaN``)
        inputs, for features without a spread in training, and for every
        feature of a model without training statistics (see
        :attr:`has_feature_stats`).
        """
        data = np.array(X, dtype=np.float64, ndmin=2)
        if data.ndim != 2 or data.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected rows of {len(self.feature_names)} features, got shape {data.shape}"
            )
        if self._stats_mean is not None and self._stats_std is not None:
            mean, std = self._stats_mean, self._stats_std
        elif self._mean is not None and self._scale is not None:
            mean, std = self._mean, np.abs(self._scale)
        else:
            return np.full(data.shape, np.nan)
        spread = np.where(std > 0, std, np.nan)
        z: _FloatArray = (data - mean) / spread
        return z


def _names(value: Any, what: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) for v in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{what} must be a non-empty list of distinct strings")
    return tuple(value)


def _vector(value: Any, length: int, what: str) -> _FloatArray | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(f"{what} must be a list of {length} numbers")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a list of {length} numbers") from exc
    if vector.shape != (length,) or not all(math.isfinite(v) for v in vector):
        raise ValueError(f"{what} must be a list of {length} finite numbers")
    return vector
