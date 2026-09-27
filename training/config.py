"""YAML configuration loading and validation for training pipelines.

A config has these sections (see ``training/README.md`` for a full
reference):

``task``
    ``ser``, ``text_to_prosody`` or ``pitch_contour``.
``model``
    ``type`` (``logistic_regression``, ``decision_tree`` or
    ``random_forest``), optionally ``labels`` (the classes the data may
    contain), and hyperparameters.
``training``
    Hyperparameters. A hyperparameter may be given in ``model`` or in
    ``training``, not both.
``data``
    Task-specific data preparation settings.
``evaluation``
    ``metrics`` and ``average`` of the report; only ``macro`` averaging is
    implemented.
``output``
    ``checkpoint_dir`` (default ``--output`` of ``train.py``) and
    ``export_format`` (default ``--format`` of ``export.py``).

Hyperparameters per model type (all optional; scikit-learn's meaning):

- ``logistic_regression``: ``C`` (inverse regularisation strength; the old
  name ``regularization`` is accepted), ``max_iter``, ``solver`` (the old
  name ``optimizer`` is accepted), ``class_weight``, ``random_state``
- ``decision_tree``: ``max_depth``, ``min_samples_split``,
  ``min_samples_leaf``, ``class_weight``, ``random_state``
- ``random_forest``: the ``decision_tree`` ones and ``n_estimators``

Invalid values raise ``ValueError``. Keys the pipeline does not use --
unknown keys, and keys such as ``epochs`` or ``learning_rate`` that the
scikit-learn baselines have no use for -- are ignored with a
:class:`UserWarning` naming them, so a setting never silently does nothing.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from prosody_protocol.datasets import DatasetEntry

from .features import CONTOUR_VALUES, PROSODY_LABELS, SER_FEATURES, TEXT_FEATURES

#: The tasks a config can train.
TASKS: tuple[str, ...] = ("ser", "text_to_prosody", "pitch_contour")

#: Hyperparameters each model type accepts (in ``model`` or ``training``).
HYPERPARAMETERS: dict[str, frozenset[str]] = {
    "logistic_regression": frozenset(
        {"C", "max_iter", "solver", "class_weight", "random_state"}
    ),
    "decision_tree": frozenset(
        {"max_depth", "min_samples_split", "min_samples_leaf", "class_weight", "random_state"}
    ),
    "random_forest": frozenset(
        {
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "class_weight", "random_state",
        }
    ),
}

# Earlier names of hyperparameters, still accepted.
_ALIASES = {"regularization": "C", "optimizer": "solver"}

_SOLVERS = frozenset({"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"})

# Keys of earlier configs that the scikit-learn baselines cannot honour.
_NOT_USED = {
    "epochs": "the scikit-learn baselines are not trained in epochs",
    "learning_rate": "the scikit-learn baselines have no learning rate",
    "batch_size": "the scikit-learn baselines fit the whole training set at once",
    "num_classes": "the classes are those of the training data",
    "format": "checkpoints are always written as model.joblib",
}

_REQUIRED_KEYS = {"task", "model", "data", "training", "evaluation", "output"}
_TOP_LEVEL_KEYS = _REQUIRED_KEYS | {"description"}
_MODEL_KEYS = frozenset({"type", "labels"})
_DATA_KEYS = {
    "ser": frozenset({"features", "label_field"}),
    "text_to_prosody": frozenset({"text_features", "labels"}),
    "pitch_contour": frozenset({"sequence_length", "contour_classes"}),
}
_EVALUATION_KEYS = frozenset({"metrics", "average"})
_OUTPUT_KEYS = frozenset({"checkpoint_dir", "export_format"})

_METRICS = ("precision", "recall", "f1")
_EXPORT_FORMATS = ("json", "pickle")
_LABEL_FIELDS = frozenset(f.name for f in fields(DatasetEntry)) - {"consent", "metadata"}


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
        return str(self.model.get("type", "unknown"))

    @property
    def labels(self) -> list[str]:
        """Get the class labels for the task."""
        if "labels" in self.model:
            return list(self.model["labels"])
        if "contour_classes" in self.data:
            return list(self.data["contour_classes"])
        return []

    @property
    def metrics(self) -> list[str]:
        return list(self.evaluation.get("metrics", list(_METRICS)))

    @property
    def average(self) -> str:
        return str(self.evaluation.get("average", "macro"))

    def model_params(self) -> dict[str, Any]:
        """Arguments for :meth:`ModelRegistry.create <training.models.ModelRegistry.create>`.

        ``type`` plus the hyperparameters of the ``model`` and ``training``
        sections under their current names; ignored keys are left out.
        """
        allowed = HYPERPARAMETERS[self.model_type]
        params: dict[str, Any] = {"type": self.model_type}
        for section in (self.model, self.training):
            for key, value in section.items():
                name = _ALIASES.get(key, key)
                if name in allowed:
                    params[name] = value
        return params


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
        If the config is missing required keys or has an invalid value.

    Keys the pipeline does not use are reported with a :class:`UserWarning`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")

    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    ignored = _validate(raw)
    for message in ignored:
        warnings.warn(f"{path}: {message}", UserWarning, stacklevel=2)

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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(raw: dict[str, Any]) -> list[str]:
    """Check *raw*; raise ``ValueError`` for invalid values, return ignored-key notes."""
    notes: list[str] = []
    for section in ("model", "data", "training", "evaluation", "output"):
        if not isinstance(raw[section], dict):
            raise ValueError(f"'{section}' must be a mapping, got {type(raw[section]).__name__}")

    task = raw["task"]
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Available: {list(TASKS)}")
    model, training = raw["model"], raw["training"]
    model_type = model.get("type")
    if model_type not in HYPERPARAMETERS:
        raise ValueError(
            f"Unknown model type {model_type!r}. Available: {sorted(HYPERPARAMETERS)} "
            "(only these scikit-learn baselines are implemented)"
        )

    notes += _unused(set(raw) - _TOP_LEVEL_KEYS, "top level")
    if "description" in raw and not isinstance(raw["description"], str):
        raise ValueError("'description' must be a string")

    # Hyperparameters, in either section.
    allowed = HYPERPARAMETERS[model_type]
    seen: dict[str, str] = {}
    for section_name, section, extra in (("model", model, _MODEL_KEYS), ("training", training, ())):
        unknown = []
        for key, value in section.items():
            name = _ALIASES.get(key, key)
            if key in extra:
                continue
            if name not in allowed:
                unknown.append(key)
                continue
            if name in seen:
                raise ValueError(
                    f"'{section_name}.{key}' sets {name!r}, which "
                    f"'{seen[name]}' already sets"
                )
            seen[name] = f"{section_name}.{key}"
            _CHECKS[name](f"{section_name}.{key}", value)
        notes += _unused(unknown, f"'{section_name}' for model type {model_type!r}")
    if "labels" in model:
        _check_names("model.labels", model["labels"], None)

    data = raw["data"]
    notes += _unused(set(data) - _DATA_KEYS[task], f"'data' for task {task!r}")
    if task == "ser":
        if "features" in data:
            _check_names("data.features", data["features"], tuple(SER_FEATURES))
        field_name = data.get("label_field", "emotion_label")
        if field_name not in _LABEL_FIELDS:
            raise ValueError(
                f"data.label_field must be a dataset entry field, one of "
                f"{sorted(_LABEL_FIELDS)}; got {field_name!r}"
            )
    elif task == "text_to_prosody":
        if "text_features" in data:
            _check_names("data.text_features", data["text_features"], TEXT_FEATURES)
        if "labels" in data:
            _check_prosody_labels(data["labels"])
    else:
        if "sequence_length" in data:
            _check_int("data.sequence_length", data["sequence_length"], minimum=2)
        if "contour_classes" in data:
            _check_names("data.contour_classes", data["contour_classes"], CONTOUR_VALUES)

    evaluation = raw["evaluation"]
    notes += _unused(set(evaluation) - _EVALUATION_KEYS, "'evaluation'")
    if "metrics" in evaluation:
        _check_names("evaluation.metrics", evaluation["metrics"], _METRICS)
    if evaluation.get("average", "macro") != "macro":
        raise ValueError(
            f"evaluation.average: only 'macro' is implemented, got {evaluation['average']!r}"
        )

    output = raw["output"]
    notes += _unused(set(output) - _OUTPUT_KEYS, "'output'")
    if "checkpoint_dir" in output and not isinstance(output["checkpoint_dir"], str):
        raise ValueError("output.checkpoint_dir must be a path string")
    if output.get("export_format", "json") not in _EXPORT_FORMATS:
        raise ValueError(
            f"output.export_format must be one of {list(_EXPORT_FORMATS)}, "
            f"got {output['export_format']!r}"
        )
    return notes


def _unused(keys: Any, where: str) -> list[str]:
    """Notes for keys in *where* that are ignored."""
    notes = []
    for key in sorted(keys):
        reason = _NOT_USED.get(key, "unknown key")
        notes.append(f"'{key}' in {where} is ignored ({reason})")
    return notes


def _check_positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")


def _check_int(name: str, value: Any, *, minimum: int, optional: bool = False) -> None:
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        allowed = " or null" if optional else ""
        raise ValueError(f"{name} must be an integer >= {minimum}{allowed}, got {value!r}")


def _check_choice(name: str, value: Any, choices: frozenset[str | None]) -> None:
    if value not in choices:
        shown = sorted("null" if c is None else c for c in choices)
        raise ValueError(f"{name} must be one of {shown}, got {value!r}")


def _check_names(name: str, value: Any, vocabulary: tuple[str, ...] | None) -> None:
    """A non-empty list of distinct strings, all in *vocabulary* when given."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) for v in value)
    ):
        raise ValueError(f"{name} must be a non-empty list of names, got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} lists a name twice: {value}")
    if vocabulary is not None:
        unknown = [v for v in value if v not in vocabulary]
        if unknown:
            raise ValueError(f"{name}: unknown name(s) {unknown}; available: {list(vocabulary)}")


def _check_prosody_labels(value: Any) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"data.labels must map label dimensions to values, got {value!r}")
    for dimension, values in value.items():
        if dimension not in PROSODY_LABELS:
            raise ValueError(
                f"data.labels: unknown dimension {dimension!r}; "
                f"available: {list(PROSODY_LABELS)}"
            )
        expected = PROSODY_LABELS[dimension]
        if not isinstance(values, list) or sorted(values) != sorted(expected):
            raise ValueError(
                f"data.labels.{dimension} must list the values {list(expected)} "
                f"(the labels read from IML markup), got {values!r}"
            )


_CHECKS: dict[str, Callable[[str, Any], None]] = {
    "C": _check_positive,
    "max_iter": lambda n, v: _check_int(n, v, minimum=1),
    "n_estimators": lambda n, v: _check_int(n, v, minimum=1),
    "max_depth": lambda n, v: _check_int(n, v, minimum=1, optional=True),
    "min_samples_split": lambda n, v: _check_int(n, v, minimum=2),
    "min_samples_leaf": lambda n, v: _check_int(n, v, minimum=1),
    "random_state": lambda n, v: _check_int(n, v, minimum=0, optional=True),
    "solver": lambda n, v: _check_choice(n, v, frozenset(_SOLVERS)),
    "class_weight": lambda n, v: _check_choice(n, v, frozenset({None, "balanced"})),
}
