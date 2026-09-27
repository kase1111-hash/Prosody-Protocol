#!/usr/bin/env python3
"""Convert dataset entries to model input format.

Features come from each entry's audio or text, and labels from its
annotations; no feature is derived from a label:

- ``ser``: prosodic features of the entry's recording, measured by
  :class:`~prosody_protocol.ProsodyAnalyzer` (``data.features``), labelled
  with the entry field ``data.label_field`` (default ``emotion_label``).
- ``text_to_prosody``: text features of each token of the entry's IML
  (``data.text_features``), labelled with the prosody markup around the
  token (``data.labels``; see :func:`training.features.token_prosody_labels`).
- ``pitch_contour``: the F0 track of the recording, resampled to
  ``data.sequence_length`` points in semitones, labelled with the
  ``pitch_contour`` of a ``<prosody>`` that covers the entry's whole text.

``ser`` and ``pitch_contour`` analyse audio, which needs the ``audio``
extra; every entry's audio file must exist. Entries that cannot be used
(no voiced speech, no contour annotation, ...) are skipped with a warning.

Writes ``train/``, ``val/`` and ``test/`` (``X.npy``, ``y.npy``) and
``prep_metadata.json`` to the output directory.

Usage:
    python training/scripts/data_prep.py \\
        --config training/configs/ser_logreg.yaml \\
        --dataset datasets/emotional-speech \\
        --output /tmp/prepared_data
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Allow imports from project root (the imports below must follow this)
# ruff: noqa: E402
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from prosody_protocol.datasets import DatasetEntry, DatasetLoader, resolve_audio_path
from prosody_protocol.exceptions import (
    AudioProcessingError,
    ProsodyProtocolError,
    TrainingError,
)
from training.config import load_config
from training.features import (
    CONTOUR_VALUES,
    DEFAULT_SER_FEATURES,
    PROSODY_LABELS,
    TEXT_FEATURES,
    contour_features,
    extract_text_features,
    feature_vector,
    has_voiced_speech,
    recording_features,
    token_prosody_labels,
    utterance_contour_label,
)

if TYPE_CHECKING:
    from prosody_protocol import SpanFeatures
    from prosody_protocol.prosody_analyzer import ProsodyAnalyzer

SPLITS = ("train", "val", "test")

#: Name of the metadata file written next to the splits.
PREP_METADATA = "prep_metadata.json"

_DEFAULT_PROSODY_DIMENSIONS = ("pitch_level", "volume_level", "rate_level")


def prepare_ser_data(
    dataset_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, int]:
    """Prepare data for Speech Emotion Recognition.

    Measures each entry's recording with ProsodyAnalyzer and summarises it
    into the features ``config["features"]`` (see
    :data:`training.features.SER_FEATURES`); unmeasured features are
    ``NaN``. Entries without voiced speech are skipped. Returns the number
    of entries in each split.
    """
    feature_names = list(config.get("features", DEFAULT_SER_FEATURES))
    label_field = config.get("label_field", "emotion_label")
    if label_field not in {f.name for f in fields(DatasetEntry)}:
        raise ValueError(f"label_field {label_field!r} is not a dataset entry field")
    splits = _load_splits(dataset_dir, check_audio=True)
    analyzer = _analyzer()
    skipped = _Skipped()

    arrays = {}
    for split_name, entries in splits.items():
        rows, labels, used = [], [], 0
        for entry in entries:
            label = getattr(entry, label_field)
            if not label:
                skipped.add(f"no {label_field}", entry)
                continue
            span = _measure(analyzer, dataset_dir, entry)
            if not has_voiced_speech([span]):
                skipped.add("no voiced speech in the audio", entry)
                continue
            rows.append(feature_vector([span], feature_names))
            labels.append(label)
            used += 1
        arrays[split_name] = (_matrix(rows, len(feature_names)), labels, used)

    return _write(output_dir, "ser", feature_names, arrays, skipped)


def prepare_text_prosody_data(
    dataset_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, int]:
    """Prepare data for text-to-prosody prediction.

    One row per token of each entry's IML: the text features
    ``config["text_features"]`` of the token, labelled with the prosody
    markup around it on the dimensions ``config["labels"]``. Returns the
    number of entries in each split.
    """
    feature_names = list(config.get("text_features", TEXT_FEATURES))
    dimensions = list(config.get("labels", _DEFAULT_PROSODY_DIMENSIONS))
    unknown = [d for d in dimensions if d not in PROSODY_LABELS]
    if unknown:
        raise ValueError(f"Unknown label dimension(s) {unknown}; available: {list(PROSODY_LABELS)}")
    splits = _load_splits(dataset_dir, check_audio=False)
    skipped = _Skipped()

    arrays = {}
    for split_name, entries in splits.items():
        blocks, labels, used = [], [], 0
        for entry in entries:
            tokens = token_prosody_labels(entry.iml, dimensions)
            if not tokens:
                skipped.add("no text in the IML", entry)
                continue
            blocks.append(extract_text_features([t for t, _ in tokens], feature_names))
            labels.extend(label for _, label in tokens)
            used += 1
        X = np.vstack(blocks) if blocks else np.zeros((0, len(feature_names)))
        arrays[split_name] = (X, labels, used)

    return _write(output_dir, "text_to_prosody", feature_names, arrays, skipped)


def prepare_pitch_contour_data(
    dataset_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, int]:
    """Prepare data for pitch contour classification.

    One row per entry whose IML has a ``pitch_contour`` covering its whole
    text (the label; it must be one of ``config["contour_classes"]``): the
    recording's F0 track as ``config["sequence_length"]`` points in
    semitones relative to its median. Other entries are skipped. Returns
    the number of entries in each split.
    """
    seq_len = int(config.get("sequence_length", 20))
    classes = list(config.get("contour_classes", CONTOUR_VALUES))
    splits = _load_splits(dataset_dir, check_audio=True)
    analyzer = _analyzer()
    skipped = _Skipped()
    feature_names = [f"f0_{i}" for i in range(seq_len)]

    arrays = {}
    for split_name, entries in splits.items():
        rows, labels, used = [], [], 0
        for entry in entries:
            label = utterance_contour_label(entry.iml)
            if label is None:
                skipped.add("no pitch_contour covering the whole utterance", entry)
                continue
            if label not in classes:
                skipped.add(f"pitch_contour {label!r} is not in contour_classes", entry)
                continue
            span = _measure(analyzer, dataset_dir, entry)
            contour = contour_features(span.f0_contour or [], seq_len)
            if contour is None:
                skipped.add("too little voiced speech for a contour", entry)
                continue
            rows.append(contour)
            labels.append(label)
            used += 1
        arrays[split_name] = (_matrix(rows, seq_len), labels, used)

    return _write(output_dir, "pitch_contour", feature_names, arrays, skipped)


def read_prep_metadata(data_dir: Path) -> dict[str, Any]:
    """The ``prep_metadata.json`` of prepared data, or ``{}`` when there is none."""
    path = Path(data_dir) / PREP_METADATA
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Skipped:
    """Entries left out of the prepared data, by reason."""

    def __init__(self) -> None:
        self.by_reason: dict[str, list[str]] = {}

    def add(self, reason: str, entry: DatasetEntry) -> None:
        self.by_reason.setdefault(reason, []).append(entry.id)

    def warn(self) -> None:
        for reason, ids in self.by_reason.items():
            shown = ", ".join(ids[:5]) + (", ..." if len(ids) > 5 else "")
            warnings.warn(
                f"Skipped {len(ids)} entries: {reason} ({shown})", UserWarning, stacklevel=4
            )


def _load_splits(dataset_dir: Path, *, check_audio: bool) -> dict[str, list[DatasetEntry]]:
    """Load and validate a dataset and split it (seed 42, speakers kept apart)."""
    loader = DatasetLoader()
    dataset = loader.load(dataset_dir, check_audio=check_audio)
    return dict(zip(SPLITS, loader.split(dataset), strict=True))


def _analyzer() -> ProsodyAnalyzer:
    try:
        from prosody_protocol.prosody_analyzer import ProsodyAnalyzer
    except ImportError as exc:
        raise TrainingError(
            f"This task measures features from audio, which needs numpy and "
            f"praat-parselmouth (pip install -e '.[ml]' installs them): {exc}"
        ) from exc
    return ProsodyAnalyzer()


def _measure(analyzer: ProsodyAnalyzer, dataset_dir: Path, entry: DatasetEntry) -> SpanFeatures:
    """Features of an entry's whole recording."""
    path = resolve_audio_path(dataset_dir, entry.audio_file)
    try:
        return recording_features(path, entry.transcript, analyzer)
    except AudioProcessingError as exc:
        raise TrainingError(f"Cannot analyse the audio of entry {entry.id!r}: {exc}") from exc


def _matrix(rows: Sequence[np.ndarray], n_features: int) -> np.ndarray:
    return np.vstack(rows) if rows else np.zeros((0, n_features))


def _write(
    output_dir: Path,
    task: str,
    feature_names: list[str],
    arrays: dict[str, tuple[np.ndarray, list[str], int]],
    skipped: _Skipped,
) -> dict[str, int]:
    """Save the splits and ``prep_metadata.json``; return entries per split."""
    output_dir = Path(output_dir)
    for split_name, (X, labels, _) in arrays.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "X.npy", X.astype(np.float64))
        np.save(split_dir / "y.npy", np.array(labels, dtype=np.str_))

    stats = {split_name: used for split_name, (_, _, used) in arrays.items()}
    meta = {
        "task": task,
        "feature_names": feature_names,
        "splits": stats,
        "rows": {split_name: len(labels) for split_name, (_, labels, _) in arrays.items()},
        "skipped": skipped.by_reason,
    }
    with open(output_dir / PREP_METADATA, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    skipped.warn()
    return stats


_TASK_PREPARERS: dict[str, Callable[[Path, dict[str, Any], Path], dict[str, int]]] = {
    "ser": prepare_ser_data,
    "text_to_prosody": prepare_text_prosody_data,
    "pitch_contour": prepare_pitch_contour_data,
}

# Errors reported as a one-line message instead of a traceback.
_USER_ERRORS = (ValueError, FileNotFoundError, ProsodyProtocolError)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", required=True, help="Path to output directory")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = _TASK_PREPARERS[config.task](Path(args.dataset), config.data, output_dir)
    except _USER_ERRORS as exc:
        parser.exit(1, f"Error: {exc}\n")

    meta = read_prep_metadata(output_dir)
    print(f"Data prepared for task '{config.task}' in {output_dir}:")
    for split, count in stats.items():
        print(f"  {split}: {count} entries, {meta['rows'][split]} rows")
    print(f"  features: {', '.join(meta['feature_names'])}")


if __name__ == "__main__":
    main()
