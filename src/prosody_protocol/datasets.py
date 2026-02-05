"""Dataset infrastructure for loading, validating, and splitting training data.

Each dataset lives in a directory with the structure::

    dataset-name/
    ├── metadata.json
    ├── entries/          # one JSON file per annotated entry
    ├── audio/            # WAV files referenced by entries
    └── README.md

See EXECUTION_GUIDE.md Phase 10.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .exceptions import DatasetError
from .validator import IMLValidator, ValidationIssue, ValidationResult

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

_VALID_SOURCES = frozenset({"mavis", "recorded", "synthetic"})
_VALID_ANNOTATORS = frozenset({"human", "model", "hybrid"})
_ISO_8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
)
_BCP47_RE = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z0-9]+)*$")


@dataclass(frozen=True)
class DatasetEntry:
    """A single annotated dataset entry."""

    id: str
    timestamp: str
    source: str
    language: str
    audio_file: str
    transcript: str
    iml: str
    emotion_label: str
    annotator: str
    consent: bool
    speaker_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Dataset:
    """A loaded dataset with its entries and metadata."""

    name: str
    entries: list[DatasetEntry]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------


class DatasetLoader:
    """Load, validate, and split training datasets."""

    def __init__(self, validate_iml: bool = True) -> None:
        self._validate_iml = validate_iml
        self._iml_validator = IMLValidator() if validate_iml else None

    def load(self, dataset_dir: str | Path) -> Dataset:
        """Load all entries from a dataset directory.

        Raises :class:`~prosody_protocol.exceptions.DatasetError`
        if the directory structure is invalid.
        """
        path = Path(dataset_dir)
        if not path.is_dir():
            raise DatasetError(f"Dataset directory does not exist: {path}")

        entries_dir = path / "entries"
        if not entries_dir.is_dir():
            raise DatasetError(f"Missing entries/ directory in {path}")

        # Load optional metadata.
        meta: dict[str, object] = {}
        meta_file = path / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise DatasetError(f"Cannot read metadata.json: {exc}") from exc

        entries: list[DatasetEntry] = []
        for entry_file in sorted(entries_dir.glob("*.json")):
            try:
                raw = json.loads(entry_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise DatasetError(
                    f"Cannot read entry {entry_file.name}: {exc}"
                ) from exc
            entries.append(self._parse_entry(raw, entry_file.name))

        return Dataset(
            name=path.name,
            entries=entries,
            metadata=meta,
        )

    def iter_entries(self, dataset_dir: str | Path) -> Iterator[DatasetEntry]:
        """Lazily iterate over entries without loading all into memory."""
        path = Path(dataset_dir)
        entries_dir = path / "entries"
        if not entries_dir.is_dir():
            raise DatasetError(f"Missing entries/ directory in {path}")

        for entry_file in sorted(entries_dir.glob("*.json")):
            try:
                raw = json.loads(entry_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise DatasetError(
                    f"Cannot read entry {entry_file.name}: {exc}"
                ) from exc
            yield self._parse_entry(raw, entry_file.name)

    def validate_entry(
        self,
        entry: dict[str, object],
        dataset_dir: Path | None = None,
    ) -> ValidationResult:
        """Validate a single entry dict against the dataset schema.

        If *dataset_dir* is provided, also checks that the referenced
        audio file exists on disk.
        """
        issues: list[ValidationIssue] = []

        # Required string fields.
        for fld in ("id", "timestamp", "source", "language",
                     "audio_file", "transcript", "iml",
                     "emotion_label", "annotator"):
            val = entry.get(fld)
            if not isinstance(val, str) or not val.strip():
                issues.append(ValidationIssue(
                    severity="error",
                    rule="D1",
                    message=f"Missing or empty required field '{fld}'",
                ))

        # consent must be true.
        consent = entry.get("consent")
        if consent is not True:
            issues.append(ValidationIssue(
                severity="error",
                rule="D2",
                message="'consent' must be true",
            ))

        # source enum.
        source = entry.get("source", "")
        if isinstance(source, str) and source and source not in _VALID_SOURCES:
            issues.append(ValidationIssue(
                severity="error",
                rule="D3",
                message=f"'source' must be one of {sorted(_VALID_SOURCES)}, got '{source}'",
            ))

        # annotator enum.
        annotator = entry.get("annotator", "")
        if isinstance(annotator, str) and annotator and annotator not in _VALID_ANNOTATORS:
            issues.append(ValidationIssue(
                severity="error",
                rule="D4",
                message=(
                    f"'annotator' must be one of {sorted(_VALID_ANNOTATORS)}, "
                    f"got '{annotator}'"
                ),
            ))

        # timestamp format.
        timestamp = entry.get("timestamp", "")
        if isinstance(timestamp, str) and timestamp and not _ISO_8601_RE.match(timestamp):
            issues.append(ValidationIssue(
                severity="warning",
                rule="D5",
                message=f"'timestamp' does not look like ISO-8601: '{timestamp}'",
            ))

        # language BCP-47 format.
        language = entry.get("language", "")
        if isinstance(language, str) and language and not _BCP47_RE.match(language):
            issues.append(ValidationIssue(
                severity="warning",
                rule="D6",
                message=f"'language' does not match BCP-47 pattern: '{language}'",
            ))

        # IML validity.
        iml = entry.get("iml", "")
        if isinstance(iml, str) and iml and self._iml_validator is not None:
            iml_result = self._iml_validator.validate(iml)
            if not iml_result.valid:
                for iml_issue in iml_result.issues:
                    if iml_issue.severity == "error":
                        issues.append(ValidationIssue(
                            severity="error",
                            rule="D7",
                            message=f"IML validation: {iml_issue.message}",
                        ))

        # Audio file existence.
        audio_file = entry.get("audio_file", "")
        if isinstance(audio_file, str) and audio_file and dataset_dir is not None:
            audio_path = Path(dataset_dir) / audio_file
            if not audio_path.exists():
                issues.append(ValidationIssue(
                    severity="error",
                    rule="D8",
                    message=f"Audio file not found: {audio_file}",
                ))

        valid = not any(i.severity == "error" for i in issues)
        return ValidationResult(valid=valid, issues=issues)

    def split(
        self,
        dataset: Dataset,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[DatasetEntry], list[DatasetEntry], list[DatasetEntry]]:
        """Split dataset entries into train/val/test sets.

        The split is deterministic given the same seed.

        Returns (train_entries, val_entries, test_entries).
        """
        if abs((train + val + test) - 1.0) > 1e-6:
            raise DatasetError(
                f"Split ratios must sum to 1.0, got {train + val + test:.4f}"
            )

        entries = list(dataset.entries)
        rng = random.Random(seed)
        rng.shuffle(entries)

        n = len(entries)
        n_train = int(n * train)
        n_val = int(n * val)

        train_set = entries[:n_train]
        val_set = entries[n_train:n_train + n_val]
        test_set = entries[n_train + n_val:]

        return (train_set, val_set, test_set)

    @staticmethod
    def _parse_entry(raw: dict[str, object], filename: str) -> DatasetEntry:
        """Parse a raw JSON dict into a DatasetEntry."""
        try:
            return DatasetEntry(
                id=str(raw.get("id", "")),
                timestamp=str(raw.get("timestamp", "")),
                source=str(raw.get("source", "")),
                language=str(raw.get("language", "")),
                audio_file=str(raw.get("audio_file", "")),
                transcript=str(raw.get("transcript", "")),
                iml=str(raw.get("iml", "")),
                emotion_label=str(raw.get("emotion_label", "")),
                annotator=str(raw.get("annotator", "")),
                consent=raw.get("consent") is True,
                speaker_id=raw.get("speaker_id") if isinstance(raw.get("speaker_id"), str) else None,
                metadata=raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {},
            )
        except Exception as exc:
            raise DatasetError(f"Cannot parse entry {filename}: {exc}") from exc
