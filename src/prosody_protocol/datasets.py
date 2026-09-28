"""Dataset infrastructure for loading, validating, and splitting training data.

Each dataset lives in a directory with the structure::

    dataset-name/
    ├── metadata.json
    ├── entries/          # one JSON file per annotated entry
    ├── audio/            # WAV files referenced by entries
    └── README.md

Entries follow ``schemas/dataset-entry.schema.json``. :class:`DatasetLoader`
checks them against these rules (the same constraints as the schema, plus
IML validity and audio existence):

  D1  required string field missing, not a string, or empty    ERROR
  D2  consent is not true (spec 8.1)                            ERROR
  D3  source is not mavis, recorded or synthetic                ERROR
  D4  annotator is not human, model or hybrid                   ERROR
  D5  timestamp does not look like ISO-8601                     WARNING
  D6  language is not a BCP-47 tag                              ERROR
  D7  iml fails IML validation                                  ERROR
  D8  audio file does not exist (only with a dataset directory) ERROR
  D9  audio_file is absolute, leaves the dataset directory, or
      contains a control character                              ERROR
  D10 another entry has the same id (load only)                 ERROR
  D11 unknown field                                             ERROR
  D12 entry is not an object, or speaker_id/metadata has the
      wrong type                                                ERROR

See EXECUTION_GUIDE.md Phase 10.
"""

from __future__ import annotations

import json
import math
import random
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import cast

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
_BCP47_RE = re.compile(r"[a-zA-Z]{2,3}(-[a-zA-Z0-9]+)*")

_REQUIRED_STR_FIELDS = (
    "id", "timestamp", "source", "language", "audio_file",
    "transcript", "iml", "emotion_label", "annotator",
)
_KNOWN_FIELDS = frozenset(_REQUIRED_STR_FIELDS) | {"consent", "speaker_id", "metadata"}

# An absolute path: POSIX root, Windows drive, or UNC/backslash root.
_ABSOLUTE_PATH_RE = re.compile(r"[/\\]|[A-Za-z]:")
_PATH_SEPARATOR_RE = re.compile(r"[/\\]")
# No file system can open a path with a NUL byte, and newlines let a path
# slip past line-oriented checks (the schema's pattern included).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")

# How many problems a DatasetError or warning lists before summarising.
_MAX_REPORTED_PROBLEMS = 20


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
    """A loaded dataset with its entries and metadata.

    ``root`` is the directory the dataset was loaded from (set by
    :meth:`DatasetLoader.load`); entries' ``audio_file`` paths are relative
    to it. It is ``None`` for datasets built in memory.
    """

    name: str
    entries: list[DatasetEntry]
    metadata: dict[str, object] = field(default_factory=dict)
    root: Path | None = None

    @property
    def size(self) -> int:
        return len(self.entries)


_GROUPABLE_FIELDS = frozenset(f.name for f in fields(DatasetEntry)) - {"metadata"}


def resolve_audio_path(dataset_dir: str | Path, audio_file: str) -> Path:
    """Return the path of an entry's audio file inside *dataset_dir*.

    Raises :class:`~prosody_protocol.exceptions.DatasetError` when
    *audio_file* is absolute (``/x``, ``C:\\x``, ``\\\\host\\x``), has a
    ``..`` segment or a control character, or resolves (through symlinks)
    outside *dataset_dir*. Entries name audio relative to their dataset, so
    such a path can only point at files that are not part of it.
    """
    if not _is_contained_path(audio_file):
        raise DatasetError(
            f"audio_file {audio_file!r} must be a relative path inside the dataset "
            "directory, without control characters"
        )
    root = Path(dataset_dir)
    path = root / audio_file
    try:
        contained = path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:  # e.g. a symlink loop
        raise DatasetError(f"audio_file {audio_file!r} cannot be resolved: {exc}") from exc
    if not contained:
        raise DatasetError(f"audio_file {audio_file!r} resolves outside the dataset directory")
    return path


def _is_contained_path(audio_file: str) -> bool:
    """Whether *audio_file* is relative, has no ``..`` segment and no
    control character."""
    if _ABSOLUTE_PATH_RE.match(audio_file) or _CONTROL_CHAR_RE.search(audio_file):
        return False
    return ".." not in _PATH_SEPARATOR_RE.split(audio_file)


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------


class DatasetLoader:
    """Load, validate, and split training datasets.

    Parameters
    ----------
    validate_iml:
        Check each entry's ``iml`` with :class:`IMLValidator` (rule D7).
    strict:
        What :meth:`load` and :meth:`iter_entries` do with entries that fail
        validation. ``True`` (the default) raises
        :class:`~prosody_protocol.exceptions.DatasetError` listing every
        problem; ``False`` skips those entries with a :class:`UserWarning`.
        Either way, no entry without ``consent: true`` is ever returned.
    """

    def __init__(self, validate_iml: bool = True, *, strict: bool = True) -> None:
        self._validate_iml = validate_iml
        self._strict = strict
        self._iml_validator = IMLValidator() if validate_iml else None

    def load(self, dataset_dir: str | Path, *, check_audio: bool = False) -> Dataset:
        """Load and validate all entries from a dataset directory.

        Every entry is checked with :meth:`validate_entry` and for duplicate
        ids (D10). *check_audio* also requires each referenced audio file to
        exist (D8); leave it off for datasets shipped without audio.

        Raises :class:`~prosody_protocol.exceptions.DatasetError` if the
        directory structure is invalid, an entry file cannot be read, or (in
        strict mode) any entry is invalid. Warns when ``metadata.json``
        declares a ``size`` that differs from the number of entries loaded.
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
                loaded = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                raise DatasetError(f"Cannot read metadata.json: {exc}") from exc
            if not isinstance(loaded, dict):
                raise DatasetError("metadata.json must contain a JSON object")
            meta = loaded

        entries: list[DatasetEntry] = []
        problems: list[str] = []
        seen_ids: dict[str, str] = {}
        entry_files = sorted(entries_dir.glob("*.json"))
        for entry_file in entry_files:
            raw = self._read_entry(entry_file)
            result = self._check_entry(raw, path if check_audio else None, seen_ids, entry_file)
            if result.errors:
                problems.extend(_describe(entry_file.name, result.errors))
                continue
            entries.append(self._parse_entry(raw, entry_file.name))

        if problems:
            skipped = len(entry_files) - len(entries)
            summary = f"{skipped} of {len(entry_files)} entries in {path} failed validation"
            if self._strict:
                raise DatasetError(
                    f"{summary} (load with DatasetLoader(strict=False) to skip them):"
                    f"{_format_problems(problems)}"
                )
            warnings.warn(
                f"Skipped invalid entries: {summary}:{_format_problems(problems)}",
                UserWarning,
                stacklevel=2,
            )

        declared = meta.get("size")
        if (
            isinstance(declared, int)
            and not isinstance(declared, bool)
            and declared != len(entries)
        ):
            warnings.warn(
                f"{path / 'metadata.json'} declares size {declared}, "
                f"but {len(entries)} entries were loaded",
                UserWarning,
                stacklevel=2,
            )

        return Dataset(
            name=path.name,
            entries=entries,
            metadata=meta,
            root=path,
        )

    def iter_entries(
        self, dataset_dir: str | Path, *, check_audio: bool = False
    ) -> Iterator[DatasetEntry]:
        """Lazily iterate over validated entries without loading all into memory.

        Entries are checked as in :meth:`load`. In strict mode the iteration
        raises :class:`~prosody_protocol.exceptions.DatasetError` at the first
        invalid entry; otherwise invalid entries are skipped with a warning.
        """
        path = Path(dataset_dir)
        entries_dir = path / "entries"
        if not entries_dir.is_dir():
            raise DatasetError(f"Missing entries/ directory in {path}")

        seen_ids: dict[str, str] = {}
        for entry_file in sorted(entries_dir.glob("*.json")):
            raw = self._read_entry(entry_file)
            result = self._check_entry(raw, path if check_audio else None, seen_ids, entry_file)
            if result.errors:
                problems = _format_problems(_describe(entry_file.name, result.errors))
                if self._strict:
                    raise DatasetError(f"Invalid entry in {path}:{problems}")
                warnings.warn(f"Skipped invalid entry:{problems}", UserWarning, stacklevel=2)
                continue
            yield self._parse_entry(raw, entry_file.name)

    def validate_entry(
        self,
        entry: dict[str, object],
        dataset_dir: Path | None = None,
    ) -> ValidationResult:
        """Validate a single entry dict against the dataset schema.

        Applies rules D1-D9 and D11-D12 (see the module docstring). If
        *dataset_dir* is provided, also checks that the referenced audio
        file exists on disk inside it.
        """
        issues: list[ValidationIssue] = []

        if not isinstance(entry, dict):
            issues.append(ValidationIssue(
                severity="error",
                rule="D12",
                message=f"Entry must be a JSON object, got {type(entry).__name__}",
            ))
            return ValidationResult(valid=False, issues=issues)

        # Required string fields.
        for fld in _REQUIRED_STR_FIELDS:
            val = entry.get(fld)
            if val is not None and not isinstance(val, str):
                message = f"Required field '{fld}' must be a string, got {type(val).__name__}"
            elif not isinstance(val, str) or not val.strip():
                message = f"Missing or empty required field '{fld}'"
            else:
                continue
            issues.append(ValidationIssue(severity="error", rule="D1", message=message))

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
        if isinstance(language, str) and language and not _BCP47_RE.fullmatch(language):
            issues.append(ValidationIssue(
                severity="error",
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

        # Audio file location and existence.
        audio_file = entry.get("audio_file", "")
        if isinstance(audio_file, str) and audio_file.strip():
            audio_path: Path | None = None
            contained = _is_contained_path(audio_file)
            if contained and dataset_dir is not None:
                try:
                    audio_path = resolve_audio_path(dataset_dir, audio_file)
                except DatasetError:  # a symlink pointing out of the dataset
                    contained = False
            if not contained:
                issues.append(ValidationIssue(
                    severity="error",
                    rule="D9",
                    message=(
                        "audio_file must be a relative path inside the dataset directory, "
                        f"without control characters, got {audio_file!r}"
                    ),
                ))
            elif audio_path is not None and not audio_path.exists():
                issues.append(ValidationIssue(
                    severity="error",
                    rule="D8",
                    message=f"Audio file not found: {audio_file}",
                ))

        # Unknown fields (the schema allows no additional properties).
        for key in entry:
            if key not in _KNOWN_FIELDS:
                issues.append(ValidationIssue(
                    severity="error",
                    rule="D11",
                    message=f"Unknown field '{key}' (put extra data under 'metadata')",
                ))

        # Optional fields.
        speaker_id = entry.get("speaker_id")
        if speaker_id is not None and not isinstance(speaker_id, str):
            issues.append(ValidationIssue(
                severity="error",
                rule="D12",
                message=f"'speaker_id' must be a string or null, got {type(speaker_id).__name__}",
            ))
        if "metadata" in entry and not isinstance(entry["metadata"], dict):
            issues.append(ValidationIssue(
                severity="error",
                rule="D12",
                message=f"'metadata' must be an object, got {type(entry['metadata']).__name__}",
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
        *,
        group_by: str | None = "speaker_id",
    ) -> tuple[list[DatasetEntry], list[DatasetEntry], list[DatasetEntry]]:
        """Split dataset entries into train/val/test sets.

        Entries are shuffled with *seed*, so the split is deterministic for a
        given dataset and seed. Split sizes follow the ratios by
        largest-remainder rounding, and every split with a non-zero ratio gets
        at least one entry when there are enough entries (5 entries at
        0.8/0.1/0.1 split 3/1/1).

        With *group_by* (default ``"speaker_id"``), entries sharing a value of
        that field always land in the same split, so no speaker is both
        trained and evaluated on; entries whose value is ``None`` are placed
        individually. Groups are placed largest first, each in the split
        furthest below its target size, so every split ends up within about
        one group's size of its target however unevenly the speakers are
        represented. When there are fewer groups than splits to fill (e.g. a
        single-speaker dataset), a group-disjoint split is impossible: the
        entries are split individually and a :class:`UserWarning` says so.
        ``group_by=None`` always splits entries individually.

        Returns (train_entries, val_entries, test_entries).
        """
        ratios = (train, val, test)
        for name, ratio in zip(("train", "val", "test"), ratios, strict=True):
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not 0.0 <= ratio <= 1.0
            ):
                raise DatasetError(f"Split ratio '{name}' must be between 0 and 1, got {ratio!r}")
        if abs((train + val + test) - 1.0) > 1e-6:
            raise DatasetError(
                f"Split ratios must sum to 1.0, got {train + val + test:.4f}"
            )
        if group_by is not None and group_by not in _GROUPABLE_FIELDS:
            raise DatasetError(
                f"Cannot group by '{group_by}'; expected one of {sorted(_GROUPABLE_FIELDS)}"
            )

        entries = list(dataset.entries)
        wanted = sum(1 for ratio in ratios if ratio > 0)
        groups = _group_entries(entries, group_by)
        if len(groups) < wanted <= len(entries):
            warnings.warn(
                f"Cannot split {len(entries)} entries into {wanted} {group_by}-disjoint "
                f"sets: they have only {len(groups)} distinct {group_by} value(s). "
                f"Splitting entries individually; evaluation sets share {group_by}s "
                "with the training set.",
                UserWarning,
                stacklevel=2,
            )
            groups = [[entry] for entry in entries]

        rng = random.Random(seed)
        rng.shuffle(groups)
        targets = _split_sizes(len(entries), ratios)
        parts = _assign_groups(groups, targets, ratios)
        train_set, val_set, test_set = ([e for g in part for e in g] for part in parts)
        return (train_set, val_set, test_set)

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _read_entry(entry_file: Path) -> object:
        try:
            return json.loads(entry_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise DatasetError(
                f"Cannot read entry {entry_file.name}: {exc}"
            ) from exc

    def _check_entry(
        self,
        raw: object,
        dataset_dir: Path | None,
        seen_ids: dict[str, str],
        entry_file: Path,
    ) -> ValidationResult:
        """Validate one raw entry and record its id for duplicate detection."""
        # validate_entry reports an entry that is not a JSON object (D12).
        result = self.validate_entry(cast("dict[str, object]", raw), dataset_dir)
        if isinstance(raw, dict):
            entry_id = raw.get("id")
            if isinstance(entry_id, str) and entry_id:
                first = seen_ids.setdefault(entry_id, entry_file.name)
                if first != entry_file.name:
                    result.issues.append(ValidationIssue(
                        severity="error",
                        rule="D10",
                        message=f"Duplicate id '{entry_id}' (first used in {first})",
                    ))
                    result.valid = False
        return result

    @staticmethod
    def _parse_entry(raw: object, filename: str) -> DatasetEntry:
        """Build a DatasetEntry from a raw JSON dict that passed validation.

        Raises DatasetError if required fields are missing or mistyped.
        """
        if not isinstance(raw, dict):
            raise DatasetError(f"Entry {filename} must be a JSON object")
        missing = [f for f in _REQUIRED_STR_FIELDS if not isinstance(raw.get(f), str)]
        if missing:
            raise DatasetError(
                f"Entry {filename} is missing required fields: {', '.join(missing)}"
            )

        speaker_id = raw.get("speaker_id")
        metadata = raw.get("metadata")
        return DatasetEntry(
            id=raw["id"],
            timestamp=raw["timestamp"],
            source=raw["source"],
            language=raw["language"],
            audio_file=raw["audio_file"],
            transcript=raw["transcript"],
            iml=raw["iml"],
            emotion_label=raw["emotion_label"],
            annotator=raw["annotator"],
            consent=raw.get("consent") is True,
            speaker_id=speaker_id if isinstance(speaker_id, str) else None,
            metadata=metadata if isinstance(metadata, dict) else {},
        )


def _describe(filename: str, issues: list[ValidationIssue]) -> list[str]:
    return [f"{filename}: {issue.rule} {issue.message}" for issue in issues]


def _format_problems(problems: list[str]) -> str:
    shown = "".join(f"\n  {p}" for p in problems[:_MAX_REPORTED_PROBLEMS])
    if len(problems) > _MAX_REPORTED_PROBLEMS:
        shown += f"\n  ... and {len(problems) - _MAX_REPORTED_PROBLEMS} more"
    return shown


def _group_entries(
    entries: list[DatasetEntry], group_by: str | None
) -> list[list[DatasetEntry]]:
    """Entries grouped by *group_by*, in order of first appearance."""
    if group_by is None:
        return [[entry] for entry in entries]
    groups: dict[object, list[DatasetEntry]] = {}
    for index, entry in enumerate(entries):
        key = getattr(entry, group_by)
        groups.setdefault(("entry", index) if key is None else ("value", key), []).append(entry)
    return list(groups.values())


def _split_sizes(n: int, ratios: tuple[float, float, float]) -> list[int]:
    """Entry counts per split: largest-remainder rounding of ``n * ratio``,
    then at least one entry for every non-zero ratio when ``n`` allows."""
    exact = [n * ratio for ratio in ratios]
    sizes = [math.floor(x) for x in exact]
    by_remainder = sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))
    for i in by_remainder[: n - sum(sizes)]:
        sizes[i] += 1
    wanted = [i for i in range(3) if ratios[i] > 0]
    if n >= len(wanted):
        for i in wanted:
            if sizes[i] == 0:
                donor = max(range(3), key=lambda j: (sizes[j], -j))
                sizes[donor] -= 1
                sizes[i] += 1
    return sizes


def _assign_groups(
    groups: list[list[DatasetEntry]],
    targets: list[int],
    ratios: tuple[float, float, float],
) -> list[list[list[DatasetEntry]]]:
    """Place whole groups in the train, val and test splits near their targets.

    With one entry per group this is a plain slice of the shuffled entries.
    Otherwise groups go largest first (equal sizes keep their shuffled order)
    to the split with the most entries still missing, ties going to the
    earlier split. A split is only chosen while it is short of its target, so
    it overshoots by less than one group; filling the splits in turn instead
    would leave test with whatever the other two did not take. Afterwards, a
    split with a non-zero ratio that received nothing takes the smallest group
    of the split that most exceeds its target (among splits holding at least
    two groups). Each split keeps its groups in shuffled order.
    """
    placed: list[list[int]] = [[], [], []]  # indices into groups
    counts = [0, 0, 0]
    splits = [k for k in range(3) if ratios[k] > 0]
    if all(len(group) == 1 for group in groups):
        for g in range(len(groups)):
            i = next((k for k in range(3) if counts[k] < targets[k]), 2)
            placed[i].append(g)
            counts[i] += 1
    else:
        for g in sorted(range(len(groups)), key=lambda g: -len(groups[g])):
            i = max(splits, key=lambda k: (targets[k] - counts[k], -k))
            placed[i].append(g)
            counts[i] += len(groups[g])

    for i in splits:
        if placed[i]:
            continue
        donors = [k for k in range(3) if len(placed[k]) >= 2]
        if not donors:
            break
        donor = max(donors, key=lambda k: (counts[k] - targets[k], -k))
        smallest = min(placed[donor], key=lambda g: (len(groups[g]), -g))
        placed[donor].remove(smallest)
        counts[donor] -= len(groups[smallest])
        placed[i].append(smallest)
        counts[i] += len(groups[smallest])
    return [[groups[g] for g in sorted(part)] for part in placed]
