"""Tests for prosody_protocol.datasets (Phase 10 -- Dataset Infrastructure).

Covers:
- Acceptance criteria from the execution guide
- DatasetLoader.load: directory loading, metadata, entry parsing, validation
- DatasetLoader.validate_entry: required fields, enums, IML, audio location
- DatasetLoader.iter_entries: lazy iteration
- DatasetLoader.split: deterministic, speaker-disjoint train/val/test split,
  optionally stratified by a field, warning about labels left out of val/test
- Data models: DatasetEntry, Dataset
- Error handling: missing dirs, bad JSON (also hostile: nested too deeply,
  integers too long), missing fields
"""

from __future__ import annotations

import json
import random
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from prosody_protocol.datasets import Dataset, DatasetEntry, DatasetLoader, resolve_audio_path
from prosody_protocol.exceptions import DatasetError

DATASETS_DIR = Path(__file__).parent / "fixtures" / "datasets"
SAMPLE_DIR = DATASETS_DIR / "sample"


@pytest.fixture()
def loader() -> DatasetLoader:
    return DatasetLoader()


@pytest.fixture()
def sample_dataset(loader: DatasetLoader) -> Dataset:
    return loader.load(SAMPLE_DIR)


@pytest.fixture()
def valid_entry_dict() -> dict[str, object]:
    return {
        "id": "test_001",
        "timestamp": "2025-01-15T10:30:00Z",
        "source": "recorded",
        "language": "en-US",
        "audio_file": "audio/sample_001.wav",
        "transcript": "Hello world.",
        "iml": "<utterance>Hello world.</utterance>",
        "speaker_id": "speaker_01",
        "emotion_label": "neutral",
        "annotator": "human",
        "consent": True,
    }


# ---------------------------------------------------------------------------
# Acceptance criteria (from EXECUTION_GUIDE.md Phase 10.4)
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    def test_loader_loads_and_validates_all_entries(self, loader: DatasetLoader) -> None:
        """DatasetLoader can load a directory of entries and validate all of them."""
        dataset = loader.load(SAMPLE_DIR)
        assert len(dataset.entries) == 3
        for entry in dataset.entries:
            raw = {
                "id": entry.id,
                "timestamp": entry.timestamp,
                "source": entry.source,
                "language": entry.language,
                "audio_file": entry.audio_file,
                "transcript": entry.transcript,
                "iml": entry.iml,
                "emotion_label": entry.emotion_label,
                "annotator": entry.annotator,
                "consent": entry.consent,
                "speaker_id": entry.speaker_id,
            }
            result = loader.validate_entry(raw, dataset_dir=SAMPLE_DIR)
            assert result.valid, f"Entry {entry.id} failed: {result.issues}"

    def test_invalid_entries_produce_clear_errors(self, loader: DatasetLoader) -> None:
        """Invalid entries (missing fields, bad IML) produce clear validation errors."""
        # Missing required fields.
        result = loader.validate_entry({"id": "x"})
        assert not result.valid
        assert len(result.issues) > 0
        assert any("Missing" in i.message for i in result.issues)

    def test_split_is_deterministic(self, sample_dataset: Dataset, loader: DatasetLoader) -> None:
        """Train/val/test split is deterministic given a seed."""
        split1 = loader.split(sample_dataset, seed=42)
        split2 = loader.split(sample_dataset, seed=42)
        assert [e.id for e in split1[0]] == [e.id for e in split2[0]]
        assert [e.id for e in split1[1]] == [e.id for e in split2[1]]
        assert [e.id for e in split1[2]] == [e.id for e in split2[2]]

    def test_audio_file_existence_checked(self, loader: DatasetLoader) -> None:
        """Audio files referenced in entries exist on disk (checked during validation)."""
        entry = {
            "id": "test_missing_audio",
            "timestamp": "2025-01-15T10:30:00Z",
            "source": "recorded",
            "language": "en-US",
            "audio_file": "audio/nonexistent.wav",
            "transcript": "Hello.",
            "iml": "<utterance>Hello.</utterance>",
            "emotion_label": "neutral",
            "annotator": "human",
            "consent": True,
        }
        result = loader.validate_entry(entry, dataset_dir=SAMPLE_DIR)
        assert not result.valid
        assert any(i.rule == "D8" for i in result.issues)

    def test_audio_file_exists_passes(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        result = loader.validate_entry(valid_entry_dict, dataset_dir=SAMPLE_DIR)
        assert result.valid


# ---------------------------------------------------------------------------
# DatasetLoader.load
# ---------------------------------------------------------------------------


class TestLoaderLoad:
    def test_load_sample_dataset(self, sample_dataset: Dataset) -> None:
        assert sample_dataset.name == "sample"
        assert sample_dataset.size == 3

    def test_load_metadata(self, sample_dataset: Dataset) -> None:
        assert sample_dataset.metadata.get("version") == "0.1.0"
        assert sample_dataset.metadata.get("description") == "Sample dataset for testing."

    def test_entries_ordered_by_filename(self, sample_dataset: Dataset) -> None:
        ids = [e.id for e in sample_dataset.entries]
        assert ids == ["entry_001", "entry_002", "entry_003"]

    def test_entry_fields_parsed(self, sample_dataset: Dataset) -> None:
        e = sample_dataset.entries[0]
        assert e.id == "entry_001"
        assert e.source == "recorded"
        assert e.language == "en-US"
        assert e.transcript == "Hello world."
        assert e.emotion_label == "neutral"
        assert e.annotator == "human"
        assert e.consent is True
        assert e.speaker_id == "speaker_01"

    def test_null_speaker_id(self, sample_dataset: Dataset) -> None:
        e = sample_dataset.entries[1]
        assert e.speaker_id is None

    def test_nonexistent_dir_raises(self, loader: DatasetLoader) -> None:
        with pytest.raises(DatasetError, match="does not exist"):
            loader.load("/nonexistent/path")

    def test_missing_entries_dir_raises(self, loader: DatasetLoader, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="Missing entries"):
            loader.load(tmp_path)

    def test_bad_json_entry_raises(self, loader: DatasetLoader, tmp_path: Path) -> None:
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()
        (entries_dir / "bad.json").write_text("{invalid json}", encoding="utf-8")
        with pytest.raises(DatasetError, match="Cannot read entry"):
            loader.load(tmp_path)


_HOSTILE_JSON = {
    "nested": "[" * 100_000,
    "long-integer": '{"id": ' + "7" * 5000 + "}",
    "not-utf8": b'{"id": "\xff"}'.decode("latin-1"),
}


class TestHostileJSON:
    """Deep nesting raised RecursionError and 5000-digit integers ValueError
    out of load(), instead of DatasetError (ProfileLoader already did this)."""

    @pytest.mark.parametrize("text", _HOSTILE_JSON.values(), ids=_HOSTILE_JSON.keys())
    def test_entry(self, loader: DatasetLoader, tmp_path: Path, text: str) -> None:
        (tmp_path / "entries").mkdir()
        (tmp_path / "entries" / "e.json").write_text(text, encoding="latin-1")
        with pytest.raises(DatasetError, match="Cannot read entry e.json"):
            loader.load(tmp_path)
        with pytest.raises(DatasetError, match="Cannot read entry e.json"):
            list(loader.iter_entries(tmp_path))

    @pytest.mark.parametrize("text", _HOSTILE_JSON.values(), ids=_HOSTILE_JSON.keys())
    def test_metadata(self, loader: DatasetLoader, tmp_path: Path, text: str) -> None:
        (tmp_path / "entries").mkdir()
        (tmp_path / "metadata.json").write_text(text, encoding="latin-1")
        with pytest.raises(DatasetError, match="Cannot read metadata.json"):
            loader.load(tmp_path)

    def test_unreadable_entry(self, loader: DatasetLoader, tmp_path: Path) -> None:
        (tmp_path / "entries" / "e.json").mkdir(parents=True)  # a directory, not a file
        with pytest.raises(DatasetError, match="Cannot read entry e.json"):
            loader.load(tmp_path)


# ---------------------------------------------------------------------------
# DatasetLoader.iter_entries
# ---------------------------------------------------------------------------


class TestIterEntries:
    def test_iter_yields_all(self, loader: DatasetLoader) -> None:
        entries = list(loader.iter_entries(SAMPLE_DIR))
        assert len(entries) == 3

    def test_iter_returns_dataset_entries(self, loader: DatasetLoader) -> None:
        for entry in loader.iter_entries(SAMPLE_DIR):
            assert isinstance(entry, DatasetEntry)

    def test_iter_missing_dir_raises(self, loader: DatasetLoader) -> None:
        with pytest.raises(DatasetError, match="Missing entries"):
            list(loader.iter_entries("/tmp/nonexistent"))


# ---------------------------------------------------------------------------
# DatasetLoader.validate_entry
# ---------------------------------------------------------------------------


class TestValidateEntry:
    def test_valid_entry(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        result = loader.validate_entry(valid_entry_dict)
        assert result.valid

    def test_missing_id(self, loader: DatasetLoader, valid_entry_dict: dict[str, object]) -> None:
        del valid_entry_dict["id"]
        result = loader.validate_entry(valid_entry_dict)
        assert not result.valid

    def test_missing_consent(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        del valid_entry_dict["consent"]
        result = loader.validate_entry(valid_entry_dict)
        assert not result.valid
        assert any(i.rule == "D2" for i in result.issues)

    def test_consent_false_invalid(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        valid_entry_dict["consent"] = False
        result = loader.validate_entry(valid_entry_dict)
        assert not result.valid

    def test_bad_source_enum(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        valid_entry_dict["source"] = "unknown_source"
        result = loader.validate_entry(valid_entry_dict)
        assert not result.valid
        assert any(i.rule == "D3" for i in result.issues)

    def test_bad_annotator_enum(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        valid_entry_dict["annotator"] = "robot"
        result = loader.validate_entry(valid_entry_dict)
        assert not result.valid
        assert any(i.rule == "D4" for i in result.issues)

    def test_bad_timestamp_warns(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        valid_entry_dict["timestamp"] = "not-a-timestamp"
        result = loader.validate_entry(valid_entry_dict)
        # Warnings don't invalidate.
        assert result.valid
        assert any(i.rule == "D5" for i in result.issues)

    def test_bad_language_is_an_error(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        """The schema requires a BCP-47 tag, so the loader rejects others."""
        valid_entry_dict["language"] = "not a language"
        result = loader.validate_entry(valid_entry_dict)
        assert not result.valid
        assert [i.rule for i in result.errors] == ["D6"]

    def test_wrong_type_is_not_coerced(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        valid_entry_dict["iml"] = {"not": "a string"}
        result = loader.validate_entry(valid_entry_dict)
        assert [(i.rule, i.message) for i in result.errors] == [
            ("D1", "Required field 'iml' must be a string, got dict"),
        ]

    def test_unknown_field_is_an_error(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        valid_entry_dict["notes"] = "extra data belongs under metadata"
        result = loader.validate_entry(valid_entry_dict)
        assert [i.rule for i in result.errors] == ["D11"]

    def test_optional_field_types(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        valid_entry_dict["speaker_id"] = 42
        valid_entry_dict["metadata"] = ["not", "an", "object"]
        result = loader.validate_entry(valid_entry_dict)
        assert [i.rule for i in result.errors] == ["D12", "D12"]

    def test_non_object_entry(self, loader: DatasetLoader) -> None:
        result = loader.validate_entry([1, 2])  # type: ignore[arg-type]
        assert not result.valid
        assert [i.rule for i in result.errors] == ["D12"]

    def test_bad_iml_flagged(self, loader: DatasetLoader) -> None:
        entry = {
            "id": "bad_iml",
            "timestamp": "2025-01-15T10:30:00Z",
            "source": "recorded",
            "language": "en-US",
            "audio_file": "audio/test.wav",
            "transcript": "Test.",
            "iml": '<utterance emotion="happy">no confidence</utterance>',
            "emotion_label": "happy",
            "annotator": "human",
            "consent": True,
        }
        result = loader.validate_entry(entry)
        assert not result.valid
        assert any(i.rule == "D7" for i in result.issues)

    def test_iml_validation_disabled(self) -> None:
        loader = DatasetLoader(validate_iml=False)
        entry = {
            "id": "bad_iml",
            "timestamp": "2025-01-15T10:30:00Z",
            "source": "recorded",
            "language": "en-US",
            "audio_file": "audio/test.wav",
            "transcript": "Test.",
            "iml": '<utterance emotion="happy">no confidence</utterance>',
            "emotion_label": "happy",
            "annotator": "human",
            "consent": True,
        }
        result = loader.validate_entry(entry)
        assert result.valid  # IML check skipped.


# ---------------------------------------------------------------------------
# DatasetLoader.split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_default_split(self, sample_dataset: Dataset, loader: DatasetLoader) -> None:
        train, val, test = loader.split(sample_dataset)
        assert len(train) + len(val) + len(test) == 3

    def test_deterministic_seed(self, sample_dataset: Dataset, loader: DatasetLoader) -> None:
        s1 = loader.split(sample_dataset, seed=123)
        s2 = loader.split(sample_dataset, seed=123)
        assert [e.id for e in s1[0]] == [e.id for e in s2[0]]

    def test_different_seed_different_order(
        self, loader: DatasetLoader,
    ) -> None:
        """With enough entries, different seeds should produce different orders."""
        entries = [
            DatasetEntry(
                id=f"e{i}", timestamp="2025-01-01T00:00:00Z",
                source="synthetic", language="en", audio_file=f"a/{i}.wav",
                transcript=f"Text {i}", iml=f"<utterance>Text {i}</utterance>",
                emotion_label="neutral", annotator="model", consent=True,
            )
            for i in range(20)
        ]
        ds = Dataset(name="test", entries=entries)
        s1 = loader.split(ds, seed=1)
        s2 = loader.split(ds, seed=2)
        # Very unlikely to be identical with different seeds and 20 entries.
        assert [e.id for e in s1[0]] != [e.id for e in s2[0]]

    def test_split_ratios_must_sum_to_one(
        self, sample_dataset: Dataset, loader: DatasetLoader
    ) -> None:
        with pytest.raises(DatasetError, match="sum to 1.0"):
            loader.split(sample_dataset, train=0.5, val=0.5, test=0.5)

    def test_custom_ratios(self, loader: DatasetLoader) -> None:
        entries = [
            DatasetEntry(
                id=f"e{i}", timestamp="2025-01-01T00:00:00Z",
                source="synthetic", language="en", audio_file=f"a/{i}.wav",
                transcript=f"Text {i}", iml=f"<utterance>Text {i}</utterance>",
                emotion_label="neutral", annotator="model", consent=True,
            )
            for i in range(10)
        ]
        ds = Dataset(name="test", entries=entries)
        train, val, test = loader.split(ds, train=0.6, val=0.2, test=0.2)
        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_dataset_entry_creation(self) -> None:
        e = DatasetEntry(
            id="e1", timestamp="2025-01-01T00:00:00Z",
            source="recorded", language="en-US",
            audio_file="audio/test.wav", transcript="Hello.",
            iml="<utterance>Hello.</utterance>",
            emotion_label="neutral", annotator="human", consent=True,
        )
        assert e.id == "e1"
        assert e.speaker_id is None
        assert e.metadata == {}

    def test_dataset_size_property(self) -> None:
        ds = Dataset(name="test", entries=[])
        assert ds.size == 0

    def test_dataset_with_entries(self, sample_dataset: Dataset) -> None:
        assert sample_dataset.size == 3
        assert sample_dataset.name == "sample"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataset_dir(self, loader: DatasetLoader, tmp_path: Path) -> None:
        (tmp_path / "entries").mkdir()
        ds = loader.load(tmp_path)
        assert ds.size == 0

    def test_no_metadata_file(self, loader: DatasetLoader, tmp_path: Path) -> None:
        (tmp_path / "entries").mkdir()
        ds = loader.load(tmp_path)
        assert ds.metadata == {}

    def test_entry_with_metadata_field(self, sample_dataset: Dataset) -> None:
        e = sample_dataset.entries[0]
        assert e.metadata.get("session") == "test_session_1"

    def test_validate_empty_dict(self, loader: DatasetLoader) -> None:
        result = loader.validate_entry({})
        assert not result.valid
        # Should have errors for all missing fields.
        assert len(result.issues) >= 9

    def test_all_three_sources_valid(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        for src in ("mavis", "recorded", "synthetic"):
            valid_entry_dict["source"] = src
            result = loader.validate_entry(valid_entry_dict)
            assert result.valid, f"Source '{src}' should be valid"

    def test_all_three_annotators_valid(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        for ann in ("human", "model", "hybrid"):
            valid_entry_dict["annotator"] = ann
            result = loader.validate_entry(valid_entry_dict)
            assert result.valid, f"Annotator '{ann}' should be valid"


# ---------------------------------------------------------------------------
# Validation on load
# ---------------------------------------------------------------------------


def _write_dataset(
    root: Path, entries: list[object], metadata: dict[str, object] | None = None
) -> Path:
    """Write *entries* as entries/NN.json (in order) under *root*."""
    (root / "entries").mkdir(parents=True)
    for i, entry in enumerate(entries):
        (root / "entries" / f"{i:02d}.json").write_text(json.dumps(entry), encoding="utf-8")
    if metadata is not None:
        (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def _entry(entry_id: str, **changes: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": entry_id,
        "timestamp": "2025-01-01T00:00:00Z",
        "source": "recorded",
        "language": "en-US",
        "audio_file": f"audio/{entry_id}.wav",
        "transcript": "hi",
        "iml": "<utterance>hi</utterance>",
        "emotion_label": "neutral",
        "annotator": "human",
        "consent": True,
    }
    entry.update(changes)
    return entry


class TestLoadValidation:
    def test_fixture_datasets_load_strictly(self) -> None:
        for name in ("sample", "training_synthetic"):
            dataset = DatasetLoader().load(DATASETS_DIR / name, check_audio=True)
            assert dataset.size == dataset.metadata["size"]

    def test_load_records_root(self, sample_dataset: Dataset) -> None:
        assert sample_dataset.root == SAMPLE_DIR

    def test_invalid_entries_raise_listing_every_problem(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [
            _entry("e1"),
            _entry("e2", consent=False),
            _entry("e1", source="youtube", annotator="crowd"),
            _entry("e4", iml="not xml <<<"),
            _entry("e5", iml='<utterance emotion="sad">x</utterance>'),
        ])
        with pytest.raises(DatasetError) as info:
            DatasetLoader().load(root)
        message = str(info.value)
        assert "4 of 5 entries" in message
        assert "01.json: D2 'consent' must be true" in message
        assert "02.json: D3" in message and "02.json: D4" in message
        assert "02.json: D10 Duplicate id 'e1' (first used in 00.json)" in message
        assert "03.json: D7" in message
        assert "04.json: D7" in message
        assert "strict=False" in message

    def test_non_strict_skips_invalid_entries(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [
            _entry("e1"),
            _entry("e2", consent=False),
            _entry("e3", iml="not xml <<<"),
            _entry("e1"),
        ])
        with pytest.warns(UserWarning, match="3 of 4 entries"):
            dataset = DatasetLoader(strict=False).load(root)
        assert [e.id for e in dataset.entries] == ["e1"]

    def test_consent_is_enforced_even_without_iml_validation(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [_entry("e1", consent=False)])
        with pytest.raises(DatasetError, match="D2"):
            DatasetLoader(validate_iml=False).load(root)
        with pytest.warns(UserWarning, match="D2"):
            assert DatasetLoader(validate_iml=False, strict=False).load(root).size == 0

    def test_iml_validation_can_be_disabled(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [_entry("e1", iml="<utterance>unclosed")])
        with pytest.raises(DatasetError, match="D7"):
            DatasetLoader().load(root)
        assert DatasetLoader(validate_iml=False).load(root).size == 1

    def test_non_object_entry_raises_dataset_error(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [[1, 2]])
        with pytest.raises(DatasetError, match="D12 Entry must be a JSON object, got list"):
            DatasetLoader().load(root)

    def test_mistyped_field_is_not_coerced(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [_entry("e1", iml={"not": "a string"})])
        with pytest.raises(DatasetError, match="'iml' must be a string"):
            DatasetLoader().load(root)

    def test_metadata_size_mismatch_warns(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [_entry("e1")], metadata={"size": 999})
        with pytest.warns(UserWarning, match="declares size 999, but 1 entries"):
            DatasetLoader().load(root)

    def test_metadata_must_be_an_object(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [_entry("e1")], metadata=[1])  # type: ignore[arg-type]
        with pytest.raises(DatasetError, match="metadata.json must contain a JSON object"):
            DatasetLoader().load(root)

    def test_audio_existence_is_opt_in(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [_entry("e1")])
        assert DatasetLoader().load(root).size == 1
        with pytest.raises(DatasetError, match="D8 Audio file not found: audio/e1.wav"):
            DatasetLoader().load(root, check_audio=True)

    def test_iter_entries_validates(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [_entry("e1"), _entry("e2", consent=False)])
        entries = DatasetLoader().iter_entries(root)
        assert next(entries).id == "e1"
        with pytest.raises(DatasetError, match="01.json: D2"):
            next(entries)
        with pytest.warns(UserWarning, match="D2"):
            ids = [e.id for e in DatasetLoader(strict=False).iter_entries(root)]
        assert ids == ["e1"]

    def test_iter_entries_detects_duplicates(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path, [_entry("e1"), _entry("e1")])
        with pytest.raises(DatasetError, match="D10"):
            list(DatasetLoader().iter_entries(root))


# ---------------------------------------------------------------------------
# Audio paths stay inside the dataset
# ---------------------------------------------------------------------------


class TestAudioPathContainment:
    @pytest.mark.parametrize(
        "audio_file",
        [
            "../../../../../../../../etc/passwd",
            "/etc/hostname",
            "audio/../../outside.wav",
            "audio\\..\\..\\outside.wav",
            "C:\\Windows\\win.ini",
            "\\\\server\\share\\a.wav",
            # Control characters: no file system opens a NUL byte, and a
            # newline hides a '..' segment from line-oriented checks.
            "audio/a\x00b.wav",
            "\n/../../etc/passwd",
            "audio/take\t1.wav",
        ],
    )
    def test_escaping_paths_rejected(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object], audio_file: str
    ) -> None:
        valid_entry_dict["audio_file"] = audio_file
        for dataset_dir in (None, SAMPLE_DIR):
            result = loader.validate_entry(valid_entry_dict, dataset_dir=dataset_dir)
            assert [i.rule for i in result.errors] == ["D9"], dataset_dir
        with pytest.raises(DatasetError, match="relative path inside the dataset"):
            resolve_audio_path(SAMPLE_DIR, audio_file)

    def test_nul_byte_is_a_d9_error_on_load(self, tmp_path: Path) -> None:
        """Checking the audio of such an entry raised a bare ValueError."""
        _write_dataset(tmp_path, [_entry("e1", audio_file="audio/a\x00b.wav")])
        with pytest.raises(DatasetError, match="D9"):
            DatasetLoader().load(tmp_path, check_audio=True)

    def test_symlink_loop_is_a_dataset_error(self, tmp_path: Path) -> None:
        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "loop.wav").symlink_to(tmp_path / "audio" / "loop.wav")
        with pytest.raises(DatasetError):
            resolve_audio_path(tmp_path, "audio/loop.wav")

    def test_symlink_out_of_the_dataset_rejected(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object], tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside.wav"
        outside.write_bytes(b"RIFF")
        dataset = tmp_path / "ds"
        (dataset / "audio").mkdir(parents=True)
        (dataset / "audio" / "link.wav").symlink_to(outside)
        valid_entry_dict["audio_file"] = "audio/link.wav"
        result = loader.validate_entry(valid_entry_dict, dataset_dir=dataset)
        assert [i.rule for i in result.errors] == ["D9"]
        with pytest.raises(DatasetError, match="resolves outside"):
            resolve_audio_path(dataset, "audio/link.wav")

    def test_names_with_dots_are_fine(self) -> None:
        assert resolve_audio_path(SAMPLE_DIR, "audio/take..2.wav") == (
            SAMPLE_DIR / "audio" / "take..2.wav"
        )


# ---------------------------------------------------------------------------
# Split sizes, validation and speaker grouping
# ---------------------------------------------------------------------------


def _entries(n: int, speakers: int | None = None) -> list[DatasetEntry]:
    return [
        DatasetEntry(
            id=f"e{i}", timestamp="2025-01-01T00:00:00Z",
            source="synthetic", language="en", audio_file=f"a/{i}.wav",
            transcript=f"Text {i}", iml=f"<utterance>Text {i}</utterance>",
            emotion_label="neutral", annotator="model", consent=True,
            speaker_id=None if speakers is None else f"spk{i % speakers}",
        )
        for i in range(n)
    ]


class TestSplitBehaviour:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [(3, [1, 1, 1]), (5, [3, 1, 1]), (9, [7, 1, 1]), (10, [8, 1, 1]), (19, [15, 2, 2]),
         (100, [80, 10, 10])],
    )
    def test_sizes_follow_ratios_and_fill_every_split(
        self, loader: DatasetLoader, n: int, expected: list[int]
    ) -> None:
        parts = loader.split(Dataset(name="t", entries=_entries(n)))
        assert [len(p) for p in parts] == expected
        assert sorted(e.id for p in parts for e in p) == sorted(e.id for e in _entries(n))

    def test_zero_ratio_split_stays_empty(self, loader: DatasetLoader) -> None:
        parts = loader.split(Dataset(name="t", entries=_entries(5)), train=0.8, val=0.0, test=0.2)
        assert [len(p) for p in parts] == [4, 0, 1]

    @pytest.mark.parametrize(
        "ratios",
        [(1.2, -0.2, 0.0), (0.5, 0.5, float("nan")), (True, 0.0, 0.0), (0.8, "0.1", 0.1)],
    )
    def test_invalid_ratios_rejected(
        self, loader: DatasetLoader, ratios: tuple[object, object, object]
    ) -> None:
        with pytest.raises(DatasetError, match="must be between 0 and 1"):
            loader.split(Dataset(name="t", entries=_entries(10)), *ratios)  # type: ignore[arg-type]

    def test_speakers_do_not_cross_splits(self, loader: DatasetLoader) -> None:
        dataset = Dataset(name="t", entries=_entries(40, speakers=8))
        for seed in range(5):
            train, val, test = loader.split(dataset, seed=seed)
            speakers = [{e.speaker_id for e in part} for part in (train, val, test)]
            assert not speakers[0] & speakers[1]
            assert not speakers[0] & speakers[2]
            assert not speakers[1] & speakers[2]
            assert [len(p) for p in (train, val, test)] == [30, 5, 5]

    def test_grouped_split_is_deterministic(self, loader: DatasetLoader) -> None:
        dataset = Dataset(name="t", entries=_entries(40, speakers=8))
        first = [[e.id for e in part] for part in loader.split(dataset, seed=3)]
        again = [[e.id for e in part] for part in loader.split(dataset, seed=3)]
        other = [[e.id for e in part] for part in loader.split(dataset, seed=4)]
        assert first == again
        assert first != other

    def test_uneven_groups_still_fill_every_split(self, loader: DatasetLoader) -> None:
        # One speaker with 8 entries, two with one each: 8/1/1 in some order.
        entries = _entries(10, speakers=1)
        entries[3] = replace(entries[3], speaker_id="solo_a")
        entries[7] = replace(entries[7], speaker_id="solo_b")
        for seed in range(6):
            parts = loader.split(Dataset(name="t", entries=entries), seed=seed)
            assert sorted(len(p) for p in parts) == [1, 1, 8]

    def test_skewed_speakers_still_fill_val_and_test(self, loader: DatasetLoader) -> None:
        """Long-tailed speaker sizes: filling train, then val, then test in
        shuffled order left test with 0-26 of its 34 entries."""
        sizes = [90, 60, 40, 30, 25, 20, 15, 12, 10, 8, 6, 5, 4, 3, 3, 2, 2, 1, 1, 1]
        entries = [
            replace(entry, id=f"s{s}_{j}", speaker_id=f"s{s}")
            for s, size in enumerate(sizes)
            for j, entry in enumerate(_entries(size))
        ]
        dataset = Dataset(name="t", entries=entries)
        for seed in range(10):
            parts = loader.split(dataset, seed=seed)
            speakers = [{e.speaker_id for e in part} for part in parts]
            assert not (speakers[0] & speakers[1] or speakers[0] & speakers[2]
                        or speakers[1] & speakers[2])
            # Targets are 270/34/34; val and test are made of small speakers,
            # so they can hit them within a few entries.
            assert [len(p) for p in parts] == pytest.approx([270, 34, 34], abs=3), seed

    def test_one_dominant_speaker(self, loader: DatasetLoader) -> None:
        """90 entries of one speaker plus 10 singletons split 90/5/5, not 90/9/1."""
        entries = _entries(90, speakers=1) + [
            replace(entry, id=f"solo{i}", speaker_id=f"solo{i}")
            for i, entry in enumerate(_entries(10))
        ]
        for seed in range(5):
            parts = loader.split(Dataset(name="t", entries=entries), seed=seed)
            assert [len(p) for p in parts] == [90, 5, 5]

    def test_unknown_speakers_are_placed_individually(self, loader: DatasetLoader) -> None:
        parts = loader.split(Dataset(name="t", entries=_entries(10)))
        assert [len(p) for p in parts] == [8, 1, 1]

    def test_single_speaker_falls_back_with_a_warning(self, loader: DatasetLoader) -> None:
        dataset = Dataset(name="t", entries=_entries(10, speakers=1))
        with pytest.warns(UserWarning, match="only 1 distinct speaker_id"):
            parts = loader.split(dataset)
        ungrouped = loader.split(dataset, group_by=None)
        assert [[e.id for e in p] for p in parts] == [[e.id for e in p] for p in ungrouped]

    def test_group_by_none_mixes_speakers(self, loader: DatasetLoader) -> None:
        train, _, test = loader.split(
            Dataset(name="t", entries=_entries(40, speakers=4)), group_by=None
        )
        assert {e.speaker_id for e in train} & {e.speaker_id for e in test}

    def test_group_by_other_field(self, loader: DatasetLoader) -> None:
        entries = [
            replace(e, source=("mavis", "recorded", "synthetic")[i % 3])
            for i, e in enumerate(_entries(9))
        ]
        parts = loader.split(Dataset(name="t", entries=entries), group_by="source")
        assert [len({e.source for e in p}) for p in parts] == [1, 1, 1]

    def test_unknown_group_by_rejected(self, loader: DatasetLoader) -> None:
        with pytest.raises(DatasetError, match="Cannot group by 'metadata'"):
            loader.split(Dataset(name="t", entries=_entries(3)), group_by="metadata")


# ---------------------------------------------------------------------------
# Stratified splits
# ---------------------------------------------------------------------------


def _labelled(labels: list[str], speakers: list[str | None] | None = None) -> Dataset:
    entries = [
        replace(entry, emotion_label=label,
                speaker_id=None if speakers is None else speakers[i])
        for i, (entry, label) in enumerate(zip(_entries(len(labels)), labels, strict=True))
    ]
    return Dataset(name="t", entries=entries)


def _label_counts(parts: tuple[list[DatasetEntry], ...], label: str) -> list[int]:
    return [sum(e.emotion_label == label for e in part) for part in parts]


class TestStratifiedSplit:
    RARE = ["rare"] * 3 + ["common"] * 27

    def test_unstratified_split_can_leave_a_rare_label_out(
        self, loader: DatasetLoader
    ) -> None:
        """Seed 3 put all three 'rare' entries in train, with no warning."""
        with pytest.warns(UserWarning, match=r"val lacks 'rare'; test lacks 'rare'; pass "
                                             r"stratify_by='emotion_label'"):
            parts = loader.split(_labelled(self.RARE), seed=3)
        assert _label_counts(parts, "rare") == [3, 0, 0]

    def test_rare_label_in_every_split(self, loader: DatasetLoader) -> None:
        dataset = _labelled(self.RARE)
        for seed in range(20):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                parts = loader.split(dataset, seed=seed, stratify_by="emotion_label")
            assert _label_counts(parts, "rare") == [1, 1, 1], seed
            assert [len(p) for p in parts] == [22, 4, 4]
            assert sorted(e.id for p in parts for e in p) == sorted(e.id for e in dataset.entries)

    def test_each_label_follows_the_ratios(self, loader: DatasetLoader) -> None:
        labels = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
        parts = loader.split(_labelled(labels), stratify_by="emotion_label")
        assert [_label_counts(parts, label) for label in "abc"] == [
            [40, 5, 5], [24, 3, 3], [16, 2, 2]
        ]

    def test_stratified_split_is_deterministic(self, loader: DatasetLoader) -> None:
        dataset = _labelled(self.RARE)

        def ids(seed: int) -> list[list[str]]:
            parts = loader.split(dataset, seed=seed, stratify_by="emotion_label")
            return [[e.id for e in p] for p in parts]

        assert ids(5) == ids(5) != ids(6)

    def test_speakers_stay_apart_and_rare_labels_spread(self, loader: DatasetLoader) -> None:
        """120 entries of 12 speakers; 'rare' is spoken by three of them."""
        rng = random.Random(0)
        speakers = [f"spk{i % 12}" for i in range(120)]
        labels = ["rare" if i in (0, 40, 80) else rng.choice("abc") for i in range(120)]
        dataset = _labelled(labels, speakers)
        for seed in range(10):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                parts = loader.split(dataset, seed=seed, stratify_by="emotion_label")
            groups = [{e.speaker_id for e in part} for part in parts]
            assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            assert _label_counts(parts, "rare") == [1, 1, 1], seed
            assert [len(p) for p in parts] == [100, 10, 10]

    def test_rare_speaker_groups_go_to_every_split(self, loader: DatasetLoader) -> None:
        """Three speakers hold 'rare', one of them most of the data."""
        speakers = ["big"] * 20 + ["s1", "s2"] + [f"x{i}" for i in range(8)]
        labels = ["rare"] + ["a"] * 19 + ["rare", "rare"] + ["a"] * 8
        for seed in range(5):
            parts = loader.split(_labelled(labels, speakers), seed=seed,
                                 stratify_by="emotion_label")
            assert _label_counts(parts, "rare") == [1, 1, 1]
            assert "big" in {e.speaker_id for e in parts[0]}

    def test_warning_names_the_missing_values(self) -> None:
        from prosody_protocol.datasets import _warn_missing_values

        entries = _labelled(["rare"] * 3 + ["a"] * 3).entries
        # 'rare' is in three groups, two of them in train and none in val.
        parts = [[entries[0:1], entries[1:2]], [entries[3:4]], [entries[2:3], entries[4:]]]
        with pytest.warns(UserWarning, match=r"val lacks 'rare'; their entries' groups could "
                                             r"not be spread further"):
            _warn_missing_values(parts, "emotion_label", suggest=False)

    def test_labels_too_rare_to_spread_do_not_warn(self, loader: DatasetLoader) -> None:
        labels = ["once"] + ["twice"] * 2 + ["common"] * 27
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loader.split(_labelled(labels), seed=3)

    def test_stratify_by_another_field(self, loader: DatasetLoader) -> None:
        entries = [
            replace(e, source="mavis" if i < 3 else "recorded")
            for i, e in enumerate(_entries(30))
        ]
        parts = loader.split(Dataset(name="t", entries=entries), seed=3, stratify_by="source")
        assert [sum(e.source == "mavis" for e in p) for p in parts] == [1, 1, 1]

    def test_unknown_stratify_by_rejected(self, loader: DatasetLoader) -> None:
        with pytest.raises(DatasetError, match="Cannot stratify by 'metadata'"):
            loader.split(Dataset(name="t", entries=_entries(3)), stratify_by="metadata")

    def test_counts_per_label_are_kept(self, loader: DatasetLoader) -> None:
        labels = [random.Random(1).choice("abcdefg") for _ in range(57)]
        parts = loader.split(_labelled(labels), stratify_by="emotion_label")
        merged = Counter(e.emotion_label for p in parts for e in p)
        assert merged == Counter(labels)
