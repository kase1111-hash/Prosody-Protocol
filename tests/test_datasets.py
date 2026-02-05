"""Tests for prosody_protocol.datasets (Phase 10 -- Dataset Infrastructure).

Covers:
- Acceptance criteria from the execution guide
- DatasetLoader.load: directory loading, metadata, entry parsing
- DatasetLoader.validate_entry: required fields, enums, IML, audio existence
- DatasetLoader.iter_entries: lazy iteration
- DatasetLoader.split: deterministic train/val/test split
- Data models: DatasetEntry, Dataset
- Error handling: missing dirs, bad JSON, missing fields
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prosody_protocol.datasets import Dataset, DatasetEntry, DatasetLoader
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

    def test_bad_language_warns(
        self, loader: DatasetLoader, valid_entry_dict: dict[str, object]
    ) -> None:
        valid_entry_dict["language"] = "not a language"
        result = loader.validate_entry(valid_entry_dict)
        assert result.valid
        assert any(i.rule == "D6" for i in result.issues)

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
