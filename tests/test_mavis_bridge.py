"""Tests for the Mavis Data Bridge (Phase 16).

Covers:
- PhonemeEvent to DatasetEntry conversion
- Feature extraction for sklearn training
- Batch feature extraction
- Dataset export
- IML generation from phoneme events
- Emotion inference from prosody
- Edge cases (empty events, single event, etc.)
- Typed text with characters XML does not allow
"""

from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np

from prosody_protocol import DatasetEntry, DatasetLoader, IMLParser, IMLValidator
from prosody_protocol.exceptions import DatasetError
from prosody_protocol.mavis_bridge import (
    MAVIS_FEATURE_NAMES,
    MavisBridge,
    PhonemeEvent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bridge() -> MavisBridge:
    return MavisBridge(language="en-US")


@pytest.fixture()
def sample_events() -> list[PhonemeEvent]:
    """Simulate typing 'the SUN is RISING' in Mavis."""
    return [
        # "the" - normal
        PhonemeEvent(phoneme="dh", start_ms=0, duration_ms=80, volume=0.5, pitch_hz=180.0),
        PhonemeEvent(phoneme="ax", start_ms=80, duration_ms=60, volume=0.5, pitch_hz=175.0),
        # "SUN" - loud emphasis
        PhonemeEvent(phoneme="s", start_ms=200, duration_ms=100, volume=0.85, pitch_hz=280.0),
        PhonemeEvent(phoneme="ah", start_ms=300, duration_ms=120, volume=0.9, pitch_hz=300.0),
        PhonemeEvent(phoneme="n", start_ms=420, duration_ms=80, volume=0.8, pitch_hz=260.0),
        # "is" - normal
        PhonemeEvent(phoneme="ih", start_ms=550, duration_ms=60, volume=0.5, pitch_hz=190.0),
        PhonemeEvent(phoneme="z", start_ms=610, duration_ms=60, volume=0.5, pitch_hz=185.0),
        # "RISING" - loud emphasis
        PhonemeEvent(phoneme="r", start_ms=720, duration_ms=80, volume=0.85, pitch_hz=270.0),
        PhonemeEvent(phoneme="ay", start_ms=800, duration_ms=120, volume=0.9, pitch_hz=320.0),
        PhonemeEvent(phoneme="z", start_ms=920, duration_ms=60, volume=0.8, pitch_hz=280.0),
        PhonemeEvent(phoneme="ih", start_ms=980, duration_ms=60, volume=0.75, pitch_hz=250.0),
        PhonemeEvent(phoneme="ng", start_ms=1040, duration_ms=80, volume=0.7, pitch_hz=230.0),
    ]


@pytest.fixture()
def quiet_events() -> list[PhonemeEvent]:
    """Simulate soft, breathy speech."""
    return [
        PhonemeEvent(phoneme="f", start_ms=0, duration_ms=100, volume=0.2,
                     pitch_hz=150.0, breathiness=0.7),
        PhonemeEvent(phoneme="ay", start_ms=100, duration_ms=120, volume=0.25,
                     pitch_hz=145.0, breathiness=0.8),
        PhonemeEvent(phoneme="n", start_ms=220, duration_ms=80, volume=0.2,
                     pitch_hz=140.0, breathiness=0.6),
    ]


# ---------------------------------------------------------------------------
# Entry conversion tests
# ---------------------------------------------------------------------------


class TestPhonemeToEntry:
    def test_basic_conversion(self, bridge: MavisBridge, sample_events: list[PhonemeEvent]) -> None:
        entry = bridge.phoneme_events_to_entry(
            events=sample_events,
            transcript="the SUN is RISING",
            session_id="test_001",
            emotion_label="joyful",
            consent=True,
        )
        assert isinstance(entry, DatasetEntry)
        assert entry.id == "mavis_test_001"
        assert entry.source == "mavis"
        assert entry.language == "en-US"
        assert entry.transcript == "the SUN is RISING"
        assert entry.emotion_label == "joyful"
        assert entry.consent is True
        # The label is the caller's, the IML the bridge's.
        assert entry.annotator == "hybrid"

    def test_consent_is_not_assumed(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        entry = bridge.phoneme_events_to_entry(sample_events, "the SUN", "s1")
        assert entry.consent is False

    def test_annotator(self, bridge: MavisBridge, sample_events: list[PhonemeEvent]) -> None:
        assert bridge.phoneme_events_to_entry(sample_events, "hi", "s1").annotator == "model"
        entry = bridge.phoneme_events_to_entry(
            sample_events, "hi", "s1", emotion_label="sad", annotator="human"
        )
        assert entry.annotator == "human"
        with pytest.raises(DatasetError, match="annotator must be one of"):
            bridge.phoneme_events_to_entry(sample_events, "hi", "s1", annotator="robot")

    @pytest.mark.parametrize("session_id", ["2025/01/01", "../escape", "", "a b", ".hidden"])
    def test_unsafe_session_id_rejected(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], session_id: str
    ) -> None:
        with pytest.raises(DatasetError, match="session_id"):
            bridge.phoneme_events_to_entry(sample_events, "hi", session_id)

    def test_empty_transcript_rejected(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        with pytest.raises(DatasetError, match="empty transcript"):
            bridge.phoneme_events_to_entry(sample_events, "  ", "s1")

    def test_non_finite_event_rejected(self, bridge: MavisBridge) -> None:
        with pytest.raises(DatasetError, match="non-finite"):
            bridge.phoneme_events_to_entry([PhonemeEvent("a", pitch_hz=float("nan"))], "a", "s1")

    def test_raw_events_are_kept(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        entry = bridge.phoneme_events_to_entry(sample_events, "the SUN is RISING", "s1")
        assert MavisBridge.events_from_entry(entry) == sample_events
        features = entry.metadata["mavis_features"]
        assert isinstance(features, dict)
        assert list(features) == MAVIS_FEATURE_NAMES
        np.testing.assert_allclose(
            list(features.values()), bridge.extract_training_features(sample_events)
        )

    def test_iml_is_valid(self, bridge: MavisBridge, sample_events: list[PhonemeEvent]) -> None:
        entry = bridge.phoneme_events_to_entry(
            events=sample_events,
            transcript="the SUN is RISING",
            session_id="test_002",
            emotion_label="joyful",
        )
        validator = IMLValidator()
        result = validator.validate(entry.iml)
        assert result.valid, f"Generated IML is invalid: {result.issues}"

    def test_iml_contains_emotion(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        entry = bridge.phoneme_events_to_entry(
            events=sample_events,
            transcript="the SUN is RISING",
            session_id="test_003",
            emotion_label="frustrated",
        )
        assert 'emotion="frustrated"' in entry.iml
        assert 'confidence=' in entry.iml

    def test_iml_parseable(self, bridge: MavisBridge, sample_events: list[PhonemeEvent]) -> None:
        entry = bridge.phoneme_events_to_entry(
            events=sample_events,
            transcript="hello world",
            session_id="test_004",
        )
        parser = IMLParser()
        doc = parser.parse(entry.iml)
        assert len(doc.utterances) == 1

    def test_speaker_id_propagated(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        entry = bridge.phoneme_events_to_entry(
            events=sample_events,
            transcript="test",
            session_id="test_005",
            speaker_id="user_42",
        )
        assert entry.speaker_id == "user_42"

    def test_empty_events_raises(self, bridge: MavisBridge) -> None:
        with pytest.raises(Exception, match="empty"):
            bridge.phoneme_events_to_entry(
                events=[],
                transcript="test",
                session_id="test_006",
            )


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    def test_feature_vector_shape(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        features = bridge.extract_training_features(sample_events)
        assert features.shape == (len(MAVIS_FEATURE_NAMES),)

    def test_feature_values_reasonable(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        features = bridge.extract_training_features(sample_events)
        # mean_pitch_hz should be between min and max pitch
        pitches = [e.pitch_hz for e in sample_events]
        assert min(pitches) <= features[0] <= max(pitches)
        # pitch_range_hz should be positive
        assert features[1] >= 0
        # mean_volume in [0, 1]
        assert 0.0 <= features[2] <= 1.0
        # speech_rate should be positive
        assert features[5] > 0

    def test_quiet_vs_loud_features(
        self,
        bridge: MavisBridge,
        sample_events: list[PhonemeEvent],
        quiet_events: list[PhonemeEvent],
    ) -> None:
        loud = bridge.extract_training_features(sample_events)
        quiet = bridge.extract_training_features(quiet_events)
        # Loud events should have higher mean volume
        assert loud[2] > quiet[2]
        # Quiet events should have higher mean breathiness
        assert quiet[4] > loud[4]

    def test_empty_events_returns_zeros(self, bridge: MavisBridge) -> None:
        features = bridge.extract_training_features([])
        assert features.shape == (len(MAVIS_FEATURE_NAMES),)
        np.testing.assert_array_equal(features, np.zeros(len(MAVIS_FEATURE_NAMES)))

    def test_single_event(self, bridge: MavisBridge) -> None:
        features = bridge.extract_training_features([
            PhonemeEvent(phoneme="a", duration_ms=100, volume=0.7, pitch_hz=200.0)
        ])
        assert features[0] == 200.0  # mean pitch
        assert features[1] == 0.0    # no pitch range with 1 event
        assert features[2] == 0.7    # mean volume

    def test_vibrato_ratio(self, bridge: MavisBridge) -> None:
        events = [
            PhonemeEvent(phoneme="a", duration_ms=100, vibrato=True),
            PhonemeEvent(phoneme="b", duration_ms=100, vibrato=False),
            PhonemeEvent(phoneme="c", duration_ms=100, vibrato=True),
            PhonemeEvent(phoneme="d", duration_ms=100, vibrato=False),
        ]
        features = bridge.extract_training_features(events)
        assert features[6] == pytest.approx(0.5)  # 2/4 vibrato


# ---------------------------------------------------------------------------
# Batch extraction tests
# ---------------------------------------------------------------------------


class TestBatchExtraction:
    def test_batch_shape(
        self,
        bridge: MavisBridge,
        sample_events: list[PhonemeEvent],
        quiet_events: list[PhonemeEvent],
    ) -> None:
        X = bridge.batch_extract_features([sample_events, quiet_events])
        assert X.shape == (2, len(MAVIS_FEATURE_NAMES))

    def test_batch_consistent_with_single(
        self,
        bridge: MavisBridge,
        sample_events: list[PhonemeEvent],
    ) -> None:
        single = bridge.extract_training_features(sample_events)
        batch = bridge.batch_extract_features([sample_events])
        np.testing.assert_array_almost_equal(batch[0], single)


# ---------------------------------------------------------------------------
# Emotion inference tests
# ---------------------------------------------------------------------------


class TestEmotionInference:
    def test_loud_high_pitch_infers_angry(self, bridge: MavisBridge) -> None:
        events = [
            PhonemeEvent(phoneme="a", volume=0.95, pitch_hz=400.0),
            PhonemeEvent(phoneme="b", volume=0.9, pitch_hz=380.0),
        ]
        entry = bridge.phoneme_events_to_entry(
            events=events, transcript="test", session_id="angry_01"
        )
        assert entry.emotion_label == "angry"

    def test_loud_low_pitch_infers_joyful(self, bridge: MavisBridge) -> None:
        events = [
            PhonemeEvent(phoneme="a", volume=0.9, pitch_hz=200.0),
            PhonemeEvent(phoneme="b", volume=0.85, pitch_hz=180.0),
        ]
        entry = bridge.phoneme_events_to_entry(
            events=events, transcript="test", session_id="joy_01"
        )
        assert entry.emotion_label == "joyful"

    def test_breathy_infers_sad(self, bridge: MavisBridge) -> None:
        events = [
            PhonemeEvent(phoneme="a", volume=0.4, breathiness=0.7),
            PhonemeEvent(phoneme="b", volume=0.35, breathiness=0.8),
        ]
        entry = bridge.phoneme_events_to_entry(
            events=events, transcript="test", session_id="sad_01"
        )
        assert entry.emotion_label == "sad"

    def test_quiet_infers_calm(self, bridge: MavisBridge) -> None:
        events = [
            PhonemeEvent(phoneme="a", volume=0.2, breathiness=0.1),
            PhonemeEvent(phoneme="b", volume=0.15, breathiness=0.2),
        ]
        entry = bridge.phoneme_events_to_entry(
            events=events, transcript="test", session_id="calm_01"
        )
        assert entry.emotion_label == "calm"

    def test_normal_infers_neutral(self, bridge: MavisBridge) -> None:
        events = [
            PhonemeEvent(phoneme="a", volume=0.5, breathiness=0.1, pitch_hz=200.0),
            PhonemeEvent(phoneme="b", volume=0.5, breathiness=0.1, pitch_hz=200.0),
        ]
        entry = bridge.phoneme_events_to_entry(
            events=events, transcript="test", session_id="neut_01"
        )
        assert entry.emotion_label == "neutral"

    def test_explicit_emotion_overrides_inference(self, bridge: MavisBridge) -> None:
        events = [PhonemeEvent(phoneme="a", volume=0.95, pitch_hz=400.0)]
        entry = bridge.phoneme_events_to_entry(
            events=events, transcript="test", session_id="override_01",
            emotion_label="surprised",
        )
        assert entry.emotion_label == "surprised"


class TestConfidence:
    """The utterance confidence used to be 0.5 + the session's pitch and
    volume range (0.5 to 0.9, under a 0.95 cap it never reached): a quiet
    session guessed as "calm" got confidence 0.9, a flat one "neutral" 0.5."""

    @pytest.mark.parametrize(
        ("volume", "breathiness", "pitch_step", "label"),
        [(0.2, 0.0, 40, "calm"), (0.5, 0.0, 0, "neutral"), (0.9, 0.0, 30, "joyful"),
         (0.4, 0.8, 30, "sad")],
    )
    def test_guessed_labels_are_not_stated_in_the_iml(
        self, bridge: MavisBridge, volume: float, breathiness: float, pitch_step: int,
        label: str,
    ) -> None:
        events = [
            PhonemeEvent("a", start_ms=i * 120, duration_ms=100, volume=volume,
                         breathiness=breathiness, pitch_hz=120 + i * pitch_step)
            for i in range(8)
        ]
        entry = bridge.phoneme_events_to_entry(events, "hello there friend", "s1", consent=True)
        assert (entry.emotion_label, entry.annotator) == (label, "model")
        utterance = IMLParser().parse(entry.iml).utterances[0]
        assert (utterance.emotion, utterance.confidence) == (None, None)
        assert IMLValidator().validate(entry.iml).valid

    def test_given_labels_are_stated_with_full_confidence(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        """A label the caller gives is an assertion, whatever the session's range."""
        flat = [PhonemeEvent("a", start_ms=i * 120, volume=0.5) for i in range(4)]
        for events in (sample_events, flat, flat[:1]):
            entry = bridge.phoneme_events_to_entry(events, "the SUN", "s1", emotion_label="sad")
            utterance = IMLParser().parse(entry.iml).utterances[0]
            assert (utterance.emotion, utterance.confidence) == ("sad", 1.0)

    def test_guessed_label_with_a_human_annotator_is_still_not_stated(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        entry = bridge.phoneme_events_to_entry(sample_events, "hi", "s1", annotator="human")
        assert IMLParser().parse(entry.iml).utterances[0].emotion is None

    def test_no_range_based_confidence_remains(self) -> None:
        assert not hasattr(MavisBridge, "_compute_confidence")


# ---------------------------------------------------------------------------
# Dataset export tests
# ---------------------------------------------------------------------------


class TestDatasetExport:
    def test_export_creates_directory_structure(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [
            {"events": sample_events, "transcript": "the SUN", "session_id": "s1",
             "emotion_label": "joyful"},
            {"events": sample_events, "transcript": "RISING", "session_id": "s2",
             "emotion_label": "neutral"},
        ]
        dataset = bridge.export_dataset(sessions, tmp_path / "mavis_ds", consent=True)

        assert (tmp_path / "mavis_ds" / "metadata.json").exists()
        assert (tmp_path / "mavis_ds" / "entries" / "mavis_s1.json").exists()
        assert (tmp_path / "mavis_ds" / "entries" / "mavis_s2.json").exists()
        assert len(dataset.entries) == 2

    def test_exported_entries_are_valid_json(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [
            {"events": sample_events, "transcript": "test", "session_id": "s1"},
        ]
        bridge.export_dataset(sessions, tmp_path / "ds", consent=True)

        entry_file = tmp_path / "ds" / "entries" / "mavis_s1.json"
        data = json.loads(entry_file.read_text())
        assert data["source"] == "mavis"
        assert data["consent"] is True

    def test_exported_metadata_correct(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [
            {"events": sample_events, "transcript": "t1", "session_id": "s1"},
            {"events": sample_events, "transcript": "t2", "session_id": "s2"},
            {"events": sample_events, "transcript": "t3", "session_id": "s3"},
        ]
        bridge.export_dataset(sessions, tmp_path / "ds", consent=True)

        meta = json.loads((tmp_path / "ds" / "metadata.json").read_text())
        assert meta["size"] == 3
        assert meta["source"] == "mavis"

    def test_exported_dataset_loadable(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        """Verify the exported dataset can be loaded back by DatasetLoader."""
        sessions = [
            {"events": sample_events, "transcript": "hello", "session_id": "s1",
             "emotion_label": "neutral"},
        ]
        bridge.export_dataset(sessions, tmp_path / "ds", consent=True)

        loader = DatasetLoader(validate_iml=True)
        loaded = loader.load(tmp_path / "ds")
        assert loaded.size == 1
        assert loaded.entries[0].source == "mavis"


class TestDatasetExportSafety:
    def test_consent_must_be_stated(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [{"events": sample_events, "transcript": "hi", "session_id": "s1"}]
        with pytest.raises(DatasetError, match="no recorded consent"):
            bridge.export_dataset(sessions, tmp_path / "ds")
        with pytest.raises(DatasetError, match="no recorded consent"):
            bridge.export_dataset(sessions, tmp_path / "ds", consent=False)
        assert not (tmp_path / "ds").exists()

    def test_per_session_consent(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [
            {"events": sample_events, "transcript": "hi", "session_id": "s1", "consent": True},
            {"events": sample_events, "transcript": "yo", "session_id": "s2", "consent": False},
        ]
        with pytest.raises(DatasetError, match="Session 's2' has no recorded consent"):
            bridge.export_dataset(sessions, tmp_path / "ds", consent=True)
        dataset = bridge.export_dataset(sessions[:1], tmp_path / "ds")
        assert [e.consent for e in dataset.entries] == [True]

    def test_duplicate_session_ids_rejected_before_writing(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [
            {"events": sample_events, "transcript": "first take", "session_id": "s1"},
            {"events": sample_events, "transcript": "second take", "session_id": "s1"},
        ]
        with pytest.raises(DatasetError, match="Duplicate session_id 's1'"):
            bridge.export_dataset(sessions, tmp_path / "ds", consent=True)
        assert not (tmp_path / "ds").exists()

    def test_missing_session_key(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        with pytest.raises(DatasetError, match="Session 0 is missing transcript"):
            bridge.export_dataset(
                [{"events": sample_events, "session_id": "s1"}], tmp_path / "ds", consent=True
            )

    def test_existing_entries_need_overwrite(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        out = tmp_path / "ds"
        old = [
            {"events": sample_events, "transcript": "old", "session_id": "old1"},
            {"events": sample_events, "transcript": "old", "session_id": "old2"},
        ]
        bridge.export_dataset(old, out, consent=True)
        new = [{"events": sample_events, "transcript": "new", "session_id": "new1"}]
        with pytest.raises(DatasetError, match="already contains 2 entries"):
            bridge.export_dataset(new, out, consent=True)
        dataset = bridge.export_dataset(new, out, consent=True, overwrite=True)
        assert [e.id for e in dataset.entries] == ["mavis_new1"]
        assert json.loads((out / "metadata.json").read_text())["size"] == 1

    def test_audio_is_copied_and_checkable(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        recording = tmp_path / "take.wav"
        recording.write_bytes(b"RIFF0000WAVE")
        sessions = [
            {"events": sample_events, "transcript": "hi", "session_id": "s1",
             "audio_path": recording},
            {"events": sample_events, "transcript": "yo", "session_id": "s2"},
        ]
        bridge.export_dataset(sessions, tmp_path / "ds", consent=True)
        assert (tmp_path / "ds" / "audio" / "s1.wav").read_bytes() == b"RIFF0000WAVE"
        # s2 has no audio: the dataset loads, but not with check_audio.
        assert DatasetLoader().load(tmp_path / "ds").size == 2
        with pytest.raises(DatasetError, match="D8 Audio file not found: audio/s2.wav"):
            DatasetLoader().load(tmp_path / "ds", check_audio=True)

    def test_missing_audio_file_rejected(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [{"events": sample_events, "transcript": "hi", "session_id": "s1",
                     "audio_path": tmp_path / "missing.wav"}]
        with pytest.raises(DatasetError, match="Audio for session 's1' not found"):
            bridge.export_dataset(sessions, tmp_path / "ds", consent=True)

    def test_export_round_trips_text_and_events(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [{"events": sample_events, "transcript": "Tom & Jerry <3",
                     "session_id": "s1", "emotion_label": "joyful", "speaker_id": "u1"}]
        dataset = bridge.export_dataset(sessions, tmp_path / "ds", consent=True)
        assert dataset.root == tmp_path / "ds"
        entry = dataset.entries[0]
        assert IMLParser().to_plain_text(IMLParser().parse(entry.iml)) == "Tom & Jerry <3"
        np.testing.assert_array_equal(
            bridge.extract_training_features(MavisBridge.events_from_entry(entry)),
            bridge.extract_training_features(sample_events),
        )



def _numpy_events() -> list[PhonemeEvent]:
    """Events as audio code produces them: numpy scalars and arrays."""
    return [
        PhonemeEvent("h", start_ms=np.int64(0), duration_ms=np.int64(90),
                     volume=np.float32(0.5), pitch_hz=np.float32(180.0)),
        PhonemeEvent("i", start_ms=np.int64(90), duration_ms=np.int32(110),
                     volume=np.float64(0.6), pitch_hz=np.float32(190.5),
                     vibrato=np.bool_(True), breathiness=np.float32(0.25),
                     harmony_intervals=np.array([3, 7])),
        PhonemeEvent("t", start_ms=np.int64(400), duration_ms=np.int64(100),
                     volume=np.float32(0.9), pitch_hz=np.float32(260.0)),
    ]


class TestExportAtomicity:
    def test_numpy_scalar_events_export(
        self, bridge: MavisBridge, tmp_path: Path
    ) -> None:
        """json cannot write numpy scalars; this raised a bare TypeError halfway
        through writing the dataset."""
        events = _numpy_events()
        dataset = bridge.export_dataset(
            [{"events": events, "transcript": "hi there", "session_id": "s1",
              "emotion_label": "sad"}],
            tmp_path / "ds",
            consent=True,
        )
        stored = json.loads((tmp_path / "ds" / "entries" / "mavis_s1.json").read_text())
        second = stored["metadata"]["phoneme_events"][1]
        assert second["start_ms"] == 90 and type(second["start_ms"]) is int
        assert second["pitch_hz"] == 190.5 and type(second["pitch_hz"]) is float
        assert second["vibrato"] is True
        assert second["harmony_intervals"] == [3, 7]
        assert MavisBridge.events_from_entry(dataset.entries[0]) == [
            PhonemeEvent("h", 0, 90, 0.5, 180.0),
            PhonemeEvent("i", 90, 110, 0.6, 190.5, True, 0.25, [3, 7]),
            PhonemeEvent("t", 400, 100, pytest.approx(0.9), 260.0),
        ]

    def _export_old(self, bridge: MavisBridge, events: list[PhonemeEvent], out: Path) -> bytes:
        recording = out.parent / "old.wav"
        recording.write_bytes(b"RIFF-old")
        bridge.export_dataset(
            [{"events": events, "transcript": "old take", "session_id": "old1",
              "audio_path": recording}],
            out,
            consent=True,
        )
        return self._snapshot(out)

    @staticmethod
    def _snapshot(out: Path) -> bytes:
        files = sorted(p for p in out.rglob("*") if p.is_file())
        return json.dumps(
            {str(p.relative_to(out)): p.read_bytes().decode("latin-1") for p in files}
        ).encode()

    def test_unserialisable_session_leaves_previous_export_intact(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        out = tmp_path / "ds"
        before = self._export_old(bridge, sample_events, out)
        odd = [PhonemeEvent("a", harmony_intervals=[Decimal(3)]), PhonemeEvent("b", 300)]
        sessions = [
            {"events": sample_events, "transcript": "new take", "session_id": "n1"},
            {"events": odd, "transcript": "odd take", "session_id": "n2"},
        ]
        with pytest.raises(DatasetError, match="Session 'n2' cannot be stored as JSON"):
            bridge.export_dataset(sessions, out, consent=True, overwrite=True)
        assert self._snapshot(out) == before

    def test_write_failure_leaves_previous_export_intact(
        self,
        bridge: MavisBridge,
        sample_events: list[PhonemeEvent],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = tmp_path / "ds"
        before = self._export_old(bridge, sample_events, out)
        recording = tmp_path / "new.wav"
        recording.write_bytes(b"RIFF-new")

        def disk_full(src: object, dst: object) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(shutil, "copyfile", disk_full)
        sessions = [{"events": sample_events, "transcript": "new take", "session_id": "n1",
                     "audio_path": recording}]
        with pytest.raises(DatasetError, match="No space left on device"):
            bridge.export_dataset(sessions, out, consent=True, overwrite=True)
        assert self._snapshot(out) == before

    def test_failed_first_export_leaves_nothing(
        self,
        bridge: MavisBridge,
        sample_events: list[PhonemeEvent],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recording = tmp_path / "take.wav"
        recording.write_bytes(b"RIFF")

        def disk_full(src: object, dst: object) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(shutil, "copyfile", disk_full)
        with pytest.raises(DatasetError):
            bridge.export_dataset(
                [{"events": sample_events, "transcript": "hi", "session_id": "s1",
                  "audio_path": recording}],
                tmp_path / "ds",
                consent=True,
            )
        assert not (tmp_path / "ds").exists()

    def test_overwrite_removes_the_earlier_exports_audio(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        """A session left out of the new export (say, consent withdrawn)
        leaves no recording behind; files the export did not write stay."""
        out = tmp_path / "ds"
        self._export_old(bridge, sample_events, out)
        (out / "audio" / "notes.txt").write_text("keep me")
        (out / "README.md").write_text("keep me too")
        recording = tmp_path / "new.wav"
        recording.write_bytes(b"RIFF-new")
        bridge.export_dataset(
            [{"events": sample_events, "transcript": "new take", "session_id": "n1",
              "audio_path": recording}],
            out,
            consent=True,
            overwrite=True,
        )
        assert sorted(p.name for p in (out / "audio").iterdir()) == ["n1.wav", "notes.txt"]
        assert sorted(p.name for p in (out / "entries").iterdir()) == ["mavis_n1.json"]
        assert (out / "README.md").read_text() == "keep me too"
        assert not [p for p in out.iterdir() if p.name.startswith(".export")]

    @pytest.mark.parametrize("text", ["[" * 100_000, '{"audio_file": ' + "9" * 5000 + "}"])
    def test_overwrite_survives_hostile_earlier_entries(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path, text: str
    ) -> None:
        """An earlier entry nested too deeply to parse raised RecursionError
        out of export_dataset(overwrite=True); it is replaced like any other."""
        out = tmp_path / "ds"
        (out / "entries").mkdir(parents=True)
        (out / "entries" / "junk.json").write_text(text)
        dataset = bridge.export_dataset(
            [{"events": sample_events, "transcript": "new", "session_id": "n1"}],
            out, consent=True, overwrite=True,
        )
        assert [e.id for e in dataset.entries] == ["mavis_n1"]
        assert sorted(p.name for p in (out / "entries").iterdir()) == ["mavis_n1.json"]

    def test_re_export_in_place(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        """Re-exporting sessions whose audio_path is the dataset's own copy
        (copyfile onto itself raised SameFileError after entries were deleted)."""
        out = tmp_path / "ds"
        self._export_old(bridge, sample_events, out)
        dataset = bridge.export_dataset(
            [{"events": sample_events, "transcript": "old take", "session_id": "old1",
              "emotion_label": "calm", "audio_path": out / "audio" / "old1.wav"}],
            out,
            consent=True,
            overwrite=True,
        )
        assert dataset.entries[0].emotion_label == "calm"
        assert (out / "audio" / "old1.wav").read_bytes() == b"RIFF-old"
        assert DatasetLoader().load(out, check_audio=True).size == 1

    @pytest.mark.parametrize(
        "event",
        [
            PhonemeEvent("a", start_ms=float("nan")),  # type: ignore[arg-type]
            PhonemeEvent("a", duration_ms="100"),  # type: ignore[arg-type]
            PhonemeEvent("a", pitch_hz=None),  # type: ignore[arg-type]
        ],
        ids=["nan start", "string duration", "missing pitch"],
    )
    def test_bad_event_values_are_dataset_errors(
        self, bridge: MavisBridge, event: PhonemeEvent
    ) -> None:
        with pytest.raises(DatasetError, match="missing or non-finite value"):
            bridge.phoneme_events_to_entry([event], "a", "s1", consent=True)

    def test_unhashable_session_id_is_a_dataset_error(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        with pytest.raises(DatasetError, match="session_id"):
            bridge.export_dataset(
                [{"events": sample_events, "transcript": "hi", "session_id": ["s1"]}],
                tmp_path / "ds",
                consent=True,
            )


# ---------------------------------------------------------------------------
# IML generation: word alignment and escaping
# ---------------------------------------------------------------------------


def _word_offsets(iml: str) -> dict[str, str | None]:
    """Each word's pitch offset in the IML (None when not marked up)."""
    doc = IMLParser().parse(iml)
    offsets: dict[str, str | None] = {}
    for child in doc.utterances[0].children:
        if isinstance(child, str):
            offsets.update({word: None for word in child.split()})
        else:
            offsets[str(child.children[0])] = child.pitch  # type: ignore[union-attr]
    return offsets


class TestWordAlignment:
    def test_words_follow_timing(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        """The loud, high phonemes of SUN and RISING mark those words up."""
        iml = bridge.phoneme_events_to_entry(sample_events, "the SUN is RISING", "s1").iml
        offsets = _word_offsets(iml)
        assert set(offsets) == {"the", "SUN", "is", "RISING"}
        assert offsets["SUN"] is not None and offsets["SUN"].startswith("+")
        assert offsets["RISING"] is not None and offsets["RISING"].startswith("+")
        assert offsets["the"] is not None and offsets["the"].startswith("-")
        assert offsets["is"] is not None and offsets["is"].startswith("-")

    def test_documented_example(self, bridge: MavisBridge) -> None:
        """docs/integrations/mavis.md: 'the' is [dh], 'SUN' is [s, ah]."""
        events = [
            PhonemeEvent("dh", start_ms=0, duration_ms=80, volume=0.5, pitch_hz=180.0),
            PhonemeEvent("s", start_ms=200, duration_ms=100, volume=0.85, pitch_hz=280.0,
                         vibrato=True),
            PhonemeEvent("ah", start_ms=300, duration_ms=120, volume=0.9, pitch_hz=300.0),
        ]
        entry = bridge.phoneme_events_to_entry(
            events, "the SUN", "session_001", emotion_label="joyful"
        )
        assert entry.iml == (
            '<utterance emotion="joyful" confidence="1.0">'
            '<prosody pitch="-29%" volume="-4dB">the</prosody> '
            '<prosody pitch="+14%" volume="+1dB">SUN</prosody></utterance>'
        )

    def test_no_event_is_dropped(self, bridge: MavisBridge) -> None:
        """Without timing, events are shared out in order and none is lost."""
        events = [PhonemeEvent("a", pitch_hz=200.0, volume=0.5)] * 6 + [
            PhonemeEvent("a", pitch_hz=400.0, volume=1.0)
        ]
        offsets = _word_offsets(bridge.phoneme_events_to_entry(events, "one two THREE", "s").iml)
        assert offsets["THREE"] is not None and offsets["THREE"].startswith("+")

    def test_fewer_events_than_words(self, bridge: MavisBridge) -> None:
        events = [PhonemeEvent("a", pitch_hz=200.0, volume=0.5),
                  PhonemeEvent("a", pitch_hz=400.0, volume=1.0)]
        offsets = _word_offsets(
            bridge.phoneme_events_to_entry(events, "one TWO three FOUR", "s").iml
        )
        assert offsets == {"one": None, "TWO": "-33%", "three": None, "FOUR": "+33%"}

    def test_phonemes_per_word(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        # All of SUN's and is's phonemes given to SUN: 'is' is left without events.
        entry = bridge.phoneme_events_to_entry(
            sample_events, "the SUN is RISING", "s1", phonemes_per_word=[2, 5, 0, 5]
        )
        assert _word_offsets(entry.iml)["is"] is None
        with pytest.raises(DatasetError, match="phonemes_per_word"):
            bridge.phoneme_events_to_entry(
                sample_events, "the SUN is RISING", "s1", phonemes_per_word=[2, 3, 2]
            )

    def test_baseline_values_are_omitted(self, bridge: MavisBridge) -> None:
        """A word that stands out in volume only used to carry pitch="+0%",
        and one that stands out in pitch only volume="+0dB" (spec 6.1: values
        equal to the speaker's baseline SHOULD be omitted)."""
        loud = [
            PhonemeEvent("a", start_ms=0, volume=0.2, pitch_hz=200.0),
            PhonemeEvent("b", start_ms=300, volume=0.9, pitch_hz=200.0),
        ]
        iml = bridge.phoneme_events_to_entry(loud, "soft LOUD", "s1").iml
        assert iml == (
            '<utterance><prosody volume="-9dB">soft</prosody> '
            '<prosody volume="+4dB">LOUD</prosody></utterance>'
        )
        assert IMLValidator().validate(iml).issues == []
        high = [
            PhonemeEvent("a", start_ms=i * 120, duration_ms=100, volume=0.2, pitch_hz=120 + i * 40)
            for i in range(8)
        ]
        iml = bridge.phoneme_events_to_entry(high, "hello there friend", "s1").iml
        assert "dB" not in iml and "+0%" not in iml
        assert IMLValidator().validate(iml).issues == []


class TestEscaping:
    @pytest.mark.parametrize("transcript", ["Tom & Jerry", "if a < b then", 'say "hi" > 3'])
    def test_text_is_escaped(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], transcript: str
    ) -> None:
        iml = bridge.phoneme_events_to_entry(sample_events, transcript, "s1").iml
        assert IMLValidator().validate(iml).valid
        assert IMLParser().to_plain_text(IMLParser().parse(iml)) == transcript

    def test_emotion_label_cannot_inject_attributes(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        label = 'sad" confidence="0.1'
        iml = bridge.phoneme_events_to_entry(sample_events, "hi", "s1", emotion_label=label).iml
        utterance = IMLParser().parse(iml).utterances[0]
        assert utterance.emotion == label
        assert utterance.confidence != 0.1


class TestCharactersXMLDoesNotAllow:
    """Typed transcripts can hold control characters (a terminal's escape
    sequences, say), which IMLParser.to_iml_string refuses to write."""

    @pytest.mark.parametrize(
        ("transcript", "cleaned", "codes"),
        [
            ("the \x1b[1mSUN\x1b[0m is up", "the [1mSUN[0m is up", "U+001B"),
            ("bell\x07 and nul\x00", "bell and nul", "U+0007, U+0000"),
            ("lone \ud800surrogate", "lone surrogate", "U+D800"),
            ("not\ufffe a\uffff char", "not a char", "U+FFFE, U+FFFF"),
            ("\x01\x02\x03\x04 four", " four", "U+0001, U+0002, U+0003, ..."),
        ],
    )
    def test_dropped_with_a_warning(
        self,
        bridge: MavisBridge,
        sample_events: list[PhonemeEvent],
        transcript: str,
        cleaned: str,
        codes: str,
    ) -> None:
        with pytest.warns(UserWarning, match="XML does not allow") as caught:
            entry = bridge.phoneme_events_to_entry(sample_events, transcript, "s1")
        assert len(caught) == 1
        assert f"({codes}) from the transcript of session 's1'" in str(caught[0].message)
        assert caught[0].filename == __file__  # points at the caller
        assert entry.transcript == cleaned
        assert IMLValidator().validate(entry.iml).valid
        assert IMLParser().to_plain_text(IMLParser().parse(entry.iml)) == " ".join(
            cleaned.split()
        )

    def test_word_separators_become_spaces(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        """U+001F splits words for str.split(), so the words (and a
        phonemes_per_word that counts them) stay the same."""
        with pytest.warns(UserWarning, match=r"Dropped 3 characters .*\(U\+001F, U\+000B\)"):
            entry = bridge.phoneme_events_to_entry(
                sample_events, "the\x1fSUN\x0bis\x1fRISING", "s1",
                phonemes_per_word=[2, 3, 2, 5],
            )
        assert entry.transcript == "the SUN is RISING"
        assert _word_offsets(entry.iml) == _word_offsets(
            bridge.phoneme_events_to_entry(
                sample_events, "the SUN is RISING", "s1", phonemes_per_word=[2, 3, 2, 5]
            ).iml
        )

    def test_nothing_else_left_is_an_empty_transcript(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        with pytest.raises(DatasetError, match="empty transcript.*XML does not allow"):
            bridge.phoneme_events_to_entry(sample_events, "\x1b\x07", "s1")

    def test_clean_transcript_gives_no_warning(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            entry = bridge.phoneme_events_to_entry(
                sample_events, "tab\tand\nnew line\u00e9\U0001F600", "s1"
            )
        assert entry.transcript == "tab\tand\nnew line\u00e9\U0001F600"

    def test_shares_the_parsers_xml_character_pattern(self) -> None:
        """One definition of the characters XML does not allow, not a copy."""
        from prosody_protocol import mavis_bridge, parser

        assert vars(mavis_bridge)["_XML_INVALID_CHAR_RE"] is parser._XML_INVALID_CHAR_RE

    def test_emotion_label_is_a_dataset_error(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        """A label is not typed text: it is refused, not changed."""
        with pytest.raises(DatasetError, match=r"emotion_label .* U\+0007, which XML does not"):
            bridge.phoneme_events_to_entry(sample_events, "hi", "s1", emotion_label="calm\x07")

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"annotator": "robot"}, "annotator must be one of"),
            ({"emotion_label": "calm\x07"}, "emotion_label"),
            ({"phonemes_per_word": [12, 0]}, "phonemes_per_word"),
        ],
        ids=["annotator", "emotion_label", "phonemes_per_word"],
    )
    def test_no_warning_when_the_entry_is_refused(
        self,
        bridge: MavisBridge,
        sample_events: list[PhonemeEvent],
        kwargs: dict[str, object],
        error: str,
    ) -> None:
        """The warning used to come before the rest of the input was checked."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(DatasetError, match=error):
                bridge.phoneme_events_to_entry(
                    sample_events, "the \x1bSUN is RISING", "s1", **kwargs  # type: ignore[arg-type]
                )

    def test_no_warning_when_an_event_is_refused(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        import warnings

        events = [*sample_events[:-1], PhonemeEvent("ng", 1040, 80, float("nan"), 230.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(DatasetError, match="non-finite"):
                bridge.phoneme_events_to_entry(events, "the \x1bSUN is RISING", "s1")

    def test_a_word_of_only_such_characters_is_removed(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        """Documented: phonemes_per_word counts the words of the cleaned text."""
        transcript = "the SUN \x1b is RISING"
        with pytest.raises(DatasetError, match="phonemes_per_word"):
            bridge.phoneme_events_to_entry(
                sample_events, transcript, "s1", phonemes_per_word=[2, 3, 0, 2, 5]
            )
        with pytest.warns(UserWarning, match="Dropped 1 character "):
            entry = bridge.phoneme_events_to_entry(
                sample_events, transcript, "s1", phonemes_per_word=[2, 3, 2, 5]
            )
        assert entry.transcript.split() == ["the", "SUN", "is", "RISING"]

    def test_export(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [
            {"events": sample_events, "transcript": "the \x1bSUN is RISING", "session_id": "s1"},
            {"events": sample_events, "transcript": "the SUN is RISING", "session_id": "s2"},
        ]
        with pytest.warns(UserWarning, match="session 's1'") as caught:
            dataset = bridge.export_dataset(sessions, tmp_path / "ds", consent=True)
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert [e.transcript for e in dataset.entries] == ["the SUN is RISING"] * 2
        assert DatasetLoader().load(tmp_path / "ds").size == 2

    @pytest.mark.parametrize("failure", ["later session", "existing entries"])
    def test_no_warning_when_the_export_fails(
        self,
        bridge: MavisBridge,
        sample_events: list[PhonemeEvent],
        tmp_path: Path,
        failure: str,
    ) -> None:
        import warnings

        sessions: list[dict[str, Any]] = [
            {"events": sample_events, "transcript": "the \x1bSUN is RISING", "session_id": "s1"},
        ]
        if failure == "later session":
            sessions.append({"events": [], "transcript": "hi", "session_id": "s2"})
        else:
            bridge.export_dataset(
                [{"events": sample_events, "transcript": "hi", "session_id": "s0"}],
                tmp_path / "ds", consent=True,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(DatasetError):
                bridge.export_dataset(sessions, tmp_path / "ds", consent=True)


# ---------------------------------------------------------------------------
# Integration: Mavis → sklearn training pipeline
# ---------------------------------------------------------------------------


class TestSklearnIntegration:
    def test_features_compatible_with_ser_model(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent]
    ) -> None:
        """Features from MavisBridge match the 7-dimensional SER model input."""
        features = bridge.extract_training_features(sample_events)
        assert features.shape == (7,)
        assert features.dtype == np.float64
        # No NaN or inf
        assert np.all(np.isfinite(features))

    def test_end_to_end_train_with_mavis_data(
        self,
        bridge: MavisBridge,
        sample_events: list[PhonemeEvent],
        quiet_events: list[PhonemeEvent],
    ) -> None:
        """Full pipeline: Mavis events → features → sklearn training."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Generate multiple "sessions" with different emotions
        sessions_data = []
        labels = []
        for _ in range(10):
            sessions_data.append(sample_events)
            labels.append("joyful")
        for _ in range(10):
            sessions_data.append(quiet_events)
            labels.append("sad")

        X = bridge.batch_extract_features(sessions_data)
        y = np.array(labels)

        assert X.shape == (20, 7)

        # Train a simple classifier
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_scaled, y)

        # Predict on new data
        new_features = bridge.extract_training_features(sample_events).reshape(1, -1)
        new_scaled = scaler.transform(new_features)
        pred = clf.predict(new_scaled)
        assert pred[0] in ("joyful", "sad")
