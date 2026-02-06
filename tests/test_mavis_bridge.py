"""Tests for the Mavis Data Bridge (Phase 16).

Covers:
- PhonemeEvent to DatasetEntry conversion
- Feature extraction for sklearn training
- Batch feature extraction
- Dataset export
- IML generation from phoneme events
- Emotion inference from prosody
- Edge cases (empty events, single event, etc.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prosody_protocol.mavis_bridge import (
    MAVIS_FEATURE_NAMES,
    MavisBridge,
    PhonemeEvent,
)
from prosody_protocol import DatasetEntry, DatasetLoader, IMLParser, IMLValidator


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
        )
        assert isinstance(entry, DatasetEntry)
        assert entry.id == "mavis_test_001"
        assert entry.source == "mavis"
        assert entry.language == "en-US"
        assert entry.transcript == "the SUN is RISING"
        assert entry.emotion_label == "joyful"
        assert entry.consent is True
        assert entry.annotator == "model"

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

    def test_iml_contains_emotion(self, bridge: MavisBridge, sample_events: list[PhonemeEvent]) -> None:
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

    def test_speaker_id_propagated(self, bridge: MavisBridge, sample_events: list[PhonemeEvent]) -> None:
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
    def test_feature_vector_shape(self, bridge: MavisBridge, sample_events: list[PhonemeEvent]) -> None:
        features = bridge.extract_training_features(sample_events)
        assert features.shape == (len(MAVIS_FEATURE_NAMES),)

    def test_feature_values_reasonable(self, bridge: MavisBridge, sample_events: list[PhonemeEvent]) -> None:
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


# ---------------------------------------------------------------------------
# Dataset export tests
# ---------------------------------------------------------------------------


class TestDatasetExport:
    def test_export_creates_directory_structure(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        sessions = [
            {"events": sample_events, "transcript": "the SUN", "session_id": "s1", "emotion_label": "joyful"},
            {"events": sample_events, "transcript": "RISING", "session_id": "s2", "emotion_label": "neutral"},
        ]
        dataset = bridge.export_dataset(sessions, tmp_path / "mavis_ds")

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
        bridge.export_dataset(sessions, tmp_path / "ds")

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
        bridge.export_dataset(sessions, tmp_path / "ds")

        meta = json.loads((tmp_path / "ds" / "metadata.json").read_text())
        assert meta["size"] == 3
        assert meta["source"] == "mavis"

    def test_exported_dataset_loadable(
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], tmp_path: Path
    ) -> None:
        """Verify the exported dataset can be loaded back by DatasetLoader."""
        sessions = [
            {"events": sample_events, "transcript": "hello", "session_id": "s1", "emotion_label": "neutral"},
        ]
        bridge.export_dataset(sessions, tmp_path / "ds")

        loader = DatasetLoader(validate_iml=True)
        loaded = loader.load(tmp_path / "ds")
        assert loaded.size == 1
        assert loaded.entries[0].source == "mavis"


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
        self, bridge: MavisBridge, sample_events: list[PhonemeEvent], quiet_events: list[PhonemeEvent]
    ) -> None:
        """Full pipeline: Mavis events → features → sklearn training."""
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
