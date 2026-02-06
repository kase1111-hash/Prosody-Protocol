"""Mavis Data Bridge -- convert Mavis game data to prosody_protocol datasets.

Mavis is a vocal typing instrument that produces PhonemeEvent data with
prosody parameters (pitch_hz, volume, breathiness, vibrato, duration_ms).
This module converts that data into the prosody_protocol Dataset format
and extracts feature vectors for sklearn training.

Usage::

    from prosody_protocol.mavis_bridge import MavisBridge

    bridge = MavisBridge()

    # Convert a Mavis session to a dataset entry
    entry = bridge.phoneme_events_to_entry(
        events=[...],
        transcript="the SUN is RISING",
        session_id="session_001",
        emotion_label="joyful",
    )

    # Extract feature vectors for training
    features = bridge.extract_training_features(events)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import Dataset, DatasetEntry, DatasetLoader
from .exceptions import DatasetError


# ---------------------------------------------------------------------------
# Mavis data types (mirrors mavis.llm_processor.PhonemeEvent)
# ---------------------------------------------------------------------------


@dataclass
class PhonemeEvent:
    """A single phoneme event from a Mavis session.

    This mirrors the ``mavis.llm_processor.PhonemeEvent`` dataclass so that
    prosody_protocol can work with Mavis data without importing Mavis.
    """

    phoneme: str
    start_ms: int = 0
    duration_ms: int = 100
    volume: float = 0.5
    pitch_hz: float = 220.0
    vibrato: bool = False
    breathiness: float = 0.0
    harmony_intervals: list[int] | None = None


# Feature names in canonical order for sklearn training
MAVIS_FEATURE_NAMES = [
    "mean_pitch_hz",
    "pitch_range_hz",
    "mean_volume",
    "volume_range",
    "mean_breathiness",
    "speech_rate",  # phonemes per second
    "vibrato_ratio",
]


# ---------------------------------------------------------------------------
# Emotion mapping from Mavis emphasis levels
# ---------------------------------------------------------------------------


_EMPHASIS_EMOTION_MAP = {
    "shout": "angry",
    "loud": "joyful",
    "soft": "sad",
    "none": "neutral",
}


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------


class MavisBridge:
    """Convert Mavis PhonemeEvent streams to prosody_protocol datasets.

    The bridge produces:
    - Dataset entries (JSON) for the dataset infrastructure
    - Feature vectors (numpy) for sklearn training
    - IML markup from phoneme prosody parameters
    """

    def __init__(self, language: str = "en-US") -> None:
        self.language = language

    def phoneme_events_to_entry(
        self,
        events: list[PhonemeEvent],
        transcript: str,
        session_id: str,
        emotion_label: str | None = None,
        speaker_id: str | None = None,
    ) -> DatasetEntry:
        """Convert a sequence of PhonemeEvents into a DatasetEntry.

        Parameters
        ----------
        events:
            Ordered list of phoneme events from a Mavis session.
        transcript:
            Plain text transcript of the vocal typing session.
        session_id:
            Unique identifier for this recording session.
        emotion_label:
            Ground-truth emotion label. If None, inferred from prosody.
        speaker_id:
            Optional speaker identifier.
        """
        if not events:
            raise DatasetError("Cannot create entry from empty event list")

        if emotion_label is None:
            emotion_label = self._infer_emotion(events)

        iml = self._events_to_iml(events, transcript, emotion_label)

        return DatasetEntry(
            id=f"mavis_{session_id}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="mavis",
            language=self.language,
            audio_file=f"audio/{session_id}.wav",
            transcript=transcript,
            iml=iml,
            emotion_label=emotion_label,
            annotator="model",
            consent=True,
            speaker_id=speaker_id,
            metadata={"mavis_events": len(events)},
        )

    def extract_training_features(
        self, events: list[PhonemeEvent]
    ) -> np.ndarray:
        """Extract a 7-dimensional feature vector from a PhonemeEvent sequence.

        Returns a 1D array of shape (7,) with features:
        [mean_pitch_hz, pitch_range_hz, mean_volume, volume_range,
         mean_breathiness, speech_rate, vibrato_ratio]
        """
        if not events:
            return np.zeros(len(MAVIS_FEATURE_NAMES))

        pitches = [e.pitch_hz for e in events]
        volumes = [e.volume for e in events]
        breathiness = [e.breathiness for e in events]
        total_duration_s = sum(e.duration_ms for e in events) / 1000.0
        vibrato_count = sum(1 for e in events if e.vibrato)

        return np.array([
            float(np.mean(pitches)),
            float(max(pitches) - min(pitches)),
            float(np.mean(volumes)),
            float(max(volumes) - min(volumes)),
            float(np.mean(breathiness)),
            len(events) / max(total_duration_s, 0.001),
            vibrato_count / len(events),
        ], dtype=np.float64)

    def batch_extract_features(
        self, sessions: list[list[PhonemeEvent]]
    ) -> np.ndarray:
        """Extract features from multiple sessions into a feature matrix.

        Returns array of shape (n_sessions, 7).
        """
        return np.vstack([
            self.extract_training_features(session) for session in sessions
        ])

    def export_dataset(
        self,
        sessions: list[dict[str, Any]],
        output_dir: str | Path,
    ) -> Dataset:
        """Export multiple Mavis sessions as a prosody_protocol dataset.

        Parameters
        ----------
        sessions:
            List of dicts with keys: 'events' (list[PhonemeEvent]),
            'transcript' (str), 'session_id' (str), and optional
            'emotion_label' (str), 'speaker_id' (str).
        output_dir:
            Directory to write the dataset to.

        Returns the loaded Dataset.
        """
        output_path = Path(output_dir)
        entries_dir = output_path / "entries"
        entries_dir.mkdir(parents=True, exist_ok=True)

        entries: list[DatasetEntry] = []
        for session in sessions:
            entry = self.phoneme_events_to_entry(
                events=session["events"],
                transcript=session["transcript"],
                session_id=session["session_id"],
                emotion_label=session.get("emotion_label"),
                speaker_id=session.get("speaker_id"),
            )
            entries.append(entry)

            # Write entry JSON
            entry_dict = {
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
                "metadata": entry.metadata,
            }
            entry_file = entries_dir / f"{entry.id}.json"
            entry_file.write_text(json.dumps(entry_dict, indent=2), encoding="utf-8")

        # Write metadata
        meta = {
            "name": output_path.name,
            "version": "0.1.0",
            "size": len(entries),
            "source": "mavis",
            "language": self.language,
        }
        (output_path / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        return Dataset(name=output_path.name, entries=entries, metadata=meta)

    # -- Private helpers ----------------------------------------------------

    def _infer_emotion(self, events: list[PhonemeEvent]) -> str:
        """Infer a simple emotion label from aggregate prosody features."""
        mean_volume = sum(e.volume for e in events) / len(events)
        mean_breathiness = sum(e.breathiness for e in events) / len(events)
        mean_pitch = sum(e.pitch_hz for e in events) / len(events)

        if mean_volume > 0.8:
            return "angry" if mean_pitch > 300 else "joyful"
        if mean_breathiness > 0.5:
            return "sad"
        if mean_volume < 0.3:
            return "calm"
        return "neutral"

    def _events_to_iml(
        self,
        events: list[PhonemeEvent],
        transcript: str,
        emotion: str,
    ) -> str:
        """Build IML markup from phoneme events and transcript."""
        # Compute overall prosody stats for the utterance
        mean_pitch = sum(e.pitch_hz for e in events) / max(len(events), 1)
        mean_volume = sum(e.volume for e in events) / max(len(events), 1)

        confidence = self._compute_confidence(events)

        # Build prosody attributes for words with significant deviation
        words = transcript.split()
        if not words:
            return (
                f'<utterance emotion="{emotion}" confidence="{confidence:.2f}">'
                f'</utterance>'
            )

        parts: list[str] = []
        events_per_word = max(1, len(events) // max(len(words), 1))

        for i, word in enumerate(words):
            start_idx = i * events_per_word
            end_idx = min(start_idx + events_per_word, len(events))
            word_events = events[start_idx:end_idx] if start_idx < len(events) else []

            if word_events:
                word_pitch = sum(e.pitch_hz for e in word_events) / len(word_events)
                word_vol = sum(e.volume for e in word_events) / len(word_events)

                pitch_dev = (word_pitch - mean_pitch) / max(mean_pitch, 1) * 100
                vol_dev = (word_vol - mean_volume) / max(mean_volume, 0.01)

                if abs(pitch_dev) > 10 or abs(vol_dev) > 0.3:
                    pitch_str = f"+{pitch_dev:.0f}%" if pitch_dev > 0 else f"{pitch_dev:.0f}%"
                    vol_db = vol_dev * 10
                    vol_str = f"+{vol_db:.0f}dB" if vol_db > 0 else f"{vol_db:.0f}dB"
                    parts.append(
                        f'<prosody pitch="{pitch_str}" volume="{vol_str}">{word}</prosody>'
                    )
                    continue

            parts.append(word)

        inner = " ".join(parts)
        return (
            f'<utterance emotion="{emotion}" confidence="{confidence:.2f}">'
            f'{inner}'
            f'</utterance>'
        )

    def _compute_confidence(self, events: list[PhonemeEvent]) -> float:
        """Compute confidence score based on prosodic distinctiveness."""
        if len(events) < 2:
            return 0.5

        volumes = [e.volume for e in events]
        pitches = [e.pitch_hz for e in events]

        vol_range = max(volumes) - min(volumes)
        pitch_range = max(pitches) - min(pitches)

        # More dynamic range → higher confidence
        confidence = 0.5 + min(vol_range * 0.3 + pitch_range / 500, 0.4)
        return min(confidence, 0.95)
