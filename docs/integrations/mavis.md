# Mavis Integration Guide

[Mavis](https://github.com/kase1111-hash/Mavis) is a vocal typing instrument that transforms keyboard input into singing through AI-powered phoneme processing. Mavis produces rich prosody data (pitch, volume, breathiness, vibrato) that maps directly to IML annotations.

## Overview

The `MavisBridge` converts Mavis `PhonemeEvent` streams into:
- **Dataset entries** for the prosody_protocol dataset infrastructure
- **Feature vectors** for sklearn-based ML training
- **IML markup** preserving the prosodic intent of each typing session

## Basic Usage

```python
from prosody_protocol import MavisBridge, PhonemeEvent

bridge = MavisBridge(language="en-US")

# Create events from a Mavis session
events = [
    PhonemeEvent(phoneme="dh", start_ms=0, duration_ms=80,
                 volume=0.5, pitch_hz=180.0),
    PhonemeEvent(phoneme="s", start_ms=200, duration_ms=100,
                 volume=0.85, pitch_hz=280.0, vibrato=True),
    PhonemeEvent(phoneme="ah", start_ms=300, duration_ms=120,
                 volume=0.9, pitch_hz=300.0),
]

# Convert to a dataset entry
entry = bridge.phoneme_events_to_entry(
    events=events,
    transcript="the SUN",
    session_id="session_001",
    emotion_label="joyful",
)

print(entry.iml)
# <utterance emotion="joyful" confidence="0.72">the <prosody pitch="+28%" volume="+4dB">SUN</prosody></utterance>
```

## Feature Extraction for Training

The bridge extracts 7-dimensional feature vectors compatible with the sklearn training pipeline:

| Feature | Description |
|---------|-------------|
| `mean_pitch_hz` | Average F0 across all phonemes |
| `pitch_range_hz` | Difference between max and min pitch |
| `mean_volume` | Average volume (0.0-1.0) |
| `volume_range` | Dynamic volume range |
| `mean_breathiness` | Average breathiness (0.0-1.0) |
| `speech_rate` | Phonemes per second |
| `vibrato_ratio` | Fraction of phonemes with vibrato |

```python
import numpy as np
from prosody_protocol import MavisBridge

bridge = MavisBridge()

# Extract features from multiple sessions
sessions = [session_1_events, session_2_events, session_3_events]
X = bridge.batch_extract_features(sessions)
# X.shape == (3, 7)

# Train with sklearn
from sklearn.linear_model import LogisticRegression
labels = np.array(["joyful", "angry", "calm"])
clf = LogisticRegression()
clf.fit(X, labels)
```

## Exporting Datasets

Export Mavis sessions as a complete prosody_protocol dataset:

```python
bridge = MavisBridge()

sessions = [
    {
        "events": session_1_events,
        "transcript": "the SUN is RISING",
        "session_id": "s001",
        "emotion_label": "joyful",
        "speaker_id": "user_42",
    },
    {
        "events": session_2_events,
        "transcript": "falling gently down",
        "session_id": "s002",
        "emotion_label": "sad",
    },
]

dataset = bridge.export_dataset(sessions, "datasets/mavis-corpus")
```

This creates:
```
datasets/mavis-corpus/
├── metadata.json
└── entries/
    ├── mavis_s001.json
    └── mavis_s002.json
```

The exported dataset can then be loaded and used with the training pipeline:

```python
from prosody_protocol import DatasetLoader

loader = DatasetLoader()
dataset = loader.load("datasets/mavis-corpus")
print(f"Loaded {dataset.size} entries from Mavis")
```

## Emotion Inference

If no `emotion_label` is provided, the bridge infers emotion from prosody:

| Prosody Pattern | Inferred Emotion |
|-----------------|-----------------|
| High volume + high pitch | `angry` |
| High volume + low pitch | `joyful` |
| High breathiness | `sad` |
| Low volume | `calm` |
| Normal range | `neutral` |

## Connecting to the Mavis Pipeline

When Mavis is installed, you can bridge directly from the game pipeline:

```python
from mavis.pipeline import create_pipeline
from prosody_protocol import MavisBridge, PhonemeEvent

# Run Mavis pipeline
pipeline = create_pipeline()
pipeline.feed_text("the SUN is RISING")

# Collect phoneme events
events = []
while pipeline.output_buffer.level > 0:
    state = pipeline.tick()
    event = pipeline.output_buffer.pop()
    events.append(PhonemeEvent(
        phoneme=event.phoneme,
        start_ms=event.start_ms,
        duration_ms=event.duration_ms,
        volume=event.volume,
        pitch_hz=event.pitch_hz,
        vibrato=event.vibrato,
        breathiness=event.breathiness,
    ))

# Convert to IML
bridge = MavisBridge()
entry = bridge.phoneme_events_to_entry(
    events=events,
    transcript="the SUN is RISING",
    session_id="live_session_001",
)
print(entry.iml)
```
