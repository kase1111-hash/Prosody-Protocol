#!/usr/bin/env python3
"""Generate the ``training_synthetic`` dataset fixture.

Writes ``tests/fixtures/datasets/training_synthetic``: ten short clips of
espeak-ng speech (16 kHz mono, 16-bit, under two seconds each) with their
dataset entries. Each clip is spoken with settings that follow its emotion
label, the way acted emotional speech tends to differ:

- angry, frustrated: louder and faster; angry also higher
- joyful: higher, with a wider pitch range
- fearful: higher and faster, but not loud
- sad: quieter, slower and lower, with a narrow pitch range
- calm: slow and soft
- sarcastic: slow, with an exaggerated pitch range
- neutral: espeak-ng's default delivery

so a model trained on the fixture has something real to learn from, unlike
the identical 440 Hz tones it replaces. It is synthetic speech whose
"emotion" is only these settings: it tests the training pipeline and does
not teach a model anything about real emotional speech.

The clips are not normalised one by one: all of them share one gain, so
the loudness differences between labels survive. A faint noise floor
(-60 dBFS, fixed seed) stands in for a quiet room. The output depends only
on the espeak-ng version (1.51 made the checked-in files).

Needs espeak-ng on PATH, numpy and praat-parselmouth
(``pip install -e '.[audio]'``). Run once:

    python tests/generate_training_fixture.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).parent / "fixtures" / "datasets" / "training_synthetic"
SAMPLE_RATE = 16_000
# Silence before and after the speech in each clip.
PADDING_S = 0.12
MAX_DURATION_S = 2.0
NOISE_FLOOR_DBFS = -60.0
# The loudest clip peaks at this level; every clip gets the same gain.
PEAK_LEVEL = 0.9
VOICE = "en-us"


@dataclass(frozen=True)
class Clip:
    id: str
    label: str
    text: str
    speed: int  # espeak-ng -s, words per minute
    pitch: int  # espeak-ng -p, 0-99
    amplitude: int  # espeak-ng -a, 0-200
    pitch_range: str  # SSML <prosody range>


CLIPS = (
    Clip("synth_001", "neutral", "The weather is nice today.", 165, 50, 60, "medium"),
    Clip("synth_002", "angry", "I cannot believe you did that!", 215, 70, 125, "high"),
    Clip("synth_003", "joyful", "This is the best day ever!", 180, 80, 90, "x-high"),
    Clip("synth_004", "sad", "I miss the old days.", 130, 25, 30, "x-low"),
    Clip("synth_005", "fearful", "Did you hear that noise outside?", 200, 85, 50, "high"),
    Clip("synth_006", "sarcastic", "Oh great, another meeting.", 150, 45, 65, "x-high"),
    Clip("synth_007", "calm", "Take a moment to breathe.", 140, 40, 45, "low"),
    Clip("synth_008", "frustrated", "This thing never works properly.", 190, 45, 100, "medium"),
    Clip("synth_009", "neutral", "Please pass me the salt.", 165, 50, 60, "medium"),
    Clip("synth_010", "angry", "Stop doing that right now!", 215, 70, 125, "high"),
)


def _speak(clip: Clip) -> Any:
    """The clip's text spoken by espeak-ng, at 16 kHz, without surrounding silence."""
    import numpy as np
    import parselmouth

    ssml = f'<speak><prosody range="{clip.pitch_range}">{clip.text}</prosody></speak>'
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "speech.wav"
        subprocess.run(
            ["espeak-ng", "-m", "-z", "-v", VOICE, "-s", str(clip.speed), "-p", str(clip.pitch),
             "-a", str(clip.amplitude), "-w", str(path), ssml],
            check=True,
        )
        samples = parselmouth.Sound(str(path)).resample(SAMPLE_RATE).values[0]
    audible = np.flatnonzero(np.abs(samples) > 1e-3)
    return samples[audible[0]:audible[-1] + 1]


def _write_wav(path: Path, samples: Any) -> None:
    import numpy as np

    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _entry(clip: Clip) -> dict[str, Any]:
    return {
        "id": clip.id,
        "timestamp": "2025-06-01T12:00:00Z",
        "source": "synthetic",
        "language": "en-US",
        "audio_file": f"audio/{clip.id}.wav",
        "transcript": clip.text,
        "iml": f'<utterance emotion="{clip.label}" confidence="0.9">{clip.text}</utterance>',
        "speaker_id": "synth_speaker",
        "emotion_label": clip.label,
        "annotator": "model",
        "consent": True,
        "metadata": {
            "synthetic": True,
            "generator": "tests/generate_training_fixture.py",
            "tts": f"espeak-ng, voice {VOICE}",
            "speed_wpm": clip.speed,
            "pitch": clip.pitch,
            "amplitude": clip.amplitude,
            "pitch_range": clip.pitch_range,
        },
    }


def generate() -> None:
    import numpy as np

    speech = [_speak(clip) for clip in CLIPS]
    gain = PEAK_LEVEL / max(float(np.abs(s).max()) for s in speech)
    pad = np.zeros(int(PADDING_S * SAMPLE_RATE))
    rng = np.random.default_rng(2025)
    (DATASET_DIR / "audio").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "entries").mkdir(parents=True, exist_ok=True)
    for clip, samples in zip(CLIPS, speech, strict=True):
        signal = np.concatenate([pad, gain * samples, pad])
        if signal.size > MAX_DURATION_S * SAMPLE_RATE:
            raise SystemExit(
                f"{clip.id} is {signal.size / SAMPLE_RATE:.2f} s long; "
                f"shorten its text or speed it up (at most {MAX_DURATION_S} s)"
            )
        signal = signal + 10 ** (NOISE_FLOOR_DBFS / 20) * rng.standard_normal(signal.size)
        _write_wav(DATASET_DIR / "audio" / f"{clip.id}.wav", signal)
        entry_path = DATASET_DIR / "entries" / f"{clip.id}.json"
        entry_path.write_text(json.dumps(_entry(clip), indent=2) + "\n", encoding="utf-8")
    metadata = {
        "name": "training_synthetic",
        "version": "0.2.0",
        "description": (
            "10 short espeak-ng clips whose speed, pitch, pitch range and loudness follow "
            "their emotion label (tests/generate_training_fixture.py). Synthetic: it tests "
            "the training pipeline, it does not teach real emotion recognition."
        ),
        "size": len(CLIPS),
    }
    (DATASET_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(CLIPS)} clips and entries to {DATASET_DIR}")


if __name__ == "__main__":
    if shutil.which("espeak-ng") is None:
        sys.exit("espeak-ng is not on PATH; the fixture was not regenerated")
    generate()
