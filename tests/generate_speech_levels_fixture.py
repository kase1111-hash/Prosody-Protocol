#!/usr/bin/env python3
"""Generate tests/fixtures/audio/speech_levels.{wav,json} (needs espeak-ng).

One espeak-ng sentence spoken three times at the same tempo but at falling
loudness (espeak-ng -a 100, 40, 25), then a loud, higher sentence (-a 200,
-p 70) followed after only 80 ms (not a pause) by a quiet one (-a 30). Each
sentence is synthesised whole (continuous speech, not word by word) and the
sentences are joined with silences of known length; the whole file is
scaled to a peak of 0.5 and given a -60 dBFS noise floor, like the other
speech fixtures. The JSON records each sentence's span, espeak-ng options,
syllable count and level (dB, relative to the first sentence, measured
before the noise is added), and the pauses (silences of 200 ms or more).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import parselmouth

SAMPLE_RATE = 16_000
NOISE_FLOOR_DBFS = -60.0
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fixtures" / "audio"

REFERENCE = "I checked the schedule this morning"


def espeak(text: str, amplitude: int, pitch: int = 50) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.wav"
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", "160", "-p", str(pitch), "-a", str(amplitude),
             "-z", "-w", str(path), text],
            check=True,
        )
        samples = parselmouth.Sound(str(path)).resample(SAMPLE_RATE).values[0]
    audible = np.flatnonzero(np.abs(samples) > 1e-4)
    return samples[audible[0]:audible[-1] + 1]


PLAN: list[tuple[str, object]] = [
    ("silence", 250),
    ("say", (REFERENCE, 100, 50, 8)),
    ("silence", 600),
    ("say", (REFERENCE, 40, 50, 8)),
    ("silence", 400),
    ("say", (REFERENCE, 25, 50, 8)),
    ("silence", 300),
    ("say", ("This is completely unacceptable", 200, 70, 10)),
    ("silence", 80),
    ("say", ("We need the report by Friday", 30, 50, 8)),
    ("silence", 250),
]


def main() -> None:
    parts: list[np.ndarray] = []
    sentences: list[dict[str, object]] = []
    pauses: list[dict[str, int]] = []
    position = 0
    for kind, arg in PLAN:
        if kind == "silence":
            assert isinstance(arg, int)
            chunk = np.zeros(int(SAMPLE_RATE * arg / 1000))
            if arg >= 200:
                pauses.append({"start_ms": round(position / SAMPLE_RATE * 1000),
                               "end_ms": round((position + chunk.size) / SAMPLE_RATE * 1000)})
        else:
            text, amplitude, pitch, syllables = arg  # type: ignore[misc]
            chunk = espeak(text, amplitude, pitch)
            sentences.append({
                "text": text, "amplitude": amplitude, "pitch": pitch, "syllables": syllables,
                "start_ms": round(position / SAMPLE_RATE * 1000),
                "end_ms": round((position + chunk.size) / SAMPLE_RATE * 1000),
                "rms": float(np.sqrt(np.mean(chunk**2))),
            })
        parts.append(chunk)
        position += chunk.size
    signal = np.concatenate(parts)
    scale = 0.5 / np.abs(signal).max()
    reference_rms = sentences[0]["rms"]
    for s in sentences:
        s["level_db"] = round(20 * float(np.log10(s.pop("rms") / reference_rms)), 2)  # type: ignore[arg-type]
    signal = signal * scale
    rng = np.random.default_rng(17)
    signal = signal + 10 ** (NOISE_FLOOR_DBFS / 20) * rng.standard_normal(signal.size)
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(OUT / "speech_levels.wav"), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    truth = {
        "sample_rate": SAMPLE_RATE,
        "noise_floor_dbfs": NOISE_FLOOR_DBFS,
        "duration_ms": round(signal.size / SAMPLE_RATE * 1000),
        "generated_with": "espeak-ng -v en-us -s 160 -p <pitch> -a <amplitude> -z; "
                          "sentences joined with silences, peak 0.5, -60 dBFS noise (seed 17)",
        "sentences": sentences,
        "pauses": pauses,
    }
    (OUT / "speech_levels.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(json.dumps(truth, indent=1))


if __name__ == "__main__":
    main()
