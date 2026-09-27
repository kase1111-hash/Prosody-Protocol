#!/usr/bin/env python3
"""Generate audio fixtures for the analysis tests.

Creates short WAV files with known acoustic properties so that
ProsodyAnalyzer and AudioToIML tests can make deterministic assertions.

The tone fixtures need only the standard library. The speech fixtures are
synthesised word by word with espeak-ng and joined with silences of known
length, so their word timings and pauses are exact; each comes with a JSON
file holding that ground truth. They need espeak-ng on PATH plus numpy and
praat-parselmouth (``pip install -e '.[audio]'``).

Run once:  python tests/generate_audio_fixtures.py
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

AUDIO_DIR = Path(__file__).parent / "fixtures" / "audio"
SAMPLE_RATE = 16_000  # 16 kHz mono

# Speech fixtures: noise floor of a quiet room, and the gap between words
# of one phrase (far below the 200 ms pause threshold).
NOISE_FLOOR_DBFS = -60.0
WORD_GAP_MS = 50


def _write_wav(path: Path, samples: list[int], sr: int = SAMPLE_RATE) -> None:
    """Write 16-bit mono PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _sine_samples(freq: float, duration_s: float, amplitude: int = 20_000) -> list[int]:
    """Generate a pure sine tone."""
    n = int(SAMPLE_RATE * duration_s)
    return [
        int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        for i in range(n)
    ]


def _silence_samples(duration_s: float) -> list[int]:
    return [0] * int(SAMPLE_RATE * duration_s)


def generate_tones() -> None:
    # 1. Pure 220 Hz tone, 1 second (known F0)
    _write_wav(AUDIO_DIR / "tone_220hz.wav", _sine_samples(220.0, 1.0))

    # 2. Pure 440 Hz tone, 0.5 second
    _write_wav(AUDIO_DIR / "tone_440hz.wav", _sine_samples(440.0, 0.5))

    # 3. Silence, 1 second
    _write_wav(AUDIO_DIR / "silence_1s.wav", _silence_samples(1.0))

    # 4. Tone with a gap (for pause detection): 0.5s tone, 0.8s silence, 0.5s tone
    samples = (
        _sine_samples(220.0, 0.5)
        + _silence_samples(0.8)
        + _sine_samples(220.0, 0.5)
    )
    _write_wav(AUDIO_DIR / "tone_gap_tone.wav", samples)

    # 5. Rising pitch: linear sweep from 150 Hz to 350 Hz over 1 second.
    # The phase is the integral of the frequency, 2*pi*(150*t + 100*t**2).
    n = int(SAMPLE_RATE * 1.0)
    sweep: list[int] = []
    for i in range(n):
        t = i / SAMPLE_RATE
        phase = 2 * math.pi * (150.0 * t + 100.0 * t * t)
        sweep.append(int(15_000 * math.sin(phase)))
    _write_wav(AUDIO_DIR / "rising_pitch.wav", sweep)

    # 6. Loud then quiet (intensity change): 0.5s loud, 0.5s quiet
    loud = _sine_samples(220.0, 0.5, amplitude=30_000)
    quiet = _sine_samples(220.0, 0.5, amplitude=3_000)
    _write_wav(AUDIO_DIR / "loud_quiet.wav", loud + quiet)

    # 7. Short utterance-length clip: 2s of 220Hz (for integration tests)
    _write_wav(AUDIO_DIR / "short_speech.wav", _sine_samples(220.0, 2.0))


# ---------------------------------------------------------------------------
# Speech fixtures (espeak-ng)
# ---------------------------------------------------------------------------


def _espeak(text: str, pitch: int = 50, amplitude: int = 100) -> Any:
    """Synthesise *text* at 16 kHz with leading and trailing silence trimmed."""
    import numpy as np
    import parselmouth

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "word.wav"
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", "160", "-p", str(pitch), "-a", str(amplitude),
             "-z", "-w", str(path), text],
            check=True,
        )
        samples = parselmouth.Sound(str(path)).resample(SAMPLE_RATE).values[0]
    audible = np.flatnonzero(np.abs(samples) > 1e-3)
    return samples[audible[0]:audible[-1] + 1]


def _build_speech(name: str, plan: list[tuple[str, Any]], seed: int) -> dict[str, Any]:
    """Join words and silences, add a noise floor, and write WAV + JSON truth.

    *plan* items are ``("word", {espeak options})`` or ``("silence", ms)``;
    silences of 200 ms or more are recorded as pauses.
    """
    import numpy as np

    parts = []
    words: list[dict[str, Any]] = []
    pauses: list[dict[str, int]] = []
    position = 0
    for item, arg in plan:
        if item == "silence":
            chunk = np.zeros(int(SAMPLE_RATE * arg / 1000))
            if arg >= 200:
                pauses.append({"start_ms": round(position / SAMPLE_RATE * 1000),
                               "end_ms": round((position + len(chunk)) / SAMPLE_RATE * 1000)})
        else:
            chunk = _espeak(item.rstrip(".,!?"), **arg)
            words.append({"word": item,
                          "start_ms": round(position / SAMPLE_RATE * 1000),
                          "end_ms": round((position + len(chunk)) / SAMPLE_RATE * 1000)})
        parts.append(chunk)
        position += len(chunk)

    signal = np.concatenate(parts)
    signal = 0.5 * signal / np.abs(signal).max()
    rng = np.random.default_rng(seed)
    signal = signal + 10 ** (NOISE_FLOOR_DBFS / 20) * rng.standard_normal(len(signal))
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767).astype(int)
    _write_wav(AUDIO_DIR / f"{name}.wav", [int(v) for v in pcm])

    truth = {
        "sample_rate": SAMPLE_RATE,
        "noise_floor_dbfs": NOISE_FLOOR_DBFS,
        "duration_ms": round(len(signal) / SAMPLE_RATE * 1000),
        "words": words,
        "pauses": pauses,
    }
    return truth


def generate_speech() -> None:
    gap = ("silence", WORD_GAP_MS)

    # "I told you -- to call me -- yesterday." Nine syllables, two pauses
    # (600 ms, 300 ms) and 250 ms of silence at each end. "told" is spoken
    # higher and louder than the rest, as an emphasised word.
    truth = _build_speech("speech_pauses", [
        ("silence", 250),
        ("I", {}), gap, ("told", {"pitch": 85, "amplitude": 170}), gap, ("you", {}),
        ("silence", 600),
        ("to", {}), gap, ("call", {}), gap, ("me", {}),
        ("silence", 300),
        ("yesterday.", {}),
        ("silence", 250),
    ], seed=7)
    truth["syllables"] = 9
    truth["emphasized"] = "told"
    (AUDIO_DIR / "speech_pauses.json").write_text(json.dumps(truth, indent=2) + "\n")

    # The same voice at its normal pitch, for use as calibration audio.
    truth = _build_speech("speech_calibration", [
        ("silence", 200),
        ("this", {}), gap, ("is", {}), gap, ("how", {}), gap, ("I", {}), gap,
        ("normally", {}), gap, ("sound.", {}),
        ("silence", 200),
    ], seed=11)
    (AUDIO_DIR / "speech_calibration.json").write_text(json.dumps(truth, indent=2) + "\n")

    # The same voice with every word raised in pitch.
    raised = {"pitch": 85}
    truth = _build_speech("speech_raised", [
        ("silence", 200),
        ("please", raised), gap, ("call", raised), gap, ("me", raised), gap, ("back.", raised),
        ("silence", 200),
    ], seed=13)
    (AUDIO_DIR / "speech_raised.json").write_text(json.dumps(truth, indent=2) + "\n")


def generate() -> None:
    generate_tones()
    if shutil.which("espeak-ng") is None:
        print("espeak-ng not found; speech fixtures were not regenerated")
    else:
        generate_speech()
    print(f"Generated {len(list(AUDIO_DIR.glob('*.wav')))} WAV fixtures in {AUDIO_DIR}")


if __name__ == "__main__":
    generate()
