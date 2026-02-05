#!/usr/bin/env python3
"""Generate synthetic audio fixtures for Phase 4 tests.

Creates short WAV files with known acoustic properties so that
ProsodyAnalyzer and AudioToIML tests can make deterministic assertions.

Run once:  python tests/generate_audio_fixtures.py
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

AUDIO_DIR = Path(__file__).parent / "fixtures" / "audio"
SAMPLE_RATE = 16_000  # 16 kHz mono


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
    import math

    n = int(SAMPLE_RATE * duration_s)
    return [
        int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        for i in range(n)
    ]


def _silence_samples(duration_s: float) -> list[int]:
    return [0] * int(SAMPLE_RATE * duration_s)


def generate() -> None:
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

    # 5. Rising pitch: sweep from 150 Hz to 350 Hz over 1 second
    import math

    n = int(SAMPLE_RATE * 1.0)
    sweep: list[int] = []
    for i in range(n):
        t = i / SAMPLE_RATE
        freq = 150.0 + 200.0 * (t / 1.0)  # linear sweep
        sweep.append(int(15_000 * math.sin(2 * math.pi * freq * t)))
    _write_wav(AUDIO_DIR / "rising_pitch.wav", sweep)

    # 6. Loud then quiet (intensity change): 0.5s loud, 0.5s quiet
    loud = _sine_samples(220.0, 0.5, amplitude=30_000)
    quiet = _sine_samples(220.0, 0.5, amplitude=3_000)
    _write_wav(AUDIO_DIR / "loud_quiet.wav", loud + quiet)

    # 7. Short utterance-length clip: 2s of 220Hz (for integration tests)
    _write_wav(AUDIO_DIR / "short_speech.wav", _sine_samples(220.0, 2.0))

    print(f"Generated {len(list(AUDIO_DIR.glob('*.wav')))} WAV fixtures in {AUDIO_DIR}")


if __name__ == "__main__":
    generate()
