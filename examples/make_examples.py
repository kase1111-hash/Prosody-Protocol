#!/usr/bin/env python3
"""Regenerate the example files in this directory.

    python examples/make_examples.py

Needs espeak-ng on PATH and the ``audio`` extra (numpy and
praat-parselmouth: ``pip install 'prosody-protocol[audio]'``). The output is
deterministic for a given espeak-ng version (these files were made with
espeak-ng 1.51).

What it writes:

``speech.wav``
    "I never said -- she STOLE my money." in a synthetic voice: 16 kHz mono,
    16-bit, about 4 s. Each word is synthesised separately with espeak-ng and the words
    are joined with 50 ms gaps, except for a 600 ms pause after "said".
    "stole" is spoken with a higher pitch and more loudly than the other
    words. A faint noise floor (-60 dBFS, fixed seed) stands in for a quiet
    room.
``speech.whisper.json``
    The exact word timings of ``speech.wav``, in the shape of an
    openai-whisper ``transcribe(..., word_timestamps=True)`` result
    (``segments[].words[]``, words with Whisper's leading space, times in
    seconds; recogniser statistics such as tokens and log-probabilities are
    left out). Since the audio is built word by word, the timings are the
    ground truth, not a recogniser's estimate; ``probability`` is 1.0.
``speech.txt``
    The plain transcript.
``monotone.wav``
    Four short sentences in a flat, monotone synthetic voice (espeak-ng with
    ``<prosody range="x-low">``), about 6.5 s, built word by word like
    ``speech.wav``: three at an ordinary rate, then "And we got the grant!"
    much faster -- the speaker of ``profile.json``, whose excitement shows as
    speed rather than pitch.
``monotone.deepgram.json``
    The exact word timings of ``monotone.wav``, in the shape of a Deepgram
    pre-recorded transcription response (``results.channels[0]
    .alternatives[0].words[]`` with ``word``, ``punctuated_word`` and times
    in seconds), to show that word timings from any supported recogniser
    work.
``profile.json`` and ``sarcasm.iml``
    Hand-written; this script checks that they are still valid.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 16_000
NOISE_FLOOR_DBFS = -60.0
WORD_GAP_MS = 50
SEED = 2024

# (token as Whisper writes it, espeak-ng options, silence after it in ms)
PLAN: list[tuple[str, dict[str, int], int]] = [
    ("I", {}, WORD_GAP_MS),
    ("never", {}, WORD_GAP_MS),
    ("said", {}, 600),
    ("she", {}, WORD_GAP_MS),
    ("stole", {"pitch": 85, "amplitude": 170}, WORD_GAP_MS),
    ("my", {}, WORD_GAP_MS),
    ("money.", {}, 0),
]
LEADING_SILENCE_MS = 250
TRAILING_SILENCE_MS = 300

# monotone.wav: (sentence, espeak-ng words per minute, gap between its words
# in ms, silence after it in ms). Every word is spoken with a flat pitch.
MONOTONE_PLAN: list[tuple[str, int, int, int]] = [
    ("I read the list.", 175, 30, 300),
    ("The room is booked.", 175, 30, 300),
    ("I have the slides.", 175, 30, 300),
    ("And we got the grant!", 300, 10, 0),
]
MONOTONE_EDGE_SILENCE_MS = 150


def _espeak(
    text: str, pitch: int = 50, amplitude: int = 100, speed: int = 160, flat: bool = False
) -> Any:
    """One word from espeak-ng at 16 kHz, with its leading and trailing silence cut."""
    import numpy as np
    import parselmouth

    command = ["espeak-ng", "-v", "en-us", "-s", str(speed), "-p", str(pitch),
               "-a", str(amplitude), "-z"]
    if flat:
        command += ["-m", f'<speak><prosody range="x-low">{text}</prosody></speak>']
    else:
        command.append(text)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "word.wav"
        subprocess.run([*command[:-1], "-w", str(path), command[-1]], check=True)
        samples = parselmouth.Sound(str(path)).resample(SAMPLE_RATE).values[0]
    audible = np.flatnonzero(np.abs(samples) > 1e-3)
    return samples[audible[0]:audible[-1] + 1]


def _ms(samples: int) -> int:
    return round(samples / SAMPLE_RATE * 1000)


def _build(
    plan: list[tuple[str, dict[str, Any], int]], leading_ms: int, trailing_ms: int, name: str
) -> list[dict[str, Any]]:
    """Speak *plan* word by word into the WAV file *name*; returns the word timings.

    Each entry is (token, espeak-ng options, silence after it in ms). Times
    are in seconds, and each token keeps its punctuation.
    """
    import numpy as np

    parts = [np.zeros(SAMPLE_RATE * leading_ms // 1000)]
    position = len(parts[0])
    words: list[dict[str, Any]] = []
    for token, options, silence_ms in plan:
        chunk = _espeak(token.rstrip(".,!?"), **options)
        words.append({
            "word": token,
            "start": round(_ms(position) / 1000, 3),
            "end": round(_ms(position + len(chunk)) / 1000, 3),
        })
        silence = np.zeros(SAMPLE_RATE * silence_ms // 1000)
        parts += [chunk, silence]
        position += len(chunk) + len(silence)
    parts.append(np.zeros(SAMPLE_RATE * trailing_ms // 1000))

    signal = np.concatenate(parts)
    signal = 0.5 * signal / np.abs(signal).max()
    rng = np.random.default_rng(SEED)
    signal = signal + 10 ** (NOISE_FLOOR_DBFS / 20) * rng.standard_normal(len(signal))
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(HERE / name), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())
    return words


def make_speech() -> None:
    timings = _build(PLAN, LEADING_SILENCE_MS, TRAILING_SILENCE_MS, "speech.wav")
    words = [
        {"word": f" {w['word']}", "start": w["start"], "end": w["end"], "probability": 1.0}
        for w in timings
    ]
    text = "".join(w["word"] for w in words)
    result = {
        "text": text,
        "segments": [{
            "id": 0,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": text,
            "words": words,
        }],
        "language": "en",
    }
    (HERE / "speech.whisper.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "speech.txt").write_text(text.strip() + "\n")


def make_monotone() -> None:
    plan: list[tuple[str, dict[str, Any], int]] = []
    for sentence, speed, gap_ms, after_ms in MONOTONE_PLAN:
        tokens = sentence.split()
        plan += [
            (token, {"speed": speed, "flat": True}, gap_ms if i < len(tokens) - 1 else after_ms)
            for i, token in enumerate(tokens)
        ]
    timings = _build(
        plan, MONOTONE_EDGE_SILENCE_MS, MONOTONE_EDGE_SILENCE_MS, "monotone.wav"
    )
    words = [
        {
            "word": w["word"].rstrip(".,!?").lower(),
            "start": w["start"],
            "end": w["end"],
            "confidence": 1.0,
            "punctuated_word": w["word"],
        }
        for w in timings
    ]
    transcript = " ".join(w["punctuated_word"] for w in words)
    result = {
        "metadata": {"channels": 1, "models": ["example"]},
        "results": {
            "channels": [{
                "alternatives": [{
                    "transcript": transcript,
                    "confidence": 1.0,
                    "words": words,
                }],
            }],
        },
    }
    (HERE / "monotone.deepgram.json").write_text(json.dumps(result, indent=2) + "\n")


def check_hand_written() -> None:
    sys.path.insert(0, str(HERE.parent / "src"))
    from prosody_protocol import IMLValidator, ProfileLoader

    result = IMLValidator().validate_file(HERE / "sarcasm.iml")
    if not result.valid or result.warnings:
        raise SystemExit(f"sarcasm.iml is not valid: {result.issues}")
    loader = ProfileLoader()
    profile = loader.load(HERE / "profile.json")
    if not loader.validate(profile).valid:
        raise SystemExit(f"profile.json is not valid: {loader.validate(profile).issues}")


def main() -> None:
    if shutil.which("espeak-ng") is None:
        raise SystemExit("espeak-ng is not on PATH; install it (e.g. apt install espeak-ng)")
    make_speech()
    make_monotone()
    check_hand_written()
    print(
        "Wrote speech.wav, speech.whisper.json, speech.txt, monotone.wav and "
        f"monotone.deepgram.json in {HERE}"
    )


if __name__ == "__main__":
    main()
