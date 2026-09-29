#!/usr/bin/env python3
"""Regenerate the committed benchmark baselines in this directory.

``training_synthetic.json`` is the report of :class:`AudioToIML` on
``tests/fixtures/datasets/training_synthetic``: ten espeak-ng clips whose
speed, pitch and loudness follow their emotion label. Each clip is converted
with its entry's transcript (``words_from="auto"``; the entries have no word
timings) and without speech recognition, against the speaker baseline of
the dataset's first neutral clip (``calibration_audio``), since a single
clip has no baseline of its own and AudioToIML would abstain on every one.
The ground truth has no pauses or pitch contours, so the report measures
emotion, IML validity and failures.

``tests/test_benchmarks.py`` runs the same benchmark and fails when it
regresses from the committed report by more than :data:`TOLERANCE`. After an
intended change in AudioToIML's output, run this script from the repository
root and commit the result::

    python tests/fixtures/benchmarks/make_baselines.py

It needs the ``audio`` extra (numpy, praat-parselmouth). The same run from
the command line, once ``prosody-protocol benchmark`` takes a calibration
recording, is ``prosody-protocol benchmark tests/fixtures/datasets/training_synthetic
--stt none`` with that recording.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from prosody_protocol import AudioToIML, Benchmark, BenchmarkReport, DatasetLoader

HERE = Path(__file__).resolve().parent
TRAINING_SYNTHETIC = HERE.parent / "datasets" / "training_synthetic"
TRAINING_SYNTHETIC_BASELINE = HERE / "training_synthetic.json"
CALIBRATION_CLIP = "audio/synth_001.wav"  # neutral

# Allowed drop of each metric against the baseline. Ten clips: one clip's
# label changing moves emotion coverage by 0.1 and accuracy by up to 0.25;
# the per-class F1 of a class with one or two clips moves by more.
TOLERANCE = 0.26
CLASS_TOLERANCE = 0.7
TRAINING_SYNTHETIC_THRESHOLDS = {"validity_rate": 1.0, "emotion_coverage": 0.1}


def benchmark_training_synthetic() -> BenchmarkReport:
    """Run the benchmark that ``training_synthetic.json`` records."""
    dataset = DatasetLoader().load(TRAINING_SYNTHETIC, check_audio=True)
    converter = AudioToIML(stt="none", calibration_audio=TRAINING_SYNTHETIC / CALIBRATION_CLIP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # degraded-output notes
        return Benchmark(dataset, converter).run()


def main() -> None:
    report = benchmark_training_synthetic()
    report.duration_seconds = 0.0  # not compared; keeps the file stable
    report.save(TRAINING_SYNTHETIC_BASELINE)
    print(f"wrote {TRAINING_SYNTHETIC_BASELINE}")
    for metric, value in report.to_dict().items():
        print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
