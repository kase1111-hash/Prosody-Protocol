"""Tests for prosody_protocol.prosody_analyzer.

Uses audio with known acoustic properties: synthetic tones, silence and
gaps (some generated on the fly with noise floors, known jitter/shimmer or
known syllable rates), synthetic voices with known F0, perturbation and
breath noise, and espeak-ng speech whose word timings and pauses are
recorded in JSON next to the WAV (see tests/generate_audio_fixtures.py).
speech_levels.wav holds whole espeak-ng sentences at known levels (one
sentence three times at the same tempo, -a 100, 40 and 25) with the pauses
between them in speech_levels.json, which says how it was made.
Tests that synthesise speech on the fly skip without espeak-ng.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
parselmouth = pytest.importorskip("parselmouth")

from prosody_protocol import prosody_analyzer
from prosody_protocol.exceptions import AudioProcessingError
from prosody_protocol.prosody_analyzer import (
    PauseInterval,
    ProsodyAnalyzer,
    SpanFeatures,
    WordAlignment,
    _classify_quality,
    _remove_octave_jumps,
    _sliding_max,
    _smooth_runs,
    detect_pauses,
)

AUDIO_DIR = Path(__file__).parent / "fixtures" / "audio"
SR = 16_000

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture()
def analyzer() -> ProsodyAnalyzer:
    return ProsodyAnalyzer()


def _write(path: Path, signal: Any, sr: int = SR, channels: int = 1) -> Path:
    """Write a float signal (samples x channels) as 16-bit PCM WAV."""
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def _write_float(path: Path, signal: Any, sr: int = SR, dtype: str = "<f4") -> Path:
    """Write a mono signal as a float WAV (32-bit, or 64-bit with
    ``dtype="<f8"``), which holds samples beyond full scale (1.0) without
    clipping."""
    samples = np.asarray(signal, dtype=dtype)
    data, width = samples.tobytes(), samples.dtype.itemsize
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 3, 1, sr, sr * width, width, 8 * width)
        + b"data" + struct.pack("<I", len(data))
    )
    path.write_bytes(header + data)
    return path


def _tone(duration_s: float, freq: float = 150.0, amplitude: float = 0.3) -> Any:
    t = np.arange(int(duration_s * SR)) / SR
    return amplitude * np.sin(2 * np.pi * freq * t)


def _voice(duration_s: float, f0: float = 150.0) -> Any:
    """A steady harmonic 'vowel' with constant amplitude."""
    phase = 2 * np.pi * f0 * np.arange(int(duration_s * SR)) / SR
    y = sum(np.sin(k * phase) / k for k in range(1, 11))
    return 0.3 * y / np.abs(y).max()


def _syllables(rate: float, duration_s: float = 2.0) -> Any:
    """A voice whose loudness rises and falls *rate* times per second."""
    t = np.arange(int(duration_s * SR)) / SR
    phase = 2 * np.pi * np.cumsum(150.0 * (1 + 0.02 * np.sin(2 * np.pi * 3 * t))) / SR
    y = sum(np.sin(k * phase) / k for k in range(1, 11))
    return 0.3 * y / np.abs(y).max() * np.sin(np.pi * rate * t) ** 2


def _noise(n: int, dbfs: float, seed: int = 0) -> Any:
    return 10 ** (dbfs / 20) * np.random.default_rng(seed).standard_normal(n)


def _phonation(
    duration_s: float,
    f0: float,
    *,
    jitter: float = 0.005,
    shimmer: float = 0.03,
    hnr_db: float = 18.0,
    seed: int = 0,
    f0_end: float | None = None,
) -> Any:
    """A voice made of glottal cycles whose periods and amplitudes vary at
    random by *jitter* and *shimmer* (relative standard deviations), with
    aspiration noise *hnr_db* below the voice and 30 ms fades at both ends.
    With *f0_end*, F0 glides evenly (in semitones) from *f0* to it."""
    rng = np.random.default_rng(seed)
    cycles: list[Any] = []
    total = 0
    while total < duration_s * SR:
        frequency = f0 * (f0_end / f0) ** (total / (duration_s * SR)) if f0_end else f0
        n = max(8, int(round(SR / frequency * (1 + jitter * rng.standard_normal()))))
        t = np.arange(n) / n
        wave_ = sum(0.7**k * np.sin(2 * np.pi * k * t) for k in range(1, 15))
        cycles.append((1 + shimmer * rng.standard_normal()) * wave_)
        total += n
    y = np.concatenate(cycles)[: int(duration_s * SR)]
    y = y / np.sqrt(np.mean(y**2)) + 10 ** (-hnr_db / 20) * rng.standard_normal(y.size)
    i = np.arange(y.size)
    fade = np.minimum(1.0, np.minimum(i, y.size - i) / (0.03 * SR))
    return 0.075 * y * fade


def _utterance(
    tmp_path: Path, odd: dict[str, float] | None = None, words: int = 8, f0: float = 120.0
) -> tuple[Path, list[WordAlignment]]:
    """*words* 350 ms 'words' of an *f0* Hz voice with 250 ms pauses between
    them; word 4 is spoken with the *odd* settings of :func:`_phonation`."""
    parts, alignments, position = [np.zeros(SR // 5)], [], 0.2
    for i in range(words):
        settings: dict[str, Any] = {"f0": f0 * (1 + 0.05 * np.sin(i)), "seed": 1 + i}
        if i == 4 and odd:
            settings.update(odd)
        parts += [_phonation(0.35, **settings), np.zeros(SR // 4)]
        alignments.append(
            WordAlignment(f"w{i}", round(position * 1000), round((position + 0.35) * 1000))
        )
        position += 0.6
    signal = np.concatenate(parts)
    path = _write(tmp_path / "utterance.wav", signal + _noise(signal.size, -80, seed=99))
    return path, alignments


def _espeak(text: str, *, pitch: int = 50, speed: int = 160) -> Any:
    """*text* spoken by espeak-ng at 16 kHz, without leading or trailing silence."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "speech.wav"
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", str(speed), "-p", str(pitch), "-z",
             "-w", str(path), text],
            check=True,
        )
        samples = parselmouth.Sound(str(path)).resample(SR).values[0]
    audible = np.flatnonzero(np.abs(samples) > 1e-3)
    return samples[audible[0]:audible[-1] + 1]


needs_espeak = pytest.mark.skipif(
    shutil.which("espeak-ng") is None, reason="espeak-ng not installed"
)


def _truth(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((AUDIO_DIR / f"{name}.json").read_text())
    return data


def _words(truth: dict[str, Any]) -> list[WordAlignment]:
    return [WordAlignment(w["word"], w["start_ms"], w["end_ms"]) for w in truth["words"]]


# ---------------------------------------------------------------------------
# Data model tests (preserved from Phase 2 stubs)
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_word_alignment(self) -> None:
        wa = WordAlignment(word="hello", start_ms=0, end_ms=500)
        assert wa.word == "hello"
        assert wa.start_ms == 0

    def test_span_features_defaults(self) -> None:
        sf = SpanFeatures(start_ms=0, end_ms=500, text="hello")
        assert sf.f0_mean is None
        assert sf.quality is None

    def test_pause_interval(self) -> None:
        pi = PauseInterval(start_ms=500, end_ms=1300)
        assert pi.duration_ms == 800


# ---------------------------------------------------------------------------
# F0 extraction
# ---------------------------------------------------------------------------


class TestF0Extraction:
    def test_220hz_tone_f0(self, analyzer: ProsodyAnalyzer) -> None:
        """A pure 220 Hz sine wave should yield F0 close to 220 Hz."""
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        assert len(results) == 1
        f = results[0]
        assert f.f0_mean is not None
        assert 218 < f.f0_mean < 222, f"Expected ~220 Hz, got {f.f0_mean}"

    def test_440hz_tone_f0(self, analyzer: ProsodyAnalyzer) -> None:
        """A pure 440 Hz sine wave should yield F0 close to 440 Hz."""
        alignments = [WordAlignment(word="tone", start_ms=50, end_ms=450)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_440hz.wav"), alignments)
        assert len(results) == 1
        f = results[0]
        assert f.f0_mean is not None
        assert 435 < f.f0_mean < 445, f"Expected ~440 Hz, got {f.f0_mean}"

    def test_f0_range_of_steady_tone_is_narrow(self, analyzer: ProsodyAnalyzer) -> None:
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        f = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)[0]
        assert f.f0_range is not None
        low, high = f.f0_range
        assert 215 < low <= high < 225

    def test_contour_samples_every_voiced_frame(self, analyzer: ProsodyAnalyzer) -> None:
        """800 ms of steady voicing gives one F0 sample per 10 ms frame."""
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        f = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)[0]
        assert f.f0_contour is not None
        assert 78 <= len(f.f0_contour) <= 81
        assert all(215 < v < 225 for v in f.f0_contour)

    def test_rising_sweep_contour(self, analyzer: ProsodyAnalyzer) -> None:
        """rising_pitch.wav sweeps linearly from 150 Hz to 350 Hz in 1 s."""
        f = analyzer.analyze(
            str(AUDIO_DIR / "rising_pitch.wav"), [WordAlignment("sweep", 0, 1000)]
        )[0]
        assert f.f0_contour is not None and f.f0_range is not None and f.f0_mean is not None
        assert 145 < f.f0_contour[0] < 170
        assert 330 < f.f0_contour[-1] < 355
        assert np.all(np.diff(f.f0_contour) > 0), "contour must rise monotonically"
        assert 240 < f.f0_mean < 260
        assert f.f0_range[0] > 145 and f.f0_range[1] < 355

    def test_octave_jumps_are_corrected(self) -> None:
        """A tracker error that doubles or halves F0 is folded back."""
        f0 = np.array([200.0, 202, 204, 412, 206, np.nan, 208, 105, 210, 212])
        cleaned = _remove_octave_jumps(f0)
        assert cleaned[3] == pytest.approx(206.0)
        assert cleaned[7] == pytest.approx(210.0)
        assert np.isnan(cleaned[5])
        assert np.array_equal(cleaned[[0, 1, 2, 4, 6, 8, 9]], f0[[0, 1, 2, 4, 6, 8, 9]])

    def test_implausible_outlier_is_dropped(self) -> None:
        f0 = np.array([200.0, 201, 202, 203, 600 * 1.3, 204, 205, 206])
        cleaned = _remove_octave_jumps(f0)
        assert np.isnan(cleaned[4])
        assert np.count_nonzero(np.isnan(cleaned)) == 1

    def test_errors_at_the_end_are_not_outvoted_by_padding(self) -> None:
        """Three spurious 650-690 Hz frames (a released /t/) ended a 70 Hz
        voice: repeating the last sample to fill the window made them the
        majority, and they survived."""
        f0 = np.array([70.0, 71, 69, 70, 72, 71, 70, 69, 659, 671, 693])
        cleaned = _remove_octave_jumps(f0)
        assert np.all(np.isnan(cleaned[-3:]))
        assert np.array_equal(cleaned[:-3], f0[:-3])
        reverse = _remove_octave_jumps(f0[::-1])
        assert np.all(np.isnan(reverse[:3]))

    def test_kept_samples_are_not_corrected(self) -> None:
        """Excursions recovered from the first pass are left alone, however
        short, but still count as neighbours."""
        f0 = np.array([100.0, 101, 102, 103, 104, 210, 212, 214, 105, 106, 107, 108, 109])
        keep = np.zeros(f0.size, dtype=bool)
        keep[5:8] = True
        assert np.allclose(_remove_octave_jumps(f0)[5:8], [105, 106, 107])
        assert np.array_equal(_remove_octave_jumps(f0, keep=keep), f0)

    def test_smooth_runs(self) -> None:
        """Runs end at unvoiced samples and at jumps of 3 semitones or more."""
        f0 = np.array([100.0, 102, np.nan, 100, 150, 152, 153, np.nan, np.nan, 90, 108, 107])
        assert _smooth_runs(f0) == [(0, 2), (3, 4), (4, 7), (9, 10), (10, 12)]
        assert _smooth_runs(np.array([np.nan, np.nan])) == []

    def test_only_sustained_strong_excursions_are_recovered(self) -> None:
        """First-pass frames outside the fitted range replace the second
        pass's only in a smooth run of at least 5 frames with a median
        voicing strength of at least 0.7. A fricative's weak "voicing" that
        follows a vowel without a break is a run of its own."""
        analysis = prosody_analyzer._AudioAnalysis.from_path(AUDIO_DIR / "tone_220hz.wav")
        n = 40
        first_times = 0.02 + 0.01 * np.arange(n)
        first, strength = np.full(n, np.nan), np.zeros(n)
        first[0:10], strength[0:10] = 100.0, 0.9  # a vowel ...
        first[10:16], strength[10:16] = 400.0, 0.55  # ... and a fricative
        first[18:22], strength[18:22] = 400.0, 0.9  # strong, but too brief
        first[25:36], strength[25:36] = 400.0, 0.9  # a shout
        second = np.where(first > 300, 200.0, first)  # the fitted range reads half
        second[10:16] = np.nan
        analysis.__dict__["_first_pass"] = (first_times, first, strength)
        analysis.__dict__["pitch_range"] = (50.0, 290.0)
        analysis._track_pitch = lambda floor, ceiling: (  # type: ignore[method-assign]
            first_times, second.copy(), np.ones(n)
        )
        _, f0, excursion = analysis._tracked_pitch
        assert np.flatnonzero(excursion).tolist() == list(range(25, 36))
        assert np.all(f0[25:36] == 400.0)
        assert np.all(np.isnan(f0[10:16])) and np.all(f0[18:22] == 200.0)
        assert np.array_equal(f0[:10], first[:10])

        # With windows of different lengths, the second pass's frames can be
        # centred half a step later: those within the shout take its F0.
        analysis.__dict__.pop("_tracked_pitch")
        analysis._track_pitch = lambda floor, ceiling: (  # type: ignore[method-assign]
            first_times + 0.005, second.copy(), np.ones(n)
        )
        _, f0, excursion = analysis._tracked_pitch
        assert np.flatnonzero(excursion).tolist() == list(range(25, 35))
        assert np.all(f0[25:35] == 400.0)

    @pytest.mark.parametrize("shout", [300.0, 400.0, 480.0])
    def test_word_far_above_the_median_is_tracked(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, shout: float
    ) -> None:
        """A range fitted to a 100 Hz voice ends near 290 Hz, and read a
        shouted word of 300-480 Hz an octave low (a fixed 75-600 Hz range
        had measured it). Its voice measures come from a range that takes
        it in, as when the word is analysed on its own: glottal pulses
        found at half the rate halved its jitter."""
        path, words = _utterance(tmp_path, {"f0": shout}, words=10, f0=100.0)
        assert prosody_analyzer._AudioAnalysis.from_path(path).pitch_range[1] < 300
        features = analyzer.analyze(path, words)
        f = features[4]
        assert f.f0_mean == pytest.approx(shout, rel=0.03)
        assert f.f0_contour is not None
        assert all(0.9 * shout < v < 1.1 * shout for v in f.f0_contour)
        assert [g.quality for g in features] == ["modal"] * 10

        word = parselmouth.Sound(str(path)).extract_part(
            from_time=words[4].start_ms / 1000 - 0.1, to_time=words[4].end_ms / 1000 + 0.1
        )
        alone_path = _write(tmp_path / "alone.wav", word.values[0])
        alone = analyzer.analyze(alone_path, [WordAlignment("w4", 100, 450)])[0]
        assert f.jitter is not None and alone.jitter is not None
        assert f.jitter == pytest.approx(alone.jitter, rel=0.2)
        assert f.shimmer == pytest.approx(alone.shimmer, rel=0.2)

    def test_rise_beyond_the_fitted_range_stays_a_rise(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """A question rising from 110 to 330 Hz at the end of a 100 Hz voice
        dropped an octave where it crossed the fitted ceiling, turning the
        rise into a rise-fall."""
        parts, words = [np.zeros(SR // 5)], []
        for i in range(9):
            parts += [_phonation(0.35, 100.0 * (1 + 0.05 * np.sin(i)), seed=i), np.zeros(SR // 4)]
            words.append(WordAlignment(f"w{i}", 200 + 600 * i, 200 + 600 * i + 350))
        parts += [_phonation(0.5, 110.0, f0_end=330.0, seed=9), np.zeros(SR // 4)]
        words.append(WordAlignment("w9", 5600, 6100))
        path = _write(tmp_path / "question.wav", np.concatenate(parts))
        contour = analyzer.analyze(path, words)[-1].f0_contour
        assert contour is not None
        steps = np.diff(np.log2(contour)) * 12
        assert np.all(steps > -0.5), "the contour must keep rising"
        assert contour[-1] > 290

    def test_creak_below_a_high_voice(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """A range fitted to a 210 Hz voice starts near 105 Hz: a creaky word
        at 75 Hz had no F0, and so could not be labelled creaky."""
        odd = {"f0": 75.0, "jitter": 0.04, "shimmer": 0.08}
        path, words = _utterance(tmp_path, odd, words=10, f0=210.0)
        features = analyzer.analyze(path, words)
        assert features[4].f0_mean == pytest.approx(76.0, rel=0.05)
        assert [f.quality for f in features] == ["modal"] * 4 + ["creaky"] + ["modal"] * 5

    def test_a_brief_second_voice(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """A 230 Hz voice says one word in twelve of a 70 Hz voice's: too
        little voicing to widen the fitted range, which read it an octave low."""
        f0s = [70.0] * 6 + [230.0] + [70.0] * 6
        parts, words = [np.zeros(SR // 5)], []
        for i, f0 in enumerate(f0s):
            parts += [_phonation(0.35, f0, seed=i), np.zeros(SR // 4)]
            words.append(WordAlignment(f"w{i}", 200 + i * 600, 200 + i * 600 + 350))
        path = _write(tmp_path / "two.wav", np.concatenate(parts))
        assert prosody_analyzer._AudioAnalysis.from_path(path).pitch_range[1] < 230
        for f, f0 in zip(analyzer.analyze(path, words), f0s, strict=True):
            assert f.f0_mean == pytest.approx(f0, rel=0.1), f.text

    def test_unvoiced_span_has_no_f0(self, analyzer: ProsodyAnalyzer) -> None:
        f = analyzer.analyze(
            str(AUDIO_DIR / "tone_gap_tone.wav"), [WordAlignment("gap", 700, 1100)]
        )[0]
        assert f.f0_mean is None and f.f0_range is None and f.f0_contour is None

    @pytest.mark.parametrize("f0", [55.0, 62.0, 70.0])
    def test_voice_below_75_hz_is_tracked(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, f0: float
    ) -> None:
        """A fixed 75-600 Hz search range found no voicing at all in these voices."""
        path, words = _utterance(tmp_path, f0=f0)
        for f in analyzer.analyze(path, words):
            assert f.f0_mean is not None and f.f0_contour is not None
            assert f.f0_mean == pytest.approx(f0, rel=0.1)
            assert all(0.8 * f0 < v < 1.25 * f0 for v in f.f0_contour)

    @pytest.mark.parametrize(
        ("path", "floor", "ceiling"),
        [
            # A 220 Hz tone: one octave below to 1.5 octaves above.
            (AUDIO_DIR / "tone_220hz.wav", 110.0, 622.3),
            # Silence has no voicing to fit the range to: Praat's default.
            (AUDIO_DIR / "silence_1s.wav", 75.0, 600.0),
        ],
    )
    def test_pitch_range_follows_the_voice(self, path: Path, floor: float, ceiling: float) -> None:
        analysis = prosody_analyzer._AudioAnalysis.from_path(path)
        assert analysis.pitch_range == pytest.approx((floor, ceiling), rel=0.01)

    def test_two_voices_an_octave_and_a_half_apart(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """A low voice (70 Hz) says most of the words and a high one (230 Hz)
        the rest. A range fitted to the median alone ends below 230 Hz,
        which then reads an octave low."""
        f0s = [70.0] * 6 + [230.0] * 3
        parts, words = [np.zeros(SR // 5)], []
        for i, f0 in enumerate(f0s):
            parts += [_phonation(0.35, f0, seed=i), np.zeros(SR // 4)]
            words.append(WordAlignment(f"w{i}", 200 + i * 600, 200 + i * 600 + 350))
        path = _write(tmp_path / "two.wav", np.concatenate(parts))
        for f, f0 in zip(analyzer.analyze(path, words), f0s, strict=True):
            assert f.f0_mean == pytest.approx(f0, rel=0.1), f.text

    def test_pitch_range_of_a_low_voice(self, tmp_path: Path) -> None:
        path, _ = _utterance(tmp_path, f0=62.0)
        floor, ceiling = prosody_analyzer._AudioAnalysis.from_path(path).pitch_range
        assert floor == 40.0  # half the median, but not below 40 Hz
        assert 150 < ceiling < 200

    @needs_espeak
    def test_low_quiet_words_have_no_octave_errors(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """A normal espeak-ng sentence, then short words spoken quietly in a
        voice of about 70 Hz. Tracked from 75 Hz up, 'miss' read 242 Hz and
        'old' 383 Hz; the samples of each word must stay near its median."""
        plan = [("I went to the store this morning", {"pitch": 50}, 1.0),
                ("I miss my old friends", {"pitch": 20, "speed": 115}, 0.4)]
        parts, words, position = [np.zeros(SR // 5)], [], SR // 5
        for sentence, settings, gain in plan:
            for word in sentence.split():
                clip = gain * _espeak(word, **settings)
                words.append(WordAlignment(
                    word, round(position / SR * 1000), round((position + clip.size) / SR * 1000)
                ))
                parts += [clip, np.zeros(SR // 25)]
                position += clip.size + SR // 25
            parts.append(np.zeros(SR // 2))
            position += SR // 2
        signal = np.concatenate(parts)
        signal = 0.3 * signal / np.abs(signal).max() + _noise(signal.size, -60)
        features = analyzer.analyze(_write(tmp_path / "story.wav", signal), words)
        for f in features[-5:]:
            assert f.f0_mean is not None and f.f0_contour is not None, f.text
            assert 55 < f.f0_mean < 95, f"{f.text}: {f.f0_mean:.0f} Hz"
            median = float(np.median(f.f0_contour))
            assert all(median / 1.4 < v < median * 1.4 for v in f.f0_contour), f.text

    @needs_espeak
    def test_fricatives_of_a_very_low_voice_are_not_voiced(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """In a voice of about 60 Hz the tracker finds 'voicing' near 400 Hz in
        the /s/ of 'miss', a few percent of all frames. The search range
        must not stretch up to take it in."""
        parts, words, position = [np.zeros(SR // 5)], [], SR // 5
        for word in ["I", "really", "miss", "my", "old", "friends", "from", "school"]:
            clip = _espeak(word, pitch=0)
            words.append(WordAlignment(
                word, round(position / SR * 1000), round((position + clip.size) / SR * 1000)
            ))
            parts += [clip, np.zeros(SR // 25)]
            position += clip.size + SR // 25
        signal = np.concatenate([*parts, np.zeros(SR // 5)])
        signal = 0.3 * signal / np.abs(signal).max() + _noise(signal.size, -60)
        path = _write(tmp_path / "low.wav", signal)
        assert prosody_analyzer._AudioAnalysis.from_path(path).pitch_range[1] < 250
        for f in analyzer.analyze(path, words):
            assert f.f0_contour is not None, f.text
            assert max(f.f0_contour) < 120, f"{f.text}: {max(f.f0_contour):.0f} Hz"


# ---------------------------------------------------------------------------
# Intensity extraction
# ---------------------------------------------------------------------------


class TestIntensityExtraction:
    def test_tone_intensity_on_praat_scale(self, analyzer: ProsodyAnalyzer) -> None:
        """A sine of amplitude 20000/32767 has RMS 0.432, i.e. 86.7 dB re 20 uPa."""
        alignments = [WordAlignment(word="tone", start_ms=100, end_ms=900)]
        f = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)[0]
        assert f.intensity_mean == pytest.approx(86.7, abs=0.5)
        assert f.intensity_range is not None and f.intensity_range < 1.0

    def test_loud_louder_than_quiet(self, analyzer: ProsodyAnalyzer) -> None:
        """The first half of loud_quiet.wav is 20 dB louder than the second half."""
        loud, quiet = analyzer.analyze(
            str(AUDIO_DIR / "loud_quiet.wav"),
            [WordAlignment("loud", 50, 450), WordAlignment("quiet", 550, 950)],
        )
        assert loud.intensity_mean is not None and quiet.intensity_mean is not None
        assert loud.intensity_mean - quiet.intensity_mean == pytest.approx(20.0, abs=0.5)

    def test_mean_is_taken_over_power(self, analyzer: ProsodyAnalyzer) -> None:
        """Averaging decibels would put the whole file 10 dB below the loud
        half; averaging power puts it 3 dB below."""
        loud, whole = analyzer.analyze(
            str(AUDIO_DIR / "loud_quiet.wav"),
            [WordAlignment("loud", 50, 450), WordAlignment("all", 0, 1000)],
        )
        assert loud.intensity_mean is not None and whole.intensity_mean is not None
        assert loud.intensity_mean - whole.intensity_mean == pytest.approx(3.0, abs=0.5)

    def test_digital_silence_is_excluded(self, analyzer: ProsodyAnalyzer) -> None:
        """Praat reports digital silence as -300 dB; it must not be averaged in.

        Before the fix the whole-file span read -76.7 dB with a 408 dB range.
        """
        tone, whole, partial = analyzer.analyze(
            str(AUDIO_DIR / "tone_gap_tone.wav"),
            [
                WordAlignment("tone", 50, 450),
                WordAlignment("all", 0, 1800),
                WordAlignment("two", 250, 650),
            ],
        )
        assert tone.intensity_mean is not None
        assert whole.intensity_mean == pytest.approx(tone.intensity_mean, abs=1.0)
        assert partial.intensity_mean == pytest.approx(tone.intensity_mean, abs=1.0)
        assert whole.intensity_range is not None and whole.intensity_range < 5.0

    def test_noise_floor_is_excluded(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        signal = np.concatenate([_tone(0.5), np.zeros(SR // 2)])
        path = _write(tmp_path / "tone_then_noise.wav", signal + _noise(len(signal), -70))
        tone, both = analyzer.analyze(
            str(path), [WordAlignment("tone", 50, 450), WordAlignment("both", 250, 750)]
        )
        assert tone.intensity_mean is not None
        assert both.intensity_mean == pytest.approx(tone.intensity_mean, abs=1.0)

    def test_silent_span_has_no_intensity(self, analyzer: ProsodyAnalyzer) -> None:
        f = analyzer.analyze(
            str(AUDIO_DIR / "silence_1s.wav"), [WordAlignment("x", 0, 1000)]
        )[0]
        assert f.intensity_mean is None and f.intensity_range is None


# ---------------------------------------------------------------------------
# Voice quality (jitter/shimmer/HNR)
# ---------------------------------------------------------------------------


def _pulse_train(periods: list[float], amplitudes: list[float]) -> Any:
    cycles = []
    for period, amplitude in zip(periods, amplitudes, strict=True):
        t = np.arange(int(round(period * SR))) / round(period * SR)
        cycles.append(amplitude * sum(np.sin(2 * np.pi * k * t) / k for k in range(1, 8)))
    return np.concatenate(cycles)


class TestVoiceQuality:
    @pytest.mark.parametrize("span_ms", [150, 250, 400, 800])
    def test_steady_voice_has_no_shimmer(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, span_ms: int
    ) -> None:
        """Constant amplitude means zero shimmer at any span length.

        A Hanning taper used to fabricate 14.8% ('tense') at 150 ms.
        """
        path = _write(tmp_path / "voice.wav", _voice(3.0))
        f = analyzer.analyze(str(path), [WordAlignment("w", 300, 300 + span_ms)])[0]
        assert f.shimmer is not None and f.shimmer < 1.0
        assert f.jitter is not None and f.jitter < 0.5
        assert f.quality == "modal"

    def test_jitter_is_in_percent(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """Periods alternating +/-1% give about 2% local jitter (spec 4.4 unit)."""
        n = 300
        periods = [(1 / 150) * (1 + 0.01 * (-1) ** i) for i in range(n)]
        path = _write(tmp_path / "jitter.wav", _pulse_train(periods, [0.3] * n))
        f = analyzer.analyze(str(path), [WordAlignment("w", 200, 1600)])[0]
        assert f.jitter is not None and 1.2 < f.jitter < 2.5

    def test_shimmer_is_in_percent(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """Amplitudes alternating +/-5% give about 10% local shimmer."""
        n = 300
        amplitudes = [0.3 * (1 + 0.05 * (-1) ** i) for i in range(n)]
        path = _write(tmp_path / "shimmer.wav", _pulse_train([1 / 150] * n, amplitudes))
        f = analyzer.analyze(str(path), [WordAlignment("w", 200, 1600)])[0]
        assert f.shimmer is not None and 6.0 < f.shimmer < 12.0

    def test_clean_tone_has_high_hnr(self, analyzer: ProsodyAnalyzer) -> None:
        f = analyzer.analyze(
            str(AUDIO_DIR / "tone_220hz.wav"), [WordAlignment("tone", 100, 900)]
        )[0]
        assert f.hnr is not None and f.hnr > 30.0

    def test_noise_has_no_voice_quality(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """Jitter, shimmer and HNR describe voicing; noise has none."""
        path = _write(tmp_path / "noise.wav", _noise(SR, -40))
        f = analyzer.analyze(str(path), [WordAlignment("n", 0, 1000)])[0]
        assert (f.jitter, f.shimmer, f.hnr, f.quality) == (None, None, None, None)


class TestQualityLabels:
    """Voice quality labels compare a span with the rest of the recording."""

    def test_espeak_words_are_not_all_tense(self, analyzer: ProsodyAnalyzer) -> None:
        """Every word of the espeak-ng fixtures used to be labelled 'tense': an
        absolute 12 % shimmer threshold, where 13-17 % is ordinary for
        running speech."""
        labels = []
        for name in ("speech_pauses", "speech_calibration", "speech_raised"):
            truth = _truth(name)
            features = analyzer.analyze(AUDIO_DIR / f"{name}.wav", _words(truth))
            labels += [f.quality for f in features]
        assert set(labels) <= {"modal", None}
        assert labels.count("modal") >= 0.75 * len(labels)

    @pytest.mark.parametrize(
        ("label", "odd"),
        [
            ("breathy", {"hnr_db": 6.0}),
            ("creaky", {"f0": 60.0, "jitter": 0.04}),
            # Irregular but not noisy (much noise would make it breathy or harsh).
            ("harsh", {"jitter": 0.02, "shimmer": 0.2, "hnr_db": 30.0}),
        ],
    )
    def test_the_odd_word_out_is_labelled(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, label: str, odd: dict[str, float]
    ) -> None:
        path, words = _utterance(tmp_path, odd)
        labels = [f.quality for f in analyzer.analyze(path, words)]
        assert labels[4] == label
        assert labels[:4] + labels[5:] == ["modal"] * 7

    def test_a_breathy_speaker_is_modal_for_herself(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """Labels are relative: a voice that is equally noisy throughout has no
        breathier words (spec 6.1: attributes at the speaker's baseline are
        omitted)."""
        parts = [_phonation(0.35, 120.0, hnr_db=6.0, seed=i) for i in range(8)]
        signal = np.concatenate([np.concatenate([p, np.zeros(SR // 4)]) for p in parts])
        path = _write(tmp_path / "breathy.wav", signal)
        words = [WordAlignment(f"w{i}", i * 600, i * 600 + 350) for i in range(8)]
        assert [f.quality for f in analyzer.analyze(path, words)] == ["modal"] * 8

    def test_whole_recording_has_no_reference(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """A span covering the whole recording leaves nothing to compare it with."""
        path, _ = _utterance(tmp_path)
        f = analyzer.analyze(path, [WordAlignment("all", 0, 2**31 - 1)])[0]
        assert f.jitter is not None and f.shimmer is not None and f.hnr is not None
        assert f.quality is None

    def test_short_voicing_is_unlabelled(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """Under 100 ms of voicing gives too few glottal periods to judge."""
        path, words = _utterance(tmp_path)
        start = words[4].start_ms
        short, whole = analyzer.analyze(
            path, [WordAlignment("x", start, start + 70), words[4]]
        )
        assert short.f0_mean is not None and short.quality is None
        assert whole.quality == "modal"

    USUAL = (1.0, 10.0, 15.0, 120.0)  # jitter %, shimmer %, HNR dB, F0 Hz

    @pytest.mark.parametrize(
        ("span", "snr_db", "label"),
        [
            ((1.1, 11.0, 14.0, 118.0), 40.0, "modal"),
            ((2.5, 11.0, 14.0, 90.0), 40.0, "creaky"),  # irregular, 4.9 st lower
            ((2.5, 11.0, 14.0, 110.0), 40.0, None),  # irregular only: unsure
            ((2.5, 25.0, 12.0, 120.0), 40.0, "harsh"),
            ((2.5, 25.0, 8.0, 120.0), 40.0, None),  # harsh or breathy
            ((1.0, 25.0, 14.0, 120.0), 40.0, None),  # shimmer only: unsure
            ((1.5, 15.0, 8.0, 120.0), 40.0, "breathy"),
            ((1.5, 15.0, 8.0, 120.0), 20.0, None),  # noise may explain the HNR
        ],
    )
    def test_rules(
        self, span: tuple[float, float, float, float], snr_db: float, label: str | None
    ) -> None:
        assert _classify_quality(span, self.USUAL, snr_db) == label

    def test_measurement_noise_is_not_a_deviation(self) -> None:
        """Against a synthetic voice (jitter near 0 %, HNR near 70 dB), tiny
        perturbations and an HNR of 40 dB are still modal."""
        usual = (0.001, 0.01, 70.0, 150.0)
        assert _classify_quality((0.3, 1.5, 40.0, 150.0), usual, 60.0) == "modal"
        assert _classify_quality((1.2, 1.5, 40.0, 150.0), usual, 60.0) is None
        assert _classify_quality((0.3, 1.5, 12.0, 150.0), usual, 60.0) == "breathy"


# ---------------------------------------------------------------------------
# Pause detection
# ---------------------------------------------------------------------------


class TestPauseDetection:
    def test_silence_file_is_one_big_pause(self, analyzer: ProsodyAnalyzer) -> None:
        pauses = analyzer.detect_pauses(str(AUDIO_DIR / "silence_1s.wav"))
        assert pauses == [PauseInterval(start_ms=0, end_ms=1000)]

    def test_tone_gap_tone_pause_is_exact(self, analyzer: ProsodyAnalyzer) -> None:
        """0.5s tone + 0.8s silence + 0.5s tone: the pause is 500-1300 ms."""
        pauses = analyzer.detect_pauses(str(AUDIO_DIR / "tone_gap_tone.wav"))
        assert len(pauses) == 1
        assert abs(pauses[0].start_ms - 500) <= 10
        assert abs(pauses[0].end_ms - 1300) <= 10

    @pytest.mark.parametrize("noise_dbfs", [None, -80.0, -60.0, -40.0])
    def test_pause_found_above_a_noise_floor(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, noise_dbfs: float | None
    ) -> None:
        """Real recordings have a noise floor; the old absolute -40 dB
        threshold only ever matched digital zeros."""
        signal = np.concatenate([_tone(1.0), np.zeros(SR // 2), _tone(1.0)])
        if noise_dbfs is not None:
            signal = signal + _noise(len(signal), noise_dbfs)
        pauses = analyzer.detect_pauses(str(_write(tmp_path / "gap.wav", signal)))
        assert len(pauses) == 1
        assert abs(pauses[0].start_ms - 1000) <= 20
        assert abs(pauses[0].end_ms - 1500) <= 20

    def test_200ms_pause_is_found(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """The spec's lower bound for a meaningful pause must not be lost to
        edge smearing."""
        signal = np.concatenate([_tone(1.0), np.zeros(SR // 5), _tone(1.0)])
        signal = signal + _noise(len(signal), -60)
        pauses = analyzer.detect_pauses(str(_write(tmp_path / "gap.wav", signal)))
        assert len(pauses) == 1
        assert abs(pauses[0].duration_ms - 200) <= 20

    def test_continuous_tone_no_pause(self, analyzer: ProsodyAnalyzer) -> None:
        """A continuous tone should not have any pauses."""
        pauses = analyzer.detect_pauses(str(AUDIO_DIR / "tone_220hz.wav"))
        assert pauses == []

    def test_quieter_sound_is_not_a_pause(self, analyzer: ProsodyAnalyzer) -> None:
        """loud_quiet.wav drops 20 dB halfway but never goes silent."""
        assert analyzer.detect_pauses(str(AUDIO_DIR / "loud_quiet.wav")) == []

    def test_silence_threshold_is_adjustable(self, analyzer: ProsodyAnalyzer) -> None:
        """With a 12 dB threshold, the half 20 dB down counts as silence.
        (Voiced sound within the threshold plus 5 dB of the speech level is
        speech, and sets the level the sound around it is judged against: at
        15 dB the tone 20.0-20.5 dB down would sit on that line.)"""
        pauses = analyzer.detect_pauses(
            str(AUDIO_DIR / "loud_quiet.wav"), silence_threshold_db=12.0
        )
        assert pauses == [PauseInterval(start_ms=500, end_ms=1000)]

    def test_click_does_not_split_a_pause(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        gap = np.zeros(SR // 2)
        gap[4000:4160] = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(160) / SR)  # 10 ms click
        signal = np.concatenate([_tone(1.0), gap, _tone(1.0)])
        pauses = analyzer.detect_pauses(str(_write(tmp_path / "click.wav", signal)))
        assert len(pauses) == 1
        assert abs(pauses[0].duration_ms - 500) <= 20

    def test_min_pause_ms_filter(self, analyzer: ProsodyAnalyzer) -> None:
        """Gaps shorter than min_pause_ms are not reported."""
        path = str(AUDIO_DIR / "tone_gap_tone.wav")
        assert analyzer.detect_pauses(path, min_pause_ms=1000) == []
        assert len(analyzer.detect_pauses(path, min_pause_ms=700)) == 1

    def test_speech_pauses_match_ground_truth(self, analyzer: ProsodyAnalyzer) -> None:
        """espeak-ng speech over a -60 dBFS noise floor: the two pauses and
        the silence at both ends, and nothing inside the phrases."""
        truth = _truth("speech_pauses")
        pauses = analyzer.detect_pauses(str(AUDIO_DIR / "speech_pauses.wav"))
        assert len(pauses) == len(truth["pauses"])
        for found, expected in zip(pauses, truth["pauses"], strict=True):
            assert abs(found.start_ms - expected["start_ms"]) <= 30
            assert abs(found.end_ms - expected["end_ms"]) <= 30

    def test_noisy_speech_pauses(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """Only ~25 dB between speech peaks and noise: pauses are still found."""
        truth = _truth("speech_pauses")
        sound = parselmouth.Sound(str(AUDIO_DIR / "speech_pauses.wav"))
        signal = sound.values[0] + _noise(sound.n_samples, -35.0, seed=3)
        pauses = analyzer.detect_pauses(str(_write(tmp_path / "noisy.wav", signal)))
        end = truth["duration_ms"]
        inner = [p for p in truth["pauses"] if p["start_ms"] > 0 and p["end_ms"] < end]
        found = [p for p in pauses if p.start_ms > 0 and p.end_ms < end]
        assert len(found) == len(inner)
        # Weak consonants at the edges of words drown in the noise.
        for got, expected in zip(found, inner, strict=True):
            assert abs(got.start_ms - expected["start_ms"]) <= 50
            assert abs(got.end_ms - expected["end_ms"]) <= 50

    def test_module_function_with_absolute_threshold(self) -> None:
        """The module-level function keeps the absolute rms_threshold_db
        (Praat dB scale): -40 dB only matches digital silence."""
        sound = parselmouth.Sound(str(AUDIO_DIR / "tone_gap_tone.wav"))
        assert detect_pauses(sound, rms_threshold_db=-40.0) == [PauseInterval(500, 1300)]
        assert detect_pauses(sound) == [PauseInterval(500, 1300)]


# ---------------------------------------------------------------------------
# Quieter speech next to louder speech
# ---------------------------------------------------------------------------


def _sentences(truth: dict[str, Any]) -> list[WordAlignment]:
    return [WordAlignment(s["text"], s["start_ms"], s["end_ms"]) for s in truth["sentences"]]


class TestQuieterSpeech:
    """Silence used to be judged against the loudest 1 % of the whole file,
    so the weak sounds of speech 7-20 dB quieter than a louder passage
    elsewhere (consonants, the onsets and ends of vowels) fell below the
    line: false pauses, a speech rate counted over too little speaking time
    (rate="145%" at an unchanged tempo), a level measured on the loudest
    frames only, and words without F0."""

    def test_pauses_match_ground_truth_at_every_level(self, analyzer: ProsodyAnalyzer) -> None:
        """The same sentence at 0, -8 and -13 dB, a sentence 5 dB louder,
        and 80 ms after it one 16 dB quieter than that: the pauses between
        them, and none inside a sentence (there were three, of 210-280 ms)."""
        truth = _truth("speech_levels")
        pauses = analyzer.detect_pauses(AUDIO_DIR / "speech_levels.wav")
        assert len(pauses) == len(truth["pauses"]), pauses
        for found, expected in zip(pauses, truth["pauses"], strict=True):
            assert abs(found.start_ms - expected["start_ms"]) <= 30
            assert abs(found.end_ms - expected["end_ms"]) <= 30

    def test_speech_rate_does_not_depend_on_level(self, analyzer: ProsodyAnalyzer) -> None:
        """The -8 and -13 dB copies read 5 % and 14 % faster."""
        truth = _truth("speech_levels")
        features = analyzer.analyze(AUDIO_DIR / "speech_levels.wav", _sentences(truth))
        reference = features[0].speech_rate
        assert reference is not None
        for sentence, f in zip(truth["sentences"], features, strict=True):
            assert f.speech_rate is not None
            expected = sentence["syllables"] / ((sentence["end_ms"] - sentence["start_ms"]) / 1000)
            assert f.speech_rate == pytest.approx(expected, rel=0.1), sentence["text"]
            if sentence["text"] == truth["sentences"][0]["text"]:
                assert f.speech_rate == pytest.approx(reference, rel=0.03), sentence["amplitude"]

    def test_level_differences_are_measured(self, analyzer: ProsodyAnalyzer) -> None:
        """-8.2 and -12.8 dB read as -6.6 and -9.2 dB: only the loudest
        frames of the quieter copies were measured."""
        truth = _truth("speech_levels")
        features = analyzer.analyze(AUDIO_DIR / "speech_levels.wav", _sentences(truth))
        reference = features[0].intensity_mean
        assert reference is not None
        for sentence, f in zip(truth["sentences"], features, strict=True):
            assert f.intensity_mean is not None
            assert f.intensity_mean - reference == pytest.approx(sentence["level_db"], abs=1.0)

    @pytest.mark.parametrize("gain_db", [10.0, 20.0])
    def test_a_louder_passage_does_not_change_the_words(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, gain_db: float
    ) -> None:
        """speech_pauses.wav followed by speech_raised.wav 10 or 20 dB up:
        at +10 dB a false pause appeared inside 'told you' and every rate
        rose by half; at +20 dB 'you', 'to' and 'me' lost F0 and intensity,
        and the rest read at 9-11 syllables/s."""
        truth = _truth("speech_pauses")
        words = _words(truth)
        first = parselmouth.Sound(str(AUDIO_DIR / "speech_pauses.wav")).values[0]
        louder = parselmouth.Sound(str(AUDIO_DIR / "speech_raised.wav")).values[0]
        signal = np.concatenate(
            [first, _noise(SR // 2, -60.0, seed=5), louder * 10 ** (gain_db / 20)]
        )
        path = _write_float(tmp_path / "louder.wav", signal)
        alone = analyzer.analyze(AUDIO_DIR / "speech_pauses.wav", words)
        together = analyzer.analyze(path, words)
        for a, b in zip(alone, together, strict=True):
            assert b.f0_mean is not None and a.f0_mean is not None, b.text
            # Praat's pitch tracker finds fewer voiced frames in speech far
            # below the loudest in the file, so F0 rests on fewer frames.
            assert b.f0_mean == pytest.approx(a.f0_mean, rel=0.08), b.text
            assert b.intensity_mean == pytest.approx(a.intensity_mean, abs=0.5), b.text
            assert b.speech_rate == pytest.approx(a.speech_rate, rel=0.05), b.text
        end = truth["duration_ms"]
        inner = [p for p in truth["pauses"] if p["start_ms"] > 0 and p["end_ms"] < end]
        found = [p for p in analyzer.detect_pauses(path) if 0 < p.start_ms < end - 30]
        assert abs(found[-1].start_ms - truth["words"][-1]["end_ms"]) <= 30  # into the gap
        assert len(found[:-1]) == len(inner)
        for got, expected in zip(found[:-1], inner, strict=True):
            assert abs(got.start_ms - expected["start_ms"]) <= 30
            assert abs(got.end_ms - expected["end_ms"]) <= 30

    def test_long_pauses_stay_whole(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """A 1.2 s pause holding a breath-like noise 30 dB below the speech,
        and a 1 s pause over a steady tone 30 dB down: far from the words,
        silence is still judged against the file's speech level, so neither
        splits the pause."""
        words = _phonation(0.6, 120.0, seed=1), _phonation(0.6, 120.0, seed=2)
        level = float(np.sqrt(np.mean(words[0] ** 2)))
        breath = np.convolve(_noise(int(0.4 * SR), 0.0, seed=4), np.ones(8) / 8, "same")
        breath *= np.hanning(breath.size) * level * 10 ** (-30 / 20) / np.std(breath)
        gap = np.concatenate([np.zeros(int(0.4 * SR)), breath, np.zeros(int(0.4 * SR))])
        signal = np.concatenate([np.zeros(SR // 4), words[0], gap, words[1], np.zeros(SR // 4)])
        signal = signal + _noise(signal.size, -70.0)
        pauses = analyzer.detect_pauses(_write(tmp_path / "breath.wav", signal))
        inner = [p for p in pauses if p.start_ms > 0 and p.end_ms < 2650]
        assert len(inner) == 1 and abs(inner[0].duration_ms - 1200) <= 60, pauses

        signal = np.concatenate([np.zeros(SR // 4), words[0], np.zeros(SR), words[1]])
        hum = _tone(signal.size / SR, 220.0, level * np.sqrt(2) * 10 ** (-30 / 20))
        signal = signal + hum + _noise(signal.size, -70.0)
        pauses = analyzer.detect_pauses(_write(tmp_path / "hum.wav", signal))
        inner = [p for p in pauses if p.start_ms > 0 and p.end_ms < 2400]
        assert len(inner) == 1 and abs(inner[0].duration_ms - 1000) <= 60, pauses

    def test_sliding_max(self) -> None:
        values = np.array([0.0, 5.0, 1.0, -np.inf, 2.0, 0.0, 0.0, 3.0])
        assert list(_sliding_max(values, 1)) == [5, 5, 5, 2, 2, 2, 3, 3]
        assert list(_sliding_max(values, 0)) == list(values)
        assert list(_sliding_max(values, 20)) == [5] * 8
        assert list(_sliding_max(np.array([7.0]), 3)) == [7.0]


# ---------------------------------------------------------------------------
# Speech rate
# ---------------------------------------------------------------------------


class TestSpeechRate:
    @pytest.mark.parametrize("rate", [3.0, 4.0, 5.0, 6.0])
    def test_known_syllable_rate(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, rate: float
    ) -> None:
        """A tapered window used to lose the edge syllables (4 -> 3.0)."""
        path = _write(tmp_path / "syllables.wav", _syllables(rate))
        f = analyzer.analyze(str(path), [WordAlignment("all", 0, 2000)])[0]
        assert f.speech_rate == pytest.approx(rate, rel=0.1)

    def test_word_spans_report_local_rate(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """One-syllable 'words' still report the surrounding rate rather than
        1 / word duration."""
        path = _write(tmp_path / "syllables.wav", _syllables(5.0))
        words = [WordAlignment(f"w{i}", i * 200, i * 200 + 200) for i in range(10)]
        for f in analyzer.analyze(str(path), words):
            assert f.speech_rate == pytest.approx(5.0, rel=0.2)

    def test_speech_rate_of_espeak_sentence(self, analyzer: ProsodyAnalyzer) -> None:
        """Nine syllables over the speaking time (pauses excluded)."""
        truth = _truth("speech_pauses")
        speaking_ms = truth["duration_ms"] - sum(
            p["end_ms"] - p["start_ms"] for p in truth["pauses"]
        )
        expected = truth["syllables"] / (speaking_ms / 1000)
        f = analyzer.analyze(
            str(AUDIO_DIR / "speech_pauses.wav"), [WordAlignment("all", 0, truth["duration_ms"])]
        )[0]
        assert f.speech_rate == pytest.approx(expected, rel=0.15)

    def test_every_word_has_a_plausible_rate(self, analyzer: ProsodyAnalyzer) -> None:
        truth = _truth("speech_pauses")
        for f in analyzer.analyze(str(AUDIO_DIR / "speech_pauses.wav"), _words(truth)):
            assert f.speech_rate is not None and 1.5 < f.speech_rate < 6.0, f

    def test_silence_and_noise_have_no_rate(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """Noise peaks used to read as 8 syllables/s."""
        noise = _write(tmp_path / "noise.wav", _noise(SR, -70))
        assert analyzer.analyze(str(noise), [WordAlignment("n", 0, 1000)])[0].speech_rate is None
        silence = analyzer.analyze(
            str(AUDIO_DIR / "silence_1s.wav"), [WordAlignment("s", 0, 1000)]
        )[0]
        assert silence.speech_rate is None

    @needs_espeak
    @pytest.mark.parametrize("speed", [110, 160, 230])
    def test_rate_follows_tempo_not_pitch(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, speed: int
    ) -> None:
        """One espeak-ng sentence at three voice pitches. A syllable used to
        count only if the pitch track was voiced in the frame of its
        intensity peak, so the low voice (-p 10, about 73 Hz) lost voicing
        and read at 0.45-0.57 of the true rate."""
        sentence, syllables = "I told you to call me yesterday about the meeting", 14
        rates = []
        for pitch in (10, 50, 90):
            clip = _espeak(sentence, pitch=pitch, speed=speed)
            signal = np.concatenate([np.zeros(SR // 4), clip, np.zeros(SR // 4)])
            signal = 0.5 * signal / np.abs(signal).max() + _noise(signal.size, -60)
            path = _write(tmp_path / f"p{pitch}.wav", signal)
            rate = analyzer.analyze(path, [WordAlignment("all", 0, 2**31 - 1)])[0].speech_rate
            assert rate == pytest.approx(syllables / (clip.size / SR), rel=0.15), pitch
            rates.append(rate)
        assert min(rates) / max(rates) > 0.85

    def test_quiet_word_is_not_split_into_syllables(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """A word more than 25 dB below the loudest speech flickers between
        silent and sounding; masking the silent frames out of the intensity
        contour turned each flicker into a dip, and so into a syllable."""
        loud = [_phonation(0.35, 120.0, seed=i) for i in range(6)]
        quiet = 0.07 * _phonation(0.6, 120.0, seed=9)  # 23 dB down
        signal = np.concatenate(
            [np.concatenate([w, np.zeros(SR // 10)]) for w in loud]
            + [np.zeros(SR // 5), quiet, np.zeros(SR // 5)]
        )
        path = _write(tmp_path / "quiet.wav", signal + _noise(signal.size, -80))
        analysis = prosody_analyzer._AudioAnalysis.from_path(path)
        start = 6 * 0.45 + 0.2
        assert analysis._silent[round(start * 100):].any()  # the flicker
        assert np.count_nonzero(analysis._nuclei >= start) == 1


# ---------------------------------------------------------------------------
# Speech with ground truth
# ---------------------------------------------------------------------------


class TestEspeakSpeech:
    def test_emphasised_word_stands_out(self, analyzer: ProsodyAnalyzer) -> None:
        """'told' was synthesised higher and louder than the other words."""
        truth = _truth("speech_pauses")
        features = analyzer.analyze(str(AUDIO_DIR / "speech_pauses.wav"), _words(truth))
        by_word = {f.text: f for f in features}
        told = by_word[truth["emphasized"]]
        others = [f for f in features if f.text != truth["emphasized"]]
        assert told.f0_mean is not None and told.intensity_mean is not None
        other_f0 = float(np.median([f.f0_mean for f in others if f.f0_mean is not None]))
        other_db = float(np.median([f.intensity_mean for f in others if f.intensity_mean]))
        assert told.f0_mean > 1.25 * other_f0
        assert told.intensity_mean > other_db + 2.0

    def test_every_word_is_voiced(self, analyzer: ProsodyAnalyzer) -> None:
        truth = _truth("speech_pauses")
        for f in analyzer.analyze(str(AUDIO_DIR / "speech_pauses.wav"), _words(truth)):
            assert f.f0_mean is not None and 60 < f.f0_mean < 200, f
            assert f.f0_contour is not None and len(f.f0_contour) >= 5
            assert f.intensity_mean is not None and 55 < f.intensity_mean < 90
            assert f.jitter is not None and f.shimmer is not None and f.hnr is not None


# ---------------------------------------------------------------------------
# The whole recording as one span
# ---------------------------------------------------------------------------


_MEASURES = (
    "f0_mean", "f0_range", "f0_contour", "intensity_mean", "intensity_range",
    "speech_rate", "jitter", "shimmer", "hnr", "quality",
)


class TestWholeRecording:
    """training.features.recording_features measures a recording as the span
    WordAlignment(text, 0, 2**31 - 1), which the analyzer clips to the audio."""

    def test_open_ended_span_is_clipped_to_the_audio(self, analyzer: ProsodyAnalyzer) -> None:
        truth = _truth("speech_pauses")
        path = AUDIO_DIR / "speech_pauses.wav"
        whole, exact = analyzer.analyze(path, [
            WordAlignment("all", 0, 2**31 - 1), WordAlignment("all", 0, truth["duration_ms"]),
        ])
        assert (whole.start_ms, whole.end_ms, whole.text) == (0, 2**31 - 1, "all")
        for name in _MEASURES:
            assert getattr(whole, name) == getattr(exact, name), name

    def test_whole_recording_summarises_the_words(self, analyzer: ProsodyAnalyzer) -> None:
        truth = _truth("speech_pauses")
        path = AUDIO_DIR / "speech_pauses.wav"
        whole = analyzer.analyze(path, [WordAlignment("", 0, 2**31 - 1)])[0]
        words = analyzer.analyze(path, _words(truth))
        # Every voiced frame lies in a word: the mean F0 is theirs.
        assert whole.f0_contour is not None and whole.f0_mean is not None
        assert len(whole.f0_contour) == sum(len(w.f0_contour or ()) for w in words)
        voiced = [v for w in words for v in w.f0_contour or ()]
        assert whole.f0_mean == pytest.approx(float(np.mean(voiced)), rel=0.01)
        speaking_ms = truth["duration_ms"] - sum(
            p["end_ms"] - p["start_ms"] for p in truth["pauses"]
        )
        assert whole.speech_rate == pytest.approx(
            truth["syllables"] / (speaking_ms / 1000), rel=0.1
        )
        assert whole.jitter is not None and whole.shimmer is not None and whole.hnr is not None
        assert whole.quality is None  # nothing left to compare it with

    def test_analyze_recording(self, analyzer: ProsodyAnalyzer) -> None:
        truth = _truth("speech_pauses")
        path = AUDIO_DIR / "speech_pauses.wav"
        recording = analyzer.analyze_recording(path, "I told you")
        open_ended = analyzer.analyze(path, [WordAlignment("", 0, 2**31 - 1)])[0]
        assert (recording.start_ms, recording.end_ms) == (0, truth["duration_ms"])
        assert recording.text == "I told you"
        for name in _MEASURES:
            assert getattr(recording, name) == getattr(open_ended, name), name

    def test_analyze_recording_of_unreadable_audio_raises(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        with pytest.raises(AudioProcessingError, match="not found"):
            analyzer.analyze_recording(tmp_path / "missing.wav")


# ---------------------------------------------------------------------------
# Reading audio
# ---------------------------------------------------------------------------


class TestReadingAudio:
    def test_nonexistent_file_raises(self, analyzer: ProsodyAnalyzer) -> None:
        with pytest.raises(AudioProcessingError, match="not found"):
            analyzer.analyze("/nonexistent.wav", [])

    def test_detect_pauses_nonexistent_file(self, analyzer: ProsodyAnalyzer) -> None:
        with pytest.raises(AudioProcessingError, match="not found"):
            analyzer.detect_pauses("/nonexistent.wav")

    def test_directory_raises(self, analyzer: ProsodyAnalyzer) -> None:
        with pytest.raises(AudioProcessingError, match="not a file"):
            analyzer.analyze(AUDIO_DIR, [])

    def test_empty_file_raises(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        path = tmp_path / "empty.wav"
        path.write_bytes(b"")
        with pytest.raises(AudioProcessingError, match="empty"):
            analyzer.detect_pauses(path)

    @pytest.mark.parametrize("ffmpeg_on_path", [True, False])
    def test_zero_sample_wav_raises(
        self,
        analyzer: ProsodyAnalyzer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ffmpeg_on_path: bool,
    ) -> None:
        if not ffmpeg_on_path:
            monkeypatch.setattr(shutil, "which", lambda name: None)
        elif shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")
        path = _write(tmp_path / "zero.wav", np.zeros(0))
        with pytest.raises(AudioProcessingError):
            analyzer.analyze(path, [WordAlignment("w", 0, 100)])

    @pytest.mark.parametrize("duration_ms", [30, 40, 60])
    def test_too_short_raises(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, duration_ms: int
    ) -> None:
        """Clips under 100 ms used to leak parselmouth.PraatError."""
        path = _write(tmp_path / "short.wav", _tone(duration_ms / 1000))
        with pytest.raises(AudioProcessingError, match="too short"):
            analyzer.analyze(path, [WordAlignment("w", 0, duration_ms)])
        with pytest.raises(AudioProcessingError, match="too short"):
            analyzer.detect_pauses(path)

    def test_non_audio_without_ffmpeg_names_the_fix(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("not audio at all\n")
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(AudioProcessingError, match="install ffmpeg"):
            analyzer.analyze(path, [])

    @needs_ffmpeg
    def test_non_audio_with_ffmpeg_raises(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        path = tmp_path / "notes.wav"
        path.write_text("not audio at all\n")
        with pytest.raises(AudioProcessingError, match="not a supported audio format"):
            analyzer.detect_pauses(path)

    @needs_ffmpeg
    @pytest.mark.parametrize("suffix,codec", [
        (".ogg", "libvorbis"), (".opus", "libopus"), (".webm", "libopus"), (".m4a", "aac"),
    ])
    def test_formats_praat_cannot_read_go_through_ffmpeg(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, suffix: str, codec: str
    ) -> None:
        encoded = tmp_path / f"gap{suffix}"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i",
             str(AUDIO_DIR / "tone_gap_tone.wav"), "-c:a", codec, str(encoded)],
            check=True,
        )
        with pytest.raises(parselmouth.PraatError):
            parselmouth.Sound(str(encoded))
        f = analyzer.analyze(encoded, [WordAlignment("tone", 100, 400)])[0]
        assert f.f0_mean is not None and 215 < f.f0_mean < 225
        pauses = analyzer.detect_pauses(encoded)
        assert len(pauses) == 1 and abs(pauses[0].duration_ms - 800) <= 40

    def test_unreadable_format_without_ffmpeg_names_the_fix(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An Ogg page header: a real container Praat does not read.
        path = tmp_path / "clip.ogg"
        path.write_bytes(b"OggS" + bytes(60))
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(AudioProcessingError, match="OGG/Opus, WebM or M4A"):
            analyzer.detect_pauses(path)

    @needs_ffmpeg
    @pytest.mark.parametrize("name,playlist", [
        # HLS opens any media file it names, by absolute path.
        ("leak.m3u8", "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\n{absolute}\n"
                      "#EXT-X-ENDLIST\n"),
        # concat refuses absolute paths but follows names next to it.
        ("leak.txt", "ffconcat version 1.0\nfile {relative}\n"),
    ], ids=["hls", "concat"])
    def test_playlists_are_not_followed(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, name: str, playlist: str
    ) -> None:
        """An uploaded HLS or concat playlist used to make ffmpeg read (and us
        analyse) other audio files on the machine that it named."""
        secret = tmp_path / "secret.wav"
        shutil.copy(AUDIO_DIR / "speech_pauses.wav", secret)
        path = tmp_path / name
        path.write_text(playlist.format(absolute=secret.resolve(), relative=secret.name))
        with pytest.raises(AudioProcessingError, match="not a supported audio format"):
            analyzer.detect_pauses(path)
        with pytest.raises(AudioProcessingError, match="not a supported audio format"):
            analyzer.analyze(path, [WordAlignment("I", 250, 400)])

    @needs_ffmpeg
    def test_slow_ffmpeg_is_an_audio_processing_error(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        encoded = tmp_path / "gap.ogg"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i",
             str(AUDIO_DIR / "tone_gap_tone.wav"), "-c:a", "libvorbis", str(encoded)],
            check=True,
        )
        monkeypatch.setattr(prosody_analyzer, "_FFMPEG_TIMEOUT_S", 0.001)
        with pytest.raises(AudioProcessingError, match="did not finish"):
            analyzer.detect_pauses(encoded)

    @pytest.mark.parametrize("head", [b"ID3\x03" + bytes(6), b"\xff\xfb\x90\x64"],
                             ids=["id3-tag", "frame-sync"])
    def test_damaged_mp3_is_an_error_not_a_crash(self, tmp_path: Path, head: bytes) -> None:
        """Praat's MP3 reader kills the process (SIGFPE) on such files; they
        now go to ffmpeg. Run in a subprocess so a crash fails the test
        instead of the test run."""
        if shutil.which("ffmpeg") is None:
            pytest.skip("without ffmpeg, MP3 is read by Praat, which crashes on this")
        path = tmp_path / "damaged.mp3"
        path.write_bytes(head + np.random.default_rng(0).bytes(5000))
        script = (
            "import sys\n"
            "from prosody_protocol.exceptions import AudioProcessingError\n"
            "from prosody_protocol.prosody_analyzer import ProsodyAnalyzer\n"
            "try:\n"
            "    ProsodyAnalyzer().detect_pauses(sys.argv[1])\n"
            "except AudioProcessingError as exc:\n"
            "    print('AudioProcessingError:', exc)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr[-500:]}"
        assert "AudioProcessingError: Cannot read audio file" in proc.stdout

    @needs_ffmpeg
    def test_mp3_is_decoded(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """Praat's MP3 reader ignores the encoder delay and put every pause
        ~60 ms late; ffmpeg's timing matches the original recording."""
        truth = _truth("speech_pauses")
        encoded = tmp_path / "speech.mp3"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i",
             str(AUDIO_DIR / "speech_pauses.wav"), "-c:a", "libmp3lame", str(encoded)],
            check=True,
        )
        end = truth["duration_ms"] - 100  # the decoded length differs by a few ms
        inner = [p for p in truth["pauses"] if p["start_ms"] > 0 and p["end_ms"] < end]
        found = [p for p in analyzer.detect_pauses(encoded) if p.start_ms > 0 and p.end_ms < end]
        assert len(found) == len(inner) == 2
        for got, expected in zip(found, inner, strict=True):
            assert abs(got.start_ms - expected["start_ms"]) <= 30
            assert abs(got.end_ms - expected["end_ms"]) <= 30

    @pytest.mark.parametrize("ffmpeg_on_path", [True, False])
    def test_truncated_wav_is_not_padded_with_silence(
        self,
        analyzer: ProsodyAnalyzer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ffmpeg_on_path: bool,
    ) -> None:
        """tone_gap_tone.wav cut off after 1 s of its 1.8 s: Praat padded the
        missing 0.8 s with digital silence, which read as a 1.3 s pause."""
        data = (AUDIO_DIR / "tone_gap_tone.wav").read_bytes()
        path = tmp_path / "truncated.wav"
        path.write_bytes(data[: 44 + SR * 2])  # 44-byte header + 1 s of 16-bit audio
        if not ffmpeg_on_path:
            monkeypatch.setattr(shutil, "which", lambda name: None)
            with pytest.raises(AudioProcessingError, match="truncated"):
                analyzer.detect_pauses(path)
            return
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")
        assert analyzer.detect_pauses(path) == [PauseInterval(500, 1000)]

    def test_stereo_is_mixed_to_mono(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        left = np.concatenate([_tone(0.5, 220), np.zeros(SR // 2), _tone(0.5, 220)])
        path = _write(tmp_path / "stereo.wav", np.stack([left, left], axis=1), channels=2)
        assert len(analyzer.detect_pauses(path)) == 1
        f = analyzer.analyze(path, [WordAlignment("tone", 100, 400)])[0]
        assert f.f0_mean is not None and 215 < f.f0_mean < 225

    @pytest.mark.parametrize(("rate", "frames"), [(1, 100), (8, 100), (100, 1000), (500, 1000)])
    def test_too_low_sample_rate_raises(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, rate: int, frames: int
    ) -> None:
        """A 1 Hz WAV of 100 frames raised IndexError from numpy (add.reduceat),
        and 500 Hz audio was analysed into nonsense (noise read as a rising
        120 Hz voice)."""
        path = _write(tmp_path / "low.wav", _noise(frames, -20), sr=rate)
        with pytest.raises(AudioProcessingError, match=f"sampled at {rate} Hz"):
            analyzer.analyze(path, [WordAlignment("w", 0, 1000)])
        with pytest.raises(AudioProcessingError, match="cannot be analysed"):
            analyzer.detect_pauses(path)
        with pytest.raises(AudioProcessingError, match="cannot be analysed"):
            detect_pauses(parselmouth.Sound(_noise(frames, -20), rate))

    def test_too_low_sample_rate_through_the_converter(self, tmp_path: Path) -> None:
        from prosody_protocol import AudioToIML

        path = _write(tmp_path / "low.wav", _noise(100, -20), sr=1)
        with pytest.raises(AudioProcessingError, match="cannot be analysed"):
            AudioToIML(stt="none").convert_detailed(path)

    def test_telephone_audio_is_analysed(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        t = np.arange(8000) / 8000
        path = _write(tmp_path / "phone.wav", 0.3 * np.sin(2 * np.pi * 150 * t), sr=8000)
        f = analyzer.analyze(path, [WordAlignment("w", 100, 900)])[0]
        assert f.f0_mean == pytest.approx(150, rel=0.01)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_samples_raise(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, bad: float
    ) -> None:
        """A damaged float WAV with NaN or infinite samples was analysed as if
        the samples were silence (F0 None, intensity 80 dB)."""
        samples = _tone(1.0).astype("<f4")
        samples[100:200] = bad
        data = samples.tobytes()
        header = (
            b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 3, 1, SR, SR * 4, 4, 32)
            + b"data" + struct.pack("<I", len(data))
        )
        path = tmp_path / "float.wav"
        path.write_bytes(header + data)
        with pytest.raises(AudioProcessingError, match="not finite"):
            analyzer.analyze(path, [WordAlignment("w", 0, 1000)])

    @staticmethod
    def _float_wav(path: Path, samples: Any) -> Path:
        """*samples* as a mono float WAV (64-bit when they are float64)."""
        data = samples.tobytes()
        width = samples.dtype.itemsize
        header = (
            b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 3, 1, SR, SR * width, width, 8 * width)
            + b"data" + struct.pack("<I", len(data))
        )
        path.write_bytes(header + data)
        return path

    @pytest.mark.parametrize(
        ("scale", "dtype"), [(1e300, "<f8"), (1e150, "<f8"), (3e38, "<f4"), (2.0**31, "<f4")]
    )
    def test_samples_beyond_any_sound_raise(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, scale: float, dtype: str
    ) -> None:
        """Samples of 1e300 overflowed the frame power into NaN, and the
        converter returned an empty <utterance></utterance>; 3e38 read as
        an intensity of 860 dB."""
        from prosody_protocol import AudioToIML

        path = self._float_wav(tmp_path / "loud.wav", (scale * _voice(1.0) / 0.3).astype(dtype))
        with pytest.raises(AudioProcessingError, match="far beyond full scale"):
            analyzer.analyze(path, [WordAlignment("w", 0, 1000)])
        with pytest.raises(AudioProcessingError, match="far beyond full scale"):
            AudioToIML(stt="none").convert_detailed(path)

    def test_samples_on_an_integer_scale_are_analysed(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path
    ) -> None:
        """Float data scaled like 16-bit integers is still sound, only 90 dB
        louder: every measure but the intensity is unchanged."""
        signal = _voice(1.0, f0=150.0)
        scaled = self._float_wav(tmp_path / "scaled.wav", (32767 * signal).astype("<f4"))
        plain = self._float_wav(tmp_path / "plain.wav", signal.astype("<f4"))
        word = [WordAlignment("w", 100, 900)]
        big, small = analyzer.analyze(scaled, word)[0], analyzer.analyze(plain, word)[0]
        assert big.f0_mean == pytest.approx(small.f0_mean, rel=1e-3)
        assert big.intensity_mean is not None and small.intensity_mean is not None
        assert big.intensity_mean - small.intensity_mean == pytest.approx(90.3, abs=0.1)


# ---------------------------------------------------------------------------
# Length limit
# ---------------------------------------------------------------------------


class TestMaxDuration:
    @pytest.mark.parametrize("fmt", ["WAV", "FLAC", "AIFF"])
    def test_long_audio_is_rejected_before_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fmt: str
    ) -> None:
        """The length comes from the header: a small FLAC can hold hours."""
        path = tmp_path / f"long.{fmt.lower()}"
        parselmouth.Sound(_tone(3.0), SR).save(str(path), fmt)
        loaded: list[str] = []
        real_sound = parselmouth.Sound

        def spy(*args: Any, **kwargs: Any) -> Any:
            loaded.append(str(args[0]) if args else "")
            return real_sound(*args, **kwargs)

        monkeypatch.setattr(parselmouth, "Sound", spy)
        limited = ProsodyAnalyzer(max_duration_s=2.0)
        with pytest.raises(AudioProcessingError, match="longer than max_duration_s=2 s"):
            limited.analyze(path, [WordAlignment("tone", 0, 500)])
        with pytest.raises(AudioProcessingError, match="longer than max_duration_s=2 s"):
            limited.detect_pauses(path)
        assert str(path) not in loaded

    def test_audio_within_the_limit_is_analysed(self) -> None:
        truth = _truth("speech_pauses")
        limited = ProsodyAnalyzer(max_duration_s=truth["duration_ms"] / 1000 + 0.1)
        pauses = limited.detect_pauses(AUDIO_DIR / "speech_pauses.wav")
        assert len(pauses) == len(truth["pauses"])

    @needs_ffmpeg
    def test_compressed_audio_is_cut_off_while_decoding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opus has no length in its header; decoding stops just past the limit."""
        encoded = tmp_path / "long.ogg"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
             "sine=frequency=220:sample_rate=48000:duration=60", "-c:a", "libopus",
             str(encoded)],
            check=True,
        )
        decoded: list[float] = []
        real_sound = parselmouth.Sound

        def spy(*args: Any, **kwargs: Any) -> Any:
            sound = real_sound(*args, **kwargs)
            decoded.append(sound.duration)
            return sound

        monkeypatch.setattr(parselmouth, "Sound", spy)
        with pytest.raises(AudioProcessingError, match="longer than max_duration_s=5 s"):
            ProsodyAnalyzer(max_duration_s=5).detect_pauses(encoded)
        assert decoded and max(decoded) <= 6.01

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
    def test_invalid_limit_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="max_duration_s"):
            ProsodyAnalyzer(max_duration_s=bad)

    def test_no_limit_by_default(self) -> None:
        assert ProsodyAnalyzer().max_duration_s is None


# ---------------------------------------------------------------------------
# Sample rate and channels
# ---------------------------------------------------------------------------


def _fast_speech(path: Path, rate: int, channels: int) -> Path:
    """speech_pauses.wav at *rate* with *channels* (the second one quieter),
    over a full-band -70 dBFS noise floor, as 16-bit WAV or (by suffix) FLAC."""
    speech = parselmouth.Sound(str(AUDIO_DIR / "speech_pauses.wav")).resample(rate).values[0]
    rows = [speech * (1.0 - 0.3 * c) + _noise(speech.size, -70.0, seed=c) for c in range(channels)]
    sound = parselmouth.Sound(np.array(rows), sampling_frequency=rate)
    if path.suffix == ".flac":
        sound.save(str(path), "FLAC")
        return path
    return _write(path, sound.values.T, sr=rate, channels=channels)


class TestSampleRate:
    """Audio used to be analysed at its own rate and channel count: a 29 MB
    FLAC of ten minutes at 192 kHz stereo took 2.8 GB and four minutes. It
    is now read as mono at no more than 16 kHz, a block at a time."""

    @pytest.mark.parametrize(
        ("rate", "channels", "suffix"),
        [(22050, 1, ".wav"), (44100, 2, ".flac"), (48000, 1, ".wav"), (192000, 2, ".flac")],
    )
    def test_fast_audio_is_analysed_at_16_khz(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, rate: int, channels: int, suffix: str
    ) -> None:
        path = _fast_speech(tmp_path / f"fast{suffix}", rate, channels)
        analysis = prosody_analyzer._AudioAnalysis.from_path(path)
        assert analysis.sound.sampling_frequency == SR
        assert analysis.sound.n_channels == 1
        truth = _truth("speech_pauses")
        assert abs(analysis.duration_ms - truth["duration_ms"]) <= 1

        # The measures match those of the 16 kHz original.
        words = _words(truth)
        original = analyzer.analyze(AUDIO_DIR / "speech_pauses.wav", words)
        level = 0.0 if channels == 1 else 20 * np.log10(0.85)  # the channels' mean
        for a, b in zip(original, analysis.features(words), strict=True):
            assert b.f0_mean == pytest.approx(a.f0_mean, rel=0.01), b.text
            assert b.intensity_mean is not None and a.intensity_mean is not None
            assert b.intensity_mean - a.intensity_mean == pytest.approx(level, abs=0.5), b.text
            assert b.speech_rate == pytest.approx(a.speech_rate, rel=0.05), b.text
        found = analysis.pauses()
        assert len(found) == len(truth["pauses"])
        for got, expected in zip(found, truth["pauses"], strict=True):
            assert abs(got.start_ms - expected["start_ms"]) <= 30
            assert abs(got.end_ms - expected["end_ms"]) <= 30

    def test_fast_audio_is_never_loaded_whole(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Praat reads it in blocks (here of 0.1 s), and the blocks join
        seamlessly: the result is the whole file mixed and resampled at
        once, to within a few steps of 16-bit audio (Praat's resampling
        filter reaches across the whole file)."""
        path = _fast_speech(tmp_path / "fast.wav", 48000, 2)
        whole = parselmouth.Sound(str(path)).convert_to_mono().resample(SR).values[0]
        monkeypatch.setattr(prosody_analyzer, "_READ_BLOCK_SAMPLES", 9600)
        monkeypatch.setattr(prosody_analyzer, "_MIN_READ_BLOCK_S", 0.1)
        loaded: list[str] = []
        largest = [0]
        real_sound, real_call = parselmouth.Sound, prosody_analyzer.call

        def spy_sound(*args: Any, **kwargs: Any) -> Any:
            loaded.append(str(args[0]) if args and isinstance(args[0], str) else "")
            return real_sound(*args, **kwargs)

        def spy_call(*args: Any) -> Any:
            result = real_call(*args)
            if len(args) > 1 and args[1] == "Extract part":
                largest[0] = max(largest[0], result.values.size)
            return result

        monkeypatch.setattr(parselmouth, "Sound", spy_sound)
        monkeypatch.setattr(prosody_analyzer, "call", spy_call)
        sound = prosody_analyzer._load_sound(path)
        assert str(path) not in loaded
        # A block and its padding: (0.1 + 2 * 0.05) s of two channels at 48 kHz.
        assert 0 < largest[0] <= 2 * 48000 * 0.2 + 4
        assert sound.sampling_frequency == SR and sound.n_samples == whole.size
        error = sound.values[0] - whole
        assert np.abs(error).max() < 3e-4 and np.sqrt(np.mean(error**2)) < 3e-5

    @pytest.mark.parametrize(("bad", "message"), [
        (float("nan"), "not finite"), (1e6, "far beyond full scale"), (1e300, "far beyond"),
    ])
    def test_damaged_fast_audio_raises(self, tmp_path: Path, bad: float, message: str) -> None:
        """Each block is checked before it is resampled, which would spread
        a NaN over the block, or overflow 1e300 into infinity."""
        signal = np.tile(_tone(1.0), 3)
        signal[20000:20100] = bad
        path = _write_float(tmp_path / "damaged.wav", signal, sr=48000, dtype="<f8")
        with pytest.raises(AudioProcessingError, match=message):
            prosody_analyzer._AudioAnalysis.from_path(path)

    @pytest.mark.parametrize("ffmpeg_on_path", [True, False])
    def test_truncated_fast_wav_is_not_padded_with_silence(
        self, analyzer: ProsodyAnalyzer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        ffmpeg_on_path: bool,
    ) -> None:
        """The header promises 4.4 s of 48 kHz stereo; 2 s are there."""
        data = _fast_speech(tmp_path / "fast.wav", 48000, 2).read_bytes()
        path = tmp_path / "truncated.wav"
        path.write_bytes(data[: 44 + 2 * 48000 * 2 * 2])
        if not ffmpeg_on_path:
            monkeypatch.setattr(shutil, "which", lambda name: None)
            with pytest.raises(AudioProcessingError, match="truncated"):
                analyzer.detect_pauses(path)
            return
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")
        analysis = prosody_analyzer._AudioAnalysis.from_path(path)
        assert analysis.duration_ms == 2000
        assert analysis.sound.sampling_frequency == SR

    def test_slow_audio_keeps_its_rate(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """Audio at 16 kHz or less is not resampled (8 kHz stays 8 kHz)."""
        t = np.arange(8000) / 8000
        signal = 0.3 * np.sin(2 * np.pi * 150 * t)
        path = _write(tmp_path / "phone.wav", np.stack([signal, signal], axis=1), 8000, 2)
        sound = prosody_analyzer._load_sound(path)
        assert (sound.sampling_frequency, sound.n_channels) == (8000, 1)
        assert np.abs(sound.values[0] - signal).max() < 1e-4


# ---------------------------------------------------------------------------
# Alignment handling and performance
# ---------------------------------------------------------------------------


class TestAlignments:
    def test_empty_alignments(self, analyzer: ProsodyAnalyzer) -> None:
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), [])
        assert results == []

    def test_alignment_beyond_audio_clipped(self, analyzer: ProsodyAnalyzer) -> None:
        """Alignment end beyond audio duration should be clamped."""
        alignments = [WordAlignment(word="long", start_ms=0, end_ms=99999)]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        assert len(results) == 1
        assert results[0].f0_mean is not None
        assert results[0].end_ms == 99999

    def test_one_result_per_alignment_in_order(self, analyzer: ProsodyAnalyzer) -> None:
        alignments = [
            WordAlignment(word="b", start_ms=500, end_ms=900),
            WordAlignment(word="a", start_ms=100, end_ms=400),
            WordAlignment(word="a", start_ms=100, end_ms=400),
            WordAlignment(word="empty", start_ms=300, end_ms=300),
            WordAlignment(word="after", start_ms=5000, end_ms=6000),
        ]
        results = analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), alignments)
        assert [(r.text, r.start_ms, r.end_ms) for r in results] == [
            (a.word, a.start_ms, a.end_ms) for a in alignments
        ]
        assert results[3].f0_mean is None and results[4].f0_mean is None

    def test_pitch_is_tracked_twice_per_file(
        self, analyzer: ProsodyAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-word re-analysis of the whole file made runtime quadratic. The
        file is tracked twice: once to find the speaker's range, once in it."""
        calls: list[int] = []
        original = parselmouth.Sound.to_pitch_ac

        def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(parselmouth.Sound, "to_pitch_ac", counting)
        words = [WordAlignment(f"w{i}", i * 10, i * 10 + 60) for i in range(90)]
        analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), words)
        assert len(calls) == 2

    def test_minute_of_speech_is_fast(self, analyzer: ProsodyAnalyzer, tmp_path: Path) -> None:
        """60 s of audio with 150 words took ~10 s before the fix."""
        truth = _truth("speech_pauses")
        clip = parselmouth.Sound(str(AUDIO_DIR / "speech_pauses.wav")).values[0]
        repeats = int(np.ceil(60_000 / truth["duration_ms"]))
        path = _write(tmp_path / "minute.wav", np.tile(clip, repeats))
        words = [WordAlignment(f"w{i}", i * 400, i * 400 + 300) for i in range(150)]
        start = time.perf_counter()
        features = analyzer.analyze(path, words)
        pauses = analyzer.detect_pauses(path)
        elapsed = time.perf_counter() - start
        assert len(features) == 150
        assert len(pauses) >= 2 * repeats
        assert elapsed < 5.0, f"analysis took {elapsed:.1f}s"
