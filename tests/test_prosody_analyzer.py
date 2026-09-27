"""Tests for prosody_protocol.prosody_analyzer.

Uses audio with known acoustic properties: synthetic tones, silence and
gaps (some generated on the fly with noise floors, known jitter/shimmer or
known syllable rates), and espeak-ng speech whose word timings and pauses
are recorded in JSON next to the WAV (see tests/generate_audio_fixtures.py).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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
    _remove_octave_jumps,
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

    def test_unvoiced_span_has_no_f0(self, analyzer: ProsodyAnalyzer) -> None:
        f = analyzer.analyze(
            str(AUDIO_DIR / "tone_gap_tone.wav"), [WordAlignment("gap", 700, 1100)]
        )[0]
        assert f.f0_mean is None and f.f0_range is None and f.f0_contour is None


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
        path = _write(tmp_path / "voice.wav", _voice(1.5))
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
        """With a 15 dB threshold, the half 20 dB down counts as silence."""
        pauses = analyzer.detect_pauses(
            str(AUDIO_DIR / "loud_quiet.wav"), silence_threshold_db=15.0
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

    def test_pitch_is_tracked_once_per_file(
        self, analyzer: ProsodyAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-word re-analysis of the whole file made runtime quadratic."""
        calls: list[int] = []
        original = parselmouth.Sound.to_pitch_ac

        def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(parselmouth.Sound, "to_pitch_ac", counting)
        words = [WordAlignment(f"w{i}", i * 10, i * 10 + 60) for i in range(90)]
        analyzer.analyze(str(AUDIO_DIR / "tone_220hz.wav"), words)
        assert len(calls) == 1

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
