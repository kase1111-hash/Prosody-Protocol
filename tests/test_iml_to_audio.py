"""Tests for prosody_protocol.iml_to_audio.

Acceptance criteria from Phase 5:
- pitch="+15%" → measurably higher F0 than baseline
- <pause duration="800"/> → ~800ms of silence in output
- <emphasis level="strong"> words are louder/higher-pitched
- Output is valid WAV

Uses parselmouth to verify acoustic properties of generated audio.
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call
import pytest

from prosody_protocol.exceptions import ConversionError
from prosody_protocol.iml_to_audio import (
    BASE_FREQ,
    IMLToAudio,
    _parse_pitch,
    _parse_volume,
)
from prosody_protocol.models import (
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Utterance,
)


@pytest.fixture()
def synth() -> IMLToAudio:
    return IMLToAudio()


def _wav_to_sound(wav_bytes: bytes) -> parselmouth.Sound:
    """Load WAV bytes into a parselmouth.Sound for analysis."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
    return parselmouth.Sound(samples, sampling_frequency=sr)


def _measure_f0(sound: parselmouth.Sound) -> float | None:
    """Measure mean F0 of a Sound object."""
    pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
    values = []
    step = 0.01
    for t in np.arange(0, sound.duration, step):
        v = call(pitch, "Get value at time", float(t), "Hertz", "Linear")
        if not np.isnan(v):
            values.append(v)
    return float(np.mean(values)) if values else None


def _measure_rms(sound: parselmouth.Sound) -> float:
    """Measure RMS amplitude of a Sound."""
    return float(np.sqrt(np.mean(sound.values**2)))


def _wav_duration(wav_bytes: bytes) -> float:
    """Get duration in seconds from WAV bytes."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class TestInterface:
    def test_instantiate(self) -> None:
        synth = IMLToAudio()
        assert synth.voice == "en_US-female-medium"
        assert synth.engine == "builtin"

    def test_custom_params(self) -> None:
        synth = IMLToAudio(voice="en_US-male-low", engine="coqui")
        assert synth.voice == "en_US-male-low"
        assert synth.engine == "coqui"


# ---------------------------------------------------------------------------
# WAV output validity
# ---------------------------------------------------------------------------


class TestWAVValidity:
    def test_returns_bytes(self, synth: IMLToAudio) -> None:
        result = synth.synthesize("<utterance>Hello world.</utterance>")
        assert isinstance(result, bytes)

    def test_valid_wav_header(self, synth: IMLToAudio) -> None:
        result = synth.synthesize("<utterance>Hello.</utterance>")
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    def test_wav_is_loadable(self, synth: IMLToAudio) -> None:
        result = synth.synthesize("<utterance>Test.</utterance>")
        buf = io.BytesIO(result)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() > 0

    def test_non_zero_audio(self, synth: IMLToAudio) -> None:
        result = synth.synthesize("<utterance>Sound.</utterance>")
        sound = _wav_to_sound(result)
        rms = _measure_rms(sound)
        assert rms > 0.0


# ---------------------------------------------------------------------------
# Pitch verification (Acceptance criterion #1)
# ---------------------------------------------------------------------------


class TestPitchModification:
    def test_higher_pitch_produces_higher_f0(self, synth: IMLToAudio) -> None:
        """pitch='+15%' should produce measurably higher F0 than baseline."""
        baseline_wav = synth.synthesize("<utterance>word</utterance>")
        high_wav = synth.synthesize(
            '<utterance><prosody pitch="+15%">word</prosody></utterance>'
        )

        baseline_f0 = _measure_f0(_wav_to_sound(baseline_wav))
        high_f0 = _measure_f0(_wav_to_sound(high_wav))

        assert baseline_f0 is not None
        assert high_f0 is not None
        assert high_f0 > baseline_f0 * 1.10, (
            f"Expected higher pitch, got baseline={baseline_f0:.1f} Hz, "
            f"high={high_f0:.1f} Hz"
        )

    def test_lower_pitch(self, synth: IMLToAudio) -> None:
        baseline_wav = synth.synthesize("<utterance>word</utterance>")
        low_wav = synth.synthesize(
            '<utterance><prosody pitch="-20%">word</prosody></utterance>'
        )

        baseline_f0 = _measure_f0(_wav_to_sound(baseline_wav))
        low_f0 = _measure_f0(_wav_to_sound(low_wav))

        assert baseline_f0 is not None
        assert low_f0 is not None
        assert low_f0 < baseline_f0 * 0.90

    def test_absolute_hz_pitch(self, synth: IMLToAudio) -> None:
        wav = synth.synthesize(
            '<utterance><prosody pitch="300Hz">word</prosody></utterance>'
        )
        f0 = _measure_f0(_wav_to_sound(wav))
        assert f0 is not None
        assert 270 < f0 < 330, f"Expected ~300 Hz, got {f0:.1f}"

    def test_semitone_pitch(self, synth: IMLToAudio) -> None:
        baseline_wav = synth.synthesize("<utterance>word</utterance>")
        up_wav = synth.synthesize(
            '<utterance><prosody pitch="+12st">word</prosody></utterance>'
        )

        baseline_f0 = _measure_f0(_wav_to_sound(baseline_wav))
        up_f0 = _measure_f0(_wav_to_sound(up_wav))

        assert baseline_f0 is not None
        assert up_f0 is not None
        # +12 semitones should roughly double the frequency.
        assert up_f0 > baseline_f0 * 1.8


# ---------------------------------------------------------------------------
# Pause verification (Acceptance criterion #2)
# ---------------------------------------------------------------------------


class TestPauseInsertion:
    def test_pause_800ms_produces_silence(self, synth: IMLToAudio) -> None:
        """<pause duration='800'/> should produce ~800ms of silence."""
        with_pause = synth.synthesize(
            '<utterance>word<pause duration="800"/>word</utterance>'
        )
        without_pause = synth.synthesize(
            "<utterance>word word</utterance>"
        )

        dur_with = _wav_duration(with_pause)
        dur_without = _wav_duration(without_pause)

        # The version with a pause should be ~0.8s longer.
        diff_ms = (dur_with - dur_without) * 1000
        assert diff_ms > 600, f"Expected ~800ms pause, got {diff_ms:.0f}ms extra"

    def test_pause_200ms(self, synth: IMLToAudio) -> None:
        with_pause = synth.synthesize(
            '<utterance>a<pause duration="200"/>b</utterance>'
        )
        without_pause = synth.synthesize(
            "<utterance>a b</utterance>"
        )
        diff_ms = (_wav_duration(with_pause) - _wav_duration(without_pause)) * 1000
        assert diff_ms > 100


# ---------------------------------------------------------------------------
# Emphasis verification (Acceptance criterion #3)
# ---------------------------------------------------------------------------


class TestEmphasis:
    def test_strong_emphasis_louder(self, synth: IMLToAudio) -> None:
        """<emphasis level='strong'> should be louder than plain text."""
        plain_wav = synth.synthesize("<utterance>word</utterance>")
        emph_wav = synth.synthesize(
            '<utterance><emphasis level="strong">word</emphasis></utterance>'
        )

        plain_rms = _measure_rms(_wav_to_sound(plain_wav))
        emph_rms = _measure_rms(_wav_to_sound(emph_wav))

        assert emph_rms > plain_rms * 1.1

    def test_strong_emphasis_higher_pitch(self, synth: IMLToAudio) -> None:
        plain_wav = synth.synthesize("<utterance>word</utterance>")
        emph_wav = synth.synthesize(
            '<utterance><emphasis level="strong">word</emphasis></utterance>'
        )

        plain_f0 = _measure_f0(_wav_to_sound(plain_wav))
        emph_f0 = _measure_f0(_wav_to_sound(emph_wav))

        assert plain_f0 is not None
        assert emph_f0 is not None
        assert emph_f0 > plain_f0


# ---------------------------------------------------------------------------
# Volume modification
# ---------------------------------------------------------------------------


class TestVolumeModification:
    def test_louder_volume(self, synth: IMLToAudio) -> None:
        baseline_wav = synth.synthesize("<utterance>word</utterance>")
        loud_wav = synth.synthesize(
            '<utterance><prosody volume="+6dB">word</prosody></utterance>'
        )

        baseline_rms = _measure_rms(_wav_to_sound(baseline_wav))
        loud_rms = _measure_rms(_wav_to_sound(loud_wav))

        assert loud_rms > baseline_rms * 1.3

    def test_quieter_volume(self, synth: IMLToAudio) -> None:
        baseline_wav = synth.synthesize("<utterance>word</utterance>")
        quiet_wav = synth.synthesize(
            '<utterance><prosody volume="-6dB">word</prosody></utterance>'
        )

        baseline_rms = _measure_rms(_wav_to_sound(baseline_wav))
        quiet_rms = _measure_rms(_wav_to_sound(quiet_wav))

        assert quiet_rms < baseline_rms * 0.8


# ---------------------------------------------------------------------------
# synthesize_to_file
# ---------------------------------------------------------------------------


class TestSynthesizeToFile:
    def test_creates_file(self, synth: IMLToAudio, tmp_path: Path) -> None:
        out = tmp_path / "output.wav"
        synth.synthesize_to_file("<utterance>Hello.</utterance>", out)
        assert out.exists()
        assert out.stat().st_size > 44  # More than just a WAV header

    def test_file_is_valid_wav(self, synth: IMLToAudio, tmp_path: Path) -> None:
        out = tmp_path / "output.wav"
        synth.synthesize_to_file("<utterance>Test.</utterance>", out)
        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getnframes() > 0


# ---------------------------------------------------------------------------
# synthesize_doc
# ---------------------------------------------------------------------------


class TestSynthesizeDoc:
    def test_from_document(self, synth: IMLToAudio) -> None:
        doc = IMLDocument(
            utterances=(Utterance(children=("Hello world.",)),),
        )
        wav = synth.synthesize_doc(doc)
        assert isinstance(wav, bytes)
        assert wav[:4] == b"RIFF"


# ---------------------------------------------------------------------------
# Multi-utterance
# ---------------------------------------------------------------------------


class TestMultiUtterance:
    def test_multiple_utterances_produce_longer_audio(
        self, synth: IMLToAudio
    ) -> None:
        single = synth.synthesize("<utterance>Hello.</utterance>")
        multi = synth.synthesize(
            '<iml version="0.1.0">'
            "<utterance>Hello.</utterance>"
            "<utterance>World.</utterance>"
            "</iml>"
        )
        assert _wav_duration(multi) > _wav_duration(single)


# ---------------------------------------------------------------------------
# Pitch/volume parsing helpers
# ---------------------------------------------------------------------------


class TestPitchParsing:
    def test_percentage(self) -> None:
        assert abs(_parse_pitch("+15%", 180.0) - 207.0) < 1.0

    def test_negative_percentage(self) -> None:
        assert abs(_parse_pitch("-20%", 180.0) - 144.0) < 1.0

    def test_semitones(self) -> None:
        # +12st should double the frequency.
        assert abs(_parse_pitch("+12st", 180.0) - 360.0) < 1.0

    def test_absolute_hz(self) -> None:
        assert _parse_pitch("300Hz", 180.0) == 300.0

    def test_none_returns_base(self) -> None:
        assert _parse_pitch(None, 200.0) == 200.0

    def test_unknown_format_returns_base(self) -> None:
        assert _parse_pitch("loud", 200.0) == 200.0


class TestVolumeParsing:
    def test_positive_db(self) -> None:
        result = _parse_volume("+6dB", 0.5)
        assert result > 0.5 * 1.5  # +6dB ≈ 2x amplitude

    def test_negative_db(self) -> None:
        result = _parse_volume("-6dB", 0.5)
        assert result < 0.5 * 0.6  # -6dB ≈ 0.5x amplitude

    def test_none_returns_base(self) -> None:
        assert _parse_volume(None, 0.5) == 0.5

    def test_unknown_format_returns_base(self) -> None:
        assert _parse_volume("very loud", 0.5) == 0.5


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_malformed_iml_raises(self, synth: IMLToAudio) -> None:
        with pytest.raises(ConversionError):
            synth.synthesize("<utterance>unclosed")

    def test_empty_string_raises(self, synth: IMLToAudio) -> None:
        with pytest.raises(ConversionError):
            synth.synthesize("")
