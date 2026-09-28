"""Tests for prosody_protocol.iml_to_audio.

Acceptance criteria from Phase 5, checked on both engines:
- pitch="+15%" → measurably higher F0 than baseline
- <pause duration="800"/> → ~800ms of silence in output
- <emphasis level="strong"> words are louder (and, in the preview, higher)
- Output is valid WAV

Plus: real speech from espeak-ng (tests skip when it is not installed),
engine and voice selection, rate/tempo, reduced emphasis, headroom,
punctuation, pitch contours, duration caps and input validation.

Uses parselmouth to verify acoustic properties of generated audio.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tracemalloc
import warnings
import wave
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("parselmouth")

import numpy as np
import parselmouth
from parselmouth.praat import call

from prosody_protocol import iml_to_audio
from prosody_protocol.exceptions import ConversionError, IMLValidationError
from prosody_protocol.iml_to_audio import (
    IMLToAudio,
    _parse_pitch,
    _parse_voice,
    _parse_volume,
    _resolve_espeak_voice,
)
from prosody_protocol.models import (
    IMLDocument,
    Pause,
    Utterance,
)

FIXTURES = Path(__file__).parent / "fixtures" / "valid"
HAS_ESPEAK = shutil.which("espeak-ng") is not None
needs_espeak = pytest.mark.skipif(not HAS_ESPEAK, reason="espeak-ng is not installed")
SENTENCE = "Please listen carefully to me now."


def _espeak_version() -> str | None:
    if not HAS_ESPEAK:
        return None
    output = subprocess.run(
        ["espeak-ng", "--version"], capture_output=True, text=True, timeout=30, check=False
    ).stdout
    m = re.search(r"\d+\.\d+", output)
    return m.group() if m else None


# The espeak-ng adaptation's pitch curve was measured on espeak-ng 1.51, so
# exact F0 ratios are only checked there; other versions are checked for the
# direction of each change.
measured_espeak = pytest.mark.skipif(
    _espeak_version() != "1.51",
    reason="exact F0 ratios are calibrated for espeak-ng 1.51",
)


@pytest.fixture()
def synth() -> IMLToAudio:
    """The tone preview: deterministic, needs only numpy."""
    return IMLToAudio(engine="tones")


@pytest.fixture()
def speech() -> IMLToAudio:
    if not HAS_ESPEAK:
        pytest.skip("espeak-ng is not installed")
    return IMLToAudio(engine="espeak")


def _samples(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0, sr


def _wav_to_sound(wav_bytes: bytes) -> parselmouth.Sound:
    """Load WAV bytes into a parselmouth.Sound for analysis."""
    samples, sr = _samples(wav_bytes)
    return parselmouth.Sound(samples, sampling_frequency=sr)


def _f0_track(wav_bytes: bytes) -> np.ndarray:
    """Voiced F0 values (Hz) in time order."""
    pitch = _wav_to_sound(wav_bytes).to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"]
    return np.asarray(f0[f0 > 0])


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


def _mean_f0(wav_bytes: bytes) -> float:
    track = _f0_track(wav_bytes)
    assert track.size > 0, "no voiced frames"
    return float(np.mean(track))


def _measure_rms(sound: parselmouth.Sound) -> float:
    """Measure RMS amplitude of a Sound."""
    return float(np.sqrt(np.mean(sound.values**2)))


def _active_rms(wav_bytes: bytes) -> float:
    """RMS over 20 ms frames that are not silent (speech has many pauses)."""
    samples, sr = _samples(wav_bytes)
    frame = int(0.02 * sr)
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return float(np.mean(rms[rms > 0.005]))


def _wav_duration(wav_bytes: bytes) -> float:
    """Get duration in seconds from WAV bytes."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _clipped(wav_bytes: bytes) -> int:
    samples, _ = _samples(wav_bytes)
    return int(np.sum(np.abs(samples) >= 0.999))


# ---------------------------------------------------------------------------
# Interface and engine selection
# ---------------------------------------------------------------------------


class TestInterface:
    def test_instantiate_defaults(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            synth = IMLToAudio()
        assert synth.voice is None
        assert synth.engine == "auto"
        assert synth.max_duration_s == 300.0
        assert synth.strict is True
        assert synth.backend == ("espeak" if HAS_ESPEAK else "tones")

    def test_tones_engine(self) -> None:
        synth = IMLToAudio(voice="en_US-male-low", engine="tones")
        assert synth.voice == "en_US-male-low"
        assert synth.backend == "tones"

    def test_builtin_is_deprecated_alias(self) -> None:
        with pytest.warns(DeprecationWarning, match="'tones'"):
            synth = IMLToAudio(engine="builtin")
        assert synth.backend == "tones"

    @pytest.mark.parametrize("engine", ["coqui", "piper", "elevenlabs"])
    def test_unimplemented_engines_rejected(self, engine: str) -> None:
        """They used to fall back to tones while claiming to be TTS engines."""
        with pytest.raises(ConversionError, match="not implemented"):
            IMLToAudio(engine=engine)  # type: ignore[arg-type]

    def test_unknown_engine_rejected(self) -> None:
        with pytest.raises(ConversionError, match="Unknown engine"):
            IMLToAudio(engine="festival")  # type: ignore[arg-type]

    def test_auto_without_espeak_warns_and_uses_tones(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(iml_to_audio.shutil, "which", lambda name: None)
        with pytest.warns(UserWarning, match="not speech"):
            synth = IMLToAudio()
        assert synth.backend == "tones"

    def test_espeak_engine_requires_espeak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(iml_to_audio.shutil, "which", lambda name: None)
        with pytest.raises(ConversionError, match="espeak-ng"):
            IMLToAudio(engine="espeak")

    @needs_espeak
    def test_auto_prefers_espeak(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert IMLToAudio().backend == "espeak"

    @pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan")])
    def test_invalid_max_duration(self, value: float) -> None:
        with pytest.raises(ValueError, match="max_duration_s"):
            IMLToAudio(engine="tones", max_duration_s=value)


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------


class TestVoiceParsing:
    @pytest.mark.parametrize(
        ("voice", "expected"),
        [
            (None, (None, None, None, None)),
            ("en_US-female-medium", ("en-US", "female", "medium", None)),
            ("fr-male", ("fr", "male", None, None)),
            ("female", (None, "female", None, None)),
            ("male-low", (None, "male", "low", None)),
            ("de", ("de", None, None, None)),
            ("en-us+f3", ("en-us", "female", None, "f3")),
            ("en-gb-x-rp+m2", ("en-gb-x-rp", "male", None, "m2")),
        ],
    )
    def test_accepted_forms(self, voice: str | None, expected: tuple[object, ...]) -> None:
        v = _parse_voice(voice)
        assert (v.language, v.gender, v.level, v.variant) == expected

    @pytest.mark.parametrize(
        "voice", ["robot", "en_GB-male-deep", "", "+", "en-us+f3-male", "female-en"]
    )
    def test_unusable_voices_rejected(self, voice: str) -> None:
        with pytest.raises(ConversionError, match="not recognised|variant"):
            IMLToAudio(voice=voice, engine="tones")

    def test_tones_voice_gender_sets_base_pitch(self, synth: IMLToAudio) -> None:
        iml = "<utterance>word</utterance>"
        female = _mean_f0(synth.synthesize(iml))
        male = _mean_f0(IMLToAudio(voice="male", engine="tones").synthesize(iml))
        assert female == pytest.approx(180, rel=0.03)
        assert male == pytest.approx(110, rel=0.03)

    def test_tones_warns_about_ignored_language(self) -> None:
        with pytest.warns(UserWarning, match="ignores the language"):
            IMLToAudio(voice="de-DE", engine="tones")

    @needs_espeak
    def test_espeak_rejects_unknown_language(self) -> None:
        with pytest.raises(ConversionError, match="no voice for the voice 'xx-YY'"):
            IMLToAudio(voice="xx-YY", engine="espeak")

    @needs_espeak
    def test_espeak_rejects_unknown_variant(self) -> None:
        with pytest.raises(ConversionError, match="variant"):
            IMLToAudio(voice="en-us+nosuchvariant", engine="espeak")

    @needs_espeak
    def test_espeak_voice_follows_document_language(self) -> None:
        binary = shutil.which("espeak-ng")
        assert binary is not None
        # Voice file paths differ between espeak-ng releases ("gmw/de", "de").
        voice_id, lang = _resolve_espeak_voice(binary, _parse_voice(None), "de-AT")
        assert lang == "de"
        assert re.fullmatch(r"(?:\w+/)?de\+f2", voice_id)
        voice_id, lang = _resolve_espeak_voice(binary, _parse_voice("en-US-male"), "de-AT")
        assert lang == "en-us"
        assert re.fullmatch(r"(?:\w+/)?en[-_]us\+m2", voice_id, re.IGNORECASE)

    def test_espeak_unknown_document_language(self, speech: IMLToAudio) -> None:
        with pytest.raises(ConversionError, match="document language 'tlh'"):
            speech.synthesize('<iml language="tlh"><utterance>Qapla</utterance></iml>')


# ---------------------------------------------------------------------------
# WAV output validity
# ---------------------------------------------------------------------------


class TestWAVValidity:
    @pytest.mark.parametrize("engine", ["tones", "espeak"])
    def test_valid_wav(self, engine: str) -> None:
        if engine == "espeak" and not HAS_ESPEAK:
            pytest.skip("espeak-ng is not installed")
        result = IMLToAudio(engine=engine).synthesize(  # type: ignore[arg-type]
            "<utterance>Hello world.</utterance>"
        )
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"
        with wave.open(io.BytesIO(result), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() > 0

    def test_non_zero_audio(self, synth: IMLToAudio) -> None:
        result = synth.synthesize("<utterance>Sound.</utterance>")
        assert _measure_rms(_wav_to_sound(result)) > 0.0


# ---------------------------------------------------------------------------
# Real speech (espeak-ng)
# ---------------------------------------------------------------------------


class TestSpeech:
    """espeak-ng output, measured: the IML values must be realized."""

    def test_different_text_gives_different_speech(self, speech: IMLToAudio) -> None:
        """The tone preview only depended on word count; speech depends on words."""
        a = speech.synthesize("<utterance>I am fine</utterance>")
        b = speech.synthesize("<utterance>xx yy zz</utterance>")
        assert a != b
        assert _f0_track(a).size > 20  # voiced speech, not silence

    def test_pitch_direction(self, speech: IMLToAudio) -> None:
        """Any espeak-ng version: each pitch value moves F0 the right way."""
        base = _mean_f0(speech.synthesize(f"<utterance>{SENTENCE}</utterance>"))
        for pitch, higher in (("+15%", True), ("-20%", False), ("+3st", True), ("250Hz", True)):
            f0 = _mean_f0(speech.synthesize(
                f'<utterance><prosody pitch="{pitch}">{SENTENCE}</prosody></utterance>'
            ))
            assert (f0 > base * 1.03) if higher else (f0 < base * 0.97), pitch

    @measured_espeak
    def test_pitch_percent_realized(self, speech: IMLToAudio) -> None:
        base = _mean_f0(speech.synthesize(f"<utterance>{SENTENCE}</utterance>"))
        up = _mean_f0(speech.synthesize(
            f'<utterance><prosody pitch="+15%">{SENTENCE}</prosody></utterance>'
        ))
        down = _mean_f0(speech.synthesize(
            f'<utterance><prosody pitch="-20%">{SENTENCE}</prosody></utterance>'
        ))
        assert up / base == pytest.approx(1.15, abs=0.05)
        assert down / base == pytest.approx(0.80, abs=0.05)

    @measured_espeak
    def test_semitones_and_absolute_hz_realized(self, speech: IMLToAudio) -> None:
        base = _mean_f0(speech.synthesize(f"<utterance>{SENTENCE}</utterance>"))
        st = _mean_f0(speech.synthesize(
            f'<utterance><prosody pitch="+3st">{SENTENCE}</prosody></utterance>'
        ))
        hz = _mean_f0(speech.synthesize(
            f'<utterance><prosody pitch="250Hz">{SENTENCE}</prosody></utterance>'
        ))
        assert st / base == pytest.approx(2 ** (3 / 12), abs=0.05)
        assert hz == pytest.approx(250, rel=0.1)

    @measured_espeak
    def test_absolute_hz_inside_emphasis(self, speech: IMLToAudio) -> None:
        """"250Hz" is absolute: an enclosing emphasis used to raise it by 15% more."""
        hz = _mean_f0(speech.synthesize(
            f'<utterance><emphasis level="strong"><prosody pitch="250Hz">{SENTENCE}'
            "</prosody></emphasis></utterance>"
        ))
        assert hz == pytest.approx(250, rel=0.07)

    def test_volume_realized(self, speech: IMLToAudio) -> None:
        base = _active_rms(speech.synthesize(f"<utterance>{SENTENCE}</utterance>"))
        loud = _active_rms(speech.synthesize(
            f'<utterance><prosody volume="+6dB">{SENTENCE}</prosody></utterance>'
        ))
        quiet = _active_rms(speech.synthesize(
            f'<utterance><prosody volume="-6dB">{SENTENCE}</prosody></utterance>'
        ))
        assert loud / base == pytest.approx(2.0, rel=0.2)
        assert quiet / base == pytest.approx(0.5, rel=0.2)

    def test_rate_realized(self, speech: IMLToAudio) -> None:
        base = _wav_duration(speech.synthesize(f"<utterance>{SENTENCE}</utterance>"))
        durations = {
            rate: _wav_duration(speech.synthesize(
                f'<utterance><prosody rate="{rate}">{SENTENCE}</prosody></utterance>'
            ))
            for rate in ("slow", "fast", "150%")
        }
        assert durations["slow"] > base * 1.1
        assert durations["fast"] < base * 0.9
        assert durations["150%"] < durations["fast"]

    def test_pause_realized(self, speech: IMLToAudio) -> None:
        with_pause = speech.synthesize('<utterance>Wait<pause duration="800"/>then go.</utterance>')
        without = speech.synthesize("<utterance>Wait then go.</utterance>")
        extra = _wav_duration(with_pause) - _wav_duration(without)
        assert extra == pytest.approx(0.8, abs=0.1)

    def test_emphasis_levels_ordered(self, speech: IMLToAudio) -> None:
        def level(lvl: str | None) -> float:
            word = f'<emphasis level="{lvl}">told</emphasis>' if lvl else "told"
            return _active_rms(speech.synthesize(f"<utterance>I {word} you</utterance>"))

        plain = level(None)
        assert level("strong") > level("moderate") > plain > level("reduced")

    def test_contour_moves_final_pitch(self, speech: IMLToAudio) -> None:
        def final_f0(contour: str | None) -> float:
            text = "Are you coming tonight?"
            if contour:
                text = f'<prosody pitch_contour="{contour}">{text}</prosody>'
            track = _f0_track(speech.synthesize(f"<utterance>{text}</utterance>"))
            return float(np.mean(track[-len(track) // 4 :]))

        plain = final_f0(None)
        assert final_f0("rise") > plain * 1.08
        assert final_f0("fall") < plain * 0.92

    def test_sarcasm_fixture_differs_from_sincere(self, speech: IMLToAudio) -> None:
        sarcastic = (FIXTURES / "sarcasm.xml").read_text()
        sincere = sarcastic.replace("sarcastic", "sincere").replace(
            ' pitch_contour="fall-sharp"', ""
        )
        a, b = speech.synthesize(sarcastic), speech.synthesize(sincere)
        assert a != b
        assert _f0_track(a)[-5:].mean() < _f0_track(b)[-5:].mean()

    def test_voice_gender(self) -> None:
        if not HAS_ESPEAK:
            pytest.skip("espeak-ng is not installed")
        iml = f"<utterance>{SENTENCE}</utterance>"
        female = _mean_f0(IMLToAudio(voice="en-US-female", engine="espeak").synthesize(iml))
        male = _mean_f0(IMLToAudio(voice="en-US-male", engine="espeak").synthesize(iml))
        assert female > 150 > male

    def test_research_fixture_does_not_clip(self, speech: IMLToAudio) -> None:
        wav = speech.synthesize((FIXTURES / "research_grade.xml").read_text())
        assert _clipped(wav) == 0

    def test_loud_span_does_not_clip(self, speech: IMLToAudio) -> None:
        wav = speech.synthesize(
            f'<utterance>Now <prosody volume="+20dB">{SENTENCE}</prosody></utterance>'
        )
        assert _clipped(wav) == 0

    @pytest.mark.parametrize(
        "body",
        [
            '<prosody volume="+20dB">{s}</prosody>',
            '<prosody volume="+6dB"><prosody volume="+40dB">{s}</prosody></prosody>',
            '<prosody volume="+12dB"><emphasis level="strong">Oye</emphasis> {s}</prosody>',
        ],
    )
    def test_loud_spans_keep_headroom_for_loud_voices(
        self, speech: IMLToAudio, body: str
    ) -> None:
        """Spanish is the loudest voice measured; it clipped with the old headroom."""
        text = "Ahora mismo voy a la playa con mis padres."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # +46 dB is clamped to +30 dB, with a warning
            wav = speech.synthesize(
                f'<iml language="es"><utterance>{body.format(s=text)}</utterance></iml>'
            )
        samples, _ = _samples(wav)
        assert np.max(np.abs(samples)) < 0.9

    def test_comments_not_spoken(self, speech: IMLToAudio) -> None:
        plain = speech.synthesize("<utterance>Hello world</utterance>")
        commented = speech.synthesize(
            "<utterance>Hello <!-- a long internal note nobody should hear --> world</utterance>"
        )
        assert _wav_duration(commented) == pytest.approx(_wav_duration(plain), abs=0.05)

    @needs_espeak
    def test_precheck_accepts_documents_that_fit(self) -> None:
        """The pre-check assumed at least 0.08 s per word; espeak-ng is faster."""
        synth = IMLToAudio(engine="espeak", max_duration_s=20)
        wav = synth.synthesize(
            '<utterance><prosody rate="fast">' + "a " * 300 + "</prosody></utterance>"
        )
        assert _wav_duration(wav) < 20

    def test_output_cap_enforced_while_streaming(self) -> None:
        """Text length is only enforced by the cap on espeak-ng's output."""
        if not HAS_ESPEAK:
            pytest.skip("espeak-ng is not installed")
        synth = IMLToAudio(engine="espeak", max_duration_s=0.6)
        with pytest.raises(ConversionError, match="exceeds max_duration_s"):
            synth.synthesize(f"<utterance>{SENTENCE}</utterance>")


# ---------------------------------------------------------------------------
# Tone preview: pitch (Acceptance criterion #1)
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
# Tone preview: pauses (Acceptance criterion #2)
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

        # The version with a pause replaces the 50 ms word gap with 800 ms.
        diff_ms = (dur_with - dur_without) * 1000
        assert diff_ms == pytest.approx(750, abs=5)

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
# Tone preview: emphasis (Acceptance criterion #3)
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

    def test_reduced_emphasis_below_baseline(self, synth: IMLToAudio) -> None:
        """Spec 3.4: reduced = de-emphasized, so quieter and lower than plain text."""
        plain_wav = synth.synthesize("<utterance>word</utterance>")
        reduced_wav = synth.synthesize(
            '<utterance><emphasis level="reduced">word</emphasis></utterance>'
        )
        assert _measure_rms(_wav_to_sound(reduced_wav)) < _measure_rms(
            _wav_to_sound(plain_wav)
        ) * 0.9
        assert _mean_f0(reduced_wav) < _mean_f0(plain_wav)

    def test_levels_ordered(self, synth: IMLToAudio) -> None:
        def rms(level: str) -> float:
            wav = synth.synthesize(
                f'<utterance><emphasis level="{level}">word</emphasis></utterance>'
            )
            return _measure_rms(_wav_to_sound(wav))

        assert rms("strong") > rms("moderate") > rms("reduced")


# ---------------------------------------------------------------------------
# Tone preview: volume, headroom
# ---------------------------------------------------------------------------


class TestVolumeModification:
    def test_louder_volume(self, synth: IMLToAudio) -> None:
        baseline_wav = synth.synthesize("<utterance>word</utterance>")
        loud_wav = synth.synthesize(
            '<utterance><prosody volume="+6dB">word</prosody></utterance>'
        )

        baseline_rms = _measure_rms(_wav_to_sound(baseline_wav))
        loud_rms = _measure_rms(_wav_to_sound(loud_wav))

        assert loud_rms / baseline_rms == pytest.approx(2.0, rel=0.05)

    def test_quieter_volume(self, synth: IMLToAudio) -> None:
        baseline_wav = synth.synthesize("<utterance>word</utterance>")
        quiet_wav = synth.synthesize(
            '<utterance><prosody volume="-6dB">word</prosody></utterance>'
        )

        baseline_rms = _measure_rms(_wav_to_sound(baseline_wav))
        quiet_rms = _measure_rms(_wav_to_sound(quiet_wav))

        assert quiet_rms < baseline_rms * 0.8

    def test_research_fixture_does_not_clip(self, synth: IMLToAudio) -> None:
        """volume="+8dB" (a valid spec example) used to clip."""
        wav = synth.synthesize((FIXTURES / "research_grade.xml").read_text())
        assert _clipped(wav) == 0

    def test_extreme_volume_scaled_not_clipped(self, synth: IMLToAudio) -> None:
        with pytest.warns(UserWarning, match="scaled down"):
            wav = synth.synthesize(
                '<utterance>quiet <prosody volume="+30dB">loud</prosody></utterance>'
            )
        samples, sr = _samples(wav)
        assert _clipped(wav) == 0
        quiet = samples[: int(0.25 * sr)]
        loud = samples[-int(0.25 * sr) :]
        # Relative level (+30 dB) is kept.
        ratio = np.sqrt(np.mean(loud**2)) / np.sqrt(np.mean(quiet**2))
        assert 20 * np.log10(ratio) == pytest.approx(30, abs=0.5)


# ---------------------------------------------------------------------------
# Tone preview: rate, tempo, timing
# ---------------------------------------------------------------------------


class TestTiming:
    @pytest.mark.parametrize(
        ("rate", "factor"), [("slow", 0.8), ("fast", 1.25), ("50%", 0.5), ("200%", 2.0)]
    )
    def test_rate_scales_duration(self, synth: IMLToAudio, rate: str, factor: float) -> None:
        plain = _wav_duration(synth.synthesize("<utterance>listen carefully</utterance>"))
        scaled = _wav_duration(synth.synthesize(
            f'<utterance><prosody rate="{rate}">listen carefully</prosody></utterance>'
        ))
        assert scaled == pytest.approx(plain / factor, abs=0.002)

    @pytest.mark.parametrize(("tempo", "longer"), [("rushed", False), ("drawn-out", True)])
    def test_segment_tempo(self, synth: IMLToAudio, tempo: str, longer: bool) -> None:
        plain = _wav_duration(synth.synthesize("<utterance>a b c d</utterance>"))
        seg = _wav_duration(synth.synthesize(
            f'<utterance><segment tempo="{tempo}">a b c d</segment></utterance>'
        ))
        assert (seg > plain * 1.1) if longer else (seg < plain * 0.9)

    def test_punctuation_and_emoji_are_silent(self, synth: IMLToAudio) -> None:
        one_word = _wav_duration(synth.synthesize("<utterance>yesterday</utterance>"))
        tail = _wav_duration(synth.synthesize(
            '<utterance><prosody pitch="+5%">yesterday</prosody>!</utterance>'
        ))
        assert tail == pytest.approx(one_word)
        two_words = _wav_duration(synth.synthesize("<utterance>Hi there</utterance>"))
        symbols = _wav_duration(synth.synthesize(
            "<utterance>Hi \u2014 there \U0001F600 !</utterance>"
        ))
        assert symbols == pytest.approx(two_words)

    def test_markup_does_not_change_word_gaps(self, synth: IMLToAudio) -> None:
        plain = synth.synthesize("<utterance>I told you</utterance>")
        marked = synth.synthesize(
            '<utterance>I <prosody volume="-1dB">told</prosody> you</utterance>'
        )
        assert _wav_duration(marked) == pytest.approx(_wav_duration(plain))
        # The 50 ms gaps are real silence.
        samples, sr = _samples(marked)
        gap = samples[int(0.26 * sr) : int(0.29 * sr)]
        assert np.max(np.abs(gap)) == 0.0


# ---------------------------------------------------------------------------
# Tone preview: pitch contours
# ---------------------------------------------------------------------------


class TestContour:
    def _ends(self, wav: bytes) -> tuple[float, float]:
        track = _f0_track(wav)
        return float(np.mean(track[:5])), float(np.mean(track[-5:]))

    def test_rise_and_fall(self, synth: IMLToAudio) -> None:
        start, end = self._ends(synth.synthesize(
            '<utterance><prosody pitch_contour="rise">word</prosody></utterance>'
        ))
        assert end > start * 1.1
        start, end = self._ends(synth.synthesize(
            '<utterance><prosody pitch_contour="fall">word</prosody></utterance>'
        ))
        assert end < start * 0.9

    def test_contour_spans_words(self, synth: IMLToAudio) -> None:
        wav = synth.synthesize(
            '<utterance><prosody pitch_contour="rise">one two three</prosody></utterance>'
        )
        samples, sr = _samples(wav)
        word_f0 = [
            _mean_f0(iml_to_audio._to_wav_bytes(samples[int(s * sr) : int((s + 0.25) * sr)]))
            for s in (0.0, 0.3, 0.6)
        ]
        assert word_f0 == sorted(word_f0)

    def test_sarcasm_fixture_differs_from_sincere(self, synth: IMLToAudio) -> None:
        sarcastic = (FIXTURES / "sarcasm.xml").read_text()
        sincere = sarcastic.replace("sarcastic", "sincere").replace(
            ' pitch_contour="fall-sharp"', ""
        )
        assert synth.synthesize(sarcastic) != synth.synthesize(sincere)


# ---------------------------------------------------------------------------
# Duration caps
# ---------------------------------------------------------------------------


class TestDurationCap:
    @pytest.mark.parametrize("engine", ["tones", "espeak"])
    def test_huge_pause_rejected_before_allocating(self, engine: str) -> None:
        if engine == "espeak" and not HAS_ESPEAK:
            pytest.skip("espeak-ng is not installed")
        synth = IMLToAudio(engine=engine)  # type: ignore[arg-type]
        tracemalloc.start()
        try:
            with pytest.raises(ConversionError, match="max_duration_s=300"):
                synth.synthesize('<utterance>a<pause duration="99999999"/>b</utterance>')
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 5_000_000

    @pytest.mark.parametrize("engine", ["tones", "espeak"])
    def test_long_text_rejected(self, engine: str) -> None:
        if engine == "espeak" and not HAS_ESPEAK:
            pytest.skip("espeak-ng is not installed")
        synth = IMLToAudio(engine=engine)  # type: ignore[arg-type]
        with pytest.raises(ConversionError, match="max_duration_s"):
            synth.synthesize("<utterance>" + "word " * 20_000 + "</utterance>")

    def test_custom_cap(self) -> None:
        synth = IMLToAudio(engine="tones", max_duration_s=1.0)
        synth.synthesize('<utterance>a<pause duration="500"/>b</utterance>')
        with pytest.raises(ConversionError, match="max_duration_s=1"):
            synth.synthesize('<utterance>a<pause duration="900"/>b</utterance>')


# ---------------------------------------------------------------------------
# Values the validator accepts but no engine can render
# ---------------------------------------------------------------------------


def _warning_messages(action: Callable[[], object]) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        action()
    return [str(w.message) for w in caught]


class TestExtremeValues:
    @pytest.mark.parametrize(
        "prosody", ['volume="+7000dB"', 'pitch="+20000st"', 'pitch="-20000st"']
    )
    @pytest.mark.parametrize("engine", ["tones", "espeak"])
    def test_rendered_not_raised(self, engine: str, prosody: str) -> None:
        """These raised OverflowError / ValueError (HTTP 500 from the API)."""
        if engine == "espeak" and not HAS_ESPEAK:
            pytest.skip("espeak-ng is not installed")
        synth = IMLToAudio(engine=engine)  # type: ignore[arg-type]
        iml = f"<utterance>one <prosody {prosody}>two</prosody> three</utterance>"
        wavs: list[bytes] = []
        messages = _warning_messages(lambda: wavs.append(synth.synthesize(iml)))
        assert any("clamped" in m for m in messages)
        assert _wav_duration(wavs[0]) > 0.5
        assert _clipped(wavs[0]) == 0

    def test_huge_volume_does_not_silence_the_document(self, synth: IMLToAudio) -> None:
        """+800 dB overflowed float32 and turned every sample into NaN (silence)."""
        iml = '<utterance>quiet <prosody volume="+800dB">loud</prosody></utterance>'
        wavs: list[bytes] = []
        messages = _warning_messages(lambda: wavs.append(synth.synthesize(iml)))
        assert any("clamped to +30 dB" in m for m in messages)
        samples, sr = _samples(wavs[0])
        quiet = samples[: int(0.25 * sr)]
        loud = samples[-int(0.25 * sr) :]
        assert np.max(np.abs(quiet)) > 0.01
        ratio = np.sqrt(np.mean(loud**2)) / np.sqrt(np.mean(quiet**2))
        assert 20 * np.log10(ratio) == pytest.approx(30, abs=0.5)

    def test_extreme_pitch_clamped_to_preview_range(self, synth: IMLToAudio) -> None:
        with pytest.warns(UserWarning, match="outside the tone preview's range"):
            wav = synth.synthesize('<utterance><prosody pitch="+20000st">hi</prosody></utterance>')
        samples, sr = _samples(wav)
        peak_hz = np.argmax(np.abs(np.fft.rfft(samples))) * sr / samples.size
        assert peak_hz == pytest.approx(2000, rel=0.01)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_pause_is_a_validation_error(self, synth: IMLToAudio) -> None:
        """Used to escape as a raw ValueError (HTTP 500 from the API)."""
        with pytest.raises(IMLValidationError) as info:
            synth.synthesize('<utterance>a<pause duration="-500"/>b</utterance>')
        assert "V6" in {issue.rule for issue in info.value.issues}

    def test_negative_pause_in_document_object(self, synth: IMLToAudio) -> None:
        doc = IMLDocument(utterances=(Utterance(children=("a", Pause(duration=-500), "b")),))
        with pytest.raises(IMLValidationError):
            synth.synthesize_doc(doc)
        with pytest.raises(ConversionError, match="duration=-500"):
            IMLToAudio(engine="tones", strict=False).synthesize_doc(doc)

    def test_emotion_without_confidence_rejected(self, synth: IMLToAudio) -> None:
        iml = '<utterance emotion="calm">Please listen.</utterance>'
        with pytest.raises(IMLValidationError, match="V3"):
            synth.synthesize(iml)
        assert IMLToAudio(engine="tones", strict=False).synthesize(iml)[:4] == b"RIFF"

    def test_comments_not_rendered(self, synth: IMLToAudio) -> None:
        plain = synth.synthesize("<utterance>Hello world</utterance>")
        commented = synth.synthesize(
            "<utterance>Hello <!-- secret note --> world</utterance>"
        )
        assert commented == plain


# ---------------------------------------------------------------------------
# synthesize_to_file / synthesize_doc / multi-utterance
# ---------------------------------------------------------------------------


class TestSynthesizeToFile:
    def test_creates_file(self, synth: IMLToAudio, tmp_path: Path) -> None:
        out = tmp_path / "output.wav"
        synth.synthesize_to_file("<utterance>Hello.</utterance>", out)
        assert out.exists()
        assert out.stat().st_size > 44  # More than just a WAV header

    def test_file_is_valid_wav(self, synth: IMLToAudio, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "output.wav"
        synth.synthesize_to_file("<utterance>Test.</utterance>", out)
        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getnframes() > 0


class TestSynthesizeDoc:
    def test_from_document(self, synth: IMLToAudio) -> None:
        doc = IMLDocument(
            utterances=(Utterance(children=("Hello world.",)),),
        )
        wav = synth.synthesize_doc(doc)
        assert wav == synth.synthesize("<utterance>Hello world.</utterance>")


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
        assert _wav_duration(multi) == pytest.approx(2 * _wav_duration(single) + 0.3)


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
