"""Tests for prosody_protocol.audio_to_iml.

Integration tests that:
- Verify the full pipeline from audio -> IML document for each transcript
  source: caller word timings, a caller transcript, Whisper (faked here, so
  no model download is needed) and placeholders when there is no transcript
- Check the output against espeak-ng speech with known words and pauses
- Check that silence and noise never get an emotion
- Round-trip: AudioToIML output parses back into a valid IMLDocument
- IMLValidator accepts the output
- Extended attributes appear only when include_extended=True
- Every failure to read audio raises AudioProcessingError
- A prosody profile sets matching utterances' emotion and is reported
- Speaker baselines: calibration from one or several recordings (cached per
  file), per-speaker baselines from speaker labels, no baseline for two
  voices without labels, and a warning whenever there is no baseline
- Warnings on what could not be measured: transcripts without timings,
  unpunctuated words, word timings that do not match the audio
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
import types
import wave
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

np = pytest.importorskip("numpy")
parselmouth = pytest.importorskip("parselmouth")

from prosody_protocol import audio_to_iml
from prosody_protocol.alignment import load_word_timings
from prosody_protocol.assembler import DEFAULT_MIN_EMOTION_CONFIDENCE, _Assembly
from prosody_protocol.audio_to_iml import PLACEHOLDER_TOKEN, AudioToIML, ConversionResult
from prosody_protocol.exceptions import (
    AudioProcessingError,
    ConversionError,
    ProfileError,
    SpeechRecognitionError,
)
from prosody_protocol.models import Emphasis, IMLDocument, Pause, Prosody, Segment, Utterance
from prosody_protocol.parser import IMLParser
from prosody_protocol.profiles import ProfileLoader, ProsodyMapping, ProsodyProfile
from prosody_protocol.prosody_analyzer import WordAlignment
from prosody_protocol.validator import IMLValidator

AUDIO_DIR = Path(__file__).parent / "fixtures" / "audio"
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
CALIBRATION = AUDIO_DIR / "speech_calibration.wav"
SPEECH = AUDIO_DIR / "speech_pauses.wav"
SR = 16_000


@pytest.fixture()
def converter() -> AudioToIML:
    return AudioToIML(include_extended=False, language="en-US", stt="none")


@pytest.fixture()
def parser() -> IMLParser:
    return IMLParser()


@pytest.fixture()
def validator() -> IMLValidator:
    return IMLValidator()


@pytest.fixture()
def no_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import whisper`` fail, whatever is installed."""
    monkeypatch.setitem(sys.modules, "whisper", None)


def _truth(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((AUDIO_DIR / f"{name}.json").read_text())
    return data


def _words(name: str, prefix: str = "") -> list[WordAlignment]:
    return [
        WordAlignment(prefix + w["word"], w["start_ms"], w["end_ms"])
        for w in _truth(name)["words"]
    ]


def _nodes(doc: IMLDocument) -> list[Any]:
    """Every element node in the document, depth first."""
    found: list[Any] = []

    def walk(children: tuple[Any, ...]) -> None:
        for child in children:
            if isinstance(child, str):
                continue
            found.append(child)
            if isinstance(child, (Prosody, Emphasis, Segment)):
                walk(child.children)

    for utterance in doc.utterances:
        walk(utterance.children)
    return found


def _pause_durations(doc: IMLDocument) -> list[int]:
    return [n.duration for n in _nodes(doc) if isinstance(n, Pause)]


def _pitch_percent(value: str) -> float:
    """A relative pitch attribute as a percentage (``+30%`` or ``+4st``)."""
    if value.endswith("st"):
        return float(2 ** (float(value[:-2]) / 12) - 1) * 100
    assert value.endswith("%"), value
    return float(value[:-1])


def _text_of(node: Any) -> str:
    return "".join(
        c if isinstance(c, str) else _text_of(c)
        for c in getattr(node, "children", ())
    )


needs_espeak = pytest.mark.skipif(
    shutil.which("espeak-ng") is None, reason="espeak-ng not installed"
)


def _espeak(word: str, pitch: int) -> Any:
    """*word* spoken by espeak-ng at 16 kHz, without leading or trailing silence."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "word.wav"
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", "160", "-p", str(pitch), "-z", "-w", str(path),
             word],
            check=True,
        )
        samples = parselmouth.Sound(str(path)).resample(SR).values[0]
    audible = np.flatnonzero(np.abs(samples) > 1e-3)
    return samples[audible[0]: audible[-1] + 1]


def _speak(
    path: Path, turns: list[tuple[str, str | None, int]], gap_ms: int = 900
) -> list[WordAlignment]:
    """Record *turns* of (sentence, speaker, espeak pitch) word by word, with
    50 ms between words and *gap_ms* between turns; returns the exact words."""
    parts: list[Any] = [np.zeros(SR // 5)]
    words: list[WordAlignment] = []
    position = len(parts[0])
    for turn, (sentence, speaker, pitch) in enumerate(turns):
        if turn:
            parts.append(np.zeros(SR * gap_ms // 1000))
            position += len(parts[-1])
        for index, token in enumerate(sentence.split()):
            if index:
                parts.append(np.zeros(SR // 20))
                position += len(parts[-1])
            clip = _espeak(token.strip(".,?!"), pitch)
            words.append(WordAlignment(
                token, round(position * 1000 / SR), round((position + len(clip)) * 1000 / SR),
                speaker,
            ))
            parts.append(clip)
            position += len(clip)
    parts.append(np.zeros(SR // 5))
    signal = np.concatenate(parts)
    signal = 0.5 * signal / np.abs(signal).max()
    signal += 1e-3 * np.random.default_rng(0).standard_normal(len(signal))
    _write(path, signal)
    return words


# A support call: a low voice (4 turns) and a high voice (2 calm turns).
LOW, HIGH = 20, 85
CALL = [
    ("I called about my order last week.", "A", LOW),
    ("Can you tell me when it ships?", "B", HIGH),
    ("It should arrive on Monday morning.", "A", LOW),
    ("Okay that sounds good to me.", "B", HIGH),
    ("Let me check the tracking number.", "A", LOW),
    ("Yes it left the warehouse today.", "A", LOW),
]


@pytest.fixture(scope="module")
def call(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[WordAlignment]]:
    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng not installed")
    path = tmp_path_factory.mktemp("call") / "call.wav"
    return path, _speak(path, CALL)


def _other_warnings(result: ConversionResult) -> list[str]:
    """The warnings besides the note that there is no speaker baseline."""
    return [w for w in result.warnings if not w.startswith("No speaker baseline")]


def _write(path: Path, signal: Any) -> Path:
    pcm = np.round(np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return path


# ---------------------------------------------------------------------------
# Constructor / interface
# ---------------------------------------------------------------------------


class TestAudioToIMLInterface:
    def test_instantiate(self) -> None:
        converter = AudioToIML()
        assert converter.stt_model == "base"
        assert converter.include_extended is False
        assert converter.stt == "auto"
        assert converter.min_emotion_confidence == DEFAULT_MIN_EMOTION_CONFIDENCE
        assert converter.calibration_audio is None
        assert converter.max_duration_s is None

    def test_custom_params(self) -> None:
        converter = AudioToIML(
            stt_model="tiny", include_extended=True, language="de-DE",
            min_emotion_confidence=0.7, calibration_audio="me.wav", stt="whisper",
        )
        assert converter.stt_model == "tiny"
        assert converter.include_extended is True
        assert converter.language == "de-DE"
        assert converter.min_emotion_confidence == 0.7
        assert converter.calibration_audio == Path("me.wav")
        assert converter.stt == "whisper"

    def test_unknown_stt_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="stt"):
            AudioToIML(stt="deepgram")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["english (US)", "en_", "", 42])
    def test_invalid_language_rejected(self, bad: Any) -> None:
        """A tag the output could not carry (V29) is refused up front."""
        with pytest.raises(ValueError, match="BCP 47"):
            AudioToIML(language=bad)

    def test_posix_language_is_normalised(self) -> None:
        """"en_US" (as in $LANG) names en-US; it used to raise ValueError."""
        converter = AudioToIML(language="en_US", stt="none")
        assert converter.language == "en-US"
        doc = converter.convert_to_doc(SPEECH, words=_words("speech_pauses"))
        assert doc.language == "en-US"

    @pytest.mark.parametrize("good", ["en", "en-US", "zh-Hant-TW", None])
    def test_valid_languages(self, good: str | None) -> None:
        assert AudioToIML(language=good).language == good

    @pytest.mark.parametrize("bad", [0, -5.0, float("nan")])
    def test_invalid_max_duration_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="max_duration_s"):
            AudioToIML(max_duration_s=bad)

    def test_result_is_frozen(self, converter: AudioToIML) -> None:
        result = converter.convert_detailed(AUDIO_DIR / "tone_220hz.wav")
        assert isinstance(result, ConversionResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.iml = ""  # type: ignore[misc]

    def test_convert_methods_agree(self, converter: AudioToIML, parser: IMLParser) -> None:
        words = _words("speech_pauses")
        result = converter.convert_detailed(SPEECH, words=words)
        assert converter.convert(SPEECH, words=words) == result.iml
        assert converter.convert_to_doc(SPEECH, words=words) == result.document
        assert parser.to_iml_string(result.document) == result.iml


# ---------------------------------------------------------------------------
# Caller-supplied word timings
# ---------------------------------------------------------------------------


class TestWords:
    def test_words_become_the_transcript(
        self, converter: AudioToIML, parser: IMLParser, validator: IMLValidator
    ) -> None:
        result = converter.convert_detailed(SPEECH, words=_words("speech_pauses"))
        assert result.transcript_source == "words"
        # One utterance without calibration: only the missing baseline is noted.
        assert _other_warnings(result) == []
        assert parser.to_plain_text(result.document) == "I told you to call me yesterday."
        assert result.document.language == "en-US"
        assert validator.validate(result.iml).valid

    def test_detected_pauses_are_marked(self, converter: AudioToIML) -> None:
        """The 600 ms and 300 ms pauses of the recording appear as <pause>."""
        doc = converter.convert_to_doc(SPEECH, words=_words("speech_pauses"))
        durations = _pause_durations(doc)
        for expected in (600, 300):
            assert any(abs(d - expected) <= 40 for d in durations), durations

    def test_emphasised_word_is_marked(self, converter: AudioToIML) -> None:
        """'told' is spoken ~40% higher and louder than the other words."""
        doc = converter.convert_to_doc(SPEECH, words=_words("speech_pauses"))
        marked = [
            n for n in _nodes(doc)
            if isinstance(n, (Prosody, Emphasis)) and "told" in _text_of(n)
        ]
        assert marked, "the emphasised word carries no prosody or emphasis"
        pitches = [
            _pitch_percent(n.pitch) for n in _nodes(doc)
            if isinstance(n, Prosody) and n.pitch and "told" in _text_of(n)
        ]
        assert pitches and pitches[0] > 20

    def test_whisper_style_leading_spaces(self, converter: AudioToIML, parser: IMLParser) -> None:
        """STT tokens often carry a leading space; the text stays clean."""
        doc = converter.convert_to_doc(SPEECH, words=_words("speech_pauses", prefix=" "))
        assert parser.to_plain_text(doc) == "I told you to call me yesterday."

    def test_contiguous_timings_keep_words_apart(
        self, converter: AudioToIML, parser: IMLParser
    ) -> None:
        """STT engines often make each word end where the next begins."""
        words = _words("speech_pauses")
        contiguous = [
            WordAlignment(w.word, w.start_ms, nxt.start_ms if nxt.start_ms - w.end_ms < 200
                          else w.end_ms)
            for w, nxt in zip(words, [*words[1:], words[-1]], strict=True)
        ]
        assert contiguous[0].end_ms == contiguous[1].start_ms
        doc = converter.convert_to_doc(SPEECH, words=contiguous)
        assert parser.to_plain_text(doc) == "I told you to call me yesterday."

    def test_words_are_put_in_time_order(self, converter: AudioToIML, parser: IMLParser) -> None:
        words = _words("speech_pauses")
        doc = converter.convert_to_doc(SPEECH, words=list(reversed(words)))
        assert parser.to_plain_text(doc) == "I told you to call me yesterday."

    def test_empty_word_list(self, converter: AudioToIML, parser: IMLParser) -> None:
        result = converter.convert_detailed(SPEECH, words=[])
        assert result.transcript_source == "words"
        assert parser.to_plain_text(result.document) == ""
        assert all(u.emotion is None for u in result.document.utterances)
        assert any("No words" in w for w in result.warnings)

    def test_words_after_the_audio_are_flagged(self, converter: AudioToIML) -> None:
        words = [*_words("speech_pauses"), WordAlignment("again", 9000, 9400)]
        result = converter.convert_detailed(SPEECH, words=words)
        assert any("after the end of the audio" in w for w in result.warnings)

    def test_words_ending_after_the_audio_are_flagged_and_clamped(
        self, validator: IMLValidator
    ) -> None:
        """A word ending far past the audio used to pass silently, and with
        include_extended its whole span became duration_ms="2999999386",
        invalid IML (V27)."""
        converter = AudioToIML(stt="none", include_extended=True)
        words = [WordAlignment("I", 250, 564), WordAlignment("never", 614, 3_000_000_000)]
        result = converter.convert_detailed(SPEECH, words=words)
        assert any("end after the end of the audio" in w for w in result.warnings)
        assert validator.validate(result.iml).valid
        durations = [n.duration_ms for n in _nodes(result.document)
                     if isinstance(n, Prosody) and n.duration_ms is not None]
        assert durations and max(durations) <= 4435

    def test_words_of_another_recording_are_flagged(self, converter: AudioToIML) -> None:
        """Timings of another recording used to give confident markup with a
        note only about the words past the end."""
        words = load_word_timings(EXAMPLES_DIR / "monotone.deepgram.json")
        result = converter.convert_detailed(EXAMPLES_DIR / "speech.wav", words=words)
        assert any("may belong to another recording" in w for w in result.warnings)

    def test_part_of_the_speech_outside_the_words_is_flagged(
        self, converter: AudioToIML
    ) -> None:
        words = _words("speech_calibration")[:2]  # "this is" of "this is how I normally sound."
        result = converter.convert_detailed(CALIBRATION, words=words)
        assert any("lie outside the words" in w for w in result.warnings)

    @pytest.mark.parametrize("name", ["speech_pauses", "speech_calibration", "speech_raised"])
    def test_matching_timings_are_not_flagged(self, converter: AudioToIML, name: str) -> None:
        result = converter.convert_detailed(AUDIO_DIR / f"{name}.wav", words=_words(name))
        assert _other_warnings(result) == []

    def test_unpunctuated_words_are_split_at_long_pauses(self, converter: AudioToIML) -> None:
        """Without sentence punctuation, only pauses of 1 s or more used to end
        an utterance, so sentences with ordinary gaps ran together, silently."""
        words = [dataclasses.replace(w, word=w.word.rstrip(".").lower())
                 for w in _words("speech_pauses")]
        result = converter.convert_detailed(SPEECH, words=words)
        assert [IMLParser().to_plain_text(dataclasses.replace(result.document, utterances=(u,)))
                for u in result.document.utterances] == ["i told you", "to call me yesterday"]
        assert any("no sentence punctuation" in w for w in result.warnings)

    def test_speakers_may_talk_over_each_other(self, converter: AudioToIML) -> None:
        """Words of different speakers may overlap by more than 500 ms."""
        words = [
            WordAlignment("I", 250, 564, "A"), WordAlignment("told", 614, 1086, "A"),
            WordAlignment("you", 1136, 1486, "A"), WordAlignment("to", 700, 2387, "B"),
        ]
        result = converter.convert_detailed(SPEECH, words=words)
        assert [u.speaker_id for u in result.document.utterances] == ["A", "B", "A"]
        crowd = [WordAlignment("hi", 0, 3000, speaker) for speaker in "ABCD"]
        with pytest.raises(ValueError, match="other speakers"):
            converter.convert(SPEECH, words=crowd)

    @pytest.mark.parametrize(
        ("speaker", "error"), [(3, TypeError), ("A\x1b", ValueError)]
    )
    def test_speaker_labels_are_checked(
        self, converter: AudioToIML, speaker: Any, error: type[Exception]
    ) -> None:
        with pytest.raises(error, match="speaker"):
            converter.convert(SPEECH, words=[WordAlignment("I", 250, 564, speaker)])

    def test_invalid_output_is_never_returned(
        self, converter: AudioToIML, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broken = IMLDocument(utterances=(Utterance(children=("hi",), emotion="angry"),))
        monkeypatch.setattr(
            converter._assembler, "_assemble", lambda *a, **k: _Assembly(broken, [], [], [])
        )
        with pytest.raises(ConversionError, match="V3"):
            converter.convert(SPEECH, words=_words("speech_pauses"))

    def test_words_may_be_a_generator(self, converter: AudioToIML, parser: IMLParser) -> None:
        """A generator used to be exhausted by validation, leaving no words."""
        words = _words("speech_pauses")
        result = converter.convert_detailed(SPEECH, words=(w for w in words))
        assert parser.to_plain_text(result.document) == "I told you to call me yesterday."
        assert _other_warnings(result) == []

    @pytest.mark.parametrize("bad", [
        WordAlignment("back", 500, 400), WordAlignment("neg", -10, 100),
        WordAlignment("nan", float("nan"), 400), WordAlignment("inf", 100, float("inf")),
    ])
    def test_invalid_timings_rejected(self, converter: AudioToIML, bad: WordAlignment) -> None:
        with pytest.raises(ValueError, match="invalid timings"):
            converter.convert(SPEECH, words=[bad])

    @pytest.mark.parametrize("bad", [
        WordAlignment(None, 100, 400),  # type: ignore[arg-type]
        WordAlignment(b"told", 100, 400),  # type: ignore[arg-type]
        WordAlignment("told", "100", 400),  # type: ignore[arg-type]
    ])
    def test_wrongly_typed_words_rejected(
        self, converter: AudioToIML, bad: WordAlignment
    ) -> None:
        with pytest.raises(TypeError, match=r"words\[0\]"):
            converter.convert(SPEECH, words=[bad])

    @pytest.mark.parametrize("text", ["a\x1bb", "nul\x00", "\ufffe"])
    def test_text_xml_cannot_hold_is_rejected(
        self, converter: AudioToIML, parser: IMLParser, text: str
    ) -> None:
        """Control characters used to produce IML that does not parse."""
        with pytest.raises(ValueError, match="XML does not allow"):
            converter.convert(SPEECH, words=[WordAlignment(text, 250, 564)])

    def test_overlapping_words_rejected_before_analysis(self, converter: AudioToIML) -> None:
        """Words that each span the recording would have it analysed once per
        word: a few kilobytes of timings used to cost minutes and gigabytes."""
        words = [WordAlignment("x", i, 596_000) for i in range(200)]
        with pytest.raises(ValueError, match=(
            r"words\[1\] \('x', 1-596000 ms\) starts 595999 ms before words\[0\] "
            r"\('x', 0-596000 ms\) ends; word timings may overlap by at most 500 ms"
        )):
            # Rejected before the audio is read: this file does not exist.
            converter.convert(Path("/nonexistent/long.wav"), words=words)

    def test_overlap_is_measured_from_the_latest_end(self, converter: AudioToIML) -> None:
        """A long word followed by short ones overlaps them all."""
        words = [
            WordAlignment("long", 0, 3000),
            WordAlignment("a", 2600, 2700),  # 400 ms before "long" ends: jitter
            WordAlignment("b", 2400, 2450),  # sorted before "a"; 600 ms: too much
        ]
        with pytest.raises(ValueError, match=r"words\[2\] \('b'.*before words\[0\]"):
            converter.convert(SPEECH, words=words)

    def test_small_overlaps_are_accepted(
        self, converter: AudioToIML, parser: IMLParser
    ) -> None:
        """Recognisers' word edges may overlap a little (up to 500 ms)."""
        words = _words("speech_pauses")
        words[0] = dataclasses.replace(words[0], end_ms=words[1].start_ms + 500)
        assert audio_to_iml.MAX_WORD_OVERLAP_MS == 500
        doc = converter.convert_to_doc(SPEECH, words=words)
        assert parser.to_plain_text(doc) == "I told you to call me yesterday."
        words[0] = dataclasses.replace(words[0], end_ms=words[1].start_ms + 501)
        with pytest.raises(ValueError, match="overlap by at most 500 ms"):
            converter.convert_to_doc(SPEECH, words=words)

    def test_silence_over_a_minute_is_reported(
        self, converter: AudioToIML, tmp_path: Path
    ) -> None:
        """Spec 6.4: a pause over 60000 ms is written as 60000 ms, with a warning."""
        t = np.arange(int(0.5 * SR)) / SR
        tone = 0.3 * np.sin(2 * np.pi * 150 * t)
        audio = _write(tmp_path / "hold.wav", np.concatenate(
            [tone, np.zeros(int(0.1 * SR)), tone, np.zeros(int(61.4 * SR)), tone]
        ))
        words = [WordAlignment("Please", 0, 500), WordAlignment("hold.", 600, 1100),
                 WordAlignment("Thanks.", 62_500, 63_000)]
        result = converter.convert_detailed(audio, words=words)
        assert 60_000 in _pause_durations(result.document)
        assert max(_pause_durations(result.document)) == 60_000
        [note] = [w for w in result.warnings if "spec 6.4" in w]
        assert note.startswith("The silence of ") and '<pause duration="60000"/>' in note
        assert not [i for i in IMLValidator().validate(result.iml).issues if i.rule == "V33"]
        with pytest.warns(UserWarning, match="spec 6.4"):
            converter.convert(audio, words=words)

    def test_words_and_transcript_together_rejected(self, converter: AudioToIML) -> None:
        with pytest.raises(ValueError, match="not both"):
            converter.convert(SPEECH, words=_words("speech_pauses"), transcript="I told you")

    def test_no_speech_recognition_runs(self, fake_whisper: types.ModuleType) -> None:
        AudioToIML(stt="whisper").convert(SPEECH, words=_words("speech_pauses"))
        assert fake_whisper.loads == []


# ---------------------------------------------------------------------------
# Caller transcript without timings
# ---------------------------------------------------------------------------


class TestTranscript:
    def test_transcript_text_without_fabricated_timing(
        self, converter: AudioToIML, parser: IMLParser, validator: IMLValidator
    ) -> None:
        result = converter.convert_detailed(SPEECH, transcript="I told you  to call me\nyesterday.")
        assert result.transcript_source == "transcript"
        assert parser.to_plain_text(result.document) == "I told you to call me yesterday."
        # Nothing says where each word is, so no word-level tags or pauses.
        assert not [n for n in _nodes(result.document) if isinstance(n, (Pause, Emphasis))]
        assert validator.validate(result.iml).valid

    def test_transcript_is_one_utterance_with_emotion(self) -> None:
        converter = AudioToIML(stt="none", min_emotion_confidence=0.0)
        doc = converter.convert_to_doc(SPEECH, transcript="I told you to call me yesterday.")
        assert len(doc.utterances) == 1
        assert doc.utterances[0].emotion is not None
        assert doc.utterances[0].confidence is not None

    def test_transcript_xml_cannot_hold_is_rejected(self, converter: AudioToIML) -> None:
        """'a\\x1bb' used to come out as IML that IMLParser rejects."""
        with pytest.raises(ValueError, match="XML does not allow"):
            converter.convert(SPEECH, transcript="I told\x1b you")

    def test_transcript_must_be_text(self, converter: AudioToIML) -> None:
        with pytest.raises(TypeError, match="transcript must be a str"):
            converter.convert(SPEECH, transcript=b"I told you")  # type: ignore[arg-type]

    def test_empty_transcript(self, converter: AudioToIML, parser: IMLParser) -> None:
        result = converter.convert_detailed(SPEECH, transcript="   ")
        assert parser.to_plain_text(result.document) == ""
        assert any("empty" in w for w in result.warnings)

    def test_transcript_conversion_says_what_was_measured(self) -> None:
        """A transcript without timings used to come back as the plain text with
        no warning, though nothing but the utterance as a whole was measured."""
        converter = AudioToIML(stt="none")
        result = converter.convert_detailed(SPEECH, transcript="I told you to call me yesterday.")
        assert any("The transcript has no word timings" in w for w in result.warnings)
        # A single utterance without calibration has no baseline either.
        assert any(w.startswith("No speaker baseline") for w in result.warnings)

    def test_transcript_gets_utterance_level_delivery_against_calibration(self) -> None:
        """With a baseline, the utterance's overall delivery is measured: every
        word of speech_raised.wav is ~45% above the speaker's usual pitch."""
        converter = AudioToIML(stt="none", calibration_audio=CALIBRATION)
        result = converter.convert_detailed(
            AUDIO_DIR / "speech_raised.wav", transcript="please call me back."
        )
        [wrapper] = result.document.utterances[0].children
        assert isinstance(wrapper, Prosody) and wrapper.pitch is not None
        assert _pitch_percent(wrapper.pitch) > 25
        assert not any(w.startswith("No speaker baseline") for w in result.warnings)
        assert any("The transcript has no word timings" in w for w in result.warnings)

    def test_transcript_over_silence_has_no_emotion(self) -> None:
        converter = AudioToIML(stt="none", min_emotion_confidence=0.0)
        result = converter.convert_detailed(AUDIO_DIR / "silence_1s.wav", transcript="hello")
        assert [u.emotion for u in result.document.utterances] == [None]
        assert [u.confidence for u in result.document.utterances] == [None]
        assert any("No voiced speech" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# No transcript: placeholders
# ---------------------------------------------------------------------------


class TestPlaceholders:
    def test_one_placeholder_per_stretch_of_speech(
        self, converter: AudioToIML, parser: IMLParser
    ) -> None:
        result = converter.convert_detailed(SPEECH)
        assert result.transcript_source == "none"
        assert any(PLACEHOLDER_TOKEN in w for w in result.warnings)
        assert parser.to_plain_text(result.document).split() == [PLACEHOLDER_TOKEN] * 3
        durations = _pause_durations(result.document)
        for expected in (600, 300):
            assert any(abs(d - expected) <= 40 for d in durations), durations

    def test_auto_without_whisper_uses_placeholders(self, no_whisper: None) -> None:
        result = AudioToIML().convert_detailed(SPEECH)
        assert result.transcript_source == "none"
        assert any("openai-whisper is not installed" in w for w in result.warnings)

    def test_whisper_mode_without_whisper_raises(self, no_whisper: None) -> None:
        with pytest.raises(AudioProcessingError, match=r"prosody-protocol\[whisper\]"):
            AudioToIML(stt="whisper").convert(SPEECH)

    def test_tone_gap_tone_has_pause_element(self, converter: AudioToIML) -> None:
        """The 800 ms gap between the tones becomes a <pause> between two
        placeholders."""
        doc = converter.convert_to_doc(AUDIO_DIR / "tone_gap_tone.wav")
        assert sum(_text_of(u).count(PLACEHOLDER_TOKEN) for u in doc.utterances) == 2
        durations = _pause_durations(doc)
        assert any(abs(d - 800) <= 20 for d in durations), durations

    def test_convert_warns_about_placeholders(self, converter: AudioToIML) -> None:
        with pytest.warns(UserWarning, match=r"\[speech\]"):
            converter.convert(AUDIO_DIR / "tone_220hz.wav")


# ---------------------------------------------------------------------------
# Emotion
# ---------------------------------------------------------------------------


class TestEmotion:
    def test_silence_has_no_emotion(self, parser: IMLParser) -> None:
        """One second of digital silence used to come out as 'sad' 0.46."""
        converter = AudioToIML(stt="none", min_emotion_confidence=0.0)
        result = converter.convert_detailed(AUDIO_DIR / "silence_1s.wav")
        assert [(u.emotion, u.confidence) for u in result.document.utterances] == [(None, None)]
        assert parser.to_plain_text(result.document) == ""
        assert any("No voiced speech" in w for w in result.warnings)

    def test_noise_has_no_emotion(self, tmp_path: Path) -> None:
        """A noise-only recording used to come out as 'angry'."""
        noise = 10 ** (-60 / 20) * np.random.default_rng(0).standard_normal(SR)
        path = _write(tmp_path / "noise.wav", noise)
        converter = AudioToIML(stt="none", min_emotion_confidence=0.0)
        for kwargs in ({}, {"words": [WordAlignment("hello", 0, 1000)]}):
            doc = converter.convert_to_doc(path, **kwargs)
            assert all(u.emotion is None and u.confidence is None for u in doc.utterances)

    def test_speech_gets_emotion_and_confidence(self) -> None:
        converter = AudioToIML(stt="none", min_emotion_confidence=0.0)
        doc = converter.convert_to_doc(SPEECH, words=_words("speech_pauses"))
        for utt in doc.utterances:
            assert utt.emotion is not None
            assert utt.confidence is not None
            assert 0.0 <= utt.confidence <= 1.0

    def test_low_confidence_emotion_is_dropped(self) -> None:
        converter = AudioToIML(stt="none", min_emotion_confidence=1.0)
        doc = converter.convert_to_doc(SPEECH, words=_words("speech_pauses"))
        assert all(u.emotion is None and u.confidence is None for u in doc.utterances)


# ---------------------------------------------------------------------------
# Whisper (a fake module stands in for openai-whisper)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A stand-in for openai-whisper that 'recognises' speech_pauses.wav."""
    module = types.ModuleType("whisper")
    module.loads = []  # type: ignore[attr-defined]
    module.calls = []  # type: ignore[attr-defined]
    module.language = "de"  # type: ignore[attr-defined]
    module.load_error = None  # type: ignore[attr-defined]
    module.transcribe_error = None  # type: ignore[attr-defined]
    module.words = [  # type: ignore[attr-defined]
        {"word": " " + w["word"], "start": w["start_ms"] / 1000, "end": w["end_ms"] / 1000}
        for w in _truth("speech_pauses")["words"]
    ]

    class Model:
        def transcribe(self, audio: Any, **options: Any) -> dict[str, Any]:
            module.calls.append((audio, options))  # type: ignore[attr-defined]
            if module.transcribe_error is not None:  # type: ignore[attr-defined]
                raise module.transcribe_error  # type: ignore[attr-defined]
            return {
                "language": module.language,  # type: ignore[attr-defined]
                "segments": [{"words": module.words}],  # type: ignore[attr-defined]
            }

    def load_model(name: str) -> Model:
        module.loads.append(name)  # type: ignore[attr-defined]
        if module.load_error is not None:  # type: ignore[attr-defined]
            raise module.load_error  # type: ignore[attr-defined]
        return Model()

    module.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", module)
    return module


class TestWhisper:
    def test_whisper_words(self, fake_whisper: types.ModuleType, parser: IMLParser) -> None:
        result = AudioToIML().convert_detailed(SPEECH)
        assert result.transcript_source == "whisper"
        assert _other_warnings(result) == []
        assert parser.to_plain_text(result.document) == "I told you to call me yesterday."

    def test_model_is_loaded_once(self, fake_whisper: types.ModuleType) -> None:
        """Loading a checkpoint per file made batch conversion crawl."""
        converter = AudioToIML(stt_model="large-v3", stt="whisper")
        for _ in range(3):
            converter.convert(SPEECH)
        assert fake_whisper.loads == ["large-v3"]
        converter.stt_model = "tiny"
        converter.convert(SPEECH)
        assert fake_whisper.loads == ["large-v3", "tiny"]

    def test_requested_language_reaches_whisper(self, fake_whisper: types.ModuleType) -> None:
        doc = AudioToIML(language="fr-FR").convert_to_doc(SPEECH)
        assert fake_whisper.calls[-1][1] == {"word_timestamps": True, "language": "fr"}
        assert doc.language == "fr-FR"

    def test_detected_language_labels_the_document(
        self, fake_whisper: types.ModuleType
    ) -> None:
        doc = AudioToIML().convert_to_doc(SPEECH)
        assert "language" not in fake_whisper.calls[-1][1]
        assert doc.language == "de"

    @pytest.mark.parametrize("rate", [16_000, 22_050, 44_100])
    def test_whisper_gets_decoded_16khz_audio(
        self, fake_whisper: types.ModuleType, tmp_path: Path, rate: int
    ) -> None:
        """Whisper gets samples, not a path, so it needs no ffmpeg and its
        timestamps refer to the analysed audio. Other rates are resampled."""
        original = parselmouth.Sound(str(SPEECH))
        path = tmp_path / f"speech_{rate}.wav"
        original.resample(rate).save(str(path), "WAV")
        AudioToIML().convert(path)
        audio = fake_whisper.calls[-1][0]
        assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
        assert abs(len(audio) - _truth("speech_pauses")["duration_ms"] * 16) <= 16
        # The same waveform as the 16 kHz original, sample for sample.
        expected = original.values[0]
        n = min(len(audio), len(expected))
        error = np.sqrt(np.mean((audio[:n] - expected[:n]) ** 2))
        assert error < 0.05 * np.sqrt(np.mean(expected**2))

    def test_control_characters_in_whisper_output_are_dropped(
        self, fake_whisper: types.ModuleType, parser: IMLParser
    ) -> None:
        fake_whisper.words[1]["word"] = " to\x1bld"
        result = AudioToIML().convert_detailed(SPEECH)
        assert parser.to_plain_text(parser.parse(result.iml)) == (
            "I told you to call me yesterday."
        )

    def test_load_failure_is_an_error_not_a_placeholder(
        self, fake_whisper: types.ModuleType
    ) -> None:
        fake_whisper.load_error = URLError("Temporary failure in name resolution")
        with pytest.raises(AudioProcessingError, match="Cannot load Whisper model"):
            AudioToIML().convert(SPEECH)

    def test_transcription_failure_is_wrapped(self, fake_whisper: types.ModuleType) -> None:
        fake_whisper.transcribe_error = RuntimeError("out of memory")
        with pytest.raises(AudioProcessingError, match="out of memory"):
            AudioToIML().convert(SPEECH)

    @pytest.mark.parametrize(
        ("failure", "prefix"),
        [("load_error", "Cannot load Whisper model"),
         ("transcribe_error", "Whisper transcription failed")],
    )
    def test_recognizer_failures_have_their_own_error(
        self, fake_whisper: types.ModuleType, failure: str, prefix: str
    ) -> None:
        """Callers (the REST API) used to tell a recognizer failure from bad
        audio by the message alone."""
        setattr(fake_whisper, failure, RuntimeError("out of memory"))
        with pytest.raises(SpeechRecognitionError, match=f"^{prefix}") as caught:
            AudioToIML().convert(SPEECH)
        assert isinstance(caught.value, AudioProcessingError)

    def test_missing_whisper_is_not_a_recognizer_failure(self, no_whisper: None) -> None:
        with pytest.raises(AudioProcessingError) as caught:
            AudioToIML(stt="whisper").convert(SPEECH)
        assert not isinstance(caught.value, SpeechRecognitionError)
        with pytest.raises(AudioProcessingError) as caught:
            AudioToIML(stt="none").convert(AUDIO_DIR / "missing.wav")
        assert not isinstance(caught.value, SpeechRecognitionError)

    def test_no_words_recognised(self, fake_whisper: types.ModuleType, parser: IMLParser) -> None:
        fake_whisper.words = [{"word": " ", "start": 0.1, "end": 0.2}]
        result = AudioToIML().convert_detailed(SPEECH)
        assert parser.to_plain_text(result.document) == ""
        assert all(u.emotion is None for u in result.document.utterances)
        assert any("no words" in w for w in result.warnings)

    def test_stt_none_skips_whisper(self, fake_whisper: types.ModuleType) -> None:
        assert AudioToIML(stt="none").convert_detailed(SPEECH).transcript_source == "none"
        assert fake_whisper.loads == []


# ---------------------------------------------------------------------------
# Speaker calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def _pitches(self, doc: IMLDocument) -> list[float]:
        return [_pitch_percent(n.pitch) for n in _nodes(doc) if isinstance(n, Prosody) and n.pitch]

    def test_calibration_sets_the_baseline(self) -> None:
        """Every word of speech_raised.wav is ~45% above the speaker's normal
        pitch. Against its own average nothing stands out; against the
        calibration recording the raised pitch is marked."""
        words = _words("speech_raised")
        plain = AudioToIML(stt="none").convert_to_doc(
            AUDIO_DIR / "speech_raised.wav", words=words
        )
        assert all(abs(p) < 20 for p in self._pitches(plain))

        calibrated = AudioToIML(
            stt="none", calibration_audio=AUDIO_DIR / "speech_calibration.wav"
        ).convert_to_doc(AUDIO_DIR / "speech_raised.wav", words=words)
        raised = [p for p in self._pitches(calibrated) if p > 25]
        assert raised, IMLParser().to_iml_string(calibrated)

    def test_calibration_audio_is_analysed_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        analysed: list[Path] = []
        original = audio_to_iml._AudioAnalysis.from_path

        def counting(cls: Any, path: Any, *args: Any) -> Any:
            analysed.append(Path(path))
            return original(path, *args)

        monkeypatch.setattr(audio_to_iml._AudioAnalysis, "from_path", classmethod(counting))
        calibration = AUDIO_DIR / "speech_calibration.wav"
        converter = AudioToIML(stt="none", calibration_audio=calibration)
        for _ in range(3):
            converter.convert(AUDIO_DIR / "speech_raised.wav", words=_words("speech_raised"))
        assert analysed.count(calibration) == 1

    def test_transcript_and_words_agree_against_calibration(self) -> None:
        """The pitch movement of a whole utterance (transcript=) used to be
        compared with that of 300 ms calibration chunks: longer spans move
        more, so neutral speech came out 'joyful' 0.59 (and with words=, no
        emotion)."""
        converter = AudioToIML(stt="none", calibration_audio=CALIBRATION)
        by_words = converter.convert_to_doc(SPEECH, words=_words("speech_pauses"))
        by_text = converter.convert_to_doc(SPEECH, transcript="I told you to call me yesterday.")
        assert [u.emotion for u in by_words.utterances] == [None]
        assert [u.emotion for u in by_text.utterances] == [None]

    @pytest.mark.parametrize("name", ["speech_calibration", "speech_pauses", "speech_raised"])
    def test_recording_against_itself_has_no_emotion(self, name: str) -> None:
        """A recording measured against itself deviates from nothing, however
        its words are grouped (speech_pauses used to be 'joyful' 0.71)."""
        path = AUDIO_DIR / f"{name}.wav"
        converter = AudioToIML(stt="none", calibration_audio=path)
        for kwargs in ({"transcript": "some words"}, {}):
            doc = converter.convert_to_doc(path, **kwargs)  # type: ignore[arg-type]
            assert all(u.emotion is None for u in doc.utterances), IMLParser().to_iml_string(doc)

    def test_single_utterance_without_calibration_is_reported(self) -> None:
        """A one-turn recording gets no utterance-level delivery or emotion,
        which used to go unmentioned."""
        words = _words("speech_raised")
        plain = AudioToIML(stt="none").convert_detailed(
            AUDIO_DIR / "speech_raised.wav", words=words
        )
        [note] = [w for w in plain.warnings if w.startswith("No speaker baseline")]
        assert "calibration_audio" in note
        calibrated = AudioToIML(stt="none", calibration_audio=CALIBRATION).convert_detailed(
            AUDIO_DIR / "speech_raised.wav", words=words
        )
        assert not [w for w in calibrated.warnings if w.startswith("No speaker baseline")]

    def test_baseline_note_is_true_for_a_classifier_without_baseline(self) -> None:
        """A plain classifier (classify(features) only, such as a trained model)
        still labels a single utterance, so the note must not say that no
        emotion was estimated."""

        class Plain:
            def classify(self, features: list[Any]) -> tuple[str, float]:
                return ("calm", 0.8)

        result = AudioToIML(stt="none", emotion_classifier=Plain()).convert_detailed(
            SPEECH, words=_words("speech_pauses")
        )
        assert [u.emotion for u in result.document.utterances] == ["calm"]
        [note] = [w for w in result.warnings if w.startswith("No speaker baseline")]
        assert "not marked" in note and "emotion" not in note

    def test_several_calibration_files(self) -> None:
        """Earlier turns of the speaker can serve as calibration together."""
        converter = AudioToIML(stt="none", calibration_audio=[CALIBRATION, str(SPEECH)])
        assert converter.calibration_audio == (CALIBRATION, SPEECH)
        doc = converter.convert_to_doc(
            AUDIO_DIR / "speech_raised.wav", words=_words("speech_raised")
        )
        assert [p for p in self._pitches(doc) if p > 25]

    def test_growing_calibration_list_analyses_each_file_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A converter reused across turns only analyses the new turn."""
        analysed: list[Path] = []
        original = audio_to_iml._AudioAnalysis.from_path

        def counting(cls: Any, path: Any, *args: Any) -> Any:
            analysed.append(Path(path))
            return original(path, *args)

        monkeypatch.setattr(audio_to_iml._AudioAnalysis, "from_path", classmethod(counting))
        converter = AudioToIML(stt="none", calibration_audio=[])
        raised = AUDIO_DIR / "speech_raised.wav"
        turns: list[Path] = []
        for turn in (CALIBRATION, SPEECH, CALIBRATION):
            converter.calibration_audio = turns
            converter.convert(raised, words=_words("speech_raised"))
            turns.append(turn)
        assert analysed.count(CALIBRATION) == 1
        assert analysed.count(SPEECH) == 1
        assert analysed.count(raised) == 3

    def test_empty_calibration_list_is_no_calibration(self) -> None:
        converter = AudioToIML(stt="none", calibration_audio=())
        result = converter.convert_detailed(SPEECH, words=_words("speech_pauses"))
        assert any(w.startswith("No speaker baseline") for w in result.warnings)

    def test_silent_calibration_file_among_others_is_skipped(self) -> None:
        silent = AUDIO_DIR / "silence_1s.wav"
        converter = AudioToIML(stt="none", calibration_audio=[silent, CALIBRATION])
        result = converter.convert_detailed(
            AUDIO_DIR / "speech_raised.wav", words=_words("speech_raised")
        )
        assert any("contain no voiced speech and were not used" in w for w in result.warnings)
        converter.calibration_audio = [silent, silent]
        with pytest.raises(AudioProcessingError, match="Calibration audio"):
            converter.convert(SPEECH, words=_words("speech_pauses"))

    @pytest.mark.parametrize("bad", [42, b"me.wav", [CALIBRATION, 3]])
    def test_calibration_audio_must_be_paths(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="calibration_audio"):
            AudioToIML(calibration_audio=bad)

    def test_silent_calibration_audio_rejected(self) -> None:
        converter = AudioToIML(stt="none", calibration_audio=AUDIO_DIR / "silence_1s.wav")
        with pytest.raises(AudioProcessingError, match="Calibration audio"):
            converter.convert(SPEECH, words=_words("speech_pauses"))

    def test_missing_calibration_audio_rejected(self) -> None:
        converter = AudioToIML(stt="none", calibration_audio="/nonexistent/me.wav")
        with pytest.raises(AudioProcessingError, match="not found"):
            converter.convert(SPEECH, words=_words("speech_pauses"))


# ---------------------------------------------------------------------------
# Several speakers
# ---------------------------------------------------------------------------


@needs_espeak
class TestSpeakers:
    def test_two_voices_without_labels_get_no_emotion(
        self, call: tuple[Path, list[WordAlignment]]
    ) -> None:
        """A low voice and a high voice without speaker labels share one
        baseline, so the high voice's calm questions came out 'fearful' 0.71
        with pitch="+93%"."""
        path, words = call
        unlabelled = [dataclasses.replace(w, speaker=None) for w in words]
        result = AudioToIML(stt="none").convert_detailed(path, words=unlabelled)
        assert [u.emotion for u in result.document.utterances] == [None] * len(CALL)
        pitches = [_pitch_percent(n.pitch) for n in _nodes(result.document)
                   if isinstance(n, Prosody) and n.pitch]
        assert all(abs(p) < 50 for p in pitches), result.iml
        assert any("as from two voices" in w for w in result.warnings)

    def test_speaker_labels_give_each_speaker_a_baseline(
        self, call: tuple[Path, list[WordAlignment]]
    ) -> None:
        path, words = call
        result = AudioToIML(stt="none", min_emotion_confidence=0.0).convert_detailed(
            path, words=words
        )
        assert [u.speaker_id for u in result.document.utterances] == [s for _, s, _ in CALL]
        for utterance in result.document.utterances:
            pitches = [_pitch_percent(n.pitch) for n in _nodes(IMLDocument((utterance,)))
                       if isinstance(n, Prosody) and n.pitch]
            assert all(abs(p) < 50 for p in pitches), result.iml
            assert utterance.emotion != "fearful"
        # Speaker B said only two utterances: too few for a baseline.
        assert any(w.startswith("Speaker 'B': No speaker baseline") for w in result.warnings)
        assert not any("two voices" in w for w in result.warnings)

    def test_calibration_is_not_used_for_several_speakers(
        self, call: tuple[Path, list[WordAlignment]]
    ) -> None:
        path, words = call
        result = AudioToIML(stt="none", calibration_audio=CALIBRATION).convert_detailed(
            path, words=words
        )
        assert any("calibration_audio describes one speaker" in w for w in result.warnings)

    def test_one_speakers_words_of_a_call(self, call: tuple[Path, list[WordAlignment]]) -> None:
        """Converting one speaker's words is the other way to use a baseline;
        the other speaker's speech is noted as lying outside the words."""
        path, words = call
        result = AudioToIML(stt="none").convert_detailed(
            path, words=[w for w in words if w.speaker == "B"]
        )
        assert [u.speaker_id for u in result.document.utterances] == ["B", "B"]
        assert any("one speaker's of several" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Prosody profiles (spec Section 7)
# ---------------------------------------------------------------------------


def _pause_profile(boost: float) -> ProsodyProfile:
    """speech_pauses.wav pauses at 2 of its 6 word boundaries: pause_frequency high."""
    return ProsodyProfile(
        profile_version="0.1.0",
        user_id="user_42",
        description=None,
        mappings=(ProsodyMapping({"pause_frequency": "high"}, "uncertain", boost),),
    )


class TestProfile:
    def test_profile_sets_the_emotion_of_real_speech(self, parser: IMLParser) -> None:
        words = _words("speech_pauses")
        plain = AudioToIML(stt="none").convert_detailed(SPEECH, words=words)
        assert plain.document.utterances[0].emotion is None
        assert plain.profile_matches == ()

        result = AudioToIML(stt="none", profile=_pause_profile(0.6)).convert_detailed(
            SPEECH, words=words
        )
        utterance = result.document.utterances[0]
        # One utterance and no calibration audio: the classifier has no
        # baseline (confidence 0), so the boost alone is the confidence.
        assert (utterance.emotion, utterance.confidence) == ("uncertain", 0.6)
        # Profile use is shown by the pattern that matched, not the user_id.
        assert 'x-profile="pause_frequency=high"' in result.iml
        assert "user_42" not in result.iml
        [match] = result.profile_matches
        assert (match.utterance, match.pattern, match.applied) == (
            0, {"pause_frequency": "high"}, True
        )
        assert match.observed["pause_frequency"] == "high"
        # The markup itself does not change.
        assert result.document.utterances[0].children == plain.document.utterances[0].children
        assert IMLValidator().validate(result.iml).valid

    def test_calibrated_confidence_is_boosted(self) -> None:
        """With calibration the classifier has a baseline; its confidence plus
        the boost (spec 7.2) is the utterance's confidence."""
        kwargs: dict[str, Any] = {
            "stt": "none",
            "calibration_audio": AUDIO_DIR / "speech_calibration.wav",
            "min_emotion_confidence": 0.0,
        }
        words = _words("speech_pauses")
        base = AudioToIML(**kwargs).convert_to_doc(SPEECH, words=words).utterances[0]
        assert base.confidence is not None
        boosted = AudioToIML(**kwargs, profile=_pause_profile(0.2)).convert_detailed(
            SPEECH, words=words
        )
        utterance = boosted.document.utterances[0]
        assert utterance.emotion == "uncertain"
        assert utterance.confidence == round(min(1.0, base.confidence + 0.2), 4)
        assert f'confidence="{utterance.confidence}"' in boosted.iml

    def test_abstention_applies_after_the_profile(self) -> None:
        result = AudioToIML(stt="none", profile=_pause_profile(0.3)).convert_detailed(
            SPEECH, words=_words("speech_pauses")
        )
        assert result.document.utterances[0].emotion is None
        assert "x-profile" not in result.iml
        [match] = result.profile_matches
        assert (match.emotion, match.confidence, match.applied) == ("uncertain", 0.3, False)

    def test_silence_gets_no_emotion_from_a_profile(self) -> None:
        """Words over silence still match pause_frequency=high, but silence is
        never given an emotion."""
        words = [WordAlignment(w, i * 300, i * 300 + 100) for i, w in enumerate("abcd")]
        result = AudioToIML(stt="none", profile=_pause_profile(0.9)).convert_detailed(
            AUDIO_DIR / "silence_1s.wav", words=words
        )
        assert result.document.utterances[0].emotion is None
        assert "x-profile" not in result.iml
        assert result.profile_matches == ()

    def test_fixture_profile_loads(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "profiles" / "autism_spectrum.json"
        profile = ProfileLoader().load(fixture)
        converter = AudioToIML(stt="none", profile=profile)
        assert converter.profile is profile
        assert AudioToIML().profile is None
        result = converter.convert_detailed(SPEECH, words=_words("speech_pauses"))
        assert IMLValidator().validate(result.iml).valid

    def test_invalid_profile_rejected(self) -> None:
        bad = dataclasses.replace(_pause_profile(0.1), profile_version="one")
        with pytest.raises(ProfileError, match="P1"):
            AudioToIML(profile=bad)


# ---------------------------------------------------------------------------
# Full pipeline on the synthetic fixtures
# ---------------------------------------------------------------------------


class TestConvertToDoc:
    def test_produces_iml_document(self, converter: AudioToIML) -> None:
        doc = converter.convert_to_doc(AUDIO_DIR / "tone_220hz.wav")
        assert isinstance(doc, IMLDocument)
        assert len(doc.utterances) == 1
        assert isinstance(doc.utterances[0], Utterance)

    def test_version_set(self, converter: AudioToIML) -> None:
        doc = converter.convert_to_doc(AUDIO_DIR / "tone_220hz.wav")
        assert doc.version == "0.1.0"

    def test_language_set(self, converter: AudioToIML) -> None:
        doc = converter.convert_to_doc(AUDIO_DIR / "tone_220hz.wav")
        assert doc.language == "en-US"


class TestConvert:
    def test_output_is_parseable(self, converter: AudioToIML, parser: IMLParser) -> None:
        xml = converter.convert(SPEECH, words=_words("speech_pauses"))
        doc = parser.parse(xml)
        assert parser.to_plain_text(doc) == "I told you to call me yesterday."

    def test_output_passes_validation(
        self, converter: AudioToIML, validator: IMLValidator
    ) -> None:
        """Key acceptance criterion: output always passes IMLValidator."""
        xml = converter.convert(SPEECH, words=_words("speech_pauses"))
        result = validator.validate(xml)
        errors = [i for i in result.issues if i.severity == "error"]
        assert result.valid, f"Validation errors: {[i.message for i in errors]}"


class TestRoundTrip:
    def test_parse_serialize_round_trip(self, converter: AudioToIML, parser: IMLParser) -> None:
        xml1 = converter.convert(SPEECH, words=_words("speech_pauses"))
        doc = parser.parse(xml1)
        xml2 = parser.to_iml_string(doc)
        assert xml2 == xml1
        assert parser.parse(xml2) == doc


class TestExtendedAttributes:
    def test_no_extended_by_default(self, converter: AudioToIML) -> None:
        xml = converter.convert(SPEECH, words=_words("speech_pauses"))
        assert "f0_mean" not in xml
        assert "jitter" not in xml

    def test_extended_when_enabled(self) -> None:
        converter = AudioToIML(include_extended=True, stt="none")
        doc = converter.convert_to_doc(SPEECH, words=_words("speech_pauses"))
        told = [
            n for n in _nodes(doc) if isinstance(n, Prosody) and "told" in _text_of(n)
        ]
        assert told and told[0].f0_mean is not None
        assert 110 < told[0].f0_mean < 160
        # Spec 4.4: jitter and shimmer are percentages, not fractions.
        assert told[0].shimmer is not None and told[0].shimmer > 1.0


class TestMultipleAudioFiles:
    def test_all_fixtures_produce_valid_output(
        self, converter: AudioToIML, validator: IMLValidator
    ) -> None:
        """Every audio fixture should produce valid IML output."""
        for wav_file in sorted(AUDIO_DIR.glob("*.wav")):
            xml = converter.convert_detailed(wav_file).iml
            result = validator.validate(xml)
            errors = [i for i in result.issues if i.severity == "error"]
            assert result.valid, (
                f"{wav_file.name} produced invalid IML: "
                + "; ".join(i.message for i in errors)
            )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _bad_inputs(tmp_path: Path) -> dict[str, Path]:
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    text = tmp_path / "README.md"
    text.write_text("# not audio\n")
    return {
        "missing": tmp_path / "missing.wav",
        "directory": tmp_path,
        "empty": empty,
        "zero samples": _write(tmp_path / "zero.wav", np.zeros(0)),
        "40 ms": _write(tmp_path / "short.wav", 0.3 * np.sin(np.arange(640) / 10)),
        "not audio": text,
    }


class TestErrors:
    def test_nonexistent_file_raises(self, converter: AudioToIML) -> None:
        with pytest.raises(AudioProcessingError, match="not found"):
            converter.convert("/nonexistent/audio.wav")

    def test_nonexistent_file_convert_to_doc_raises(self, converter: AudioToIML) -> None:
        with pytest.raises(AudioProcessingError, match="not found"):
            converter.convert_to_doc("/nonexistent/audio.wav")

    @pytest.mark.parametrize("method", ["convert", "convert_to_doc", "convert_detailed"])
    @pytest.mark.parametrize("stt", ["auto", "none"])
    def test_unreadable_audio_raises_audio_processing_error(
        self, tmp_path: Path, method: str, stt: str, no_whisper: None
    ) -> None:
        """These used to leak parselmouth.PraatError (HTTP 500 in the API)."""
        converter = AudioToIML(stt=stt)  # type: ignore[arg-type]
        for name, path in _bad_inputs(tmp_path).items():
            with pytest.raises(AudioProcessingError):
                getattr(converter, method)(path)
            with pytest.raises(AudioProcessingError):
                getattr(converter, method)(path, words=[WordAlignment("hi", 0, 100)])
            with pytest.raises(AudioProcessingError):
                getattr(converter, method)(path, transcript="hi")
            assert name  # keeps the failing input visible in tracebacks

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_ogg_opus_upload_is_decoded(self, tmp_path: Path, converter: AudioToIML) -> None:
        """Browser recordings (Opus in WebM/Ogg) are decoded with ffmpeg."""
        encoded = tmp_path / "recording.webm"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(SPEECH),
             "-c:a", "libopus", str(encoded)],
            check=True,
        )
        result = converter.convert_detailed(encoded, words=_words("speech_pauses"))
        assert IMLParser().to_plain_text(result.document) == "I told you to call me yesterday."
        durations = _pause_durations(result.document)
        assert any(abs(d - 600) <= 50 for d in durations), durations

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_playlist_upload_is_rejected(self, tmp_path: Path, converter: AudioToIML) -> None:
        """An uploaded .m3u8 naming another file used to return that file's
        prosody (and, with Whisper, its transcript)."""
        playlist = tmp_path / "upload.m3u8"
        playlist.write_text(
            "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\n"
            f"{SPEECH.resolve()}\n#EXT-X-ENDLIST\n"
        )
        with pytest.raises(AudioProcessingError, match="not a supported audio format"):
            converter.convert_detailed(playlist)


class TestMaxDuration:
    def test_audio_longer_than_the_limit_is_rejected(self) -> None:
        converter = AudioToIML(stt="none", max_duration_s=2.0)
        with pytest.raises(AudioProcessingError, match="max_duration_s=2 s"):
            converter.convert_detailed(SPEECH, words=_words("speech_pauses"))

    def test_audio_within_the_limit_is_converted(self, parser: IMLParser) -> None:
        converter = AudioToIML(stt="none", max_duration_s=5.0)
        doc = converter.convert_to_doc(SPEECH, words=_words("speech_pauses"))
        assert parser.to_plain_text(doc) == "I told you to call me yesterday."

    def test_limit_applies_to_calibration_audio(self) -> None:
        converter = AudioToIML(
            stt="none", max_duration_s=1.5, calibration_audio=AUDIO_DIR / "speech_calibration.wav"
        )
        with pytest.raises(AudioProcessingError, match="speech_calibration.wav"):
            converter.convert(AUDIO_DIR / "tone_220hz.wav", words=[WordAlignment("a", 0, 500)])
