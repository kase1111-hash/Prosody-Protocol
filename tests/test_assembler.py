"""Tests for prosody_protocol.assembler.

Tests word spacing, utterance grouping, pause placement, speaker
baselines, prosody and emphasis markup, pitch contours, voice quality,
extended attributes and emotion thresholds -- mostly by comparing the
exact IML produced from hand-made features.
"""

from __future__ import annotations

import dataclasses
import math
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from prosody_protocol import PauseInterval, SpanFeatures, WordAlignment
from prosody_protocol.assembler import (
    DEFAULT_MIN_EMOTION_CONFIDENCE,
    MAX_PAUSE_MS,
    PROFILE_ATTRIBUTE,
    IMLAssembler,
    ProfileMatch,
    _Assembly,
)
from prosody_protocol.emotion_classifier import RuleBasedEmotionClassifier, SpeakerBaseline
from prosody_protocol.exceptions import ProfileError
from prosody_protocol.models import (
    ChildNode,
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
)
from prosody_protocol.parser import IMLParser
from prosody_protocol.profiles import ProfileLoader, ProsodyMapping, ProsodyProfile
from prosody_protocol.validator import IMLValidator

PROFILES_DIR = Path(__file__).parent / "fixtures" / "profiles"

# A word in the tables below: (text, start_ms, end_ms, f0_hz, intensity_db),
# with optional extra SpanFeatures fields as a dict.
Word = tuple[str, int, int, float, float] | tuple[str, int, int, float, float, Mapping[str, object]]


@pytest.fixture()
def assembler() -> IMLAssembler:
    return IMLAssembler(include_extended=False)


@pytest.fixture()
def assembler_extended() -> IMLAssembler:
    return IMLAssembler(include_extended=True)


@pytest.fixture()
def validator() -> IMLValidator:
    return IMLValidator()


def _make_alignment(word: str, start_ms: int, end_ms: int) -> WordAlignment:
    return WordAlignment(word=word, start_ms=start_ms, end_ms=end_ms)


def _make_features(
    word: str,
    start_ms: int,
    end_ms: int,
    f0_mean: float = 180.0,
    intensity_mean: float = 65.0,
    **kwargs: object,
) -> SpanFeatures:
    return SpanFeatures(
        start_ms=start_ms,
        end_ms=end_ms,
        text=word,
        f0_mean=f0_mean,
        intensity_mean=intensity_mean,
        **kwargs,  # type: ignore[arg-type]
    )


def _assemble(
    words: Sequence[Word],
    pauses: Sequence[PauseInterval] = (),
    assembler: IMLAssembler | None = None,
    **kwargs: object,
) -> IMLDocument:
    alignments = [_make_alignment(w[0], w[1], w[2]) for w in words]
    features = [
        _make_features(w[0].strip(), w[1], w[2], w[3], w[4], **(w[5] if len(w) > 5 else {}))
        for w in words
    ]
    return (assembler or IMLAssembler()).assemble(
        alignments, features, list(pauses), **kwargs  # type: ignore[arg-type]
    )


def _features(words: Sequence[Word]) -> list[SpanFeatures]:
    return [_make_features(w[0], w[1], w[2], w[3], w[4]) for w in words]


def _xml(doc: IMLDocument) -> str:
    return IMLParser().to_iml_string(doc)


def _plain(doc: IMLDocument) -> str:
    return IMLParser().to_plain_text(doc)


def _sentence(text: str, start_ms: int = 0, f0: float = 120.0, db: float = 65.0) -> list[Word]:
    """Evenly spoken words, 250 ms each with 50 ms gaps."""
    return [
        (w, start_ms + i * 300, start_ms + i * 300 + 250, f0, db)
        for i, w in enumerate(text.split())
    ]


_EXTENDED_FIELDS = (
    "f0_mean", "f0_range", "f0_contour", "intensity_mean", "intensity_range", "speech_rate",
    "duration_ms", "jitter", "shimmer", "hnr",
)


def _without_extended(children: Sequence[ChildNode]) -> list[ChildNode]:
    """*children* with the extended attributes removed, and <prosody> elements
    that had nothing else unwrapped (adjacent text joined)."""
    out: list[ChildNode] = []
    for child in children:
        parts: list[ChildNode] = [child]
        if isinstance(child, Prosody):
            inner = _without_extended(child.children)
            bare = dataclasses.replace(
                child, children=tuple(inner), **dict.fromkeys(_EXTENDED_FIELDS)
            )
            parts = inner if bare == Prosody(children=tuple(inner)) else [bare]
        elif isinstance(child, Emphasis):
            parts = [dataclasses.replace(child, children=tuple(_without_extended(child.children)))]
        for part in parts:
            if isinstance(part, str) and out and isinstance(out[-1], str):
                out[-1] += part
            else:
                out.append(part)
    return out


def _nodes(children: Sequence[ChildNode]) -> list[ChildNode]:
    out: list[ChildNode] = []
    for child in children:
        out.append(child)
        if not isinstance(child, str | Pause):
            out.extend(_nodes(child.children))
    return out


# ---------------------------------------------------------------------------
# Basic assembly
# ---------------------------------------------------------------------------


class TestBasicAssembly:
    def test_single_word(self, assembler: IMLAssembler) -> None:
        alignments = [_make_alignment("Hello.", 0, 500)]
        features = [_make_features("Hello.", 0, 500)]
        doc = assembler.assemble(alignments, features, pauses=[])
        assert _xml(doc) == '<iml version="0.1.0"><utterance>Hello.</utterance></iml>'

    def test_version_set(self, assembler: IMLAssembler) -> None:
        alignments = [_make_alignment("Hi.", 0, 500)]
        features = [_make_features("Hi.", 0, 500)]
        doc = assembler.assemble(alignments, features, pauses=[])
        assert doc.version == "0.1.0"

    def test_language_passed_through(self, assembler: IMLAssembler) -> None:
        alignments = [_make_alignment("Hi.", 0, 500)]
        features = [_make_features("Hi.", 0, 500)]
        doc = assembler.assemble(alignments, features, pauses=[], language="en-US")
        assert doc.language == "en-US"

    def test_no_words_gives_one_empty_utterance(
        self, assembler: IMLAssembler, validator: IMLValidator
    ) -> None:
        doc = assembler.assemble([_make_alignment("  ", 0, 100)], [], pauses=[])
        assert _xml(doc) == '<iml version="0.1.0"><utterance></utterance></iml>'
        assert validator.validate(_xml(doc)).valid

    def test_features_paired_by_span_when_not_one_to_one(
        self, assembler: IMLAssembler
    ) -> None:
        alignments = [
            _make_alignment(w, i * 300, i * 300 + 250) for i, w in enumerate(["a", "b", "c", "d"])
        ]
        features = [  # out of order, one missing
            _make_features("d", 900, 1150),
            _make_features("b", 300, 550, f0_mean=260.0),
            _make_features("a", 0, 250),
        ]
        doc = assembler.assemble(alignments, features, pauses=[])
        assert _xml(doc) == (
            '<iml version="0.1.0"><utterance>a <emphasis level="moderate">'
            '<prosody pitch="+44%">b</prosody></emphasis> c d</utterance></iml>'
        )


# ---------------------------------------------------------------------------
# Word spacing
# ---------------------------------------------------------------------------


class TestSpacing:
    def test_contiguous_words_around_markup(self) -> None:
        """STT timings usually touch; words next to tags used to be glued together."""
        doc = _assemble([
            ("I", 0, 300, 180, 65), ("really", 300, 600, 180, 65), ("told", 600, 900, 250, 75),
            ("you", 900, 1200, 180, 65), ("so.", 1200, 1500, 180, 65),
        ])
        assert _xml(doc) == (
            '<iml version="0.1.0"><utterance>I really <emphasis level="strong">'
            '<prosody pitch="+39%" volume="+10dB">told</prosody></emphasis> you so.'
            "</utterance></iml>"
        )
        assert _plain(doc) == "I really told you so."

    def test_whisper_leading_spaces(self) -> None:
        doc = _assemble([
            (" I", 0, 200, 180, 65), (" told", 200, 500, 240, 74),
            (" you", 500, 700, 180, 65), (" yesterday!", 700, 1200, 182, 65),
        ])
        assert _xml(doc) == (
            '<iml version="0.1.0"><utterance>I <emphasis level="strong">'
            '<prosody pitch="+33%" volume="+9dB">told</prosody></emphasis> you yesterday!'
            "</utterance></iml>"
        )

    def test_punctuation_tokens_attach(self) -> None:
        doc = _assemble([
            ("Well", 0, 300, 180, 65), (",", 300, 300, 180, 65), ("(", 400, 400, 180, 65),
            ("I", 400, 500, 180, 65), (")", 500, 500, 180, 65), ("think", 500, 800, 180, 65),
            ("?!", 800, 800, 180, 65),
        ])
        assert _plain(doc) == "Well, (I) think?!"

    def test_pause_keeps_one_space(self) -> None:
        doc = _assemble(_sentence("I think") + [("so.", 1300, 1550, 120, 65)])
        assert _xml(doc) == (
            '<iml version="0.1.0"><utterance>I think<pause duration="750"/> so.</utterance></iml>'
        )
        assert _plain(doc) == "I think so."

    @pytest.mark.parametrize(
        ("tokens", "text"),
        [
            ("rock & roll", "rock & roll"),
            ("and / or", "and / or"),
            ("2010 – 2020", "2010 – 2020"),
            ("wait - what", "wait - what"),
            ("Wait … what", "Wait… what"),
            ("50 % off", "50% off"),
            ("¿ Qué ?", "¿Qué?"),
            ('He said " stop " twice', 'He said "stop" twice'),
            ('" Go " , he said', '"Go", he said'),
        ],
    )
    def test_only_closing_punctuation_attaches(self, tokens: str, text: str) -> None:
        """Every punctuation token used to attach left: 'rock& roll', 'and/ or'."""
        assert _plain(_assemble(_sentence(tokens))) == text


# ---------------------------------------------------------------------------
# Utterance grouping
# ---------------------------------------------------------------------------


class TestUtteranceGrouping:
    def test_sentence_boundary_splits_utterances(
        self, assembler: IMLAssembler
    ) -> None:
        """Words ending with '.' should trigger a new utterance."""
        alignments = [
            _make_alignment("Hello.", 0, 500),
            _make_alignment("World.", 600, 1100),
        ]
        features = [
            _make_features("Hello.", 0, 500),
            _make_features("World.", 600, 1100),
        ]
        doc = assembler.assemble(alignments, features, pauses=[])
        assert [u.children for u in doc.utterances] == [("Hello.",), ("World.",)]

    def test_long_pause_splits_unpunctuated_text(self, assembler: IMLAssembler) -> None:
        """Without punctuation, a long pause splits -- and is kept."""
        alignments = [
            _make_alignment("Before", 0, 400),
            _make_alignment("after", 1500, 1900),
        ]
        features = [
            _make_features("Before", 0, 400),
            _make_features("after", 1500, 1900),
        ]
        pauses = [PauseInterval(start_ms=400, end_ms=1500)]
        doc = assembler.assemble(alignments, features, pauses=pauses)
        assert [u.children for u in doc.utterances] == [
            ("Before",), (Pause(duration=1100), "after"),
        ]

    def test_long_pause_inside_a_sentence_does_not_split(self) -> None:
        words = _sentence("I think") + [("so,", 2000, 2250, 120, 65), ("yes.", 2300, 2550, 120, 65)]
        doc = _assemble(words)
        assert _xml(doc) == (
            '<iml version="0.1.0"><utterance>I think<pause duration="1450"/> so, yes.'
            "</utterance></iml>"
        )

    def test_pause_after_sentence_end_starts_next_utterance(self) -> None:
        words = _sentence("Are you okay?") + [("Yes.", 3850, 4100, 120, 65)]
        doc = _assemble(words)
        assert [u.children for u in doc.utterances] == [
            ("Are you okay?",), (Pause(duration=3000), "Yes."),
        ]

    def test_abbreviations_and_initials_do_not_split(self) -> None:
        doc = _assemble(_sentence("I met Dr. Smith and J. Doe at 5 p.m. in the U.S. office."))
        assert len(doc.utterances) == 1

    @pytest.mark.parametrize(
        ("text", "sentences"),
        [
            ("So did I. Then we left.", ["So did I.", "Then we left."]),
            ("It was Plan B. We had no choice.", ["It was Plan B.", "We had no choice."]),
            ("Take vitamin C. It helps.", ["Take vitamin C.", "It helps."]),
            ("It was Plan B. It’s over.", ["It was Plan B.", "It’s over."]),
            ("Call me at 5 p.m. The office closes.", ["Call me at 5 p.m.", "The office closes."]),
            ("We moved to the U.S. Army base.", ["We moved to the U.S. Army base."]),
            ("I met John F. Kennedy once.", ["I met John F. Kennedy once."]),
            ("We bought pears etc. Then we left.", ["We bought pears etc.", "Then we left."]),
        ],
    )
    def test_initials_and_abbreviations_at_sentence_ends(
        self, text: str, sentences: list[str]
    ) -> None:
        """'I.' and a final 'B.' or 'C.' used to count as initials and join two sentences."""
        doc = _assemble(_sentence(text))
        assert [u.children[-1] for u in doc.utterances] == sentences

    def test_closing_quote_after_period_splits(self) -> None:
        doc = _assemble(_sentence('He said "stop." Then he left.'))
        assert _plain(doc) == 'He said "stop." Then he left.'
        assert len(doc.utterances) == 2

    def test_ellipsis_and_lowercase_continuation_do_not_split(self) -> None:
        doc = _assemble(_sentence("Well... I guess. so it goes."))
        assert len(doc.utterances) == 1

    def test_all_lowercase_transcript_still_splits(self) -> None:
        """Some recognisers write no capitals; their periods still end sentences."""
        doc = _assemble(_sentence("hello there. how are you? fine."))
        assert [u.children for u in doc.utterances] == [
            ("hello there.",), ("how are you?",), ("fine.",),
        ]


# ---------------------------------------------------------------------------
# Pause insertion
# ---------------------------------------------------------------------------


    def test_half_second_pause_splits_unpunctuated_text(self) -> None:
        """Unpunctuated sentences with ordinary gaps (0.5-1 s) used to run
        together into one utterance, silently; shorter pauses do not split."""
        words = (
            _sentence("i checked the schedule")
            + _sentence("the train leaves at nine", 1800)
            + _sentence("we should get there", 3250 + 400)  # 400 ms after "nine"
        )
        assembly = IMLAssembler()._assemble(
            [_make_alignment(*w[:3]) for w in words], _features(words), []
        )
        assert [_plain(IMLDocument((u,))) for u in assembly.document.utterances] == [
            "i checked the schedule", "the train leaves at nine we should get there",
        ]
        [note] = [n for n in assembly.info if "no sentence punctuation" in n]
        assert "0.5 s" in note

    def test_no_punctuation_note_for_punctuated_or_placeholder_text(self) -> None:
        words = _sentence("I checked it.") + _sentence("The train left.", 1800)
        assembly = IMLAssembler()._assemble(
            [_make_alignment(*w[:3]) for w in words], _features(words), []
        )
        assert not [n for n in assembly.info if "punctuation" in n]
        bare = [(w[0].rstrip("."), *w[1:]) for w in words]
        assembly = IMLAssembler()._assemble(
            [_make_alignment(*w[:3]) for w in bare], _features(bare), [],
            check_punctuation=False,
        )
        assert not [n for n in assembly.info if "punctuation" in n]


# ---------------------------------------------------------------------------
# Speakers and baselines
# ---------------------------------------------------------------------------


def _spoken(words: Sequence[Word], speaker: str | None) -> list[WordAlignment]:
    return [WordAlignment(w[0], w[1], w[2], speaker) for w in words]


def _assembly(
    turns: Sequence[tuple[Sequence[Word], str | None]],
    assembler: IMLAssembler | None = None,
    **kwargs: Any,
) -> _Assembly:
    alignments = [a for words, speaker in turns for a in _spoken(words, speaker)]
    features = [f for words, _ in turns for f in _features(words)]
    return (assembler or IMLAssembler())._assemble(alignments, features, [], **kwargs)


def _call(labelled: bool) -> list[tuple[list[Word], str | None]]:
    """Three turns each of a low voice (120 Hz) and a high voice (220 Hz)."""
    turns: list[tuple[list[Word], str | None]] = []
    for i in range(6):
        low = i % 2 == 0
        words = _sentence("okay that sounds good.", i * 2000, 120 if low else 220, 65)
        turns.append((words, ("A" if low else "B") if labelled else None))
    return turns


class TestSpeakers:
    def test_speaker_change_starts_an_utterance(self) -> None:
        assembly = _assembly([(_sentence("yes i know"), "A"), (_sentence("right", 1000), "B")])
        utterances = assembly.document.utterances
        assert [(u.speaker_id, _plain(IMLDocument((u,)))) for u in utterances] == [
            ("A", "yes i know"), ("B", "right"),
        ]

    def test_each_speaker_has_their_own_baseline(self) -> None:
        """Against one baseline for both, the high voice was 'much higher'."""
        assembly = _assembly(_call(labelled=True), IMLAssembler(_Fixed("angry", 0.9)))
        assert [u.speaker_id for u in assembly.document.utterances] == ["A", "B"] * 3
        assert not [n for u in assembly.document.utterances for n in _nodes(u.children)
                    if isinstance(n, Prosody) and n.pitch]
        assert all(u.emotion == "angry" for u in assembly.document.utterances)
        assert not [n for n in assembly.info if "baseline" in n or "voices" in n]

    def test_two_voices_without_labels_have_no_baseline(self) -> None:
        """The high voice's calm sentences used to be 'fearful' with pitch="+93%"."""
        recorder = _BaselineRecorder()
        assembly = _assembly(_call(labelled=False), IMLAssembler(recorder))
        utterances = assembly.document.utterances
        assert recorder.baselines == [SpeakerBaseline()] * 6
        assert not [n for u in utterances for n in _nodes(u.children)
                    if isinstance(n, Prosody) and n.pitch]
        [note] = assembly.info
        assert "(about 120 Hz and 220 Hz), as from two voices" in note
        assert "the emotion classifier got no baseline to compare with" in note
        default = _assembly(_call(labelled=False), IMLAssembler(min_emotion_confidence=0.0))
        assert [(u.emotion, u.confidence) for u in default.document.utterances] == [
            ("neutral", 0.0)
        ] * 6
        [note] = default.info
        assert "and no emotion was estimated" in note

    def test_notes_do_not_claim_abstention_for_a_classifier_without_baseline(self) -> None:
        """A classifier that does not use the baseline (a trained model, say)
        still labels the utterances; the notes used to say no emotion was
        estimated."""
        plain = IMLAssembler(_Fixed("angry", 0.9))
        one = _assembly([(_sentence("I see."), None)], plain)
        assert [u.emotion for u in one.document.utterances] == ["angry"]
        [note] = one.info
        assert note.startswith("No speaker baseline") and "emotion" not in note
        voices = _assembly(_call(labelled=False), plain)
        assert [u.emotion for u in voices.document.utterances] == ["angry"] * 6
        [note] = voices.info
        assert "two voices" in note and "emotion" not in note
        fixed = IMLAssembler(RuleBasedEmotionClassifier(baseline_f0=120.0))
        [note] = _assembly([(_sentence("I see."), None)], fixed).info
        assert "used only the baseline it was constructed with" in note

    @pytest.mark.parametrize(
        ("semitones", "two_voices"),
        [
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], False),  # ever more excited: no gap
            ([0, 0.5, 1, 5, 5.5, 6], False),  # two groups, but only 5 semitones apart
            ([0, 0.5, 1, 1.5, 9, 9.5], True),
        ],
    )
    def test_two_voices_need_two_distant_groups(
        self, semitones: list[float], two_voices: bool
    ) -> None:
        turns = [
            (_sentence("okay then.", i * 2000, 120 * 2 ** (st / 12), 65), None)
            for i, st in enumerate(semitones)
        ]
        assembly = _assembly(turns)
        assert bool([n for n in assembly.info if "two voices" in n]) is two_voices

    def test_one_outlier_is_not_a_second_voice(self) -> None:
        turns = [(_sentence("okay then.", i * 2000, f0, 65), None)
                 for i, f0 in enumerate([120, 122, 118, 121, 240])]
        assert not [n for n in _assembly(turns).info if "two voices" in n]

    def test_calibration_and_profile_describe_one_speaker(self) -> None:
        reference = _features(_sentence("this is how I sound.", f0=120))
        assembler = IMLAssembler(_Fixed("angry", 0.9), profile=SPIKE_PROFILE)
        assembly = _assembly(_call(labelled=True), assembler, reference_features=reference)
        assert any(n.startswith("calibration_audio describes one speaker") for n in assembly.info)
        assert any(n.startswith("The prosody profile describes one speaker") for n in assembly.info)
        assert assembly.matches == []

    def test_no_baseline_is_noted(self) -> None:
        one = _assembly([(_sentence("I see."), None)])
        assert [n[:60] for n in one.info] == [
            "No speaker baseline: a single utterance without calibration_"
        ]
        two = _assembly([(_sentence("I see."), None), (_sentence("Fine.", 2000), None)])
        assert any("this one has 2." in n for n in two.info)
        calibrated = _assembly(
            [(_sentence("I see."), None)], reference_features=_features(_sentence("so it is."))
        )
        assert calibrated.info == []
        per_speaker = _assembly(_call(labelled=True)[:3])  # A, B, A
        assert [n.split(":")[0] for n in per_speaker.info] == ["Speaker 'A'", "Speaker 'B'"]


class TestPauseInsertion:
    def test_gap_produces_pause_element(self, assembler: IMLAssembler) -> None:
        """A 500ms gap between words should produce a <pause> element."""
        alignments = [
            _make_alignment("before", 0, 300),
            _make_alignment("after.", 800, 1100),
        ]
        features = [
            _make_features("before", 0, 300),
            _make_features("after.", 800, 1100),
        ]
        doc = assembler.assemble(alignments, features, pauses=[])
        assert doc.utterances[0].children == ("before", Pause(duration=500), " after.")

    def test_detected_pause_inside_contiguous_timings(self) -> None:
        """Whisper-style timings absorb silence; the detected pause still shows."""
        words: list[Word] = [
            ("one", 0, 300, 120, 65), ("two", 300, 1100, 120, 65),
            ("three", 1100, 1400, 120, 65), ("four.", 1400, 1700, 120, 65),
        ]
        doc = _assemble(words, pauses=[PauseInterval(500, 1230)])
        assert doc.utterances[0].children == ("one two", Pause(duration=730), " three four.")

    def test_one_pause_per_gap(self) -> None:
        """A detected pause and a word-timing gap at the same place give one <pause>."""
        words: list[Word] = [("one", 0, 300, 120, 65), ("two.", 900, 1200, 120, 65)]
        doc = _assemble(words, pauses=[PauseInterval(320, 880)])
        assert doc.utterances[0].children == ("one", Pause(duration=560), " two.")

    def test_silence_before_and_after_speech_is_not_a_pause(self) -> None:
        words = _sentence("one two three.", start_ms=1000)
        doc = _assemble(words, pauses=[PauseInterval(0, 1000), PauseInterval(1850, 3000)])
        assert doc.utterances[0].children == ("one two three.",)

    def test_short_silence_is_not_a_pause(self) -> None:
        doc = _assemble(_sentence("one two."), pauses=[PauseInterval(250, 440)])
        assert doc.utterances[0].children == ("one two.",)

    def test_pause_before_punctuation_moves_after_it(self) -> None:
        words: list[Word] = [
            ("Wait", 0, 300, 120, 65), (",", 900, 900, 120, 65), ("what?", 900, 1200, 120, 65),
        ]
        doc = _assemble(words)
        assert doc.utterances[0].children == ("Wait,", Pause(duration=600), " what?")

    def test_silence_before_final_punctuation_is_trailing_silence(self) -> None:
        """With nothing after the '?', the gap is silence after the last word."""
        doc = _assemble([("Wait", 0, 300, 120, 65), ("?", 1500, 1500, 120, 65)])
        assert doc.utterances[0].children == ("Wait?",)

    def test_silence_over_a_minute_is_a_one_minute_pause(self, validator: IMLValidator) -> None:
        """Spec 6.4: producers SHOULD NOT emit pauses over 60000 ms (V33)."""
        words = [("Please", 0, 400, 120, 65), ("hold.", 450, 900, 120, 65),
                 ("Thanks", 71_440, 71_800, 120, 65), ("for", 71_850, 72_000, 120, 65),
                 ("waiting.", 72_050, 72_600, 120, 65)]
        with pytest.warns(UserWarning, match=r"The silence of 70\.5 s after 'hold\.' is "
                          r'written as <pause duration="60000"/>'):
            doc = _assemble(words)
        assert [u.children for u in doc.utterances] == [
            ("Please hold.",), (Pause(duration=MAX_PAUSE_MS), "Thanks for waiting."),
        ]
        result = validator.validate(_xml(doc))
        assert result.valid and not result.issues

    def test_several_long_silences_are_reported_once(self) -> None:
        words = [("one", 0, 300, 120, 65), ("two", 61_300, 61_600, 120, 65),
                 ("three", 161_600, 161_900, 120, 65), ("four", 162_000, 162_300, 120, 65)]
        assembly = IMLAssembler()._assemble(
            [_make_alignment(*w[:3]) for w in words], _features(words), []
        )
        pauses = [n for u in assembly.document.utterances for n in _nodes(u.children)
                  if isinstance(n, Pause)]
        assert pauses == [Pause(duration=MAX_PAUSE_MS)] * 2
        assert assembly.notes == [
            "2 silences longer than 60 s (the longest 100.0 s, after 'two') are written as "
            '<pause duration="60000"/>, the longest pause spec 6.4 recommends.'
        ]

    def test_one_minute_pause_is_kept(self) -> None:
        words = [("one", 0, 300, 120, 65), ("two.", 60_300, 60_600, 120, 65)]
        assembly = IMLAssembler()._assemble(
            [_make_alignment(*w[:3]) for w in words], _features(words), []
        )
        assert assembly.document.utterances[0].children == (
            "one", Pause(duration=60_000), " two."
        )
        assert assembly.notes == []

    def test_fractional_millisecond_timings(self, validator: IMLValidator) -> None:
        """Float timings used to give duration="730.5", which is invalid IML."""
        alignments = [
            WordAlignment("I", 0.0, 200.5),  # type: ignore[arg-type]
            WordAlignment("think", 200.5, 500.25),  # type: ignore[arg-type]
            WordAlignment("so,", 1230.75, 1500.0),  # type: ignore[arg-type]
            WordAlignment("yes.", 2000.0, 2300.0),  # type: ignore[arg-type]
        ]
        features = [
            _make_features(a.word, a.start_ms, a.end_ms, 120.0) for a in alignments
        ]
        pauses = [PauseInterval(1550.5, 1950.2)]  # type: ignore[arg-type]
        doc = IMLAssembler(include_extended=True).assemble(alignments, features, pauses)
        assert [n for n in doc.utterances[0].children if isinstance(n, Pause)] == [
            Pause(duration=730), Pause(duration=400),
        ]
        assert [n.duration_ms for n in _nodes(doc.utterances[0].children)
                if isinstance(n, Prosody)] == [200, 300, 269, 300]
        assert not validator.validate(_xml(doc)).issues


# ---------------------------------------------------------------------------
# Speaker baseline and utterance-level prosody
# ---------------------------------------------------------------------------


CALM = (
    _sentence("It was a long day.", 0, 180, 62)
    + _sentence("We went home early.", 2000, 178, 62)
    + _sentence("I'm fine.", 4000, 176, 62)
)
SHOUTED = _sentence("I said I'm fine!", 5000, 265, 80)


class TestSpeakerBaseline:
    def test_shouted_utterance_is_marked(self) -> None:
        """Each utterance used to be its own baseline, so shouting left no trace."""
        doc = _assemble(CALM + SHOUTED, assembler=IMLAssembler(min_emotion_confidence=0.0))
        # The median of the four utterances (179 Hz, 62 dB) is the baseline.
        assert [u.children for u in doc.utterances] == [
            ("It was a long day.",), (Pause(duration=550), "We went home early."),
            (Pause(duration=850), "I'm fine."),
            (
                Pause(duration=450),
                Prosody(children=("I said I'm fine!",), pitch="+48%", volume="+18dB"),
            ),
        ]
        assert [u.emotion for u in doc.utterances] == ["neutral"] * 3 + ["angry"]

    def test_long_emotional_utterance_does_not_become_the_baseline(self) -> None:
        """With more shouted words than calm ones, a word-weighted baseline flipped:
        the four calm sentences were marked lower and quieter, and sad."""
        calm = (
            _sentence("I am fine thanks.", 0) + _sentence("It was a long day.", 2000)
            + _sentence("We went home early.", 4000) + _sentence("Then I made dinner.", 6000)
        )
        shouted = _sentence(
            "I said I am absolutely fine and I do not need any help!", 8000, 160, 77
        )
        doc = _assemble(calm + shouted)
        assert [u.children[-1] for u in doc.utterances] == [
            "I am fine thanks.", "It was a long day.", "We went home early.",
            "Then I made dinner.",
            Prosody(
                children=("I said I am absolutely fine and I do not need any help!",),
                pitch="+33%", volume="+12dB",
            ),
        ]
        assert [u.emotion for u in doc.utterances[:4]] == [None] * 4

    def test_one_shouted_utterance_among_three(self) -> None:
        """Comparing each utterance with the median of the *others* would put the
        calm sentences half-way to the shouted one."""
        words = (
            _sentence("I'm fine.", 0, 176, 62) + _sentence("It was okay.", 1000, 178, 62)
            + _sentence("I said I'm fine!", 2500, 265, 80)
        )
        doc = _assemble(words)
        assert [u.children[-1] for u in doc.utterances] == [
            "I'm fine.", "It was okay.",
            Prosody(children=("I said I'm fine!",), pitch="+49%", volume="+18dB"),
        ]

    def test_reference_features_define_the_baseline(self) -> None:
        calm, shouted = _sentence("I'm fine.", 0, 176, 62), _sentence("I said so!", 1000, 265, 80)
        reference = _features(calm)
        doc = _assemble(calm + shouted, reference_features=reference)
        assert [u.children for u in doc.utterances] == [
            ("I'm fine.",),
            (Pause(duration=450), Prosody(children=("I said so!",), pitch="+51%", volume="+18dB")),
        ]

    def test_two_utterances_are_marked_from_their_midpoint(self) -> None:
        """Without a reference, two utterances cannot show which one is typical."""
        calm, shouted = _sentence("I'm fine.", 0, 176, 62), _sentence("I said so!", 1000, 265, 80)
        doc = _assemble(calm + shouted)
        wrappers = [u.children[-1] for u in doc.utterances]
        assert [(w.pitch, w.volume) for w in wrappers if isinstance(w, Prosody)] == [
            ("-20%", "-9dB"), ("+20%", "+9dB"),
        ]

    def test_single_utterance_has_no_utterance_level_offsets(self) -> None:
        doc = _assemble(_sentence("please call me back.", f0=265, db=80))
        assert _xml(doc) == (
            '<iml version="0.1.0"><utterance>please call me back.</utterance></iml>'
        )

    def test_one_word_utterance_is_one_element(self) -> None:
        rising = {"f0_contour": _contour(250, 330)}
        doc = _assemble(CALM + [("No!", 5000, 5400, 265, 80, rising)])
        assert doc.utterances[-1].children == (
            Pause(duration=450),
            Prosody(children=("No!",), pitch="+48%", volume="+18dB", pitch_contour="rise"),
        )

    @pytest.mark.parametrize(("fast_rate", "expected"), [(6.75, "150%"), (5.85, None)])
    def test_utterance_rate(self, fast_rate: float, expected: str | None) -> None:
        """Only clear changes are marked: measured rates at one tempo vary by 25 %."""
        fast = [(*w, {"speech_rate": fast_rate}) for w in _sentence("Hurry up now!", 4000)]
        steady = [
            (*w, {"speech_rate": 4.5}) for w in _sentence("It was a long day.") +
            _sentence("We went home early.", 2000)
        ]
        doc = _assemble(steady + fast)
        assert doc.utterances[-1].children[-1] == (
            "Hurry up now!" if expected is None
            else Prosody(children=("Hurry up now!",), rate=expected)
        )

    def test_implausible_rate_is_not_emitted(self) -> None:
        """One mis-measured sentence used to give rate="10%" and rate="3000%"."""
        slow = [
            (*w, {"speech_rate": 0.3}) for w in _sentence("It was a long day.") +
            _sentence("We went home early.", 2000)
        ]
        fast = [(*w, {"speech_rate": 9.0}) for w in _sentence("Hurry up now!", 4000)]
        doc = _assemble(slow + fast)
        assert "rate=" not in _xml(doc)

    def test_digital_silence_gives_no_absurd_volume(self, validator: IMLValidator) -> None:
        """Praat's -300 dB floor used to produce volume="-83dB" and +28dB elsewhere."""
        words = _sentence("one two three four.")
        words[1] = ("two", 300, 550, 120, -300.0)
        doc = _assemble(words, assembler=IMLAssembler(include_extended=True))
        two = next(n for n in _nodes(doc.utterances[0].children)
                   if isinstance(n, Prosody) and n.children == ("two",))
        assert two.volume is None and two.intensity_mean is None
        assert all(
            n.volume is None for n in _nodes(doc.utterances[0].children) if isinstance(n, Prosody)
        )
        assert not validator.validate(_xml(doc)).issues


# ---------------------------------------------------------------------------
# Prosody wrapping and emphasis
# ---------------------------------------------------------------------------


class TestProsodyWrapping:
    def test_high_f0_gets_prosody_tag(self, assembler: IMLAssembler) -> None:
        alignments = [_make_alignment(w, i * 300, i * 300 + 250) for i, w in enumerate(
            ["a", "normal", "then", "high", "word."]
        )]
        features = [
            _make_features(a.word, a.start_ms, a.end_ms, f0_mean=250 if a.word == "high" else 180)
            for a in alignments
        ]
        doc = assembler.assemble(alignments, features, pauses=[])
        assert _xml(doc) == (
            '<iml version="0.1.0"><utterance>a normal then <emphasis level="moderate">'
            '<prosody pitch="+39%">high</prosody></emphasis> word.</utterance></iml>'
        )

    def test_normal_features_no_prosody_tag(self, assembler: IMLAssembler) -> None:
        """Words near baseline should not get prosody wrapping."""
        doc = _assemble(_sentence("nothing to see here."))
        assert doc.utterances[0].children == ("nothing to see here.",)

    def test_quieter_word(self) -> None:
        words = _sentence("I really do not know.")
        words[2] = ("do", 600, 850, 120, 57.0)
        doc = _assemble(words)
        assert _xml(doc) == (
            '<iml version="0.1.0"><utterance>I really <prosody volume="-8dB">do</prosody>'
            " not know.</utterance></iml>"
        )


class TestEmphasis:
    def test_lowered_final_pitch_is_not_emphasis(self) -> None:
        """Final lowering used to be tagged <emphasis level="strong">."""
        doc = _assemble([
            ("I", 0, 300, 200, 70), ("went", 400, 700, 198, 70), ("home", 800, 1100, 195, 70),
            ("today.", 1200, 1500, 140, 70),
        ])
        assert _xml(doc) == (
            '<iml version="0.1.0"><utterance>I went home <prosody pitch="-29%">today.'
            "</prosody></utterance></iml>"
        )

    def test_word_is_not_its_own_context(self) -> None:
        """A +10 dB word used to sit only +5 dB above a mean that included it."""
        doc = _assemble([("No", 0, 300, 180, 65), ("way!", 400, 700, 180, 75)])
        assert isinstance(doc.utterances[0].children[-1], Emphasis)
        assert not isinstance(doc.utterances[0].children[0], Emphasis)

    def test_true_offsets_are_reported(self) -> None:
        """A +9 dB / +46 % word used to be reported as +7dB / +31%."""
        words = _sentence("she never said that.", f0=150, db=65)
        words[1] = ("never", 300, 550, 219, 74)
        doc = _assemble(words)
        emphasis = doc.utterances[0].children[1]
        assert emphasis == Emphasis(
            level="strong", children=(Prosody(children=("never",), pitch="+46%", volume="+9dB"),)
        )

    @pytest.mark.parametrize(
        ("louder_db", "level"), [(4.0, None), (6.0, "moderate"), (12.0, "strong")]
    )
    def test_levels(self, louder_db: float, level: str | None) -> None:
        words = _sentence("I want the red one.")
        words[3] = ("red", 900, 1150, 120, 65 + louder_db)
        doc = _assemble(words)
        levels = [n.level for n in _nodes(doc.utterances[0].children) if isinstance(n, Emphasis)]
        assert levels == ([level] if level else [])

    def test_pitch_tracking_error_is_not_emphasis(self) -> None:
        """An F0 more than an octave above the neighbours is an octave jump."""
        words = _sentence("I miss my old friends.", f0=77)
        words[1] = ("miss", 300, 550, 242, 65)
        doc = _assemble(words)
        assert doc.utterances[0].children == ("I miss my old friends.",)

    def test_bare_emphasis_stays_inside_utterance_prosody(self) -> None:
        loud = _sentence("I said no now.", 5000, 265, 80)
        loud[2] = ("no", 5600, 5850, 297.4, 84)  # +2 st, +4 dB: emphasis, but no offsets
        doc = _assemble(CALM + loud)
        assert doc.utterances[-1].children[-1] == Prosody(
            children=("I said ", Emphasis(level="moderate", children=("no",)), " now."),
            pitch="+48%",
            volume="+18dB",
        )

    def test_final_contour_of_emphasized_word_is_kept(self, validator: IMLValidator) -> None:
        """The rise of a question on an emphasized last word cannot go inside the
        utterance's <prosody> (<emphasis><prosody> would be three deep). The word
        stands alone with exactly the utterance's attributes plus its contour, so
        the pieces still describe one delivery; its own offset used to be added
        to the utterance's (+96%, next to +48%)."""
        question = _sentence("You did what?", 5000, 265, 80)
        question[2] = ("what?", 5600, 5850, 350, 88, {"f0_contour": _contour(280, 420)})
        doc = _assemble(CALM + question, assembler=IMLAssembler(min_emotion_confidence=1.0))
        assert doc.utterances[-1].children == (
            Pause(duration=450),
            Prosody(children=("You did",), pitch="+48%", volume="+18dB"),
            " ",
            Emphasis(level="strong", children=(Prosody(
                children=("what?",), pitch="+48%", volume="+18dB", pitch_contour="rise-sharp",
            ),)),
        )
        assert not [i for i in validator.validate(_xml(doc)).issues if i.severity != "info"]

    def test_emphasized_word_stays_inside_the_utterance_prosody(
        self, validator: IMLValidator
    ) -> None:
        """An emphasized word louder than its shouted utterance used to step out
        of the utterance's <prosody> and split it in three, so readers saw the
        utterance's delivery three times, piecewise, and no overall delivery."""
        loud = _sentence("I said no way.", 5000, 265, 80)
        loud[2] = ("no", 5600, 5850, 265, 92, {"f0_contour": _contour(250, 300)})
        doc = _assemble(CALM + loud)
        assert doc.utterances[-1].children[1:] == (
            Prosody(
                children=("I said ", Emphasis(level="strong", children=("no",)), " way."),
                pitch="+48%",
                volume="+18dB",
            ),
        )
        assert _plain(doc).endswith("I said no way.")
        assert not [i for i in validator.validate(_xml(doc)).issues if i.severity != "info"]


# ---------------------------------------------------------------------------
# Pitch contour
# ---------------------------------------------------------------------------


def _contour(start_hz: float, end_hz: float, samples: int = 20) -> list[float]:
    return [start_hz * (end_hz / start_hz) ** (i / (samples - 1)) for i in range(samples)]


def _rise_fall(low: float, high: float, samples: int = 21) -> list[float]:
    half = samples // 2
    return _contour(low, high, half + 1)[:-1] + _contour(high, low, samples - half)


class TestPitchContour:
    @pytest.mark.parametrize(
        ("contour", "end_ms", "expected"),
        [
            (_contour(120, 240), 1000, "rise"),
            (_contour(240, 120), 1000, "fall"),
            (_contour(120, 240), 300, "rise-sharp"),
            (_contour(240, 120), 300, "fall-sharp"),
            (_rise_fall(120, 180), 600, "rise-fall"),
            (_rise_fall(180, 120), 600, "fall-rise"),
        ],
    )
    def test_final_word_contour(self, contour: list[float], end_ms: int, expected: str) -> None:
        doc = _assemble([("Really?", 0, end_ms, 170, 65, {"f0_contour": contour})])
        assert _xml(doc) == (
            f'<iml version="0.1.0"><utterance><prosody pitch_contour="{expected}">Really?'
            "</prosody></utterance></iml>"
        )

    @pytest.mark.parametrize(
        "contour",
        [
            [150.0 + (i % 2) for i in range(20)],  # flat: the default, not written
            _contour(120, 240, samples=5),  # too few voiced samples to judge
        ],
        ids=["flat", "few-samples"],
    )
    def test_no_contour(self, contour: list[float]) -> None:
        doc = _assemble([("Really?", 0, 1000, 150, 65, {"f0_contour": contour})])
        assert doc.utterances[0].children == ("Really?",)

    def test_octave_errors_are_ignored(self) -> None:
        contour = _contour(120, 240)
        contour[2] = contour[2] * 2  # the tracker jumped an octave
        contour[15] = contour[15] / 2
        doc = _assemble([("Really?", 0, 1000, 170, 65, {"f0_contour": contour})])
        assert _xml(doc).endswith(
            '<prosody pitch_contour="rise">Really?</prosody></utterance></iml>'
        )

    def test_only_final_emphasized_or_marked_words_get_contours(self) -> None:
        rising = {"f0_contour": _contour(110, 150)}
        words: list[Word] = [
            ("so", 0, 250, 120, 65, rising), ("you", 300, 550, 120, 65, rising),
            ("came", 600, 850, 120, 65, rising), ("back?", 900, 1150, 120, 65, rising),
        ]
        doc = _assemble(words)
        assert doc.utterances[0].children == (
            "so you came ", Prosody(children=("back?",), pitch_contour="rise"),
        )

    def test_rising_and_falling_audio_differ(self, tmp_path: Path) -> None:
        """Rising and falling speech used to give identical IML."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("parselmouth")
        from prosody_protocol.prosody_analyzer import ProsodyAnalyzer

        rate = 16_000
        t = np.arange(rate) / rate
        results = {}
        for name, f0 in [("rise", 120 + 120 * t), ("fall", 240 - 120 * t)]:
            phase = 2 * np.pi * np.cumsum(f0) / rate
            signal = sum(np.sin(k * phase) / k for k in range(1, 11))
            path = tmp_path / f"{name}.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes((0.3 * signal / np.abs(signal).max() * 32767).astype("<i2"))
            alignments = [_make_alignment("Really?", 0, 1000)]
            features = ProsodyAnalyzer().analyze(path, alignments)
            results[name] = _xml(IMLAssembler().assemble(alignments, features, []))
        assert 'pitch_contour="rise"' in results["rise"]
        assert 'pitch_contour="fall"' in results["fall"]


# ---------------------------------------------------------------------------
# Voice quality
# ---------------------------------------------------------------------------


def _voice(quality: str, hnr: float = 18.0, jitter: float = 0.8, shimmer: float = 4.0) -> dict[
    str, object
]:
    return {"quality": quality, "hnr": hnr, "jitter": jitter, "shimmer": shimmer}


class TestQuality:
    def test_breathy_word(self) -> None:
        words: list[Word] = [(*w, _voice("modal")) for w in _sentence("I really do care.")]
        words[2] = ("do", 600, 850, 120, 65, _voice("breathy", hnr=6.0))
        doc = _assemble(words)
        assert doc.utterances[0].children == (
            "I really ", Prosody(children=("do",), quality="breathy"), " care.",
        )

    def test_label_typical_of_the_speaker_is_not_emitted(self) -> None:
        """The analyzer's absolute thresholds call many ordinary voices "tense"."""
        words: list[Word] = [
            (*w, _voice("tense", shimmer=14.0)) for w in _sentence("this is how I talk.")
        ]
        words[1] = ("is", 300, 550, 120, 65, _voice("tense", shimmer=16.0))
        doc = _assemble(words)
        assert doc.utterances[0].children == ("this is how I talk.",)

    def test_whole_utterance_quality_is_stated_once(self) -> None:
        steady = [
            (*w, _voice("modal"))
            for w in _sentence("It was a long day.") + _sentence("We went home.", 2000)
        ]
        creaky = [
            (*w, _voice("creaky", jitter=2.5)) for w in _sentence("Whatever you say.", 4000)
        ]
        doc = _assemble(steady + creaky)
        assert doc.utterances[2].children == (
            Pause(duration=1150),
            Prosody(children=("Whatever you say.",), quality="creaky"),
        )


# ---------------------------------------------------------------------------
# Extended attributes
# ---------------------------------------------------------------------------


RESEARCH = {
    "f0_range": (101.4, 138.9),
    "f0_contour": [101.4 + i * 2.5 for i in range(16)],
    "intensity_range": 12.34,
    "speech_rate": 4.21,
    "jitter": 1.234,
    "shimmer": 3.876,
    "hnr": 17.66,
}


class TestExtendedAttrs:
    def test_extended_attrs_on_every_word(
        self, assembler_extended: IMLAssembler, validator: IMLValidator
    ) -> None:
        """Steady speech used to get no measurements at all."""
        words: list[Word] = [(*w, RESEARCH) for w in _sentence("so steady.")]
        doc = _assemble(words, assembler=assembler_extended)
        xml = _xml(doc)
        assert xml == (
            '<iml version="0.1.0"><utterance><prosody f0_mean="120.0" f0_range="101-139" '
            'f0_contour="103,106,110,114,118,123,126,130,134,138" intensity_mean="65.0" '
            'intensity_range="12.3" speech_rate="4.2" duration_ms="250" jitter="1.23" '
            'shimmer="3.88" hnr="17.7">so</prosody> <prosody pitch_contour="rise" '
            'f0_mean="120.0" f0_range="101-139" '
            'f0_contour="103,106,110,114,118,123,126,130,134,138" '
            'intensity_mean="65.0" intensity_range="12.3" speech_rate="4.2" duration_ms="250" '
            'jitter="1.23" shimmer="3.88" hnr="17.7">steady.</prosody></utterance></iml>'
        )
        assert validator.validate(xml).valid

    def test_jitter_and_shimmer_stay_in_percent(self, assembler_extended: IMLAssembler) -> None:
        """Spec 4.4 uses percent; SpanFeatures already carries percent."""
        doc = _assemble([("loud.", 0, 500, 250, 80, RESEARCH)], assembler=assembler_extended)
        prosody = doc.utterances[0].children[0]
        assert isinstance(prosody, Prosody)
        assert (prosody.jitter, prosody.shimmer) == (1.23, 3.88)

    def test_emphasized_word_keeps_core_and_extended(
        self, assembler_extended: IMLAssembler
    ) -> None:
        words: list[Word] = [(*w, RESEARCH) for w in _sentence("I told you.", f0=120, db=65)]
        words[1] = ("told", 300, 550, 170, 72, RESEARCH)
        doc = _assemble(words, assembler=assembler_extended)
        emphasis = doc.utterances[0].children[2]
        assert isinstance(emphasis, Emphasis) and emphasis.level == "strong"
        inner = emphasis.children[0]
        assert isinstance(inner, Prosody)
        assert (inner.pitch, inner.volume, inner.f0_mean, inner.intensity_mean) == (
            "+42%", "+7dB", 170.0, 72.0,
        )

    def test_extended_attributes_do_not_change_the_markup(
        self, assembler_extended: IMLAssembler
    ) -> None:
        """With measurements, an emphasized word in a shouted utterance used to
        step out of the utterance's <prosody> (to hold them in a <prosody> of
        its own), which split the utterance-level delivery into pieces: the
        text a reader saw changed with include_extended. Now the markup is the
        same, with the measurements added; the emphasized word inside the
        utterance's <prosody> goes without them (they would nest three deep)."""
        loud: list[Word] = [(*w, RESEARCH) for w in _sentence("I said no now.", 5000, 265, 80)]
        loud[2] = ("no", 5600, 5850, 265, 92, RESEARCH)
        extended = _assemble(CALM + loud, assembler=assembler_extended)
        plain = _assemble(CALM + loud)
        assert [_without_extended(u.children) for u in extended.utterances] == [
            list(u.children) for u in plain.utterances
        ]
        wrapper = extended.utterances[-1].children[-1]
        assert isinstance(wrapper, Prosody)
        assert (wrapper.pitch, wrapper.volume) == ("+48%", "+18dB")
        assert Emphasis(level="strong", children=("no",)) in wrapper.children
        said = wrapper.children[0]
        assert isinstance(said, Prosody) and said.jitter == 1.23

    def test_no_duration_for_a_span_over_an_hour(self) -> None:
        """A word timing ending days later became duration_ms="2999999386",
        invalid IML (V27); a span over an hour is not a word."""
        words: list[Word] = [("never", 614, 3_000_000_000, 120, 65)]
        doc = _assemble(words, assembler=IMLAssembler(include_extended=True))
        assert "duration_ms" not in _xml(doc)
        assert IMLValidator().validate(_xml(doc)).valid

    def test_no_extended_by_default(self, assembler: IMLAssembler) -> None:
        words: list[Word] = [(*w, RESEARCH) for w in _sentence("so steady.")]
        assert "f0_mean" not in _xml(_assemble(words, assembler=assembler))


# ---------------------------------------------------------------------------
# Emotion
# ---------------------------------------------------------------------------


class _Fixed:
    """A plain EmotionClassifier: classify(features) only."""

    def __init__(self, emotion: str, confidence: float) -> None:
        self.result = (emotion, confidence)
        self.calls: list[list[SpanFeatures]] = []

    def classify(self, features: list[SpanFeatures]) -> tuple[str, float]:
        self.calls.append(features)
        return self.result


class _BaselineRecorder(_Fixed):
    def __init__(self) -> None:
        super().__init__("calm", 0.8)
        self.baselines: list[SpeakerBaseline] = []

    def classify_relative(
        self, features: Sequence[SpanFeatures], baseline: SpeakerBaseline
    ) -> tuple[str, float]:
        self.baselines.append(baseline)
        return self.result


class TestEmotion:
    def test_default_threshold(self) -> None:
        assert DEFAULT_MIN_EMOTION_CONFIDENCE == 0.5

    @pytest.mark.parametrize(
        ("threshold", "expected"), [(0.5, (None, None)), (0.4, ("sad", 0.45)), (0.0, ("sad", 0.45))]
    )
    def test_low_confidence_emotion_is_dropped(
        self, threshold: float, expected: tuple[str | None, float | None]
    ) -> None:
        assembler = IMLAssembler(_Fixed("sad", 0.45), min_emotion_confidence=threshold)
        doc = _assemble(_sentence("I see."), assembler=assembler)
        utterance = doc.utterances[0]
        assert (utterance.emotion, utterance.confidence) == expected

    def test_plain_classifier_still_works(self) -> None:
        classifier = _Fixed("frustrated", 0.9)
        doc = _assemble(_sentence("Not again."), assembler=IMLAssembler(classifier))
        assert (doc.utterances[0].emotion, doc.utterances[0].confidence) == ("frustrated", 0.9)
        assert [f.text for f in classifier.calls[0]] == ["Not", "again."]

    def test_baseline_aware_classifier_gets_the_speaker_baseline(self) -> None:
        classifier = _BaselineRecorder()
        _assemble(CALM + SHOUTED, assembler=IMLAssembler(classifier))
        assert classifier.calls == []
        # The median of the utterances' medians: 176, 178, 180 and 265 Hz; 62 and 80 dB.
        assert {(b.f0_mean, b.intensity_mean) for b in classifier.baselines} == {(179.0, 62.0)}

    def test_single_utterance_is_not_judged_against_itself(self) -> None:
        doc = _assemble(SHOUTED, assembler=IMLAssembler(min_emotion_confidence=0.0))
        assert (doc.utterances[0].emotion, doc.utterances[0].confidence) == ("neutral", 0.0)

    def test_two_utterances_get_no_emotion(self) -> None:
        """Judged only against each other, the calm one of two utterances used to
        get emotion="sad" confidence="0.71"."""
        words = _sentence("I'm fine.", 0, 176, 62) + _sentence("I said I'm fine!", 1000, 265, 80)
        doc = _assemble(words)
        assert [(u.emotion, u.confidence) for u in doc.utterances] == [(None, None)] * 2

        classifier = _BaselineRecorder()
        _assemble(words, assembler=IMLAssembler(classifier))
        assert classifier.baselines == [SpeakerBaseline()] * 2

    def test_recording_without_a_typical_level_gets_no_emotion(self) -> None:
        """Two calm and two shouted sentences: which level is the speaker's usual one?"""
        words = (
            _sentence("I'm fine.", 0, 176, 62) + _sentence("I said so!", 1000, 265, 80)
            + _sentence("It was okay.", 2500, 178, 62) + _sentence("Leave me alone!", 4000, 262, 80)
        )
        doc = _assemble(words, assembler=IMLAssembler(min_emotion_confidence=0.0))
        assert [(u.emotion, u.confidence) for u in doc.utterances] == [("neutral", 0.0)] * 4

    def test_unmeasured_utterances_do_not_establish_a_baseline(self) -> None:
        words = _sentence("I'm fine.", 0, 176, 62) + _sentence("I said so!", 1000, 265, 80)
        unmeasured = _sentence("Music plays. Doors close.", 2500)
        alignments = [_make_alignment(w[0], w[1], w[2]) for w in words + unmeasured]
        classifier = _BaselineRecorder()
        IMLAssembler(classifier).assemble(alignments, _features(words), [])
        assert classifier.baselines == [SpeakerBaseline()] * 4

    def test_calibration_makes_two_utterances_comparable(self) -> None:
        calm = _sentence("I'm fine.", 0, 176, 62)
        reference = _features(calm)
        classifier = _BaselineRecorder()
        _assemble(
            calm + _sentence("I said so!", 1000, 265, 80),
            assembler=IMLAssembler(classifier),
            reference_features=reference,
        )
        assert [(b.f0_mean, b.intensity_mean) for b in classifier.baselines] == [(176.0, 62.0)] * 2

    def test_emotion_always_comes_with_confidence(self, validator: IMLValidator) -> None:
        for threshold in (0.0, 0.5, 1.0):
            assembler = IMLAssembler(min_emotion_confidence=threshold)
            doc = _assemble(CALM + SHOUTED, assembler=assembler)
            for utterance in doc.utterances:
                assert (utterance.emotion is None) == (utterance.confidence is None)
                assert utterance.confidence is None or 0.0 <= utterance.confidence <= 1.0
            assert validator.validate(_xml(doc)).valid

    @pytest.mark.parametrize("threshold", [-0.1, 1.5, math.nan])
    def test_invalid_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="min_emotion_confidence"):
            IMLAssembler(min_emotion_confidence=threshold)


class TestDeprecatedConstants:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("F0_DEVIATION_PCT", 15.0), ("INTENSITY_DEVIATION_DB", 5.0),
            ("EMPHASIS_INTENSITY_DB", 6.0), ("EMPHASIS_F0_PCT", 20.0),
        ],
    )
    def test_old_thresholds_still_import(self, name: str, value: float) -> None:
        from prosody_protocol import assembler

        with pytest.warns(DeprecationWarning, match=name):
            assert getattr(assembler, name) == value

    def test_from_import(self) -> None:
        """Removing the names made this line raise ImportError."""
        with pytest.warns(DeprecationWarning, match="no longer used"):
            from prosody_protocol.assembler import F0_DEVIATION_PCT
        assert F0_DEVIATION_PCT == 15.0

    def test_unknown_name(self) -> None:
        from prosody_protocol import assembler

        with pytest.raises(AttributeError, match="NO_SUCH_THRESHOLD"):
            assembler.NO_SUCH_THRESHOLD  # noqa: B018


# ---------------------------------------------------------------------------
# Prosody profiles (spec Section 7)
# ---------------------------------------------------------------------------


def _profile(
    *mappings: tuple[dict[str, str], str, float], user_id: str = "user_1"
) -> ProsodyProfile:
    return ProsodyProfile(
        profile_version="0.1.0",
        user_id=user_id,
        description=None,
        mappings=tuple(ProsodyMapping(dict(p), emotion, boost) for p, emotion, boost in mappings),
    )


SPIKE_PROFILE = _profile(({"volume": "spike"}, "sincere", 0.2))


def _spiked(text: str = "I did not say that.", louder: int = 2) -> list[Word]:
    """Evenly spoken words, one of them 13 dB louder (a volume "spike")."""
    words = _sentence(text)
    word = words[louder]
    words[louder] = (word[0], word[1], word[2], word[3], 78.0)
    return words


def _flat_fast(text: str, start_ms: int = 0) -> list[Word]:
    """Monotone words (flat F0 contours) at 7 syllables per second."""
    return [
        (w[0], w[1], w[2], 120.0, 65.0, {"f0_contour": [120.0] * 10, "speech_rate": 7.0})
        for w in _sentence(text, start_ms)
    ]


class TestProfiles:
    def test_matching_mapping_sets_emotion_and_boosts_confidence(
        self, validator: IMLValidator
    ) -> None:
        """Spec 7.2: the mapping takes precedence, confidence = classifier's + boost."""
        without = _assemble(_spiked(), assembler=IMLAssembler(_Fixed("angry", 0.6)))
        assert (without.utterances[0].emotion, without.utterances[0].confidence) == ("angry", 0.6)

        assembler = IMLAssembler(_Fixed("angry", 0.6), profile=SPIKE_PROFILE)
        doc = _assemble(_spiked(), assembler=assembler)
        utterance = doc.utterances[0]
        assert (utterance.emotion, utterance.confidence) == ("sincere", pytest.approx(0.8))
        assert utterance.extra_attributes == ((PROFILE_ATTRIBUTE, "volume=spike"),)
        xml = _xml(doc)
        assert 'emotion="sincere"' in xml and 'x-profile="volume=spike"' in xml
        result = validator.validate(xml)
        assert result.valid and not result.issues

    def test_profile_marker_does_not_name_the_speaker(self) -> None:
        """x-profile holds the matched pattern (sorted by key), never the user_id."""
        profile = _profile(
            ({"volume": "spike", "pitch_contour": "flat"}, "sincere", 0.2),
            user_id="jane.doe@example.com",
        )
        words = [(*w[:5], {"f0_contour": [120.0] * 10}) for w in _spiked()]
        doc = _assemble(words, assembler=IMLAssembler(_Fixed("angry", 0.6), profile=profile))
        assert doc.utterances[0].extra_attributes == (
            (PROFILE_ATTRIBUTE, "pitch_contour=flat volume=spike"),
        )
        assert "jane.doe" not in _xml(doc)

    def test_matches_are_reported(self) -> None:
        assembler = IMLAssembler(_Fixed("angry", 0.6), profile=SPIKE_PROFILE)
        words = _sentence("It was fine.") + _spiked("Then you said that.", 3)
        words = words[:3] + [(w[0], w[1] + 2000, w[2] + 2000, *w[3:]) for w in words[3:]]
        alignments = [_make_alignment(w[0], w[1], w[2]) for w in words]
        features = [
            _make_features(w[0], w[1], w[2], w[3], w[4], **(w[5] if len(w) > 5 else {}))
            for w in words
        ]
        doc, matches, *_ = assembler._assemble(alignments, features, [])
        assert len(doc.utterances) == 2
        assert matches == [ProfileMatch(
            utterance=1,
            observed=matches[0].observed,
            pattern={"volume": "spike"},
            emotion="sincere",
            confidence=pytest.approx(0.8),
            applied=True,
        )]
        assert matches[0].observed["volume"] == "spike"
        assert doc.utterances[0].extra_attributes == ()
        assert doc.utterances[0].emotion == "angry"

    def test_confidence_is_capped_at_one(self) -> None:
        assembler = IMLAssembler(_Fixed("angry", 0.9), profile=SPIKE_PROFILE)
        assert _assemble(_spiked(), assembler=assembler).utterances[0].confidence == 1.0

    def test_abstention_applies_after_the_profile(self) -> None:
        """A profile adjusts the classifier's estimate; it cannot make up confidence."""
        assembler = IMLAssembler(_Fixed("neutral", 0.1), profile=SPIKE_PROFILE)
        doc, matches, *_ = assembler._assemble(
            [_make_alignment(w[0], w[1], w[2]) for w in _spiked()], _features(_spiked()), []
        )
        utterance = doc.utterances[0]
        assert (utterance.emotion, utterance.confidence, utterance.extra_attributes) == (
            None, None, ()
        )
        assert [(m.emotion, m.confidence, m.applied) for m in matches] == [
            ("sincere", pytest.approx(0.3), False)
        ]
        assert "x-profile" not in _xml(doc)

        lenient = IMLAssembler(
            _Fixed("neutral", 0.1), profile=SPIKE_PROFILE, min_emotion_confidence=0.25
        )
        utterance = _assemble(_spiked(), assembler=lenient).utterances[0]
        assert (utterance.emotion, utterance.confidence) == ("sincere", pytest.approx(0.3))

    def test_no_match_leaves_the_classification(self) -> None:
        assembler = IMLAssembler(_Fixed("angry", 0.6), profile=SPIKE_PROFILE)
        doc, matches, *_ = assembler._assemble(
            [_make_alignment(w[0], w[1], w[2]) for w in _sentence("I see.")],
            _features(_sentence("I see.")),
            [],
        )
        assert (doc.utterances[0].emotion, doc.utterances[0].confidence) == ("angry", 0.6)
        assert doc.utterances[0].extra_attributes == ()
        assert matches == []

    def test_most_specific_mapping_wins(self) -> None:
        profile = _profile(
            ({"volume": "spike"}, "sincere", 0.1),
            ({"volume": "spike", "rate": "fast"}, "joyful", 0.1),
            ({"rate": "fast", "volume": "spike"}, "surprised", 0.1),
        )
        words = [(*w[:5], {"speech_rate": 7.0}) for w in _spiked()]
        doc = _assemble(words, assembler=IMLAssembler(_Fixed("angry", 0.6), profile=profile))
        # Two keys beat one; of the two with two keys, the first listed wins.
        assert doc.utterances[0].emotion == "joyful"

    def test_fixture_profile_flat_and_fast_is_excitement(self) -> None:
        """The spec 7.1 example: monotone, fast speech from this speaker is excitement."""
        profile = ProfileLoader().load(PROFILES_DIR / "autism_spectrum.json")
        words = _flat_fast("I just got the new keyboard.")
        doc, matches, *_ = IMLAssembler(_Fixed("angry", 0.63), profile=profile)._assemble(
            [_make_alignment(w[0], w[1], w[2]) for w in words],
            [_make_features(w[0], w[1], w[2], w[3], w[4], **w[5]) for w in words],  # type: ignore[misc]
            [],
        )
        utterance = doc.utterances[0]
        assert (utterance.emotion, utterance.confidence) == ("excitement", pytest.approx(0.78))
        assert utterance.extra_attributes == (("x-profile", "pitch_contour=flat rate=fast"),)
        assert matches[0].observed["pitch_contour"] == "flat"
        assert matches[0].observed["rate"] == "fast"

    def test_mapping_is_chosen_by_profile_applier_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The assembler uses the public matching rule, not a copy of it."""
        from prosody_protocol.profiles import ProfileApplier

        calls: list[dict[str, str]] = []
        original = ProfileApplier.match

        def spy(
            self: ProfileApplier, profile: ProsodyProfile, features: Mapping[str, str]
        ) -> ProsodyMapping | None:
            calls.append(dict(features))
            return original(self, profile, features)

        monkeypatch.setattr(ProfileApplier, "match", spy)
        doc = _assemble(
            _spiked(), assembler=IMLAssembler(_Fixed("angry", 0.6), profile=SPIKE_PROFILE)
        )
        assert doc.utterances[0].emotion == "sincere"
        assert calls and calls[0]["volume"] == "spike"

        # Whatever match() decides is what the assembler applies.
        monkeypatch.setattr(ProfileApplier, "match", lambda self, profile, features: None)
        doc = _assemble(
            _spiked(), assembler=IMLAssembler(_Fixed("angry", 0.6), profile=SPIKE_PROFILE)
        )
        assert (doc.utterances[0].emotion, doc.utterances[0].extra_attributes) == ("angry", ())

    def test_profile_property(self) -> None:
        assert IMLAssembler(profile=SPIKE_PROFILE).profile is SPIKE_PROFILE
        assert IMLAssembler().profile is None

    def test_pitch_level_needs_a_baseline(self) -> None:
        """"pitch" is only judged against the speaker's own speech (reference
        features, or the recording's typical utterances)."""
        profile = _profile(({"pitch": "high"}, "surprised", 0.2))
        high = _sentence("Is that so?", 0, 150, 65)
        alone = _assemble(high, assembler=IMLAssembler(_Fixed("neutral", 0.4), profile=profile))
        assert alone.utterances[0].emotion is None

        calibrated = _assemble(
            high,
            assembler=IMLAssembler(_Fixed("neutral", 0.4), profile=profile),
            reference_features=_features(_sentence("This is my voice.", 0, 120, 65)),
        )
        assert (calibrated.utterances[0].emotion, calibrated.utterances[0].confidence) == (
            "surprised", pytest.approx(0.6)
        )

        doc = _assemble(
            CALM + SHOUTED, assembler=IMLAssembler(_Fixed("neutral", 0.4), profile=profile)
        )
        assert [u.emotion for u in doc.utterances] == [None, None, None, "surprised"]

    def test_profile_with_unclassifiable_confidence(self) -> None:
        assembler = IMLAssembler(
            _Fixed("angry", math.nan), profile=SPIKE_PROFILE, min_emotion_confidence=0.0
        )
        utterance = _assemble(_spiked(), assembler=assembler).utterances[0]
        assert (utterance.emotion, utterance.confidence) == ("sincere", pytest.approx(0.2))
        plain = IMLAssembler(_Fixed("angry", math.nan), min_emotion_confidence=0.0)
        assert _assemble(_spiked(), assembler=plain).utterances[0].emotion is None

    @pytest.mark.parametrize(
        ("fixture", "rule"), [("invalid_bad_version", "P1"), ("invalid_empty_mappings", "P3")]
    )
    def test_invalid_fixture_profiles_rejected(self, fixture: str, rule: str) -> None:
        """ProfileLoader.load reads these; the assembler refuses to apply them."""
        profile = ProfileLoader().load(PROFILES_DIR / f"{fixture}.json")
        with pytest.raises(ProfileError, match=rule):
            IMLAssembler(profile=profile)

    def test_minimal_fixture_profile_needs_a_baseline(self) -> None:
        """Its one mapping is pitch=high, which is judged against a baseline only."""
        profile = ProfileLoader().load(PROFILES_DIR / "minimal_valid.json")
        high = _sentence("Is that so?", 0, 150, 65)
        doc = _assemble(
            high,
            assembler=IMLAssembler(_Fixed("neutral", 0.45), profile=profile),
            reference_features=_features(_sentence("This is my voice.", 0, 120, 65)),
        )
        assert (doc.utterances[0].emotion, doc.utterances[0].confidence) == ("excited", 0.55)

    def test_invalid_profile_rejected(self) -> None:
        with pytest.raises(ProfileError, match="P5"):
            IMLAssembler(profile=_profile(({"loudness": "high"}, "angry", 0.1)))
        with pytest.raises(ProfileError, match="P8"):
            IMLAssembler(profile=_profile(({"volume": "loud"}, "angry", 1.5)))
        with pytest.raises(ProfileError, match=r"prosody_mappings\[1\].*XML does not allow"):
            IMLAssembler(profile=_profile(
                ({"volume": "loud"}, "angry", 0.1), ({"volume": "spike"}, "calm\x01", 0.1)
            ))
        # The user_id is never written to the IML, so any string will do.
        IMLAssembler(profile=_profile(({"volume": "loud"}, "angry", 0.1), user_id="a\x01b"))
        with pytest.raises(TypeError, match="ProsodyProfile"):
            IMLAssembler(profile={"user_id": "x"})  # type: ignore[arg-type]

    def test_shares_the_parsers_xml_character_pattern(self) -> None:
        """One definition of the characters XML does not allow, not a copy."""
        from prosody_protocol import assembler, parser

        assert vars(assembler)["_XML_INVALID_CHAR_RE"] is parser._XML_INVALID_CHAR_RE


# ---------------------------------------------------------------------------
# Validation of output
# ---------------------------------------------------------------------------


class TestOutputValidation:
    def test_assembled_docs_are_valid_without_warnings(self, validator: IMLValidator) -> None:
        """Every document built in these scenarios is valid and within depth 2."""
        loud = _sentence("I said no.", 5000, 265, 80)
        loud[2] = ("no.", 5600, 5850, 265, 92, RESEARCH)
        scenarios = [
            _assemble(CALM + SHOUTED),
            _assemble(CALM + loud, assembler=IMLAssembler(include_extended=True)),
            _assemble(_sentence("Are you okay?") + [("Yes.", 3850, 4100, 120, 65)]),
        ]
        for doc in scenarios:
            result = validator.validate(_xml(doc))
            assert result.valid
            assert not [i for i in result.issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# Package root and entry point (prosody_protocol/__init__.py, __main__.py)
# ---------------------------------------------------------------------------


def test_profile_match_is_exported_from_the_package() -> None:
    import prosody_protocol

    assert prosody_protocol.ProfileMatch is ProfileMatch
    assert "ProfileMatch" in prosody_protocol.__all__
    # Core (lxml only): imported eagerly, not through the lazy loader.
    assert "ProfileMatch" not in prosody_protocol._LAZY
    assert "ProfileMatch" in vars(prosody_protocol)


def test_python_dash_m_runs_the_cli() -> None:
    import subprocess
    import sys

    from prosody_protocol import __version__

    version = subprocess.run(
        [sys.executable, "-m", "prosody_protocol", "--version"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert (version.returncode, version.stdout.strip()) == (0, f"prosody-protocol {__version__}")
    validate = subprocess.run(
        [sys.executable, "-m", "prosody_protocol", "validate", "-"],
        input='<utterance emotion="calm">Hi.</utterance>',
        capture_output=True, text=True, check=False, timeout=60,
    )
    # Exit 1: an invalid document (emotion without confidence, spec 3.1).
    assert validate.returncode == 1
    assert "invalid" in validate.stdout


def test_importing_main_module_does_not_run_the_cli() -> None:
    import importlib

    module = importlib.import_module("prosody_protocol.__main__")
    assert callable(module.main)
