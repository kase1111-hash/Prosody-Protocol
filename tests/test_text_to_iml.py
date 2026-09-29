"""Tests for prosody_protocol.text_to_iml (Phase 6a -- rule-based baseline).

Covers:
- Acceptance criteria from the execution guide
- Exact text preservation (corpus of varied inputs) and valid output
- ALL CAPS -> <emphasis level="strong">, but not acronyms
- ? -> pitch_contour="rise", ?! -> "rise-sharp"
- ! -> pitch="+5%" volume="+3dB" (not an emotion by itself)
- ... -> <pause duration="500"/>
- Quoted speech -> separate <utterance>, in place
- Abbreviation-aware sentence splitting
- Emotion cues, negation, sarcasm, honest confidence, context
- Edge cases: empty input, control characters, multi-sentence
- predict_document round-trip
- Constructor validation; unsupported model raises NotImplementedError
"""

from __future__ import annotations

import re
import time

import pytest
from lxml import etree

from prosody_protocol.models import Emphasis, Pause, Prosody
from prosody_protocol.parser import IMLParser
from prosody_protocol.text_to_iml import TextToIML
from prosody_protocol.validator import IMLValidator


@pytest.fixture()
def predictor() -> TextToIML:
    return TextToIML()


@pytest.fixture()
def validator() -> IMLValidator:
    return IMLValidator()


def _normalized(text: str) -> str:
    """The text an IML document for ``text`` carries (module docstring)."""
    text = re.sub("[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    return re.sub(r"[ \t\r\n]+", " ", text).strip(" ")


def _utterance_texts(xml: str) -> list[str]:
    root = etree.fromstring(xml.encode("utf-8"))
    return ["".join(u.itertext()) for u in root.iter("utterance")]


# ---------------------------------------------------------------------------
# Acceptance criteria (from EXECUTION_GUIDE.md Phase 6.3)
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """Direct tests for the acceptance criteria listed in Phase 6.3."""

    def test_great_produces_emphasis(self, predictor: TextToIML, validator: IMLValidator) -> None:
        """'Oh, that's GREAT.' produces <emphasis level="strong">GREAT</emphasis>."""
        xml = predictor.predict("Oh, that's GREAT.")
        result = validator.validate(xml)
        assert result.valid, f"Validation failed: {result.issues}"
        assert '<emphasis level="strong">GREAT</emphasis>.' in xml

    def test_really_question_ellipsis(
        self, predictor: TextToIML, validator: IMLValidator
    ) -> None:
        """'Really?...' produces pitch_contour="rise" and a <pause>."""
        xml = predictor.predict("Really?...")
        result = validator.validate(xml)
        assert result.valid, f"Validation failed: {result.issues}"
        assert 'pitch_contour="rise"' in xml
        assert "<pause" in xml

    def test_all_output_passes_validator(
        self, predictor: TextToIML, validator: IMLValidator
    ) -> None:
        """All output from various inputs passes IMLValidator."""
        test_inputs = [
            "Hello world.",
            "I can't believe this!",
            "Really?",
            "Oh, that's GREAT.",
            "Maybe... I'm not sure.",
            'She said "hello" and left.',
            "This is ABSOLUTELY RIDICULOUS!",
            "",
            "  ",
        ]
        for text in test_inputs:
            xml = predictor.predict(text)
            result = validator.validate(xml)
            assert result.valid, f"Validation failed for {text!r}: {result.issues}"

    def test_no_external_dependencies(self) -> None:
        """Rule-based model requires zero external dependencies beyond lxml."""
        import prosody_protocol.text_to_iml as mod

        source = open(mod.__file__).read()  # noqa: SIM115
        assert "import torch" not in source
        assert "import transformers" not in source


# ---------------------------------------------------------------------------
# Text preservation and validity over a varied corpus
# ---------------------------------------------------------------------------

CORPUS = [
    "Hello world.",
    "First sentence. Second one! Third one?",
    'She said "I love it" and left.',
    "\u201cGoodbye\u201d she whispered.",
    'A "b" c "d" e.',
    'He yelled "stop!", then left.',
    '"Stop. Now." he said.',
    "Help! I'm so scared!",
    "Well... I suppose that could work.",
    "I... don't... know.",
    "I...don't...know",
    "Wait \u2026 what?",
    "Dr. Smith lives in the U.S. near St. Louis.",
    "It's 3.5 miles, i.e. far.",
    "I moved to the U.S. Then I left.",
    "Meet me at 5 p.m. tomorrow, e.g. at the caf\u00e9.",
    "NASA and the FBI use an API.",
    "DON'T do that, I'M SERIOUS.",
    "That was SUPER-FAST and \u00c9T\u00c9 was hot.",
    "I CANNOT BELIEVE THIS!",
    "What?!",
    "Oh, that's GREAT.",
    "Yeah right, like that will work.",
    "Ctrl\x01char and form\x0cfeed\x1b end.",
    "Tabs\tand\nnew\r\nlines   collapse.",
    "A & B < C > D.",
    "Emoji \U0001F600 and symbols \u2014 stay put.",
    "10\u00a0km is far (really far).",
    "Well...",
    "!!!",
    "(Parenthetical aside.) Then more text.",
    "\u00bfD\u00f3nde est\u00e1? \u00a1Aqu\u00ed!",
    "Is it 'quoted' or not?",
    "Trailing spaces and leading ones   ",
]


class TestCorpus:
    @pytest.mark.parametrize("text", CORPUS)
    def test_valid_and_text_exact(
        self, predictor: TextToIML, validator: IMLValidator, text: str
    ) -> None:
        xml = predictor.predict(text)
        result = validator.validate(xml)
        assert result.valid, result.issues
        assert result.warnings == []
        # Stripping the tags (spec 6.2) gives the normalized input back exactly.
        root = etree.fromstring(xml.encode("utf-8"))
        assert "".join(root.itertext()) == _normalized(text)
        parser = IMLParser()
        assert parser.to_plain_text(parser.parse(xml)) == _normalized(text)

    @pytest.mark.parametrize("text", CORPUS)
    def test_emotion_always_has_confidence(self, predictor: TextToIML, text: str) -> None:
        for utt in predictor.predict_document(text).utterances:
            assert (utt.emotion is None) == (utt.confidence is None)
            if utt.confidence is not None:
                assert 0.5 <= utt.confidence <= 1.0


# ---------------------------------------------------------------------------
# ALL CAPS -> emphasis
# ---------------------------------------------------------------------------


class TestAllCapsEmphasis:
    def test_single_caps_word(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("That is AMAZING work.")
        utt = doc.utterances[0]
        emphasis_nodes = [c for c in utt.children if isinstance(c, Emphasis)]
        assert emphasis_nodes == [Emphasis(level="strong", children=("AMAZING",))]

    def test_multiple_caps_words(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("This is ABSOLUTELY RIDICULOUS.")
        utt = doc.utterances[0]
        emphasis_nodes = [c for c in utt.children if isinstance(c, Emphasis)]
        assert len(emphasis_nodes) == 2

    def test_punctuation_stays_outside_emphasis(self, predictor: TextToIML) -> None:
        xml = predictor.predict('He said "NO!" twice.')
        assert '&quot;<emphasis level="strong">NO</emphasis>!&quot;' in xml

    def test_single_letter_not_emphasized(self, predictor: TextToIML) -> None:
        """Single capital letter (e.g. 'I') should NOT be emphasized."""
        xml = predictor.predict("I am fine.")
        assert "<emphasis" not in xml

    def test_mixed_case_not_emphasized(self, predictor: TextToIML) -> None:
        """Mixed-case words should not be emphasized."""
        xml = predictor.predict("Hello World.")
        assert "<emphasis" not in xml

    @pytest.mark.parametrize("text", ["NASA and the FBI use an API.", "OK fine.", "The USA won."])
    def test_acronyms_not_emphasized(self, predictor: TextToIML, text: str) -> None:
        assert "<emphasis" not in predictor.predict(text)

    @pytest.mark.parametrize(
        ("text", "word"),
        [
            ("DON'T do that.", "DON'T"),
            ("I'M here.", "I'M"),
            ("A SUPER-FAST car.", "SUPER-FAST"),
            ("Das ist \u00dcBERALL so.", "\u00dcBERALL"),
            ("I said NO.", "NO"),
            ("STOP that.", "STOP"),
        ],
    )
    def test_shouted_words_emphasized(self, predictor: TextToIML, text: str, word: str) -> None:
        assert f'<emphasis level="strong">{word}</emphasis>' in predictor.predict(text)

    def test_all_caps_sentence_is_louder_not_emphasized_word_by_word(
        self, predictor: TextToIML
    ) -> None:
        xml = predictor.predict("I CANNOT BELIEVE THIS!")
        assert "<emphasis" not in xml
        assert 'volume="+6dB"' in xml


# ---------------------------------------------------------------------------
# Question marks -> pitch_contour="rise"
# ---------------------------------------------------------------------------


class TestQuestionPitch:
    def test_simple_question(self, predictor: TextToIML) -> None:
        xml = predictor.predict("Are you sure?")
        assert 'pitch_contour="rise"' in xml

    def test_question_with_ellipsis(self, predictor: TextToIML) -> None:
        xml = predictor.predict("Really?...")
        assert 'pitch_contour="rise"' in xml
        assert "<pause" in xml

    def test_non_question_no_rise(self, predictor: TextToIML) -> None:
        xml = predictor.predict("This is a statement.")
        assert "pitch_contour" not in xml

    def test_interrobang_is_a_surprised_question(self, predictor: TextToIML) -> None:
        """'What?!' used to be labelled joyful with no rise."""
        doc = predictor.predict_document("What?!")
        utt = doc.utterances[0]
        assert isinstance(utt.children[0], Prosody)
        assert utt.children[0].pitch_contour == "rise-sharp"
        assert utt.emotion == "surprised"

    def test_polite_request_not_uncertain(self, predictor: TextToIML) -> None:
        """'could'/'might' are not hedges in 'Could you pass the salt?'."""
        doc = predictor.predict_document("Could you pass the salt?")
        assert doc.utterances[0].emotion is None


# ---------------------------------------------------------------------------
# Exclamation -> louder and higher, but not an emotion by itself
# ---------------------------------------------------------------------------


class TestExclamationHandling:
    def test_exclamation_gets_pitch_and_volume_boost(self, predictor: TextToIML) -> None:
        xml = predictor.predict("Let's go!")
        assert '<prosody pitch="+5%" volume="+3dB">' in xml

    def test_exclamation_with_negative_word(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("This is terrible!")
        utt = doc.utterances[0]
        assert utt.emotion == "frustrated"
        assert utt.confidence == 0.6

    def test_exclamation_with_positive_word(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("This is wonderful!")
        utt = doc.utterances[0]
        assert utt.emotion == "joyful"
        assert utt.confidence == 0.6

    def test_exclamation_alone_is_not_an_emotion(self, predictor: TextToIML) -> None:
        """'!' used to default to joyful, even for alarm ('Watch out!')."""
        for text in ("Let's go!", "Get out of my house!", "I can't believe this happened!"):
            assert predictor.predict_document(text).utterances[0].emotion is None

    @pytest.mark.parametrize("text", ["Watch out!", "Help!", "I'm so scared!"])
    def test_alarm_is_fearful(self, predictor: TextToIML, text: str) -> None:
        assert predictor.predict_document(text).utterances[0].emotion == "fearful"


# ---------------------------------------------------------------------------
# Ellipsis -> pause
# ---------------------------------------------------------------------------


class TestEllipsisPause:
    def test_ellipsis_produces_pause(self, predictor: TextToIML) -> None:
        xml = predictor.predict("Well... I suppose so.")
        assert 'Well...<pause duration="500"/> I suppose so.' in xml

    def test_unicode_ellipsis(self, predictor: TextToIML) -> None:
        xml = predictor.predict("Well\u2026 I suppose so.")
        assert '<pause duration="500"/>' in xml

    def test_multiple_ellipses(self, predictor: TextToIML) -> None:
        xml = predictor.predict("I... don't... know.")
        assert xml.count("<pause") == 2

    def test_ellipsis_does_not_end_the_sentence(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("Well... I suppose that could work.")
        assert len(doc.utterances) == 1

    def test_leading_ellipsis(self, predictor: TextToIML) -> None:
        xml = predictor.predict("And then ...nothing.")
        assert 'And then <pause duration="500"/>...nothing.' in xml

    def test_standalone_ellipsis_is_one_pause(self, predictor: TextToIML) -> None:
        xml = predictor.predict("Wait … what?")
        assert xml.count("<pause") == 1


# ---------------------------------------------------------------------------
# Quoted speech -> separate utterances, in place
# ---------------------------------------------------------------------------


class TestQuotedSpeech:
    def test_quoted_speech_separate_utterance(self, predictor: TextToIML) -> None:
        xml = predictor.predict('She said "hello there" and left.')
        assert _utterance_texts(xml) == ["She said", '"hello there"', "and left."]

    def test_quoted_speech_content(self, predictor: TextToIML) -> None:
        xml = predictor.predict('He yelled "stop right now" at the dog.')
        assert _utterance_texts(xml) == ["He yelled", '"stop right now"', "at the dog."]

    def test_smart_quotes(self, predictor: TextToIML) -> None:
        xml = predictor.predict("\u201cGoodbye\u201d she whispered.")
        assert _utterance_texts(xml) == ["\u201cGoodbye\u201d", "she whispered."]

    def test_quote_keeps_attached_punctuation(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document('He yelled "stop!", then left.')
        texts = [IMLParser().to_plain_text(type(doc)(utterances=(u,))) for u in doc.utterances]
        assert texts == ["He yelled", '"stop!",', "then left."]
        # The quoted exclamation is still an exclamation.
        assert isinstance(doc.utterances[1].children[0], Prosody)

    def test_quote_gets_its_own_emotion(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document('She said "I love it" and left.')
        assert [u.emotion for u in doc.utterances] == [None, "joyful", None]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


class TestSentenceSplitting:
    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith lives in the U.S. near St. Louis.",
            "It's 3.5 miles, i.e. far.",
            "Mr. Jones said hi.",
            "Meet me at 5 p.m. tomorrow.",
            "Books by J. R. R. Tolkien are long.",
            "Apples, pears, etc. are fruit.",
        ],
    )
    def test_abbreviations_do_not_split(self, predictor: TextToIML, text: str) -> None:
        assert len(predictor.predict_document(text).utterances) == 1

    @pytest.mark.parametrize(
        ("text", "count"),
        [
            ("Hello. How are you? I'm fine!", 3),
            ("I moved to the U.S. Then I left.", 2),
            ("He works at Acme Inc. The pay is good.", 2),
            # "No." is the number abbreviation only before a number.
            ("Yes. No. Maybe.", 3),
            ("I said no. Nobody cares.", 2),
            ("See No. 5 for details.", 1),
        ],
    )
    def test_sentences_split(self, predictor: TextToIML, text: str, count: int) -> None:
        assert len(predictor.predict_document(text).utterances) == count

    def test_utterances_separated_by_whitespace(self, predictor: TextToIML) -> None:
        """Stripping the tags must not run sentences together (spec 6.2)."""
        xml = predictor.predict("First sentence. Second one! Third one?")
        root = etree.fromstring(xml.encode("utf-8"))
        assert "".join(root.itertext()) == "First sentence. Second one! Third one?"


# ---------------------------------------------------------------------------
# Emotion cues
# ---------------------------------------------------------------------------


class TestEmotionCues:
    def test_negative_lexicon_frustrated(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("This is a terrible situation.")
        utt = doc.utterances[0]
        assert utt.emotion == "frustrated"
        assert utt.confidence == 0.5

    def test_positive_lexicon_joyful(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("What a wonderful day.")
        utt = doc.utterances[0]
        assert utt.emotion == "joyful"
        assert utt.confidence == 0.5

    def test_uncertain_lexicon(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("Maybe we should wait, perhaps not.")
        utt = doc.utterances[0]
        assert utt.emotion == "uncertain"
        assert utt.confidence == 0.7

    def test_no_lexicon_match_no_emotion(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("The cat sat on the mat.")
        utt = doc.utterances[0]
        assert utt.emotion is None
        assert utt.confidence is None

    @pytest.mark.parametrize("text", ["I'm not angry!", "I am not happy.", "Never sad."])
    def test_negated_cue_ignored(self, predictor: TextToIML, text: str) -> None:
        assert predictor.predict_document(text).utterances[0].emotion is None

    def test_conflicting_cues_give_no_emotion(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("I love it and I hate it.")
        assert doc.utterances[0].emotion is None

    @pytest.mark.parametrize(
        "text",
        [
            "Oh, that's GREAT.",
            "Oh great, another meeting.",
            "Yeah right, like that will work.",
            "Just what I needed today.",
            "That's just perfect, thanks.",
            "Oh, that's just wonderful.",
            "oh yeah, that's just so really great.",
        ],
    )
    def test_sarcasm_cues(self, predictor: TextToIML, text: str) -> None:
        doc = predictor.predict_document(text)
        utt = doc.utterances[0]
        assert utt.emotion == "sarcastic"
        # Spec 3.2: fall-rise marks sarcasm.
        assert isinstance(utt.children[0], Prosody)
        assert utt.children[0].pitch_contour == "fall-rise"

    def test_sincere_praise_is_not_sarcastic(self, predictor: TextToIML) -> None:
        assert predictor.predict_document("That is great news.").utterances[0].emotion == "joyful"

    @pytest.mark.parametrize(
        "text",
        [
            "Oh, that's great news!",
            "Oh great, you made it!",
            "Oh, how wonderful to see you!",
            "Oh, that's great news.",
        ],
    )
    def test_sincere_oh_exclamations_are_joyful(self, predictor: TextToIML, text: str) -> None:
        """These were labelled sarcastic at 0.7, with a fall-rise contour."""
        utt = predictor.predict_document(text).utterances[0]
        assert utt.emotion == "joyful"
        assert "fall-rise" not in predictor.predict(text)

    def test_exclamation_is_not_evidence_of_sarcasm(self, predictor: TextToIML) -> None:
        assert predictor.predict_document("Oh nice, thanks so much!").utterances[0].emotion is None

    @pytest.mark.parametrize(
        ("text", "confidence"),
        [
            ("Oh great.", 0.5),  # a bare frame is a weak cue
            ("That's just perfect, thanks.", 0.5),
            ("Oh, that's GREAT.", 0.6),  # reinforced by capitals
            ("Oh great, another meeting.", 0.6),  # ... by a follow-up
            ("Oh great...", 0.6),  # ... by trailing off (not hesitation)
            ("Yeah right...", 0.6),
            ("Yeah right, just what I needed.", 0.6),  # cues do not add up
            ("Just what I needed!", 0.6),  # no exclamation bonus
        ],
    )
    def test_sarcasm_confidence_is_capped(
        self, predictor: TextToIML, text: str, confidence: float
    ) -> None:
        utt = predictor.predict_document(text).utterances[0]
        assert (utt.emotion, utt.confidence) == ("sarcastic", confidence)


# ---------------------------------------------------------------------------
# Confidence values follow spec rules
# ---------------------------------------------------------------------------


class TestConfidenceRules:
    def test_confidence_present_when_emotion_set(self, predictor: TextToIML) -> None:
        """Spec rule: confidence MUST be present when emotion is set."""
        doc = predictor.predict_document("This is awful!")
        assert doc.utterances[0].emotion is not None
        assert doc.utterances[0].confidence is not None

    def test_confidence_absent_when_no_emotion(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("The sky is blue.")
        for utt in doc.utterances:
            assert utt.emotion is None
            assert utt.confidence is None

    def test_weak_cues_are_not_inflated(self, predictor: TextToIML) -> None:
        """A 0.4 guess used to be reported as 0.6, above spec 6.2's 0.5 line."""
        # An ellipsis alone is a weak hesitation cue (confidence 0.4).
        assert predictor.predict_document("So... we go.").utterances[0].emotion is None
        lenient = TextToIML(min_confidence=0.0).predict_document("So... we go.")
        assert lenient.utterances[0].emotion == "uncertain"
        assert lenient.utterances[0].confidence == 0.4

    def test_confidence_reflects_cue_strength(self, predictor: TextToIML) -> None:
        def confidence(text: str) -> float | None:
            return predictor.predict_document(text).utterances[0].confidence

        assert confidence("I hate it.") == 0.5
        assert confidence("I hate it!") == 0.6
        assert confidence("This is stupid and useless.") == 0.7
        # Cues for different emotions only count by their margin.
        assert confidence("I hate this stupid, useless thing.") == 0.5

    def test_default_confidence_is_an_opt_in_floor(self) -> None:
        """Confidence should be at least the default_confidence when one is given."""
        p = TextToIML(default_confidence=0.7)
        doc = p.predict_document("This is terrible!")
        utt = doc.utterances[0]
        assert utt.confidence == 0.7
        # The floor does not let weak guesses through.
        assert p.predict_document("So... we go.").utterances[0].emotion is None

    def test_confidence_in_valid_range(self, predictor: TextToIML) -> None:
        test_inputs = [
            "I hate this!",
            "Amazing!",
            "Maybe tomorrow.",
            "Really?",
        ]
        for text in test_inputs:
            doc = predictor.predict_document(text)
            for utt in doc.utterances:
                if utt.confidence is not None:
                    assert 0.0 <= utt.confidence <= 1.0


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class TestContext:
    @pytest.mark.parametrize(
        ("text", "confidence"),
        [
            ("I can't believe this happened!", 0.6),
            # A declarative sentence without cues used to ignore the context.
            ("I can't believe this happened.", 0.5),
            ("The app crashed again.", 0.5),
        ],
    )
    def test_context_labels_sentence_without_cues(
        self, predictor: TextToIML, text: str, confidence: float
    ) -> None:
        """The documented example: context decides an otherwise neutral sentence."""
        assert predictor.predict_document(text).utterances[0].emotion is None
        for context in ("frustrated", "user_frustrated", "The user is annoyed."):
            utt = predictor.predict_document(text, context=context).utterances[0]
            assert (utt.emotion, utt.confidence) == ("frustrated", confidence)

    def test_context_raises_agreeing_confidence(self, predictor: TextToIML) -> None:
        plain = predictor.predict_document("This is terrible.").utterances[0]
        agreed = predictor.predict_document("This is terrible.", context="frustrated")
        assert agreed.utterances[0].emotion == plain.emotion == "frustrated"
        assert agreed.utterances[0].confidence > plain.confidence  # type: ignore[operator]

    def test_context_does_not_override_clear_cues(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("What a wonderful surprise, I love it!", context="sad")
        assert doc.utterances[0].emotion == "joyful"

    def test_contrary_context_counts_as_one_cue(self, predictor: TextToIML) -> None:
        # One cue word against the context's one cue: a tie, so no label.
        assert predictor.predict_document("I love it.", context="sad").utterances[0].emotion is None

    def test_context_without_cues_has_no_effect(self, predictor: TextToIML) -> None:
        text = "Oh, that's GREAT."
        assert predictor.predict(text, context="Previous conversation.") == predictor.predict(
            text
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self, predictor: TextToIML, validator: IMLValidator) -> None:
        xml = predictor.predict("")
        result = validator.validate(xml)
        assert result.valid

    def test_whitespace_only(self, predictor: TextToIML, validator: IMLValidator) -> None:
        xml = predictor.predict("   ")
        result = validator.validate(xml)
        assert result.valid

    def test_single_word(self, predictor: TextToIML, validator: IMLValidator) -> None:
        xml = predictor.predict("Hello")
        result = validator.validate(xml)
        assert result.valid

    def test_multi_sentence(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("Hello. How are you? I'm fine!")
        assert len(doc.utterances) == 3

    def test_context_parameter_accepted(self, predictor: TextToIML) -> None:
        """Context parameter should be accepted without error."""
        xml = predictor.predict("Hello.", context="Previous conversation context.")
        assert xml

    def test_xml_special_characters(
        self, predictor: TextToIML, validator: IMLValidator
    ) -> None:
        """Text with XML special chars should produce valid output."""
        xml = predictor.predict("A & B < C > D.")
        result = validator.validate(xml)
        assert result.valid
        assert "A &amp; B &lt; C &gt; D." in xml

    def test_control_characters(self, predictor: TextToIML, validator: IMLValidator) -> None:
        """C0 control characters used to make the output malformed XML."""
        xml = predictor.predict("Ctrl\x01char and form\x0cfeed")
        assert validator.validate(xml).valid
        assert IMLParser().to_plain_text(IMLParser().parse(xml)) == "Ctrl char and form feed"

    @pytest.mark.parametrize(
        "unit",
        [
            "not sure ",  # phrase matching and text building
            "oh great, ",  # sarcasm frames
            ".",  # sentence-final punctuation runs
            "!",
            "\u201ea ",  # quotes that are never closed
            '"a"',  # quotes in a text without spaces
            "just ",  # a sarcasm-frame opener that was also a filler word
            "oh so ",
            "oh really so yeah ",
            "Oh, that's just ",
        ],
    )
    def test_long_input_takes_linear_time(self, predictor: TextToIML, unit: str) -> None:
        """Each of these was quadratic: 200 KB took from 20 s to several minutes
        ("just just ...": about 3.5 minutes)."""
        text = unit * (200_000 // len(unit))
        start = time.perf_counter()
        doc = predictor.predict_document(text)
        assert time.perf_counter() - start < 5.0
        assert IMLParser().to_plain_text(doc) == text.strip()

    @pytest.mark.parametrize("unit", ["just ", "oh so ", "Oh, that's just "])
    def test_long_context_takes_linear_time(self, predictor: TextToIML, unit: str) -> None:
        """The context is scored for cues too (the REST route accepts 100,000
        characters in each field)."""
        context = unit * (200_000 // len(unit))
        start = time.perf_counter()
        predictor.predict("Fine.", context=context)
        assert time.perf_counter() - start < 5.0

    def test_one_megabyte(self, predictor: TextToIML) -> None:
        units = [
            "just just just great. ", "Oh, that's just so really great. ", "not sure ",
            "Well... ", "I LOVE it! ", "Are you coming?! ", "\u201cso ", "e.g. 5 p.m. ",
        ]
        text = "".join(units) * (1_000_000 // len("".join(units)))
        start = time.process_time()
        doc = predictor.predict_document(text)
        assert time.process_time() - start < 20.0
        assert IMLParser().to_plain_text(doc) == text.strip()

    def test_predict_document_round_trip(self, predictor: TextToIML) -> None:
        """predict_document should return a parseable IMLDocument."""
        doc = predictor.predict_document("Oh, that's GREAT.")
        assert len(doc.utterances) == 1
        assert doc.version == "0.1.0"

    def test_pause_nodes_are_pauses(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("Well... fine.")
        assert Pause(duration=500) in doc.utterances[0].children


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_params(self) -> None:
        p = TextToIML()
        assert p.model == "rule-based"
        assert p.default_confidence is None
        assert p.min_confidence == 0.5

    def test_custom_confidence(self) -> None:
        p = TextToIML(default_confidence=0.8, min_confidence=0.3)
        assert p.default_confidence == 0.8
        assert p.min_confidence == 0.3

    @pytest.mark.parametrize(
        "kwargs", [{"default_confidence": 1.5}, {"default_confidence": -0.1},
                   {"min_confidence": 2.0}]
    )
    def test_out_of_range_confidence_rejected(self, kwargs: dict[str, float]) -> None:
        """default_confidence=1.5 used to produce confidence="1.5" (invalid IML)."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            TextToIML(**kwargs)  # type: ignore[arg-type]

    def test_unsupported_model_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            TextToIML(model="prosody-bert-large")


# ---------------------------------------------------------------------------
# Combined cues
# ---------------------------------------------------------------------------


class TestCombinedCues:
    def test_caps_in_exclamation(
        self, predictor: TextToIML, validator: IMLValidator
    ) -> None:
        """ALL CAPS + exclamation should produce both emphasis and pitch boost."""
        xml = predictor.predict("That is OUTRAGEOUS!")
        result = validator.validate(xml)
        assert result.valid
        assert '<emphasis level="strong">OUTRAGEOUS</emphasis>' in xml
        assert 'pitch="+5%"' in xml

    def test_question_with_caps(
        self, predictor: TextToIML, validator: IMLValidator
    ) -> None:
        """Question + ALL CAPS should produce both rise and emphasis."""
        xml = predictor.predict("Are you SERIOUS?")
        result = validator.validate(xml)
        assert result.valid
        assert 'pitch_contour="rise"' in xml
        assert '<emphasis level="strong">SERIOUS</emphasis>' in xml

    def test_ellipsis_then_question(
        self, predictor: TextToIML, validator: IMLValidator
    ) -> None:
        xml = predictor.predict("Well... really?")
        result = validator.validate(xml)
        assert result.valid
        assert '<pause duration="500"/>' in xml
        assert 'pitch_contour="rise"' in xml
