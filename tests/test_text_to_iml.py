"""Tests for prosody_protocol.text_to_iml (Phase 6a -- rule-based baseline).

Covers:
- Acceptance criteria from the execution guide
- ALL CAPS -> <emphasis level="strong">
- ? -> pitch_contour="rise"
- ! -> emotion + pitch="+5%"
- ... -> <pause duration="500"/>
- Quoted speech -> separate <utterance>
- Sentiment lexicon emotion detection
- All output passes IMLValidator
- Edge cases: empty input, multi-sentence, mixed cues
- predict_document round-trip
- Unsupported model raises NotImplementedError
"""

from __future__ import annotations

import pytest

from prosody_protocol.models import Emphasis, Pause, Prosody
from prosody_protocol.text_to_iml import TextToIML
from prosody_protocol.validator import IMLValidator


@pytest.fixture()
def predictor() -> TextToIML:
    return TextToIML()


@pytest.fixture()
def validator() -> IMLValidator:
    return IMLValidator()


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
        assert '<emphasis level="strong">' in xml
        assert "GREAT" in xml

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
# ALL CAPS -> emphasis
# ---------------------------------------------------------------------------


class TestAllCapsEmphasis:
    def test_single_caps_word(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("That is AMAZING work.")
        utt = doc.utterances[0]
        emphasis_nodes = [c for c in utt.children if isinstance(c, Emphasis)]
        assert len(emphasis_nodes) >= 1
        caps_em = [e for e in emphasis_nodes if "AMAZING" in str(e.children)]
        assert len(caps_em) == 1
        assert caps_em[0].level == "strong"

    def test_multiple_caps_words(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("This is ABSOLUTELY RIDICULOUS.")
        utt = doc.utterances[0]
        emphasis_nodes = [c for c in utt.children if isinstance(c, Emphasis)]
        assert len(emphasis_nodes) >= 2

    def test_single_letter_not_emphasized(self, predictor: TextToIML) -> None:
        """Single capital letter (e.g. 'I') should NOT be emphasized."""
        xml = predictor.predict("I am fine.")
        assert '<emphasis level="strong">I</emphasis>' not in xml

    def test_mixed_case_not_emphasized(self, predictor: TextToIML) -> None:
        """Mixed-case words should not be emphasized."""
        xml = predictor.predict("Hello World.")
        assert "<emphasis" not in xml


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
        assert 'pitch_contour="rise"' not in xml


# ---------------------------------------------------------------------------
# Exclamation -> emotion + pitch boost
# ---------------------------------------------------------------------------


class TestExclamationHandling:
    def test_exclamation_gets_pitch_boost(self, predictor: TextToIML) -> None:
        xml = predictor.predict("Watch out!")
        assert 'pitch="+5%"' in xml

    def test_exclamation_with_negative_word(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("This is terrible!")
        utt = doc.utterances[0]
        assert utt.emotion == "frustrated"
        assert utt.confidence is not None
        assert utt.confidence >= 0.55

    def test_exclamation_with_positive_word(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("This is wonderful!")
        utt = doc.utterances[0]
        assert utt.emotion == "joyful"
        assert utt.confidence is not None

    def test_exclamation_default_joyful(self, predictor: TextToIML) -> None:
        """Exclamation with neutral words defaults to joyful."""
        doc = predictor.predict_document("Let's go!")
        utt = doc.utterances[0]
        assert utt.emotion == "joyful"


# ---------------------------------------------------------------------------
# Ellipsis -> pause
# ---------------------------------------------------------------------------


class TestEllipsisPause:
    def test_ellipsis_produces_pause(self, predictor: TextToIML) -> None:
        xml = predictor.predict("Well... I suppose so.")
        assert '<pause duration="500"/>' in xml

    def test_unicode_ellipsis(self, predictor: TextToIML) -> None:
        xml = predictor.predict("Well\u2026 I suppose so.")
        assert '<pause duration="500"/>' in xml

    def test_multiple_ellipses(self, predictor: TextToIML) -> None:
        xml = predictor.predict("I... don't... know.")
        assert xml.count("<pause") >= 2


# ---------------------------------------------------------------------------
# Quoted speech -> separate utterances
# ---------------------------------------------------------------------------


class TestQuotedSpeech:
    def test_quoted_speech_separate_utterance(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document('She said "hello there" and left.')
        assert len(doc.utterances) >= 2

    def test_quoted_speech_content(self, predictor: TextToIML) -> None:
        xml = predictor.predict('He yelled "stop right now" at the dog.')
        assert "stop right now" in xml
        assert xml.count("<utterance") >= 2

    def test_smart_quotes(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("\u201cGoodbye\u201d she whispered.")
        assert len(doc.utterances) >= 2


# ---------------------------------------------------------------------------
# Sentiment lexicon
# ---------------------------------------------------------------------------


class TestSentimentLexicon:
    def test_negative_lexicon_frustrated(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("This is a terrible situation.")
        utt = doc.utterances[0]
        assert utt.emotion == "frustrated"
        assert utt.confidence is not None

    def test_positive_lexicon_joyful(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("What a wonderful day.")
        utt = doc.utterances[0]
        assert utt.emotion == "joyful"
        assert utt.confidence is not None

    def test_uncertain_lexicon(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("Maybe we should wait, perhaps not.")
        utt = doc.utterances[0]
        assert utt.emotion == "uncertain"
        assert utt.confidence is not None

    def test_no_lexicon_match_no_emotion(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("The cat sat on the mat.")
        utt = doc.utterances[0]
        assert utt.emotion is None
        assert utt.confidence is None


# ---------------------------------------------------------------------------
# Confidence values follow spec rules
# ---------------------------------------------------------------------------


class TestConfidenceRules:
    def test_confidence_present_when_emotion_set(self, predictor: TextToIML) -> None:
        """Spec rule: confidence MUST be present when emotion is set."""
        doc = predictor.predict_document("This is awful!")
        for utt in doc.utterances:
            if utt.emotion is not None:
                assert utt.confidence is not None

    def test_confidence_absent_when_no_emotion(self, predictor: TextToIML) -> None:
        doc = predictor.predict_document("The sky is blue.")
        for utt in doc.utterances:
            if utt.emotion is None:
                assert utt.confidence is None

    def test_confidence_uses_default_floor(self) -> None:
        """Confidence should be at least the default_confidence."""
        p = TextToIML(default_confidence=0.7)
        doc = p.predict_document("This is terrible!")
        utt = doc.utterances[0]
        assert utt.confidence is not None
        assert utt.confidence >= 0.7

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
        assert "&amp;" in xml

    def test_predict_document_round_trip(self, predictor: TextToIML) -> None:
        """predict_document should return a parseable IMLDocument."""
        doc = predictor.predict_document("Oh, that's GREAT.")
        assert len(doc.utterances) >= 1
        assert doc.version == "0.1.0"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_params(self) -> None:
        p = TextToIML()
        assert p.model == "rule-based"
        assert p.default_confidence == 0.6

    def test_custom_confidence(self) -> None:
        p = TextToIML(default_confidence=0.8)
        assert p.default_confidence == 0.8

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
        assert '<emphasis level="strong">' in xml
        assert 'pitch="+5%"' in xml

    def test_question_with_caps(
        self, predictor: TextToIML, validator: IMLValidator
    ) -> None:
        """Question + ALL CAPS should produce both rise and emphasis."""
        xml = predictor.predict("Are you SERIOUS?")
        result = validator.validate(xml)
        assert result.valid
        assert 'pitch_contour="rise"' in xml
        assert '<emphasis level="strong">' in xml

    def test_ellipsis_then_question(
        self, predictor: TextToIML, validator: IMLValidator
    ) -> None:
        xml = predictor.predict("Well... really?")
        result = validator.validate(xml)
        assert result.valid
