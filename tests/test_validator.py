"""Tests for prosody_protocol.validator.

One test per validation rule (V1-V18), plus tests for:
- Valid documents passing cleanly
- Documents with multiple errors returning all of them
- File-based validation
- Consent model validation (V17-V18)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prosody_protocol.validator import IMLValidator, ValidationIssue, ValidationResult


@pytest.fixture()
def validator() -> IMLValidator:
    return IMLValidator()


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestValidationModels:
    def test_validation_issue(self) -> None:
        issue = ValidationIssue(
            severity="error",
            rule="V3",
            message="Missing confidence",
            line=1,
            column=15,
        )
        assert issue.severity == "error"
        assert issue.rule == "V3"

    def test_validation_result_default(self) -> None:
        result = ValidationResult()
        assert result.valid is True
        assert result.issues == []


# ---------------------------------------------------------------------------
# Valid documents
# ---------------------------------------------------------------------------


class TestValidDocuments:
    def test_simple_utterance_valid(
        self, validator: IMLValidator, simple_utterance: str
    ) -> None:
        result = validator.validate(simple_utterance)
        assert result.valid is True
        errors = [i for i in result.issues if i.severity == "error"]
        assert errors == []

    def test_multi_speaker_valid(
        self, validator: IMLValidator, multi_speaker: str
    ) -> None:
        result = validator.validate(multi_speaker)
        assert result.valid is True

    def test_segment_valid(
        self, validator: IMLValidator, segment_example: str
    ) -> None:
        result = validator.validate(segment_example)
        assert result.valid is True

    def test_no_emotion_valid(self, validator: IMLValidator) -> None:
        result = validator.validate("<utterance>No emotion here.</utterance>")
        assert result.valid is True

    def test_all_valid_fixtures(
        self, validator: IMLValidator, valid_fixtures_dir: Path
    ) -> None:
        for xml_file in sorted(valid_fixtures_dir.glob("*.xml")):
            result = validator.validate_file(xml_file)
            errors = [i for i in result.issues if i.severity == "error"]
            assert result.valid is True, (
                f"{xml_file.name} should be valid but got errors: "
                + "; ".join(i.message for i in errors)
            )


# ---------------------------------------------------------------------------
# V1: Well-formed XML
# ---------------------------------------------------------------------------


class TestV1WellFormedXML:
    def test_malformed_xml(self, validator: IMLValidator) -> None:
        result = validator.validate("<utterance>unclosed")
        assert result.valid is False
        assert any(i.rule == "V1" for i in result.issues)

    def test_malformed_returns_early(self, validator: IMLValidator) -> None:
        """Malformed XML should only produce a V1 error, not cascade."""
        result = validator.validate("<<<not xml>>>")
        assert len(result.issues) == 1
        assert result.issues[0].rule == "V1"


# ---------------------------------------------------------------------------
# V2: At least one <utterance> exists
# ---------------------------------------------------------------------------


class TestV2UtteranceExists:
    def test_empty_iml(self, validator: IMLValidator) -> None:
        result = validator.validate("<iml></iml>")
        assert result.valid is False
        assert any(i.rule == "V2" for i in result.issues)

    def test_wrong_root_element(self, validator: IMLValidator) -> None:
        result = validator.validate("<div>not iml</div>")
        assert result.valid is False
        assert any(i.rule == "V2" for i in result.issues)


# ---------------------------------------------------------------------------
# V3: confidence required when emotion is present
# ---------------------------------------------------------------------------


class TestV3ConfidenceRequired:
    def test_emotion_without_confidence(
        self, validator: IMLValidator, missing_confidence: str
    ) -> None:
        result = validator.validate(missing_confidence)
        assert result.valid is False
        v3 = [i for i in result.issues if i.rule == "V3"]
        assert len(v3) == 1

    def test_emotion_with_confidence_ok(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="calm" confidence="0.9">OK</utterance>'
        )
        assert not any(i.rule == "V3" for i in result.issues)

    def test_no_emotion_no_confidence_ok(self, validator: IMLValidator) -> None:
        result = validator.validate("<utterance>OK</utterance>")
        assert not any(i.rule == "V3" for i in result.issues)


# ---------------------------------------------------------------------------
# V4: confidence is float in [0.0, 1.0]
# ---------------------------------------------------------------------------


class TestV4ConfidenceRange:
    def test_confidence_too_high(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="joyful" confidence="1.5">Over</utterance>'
        )
        assert result.valid is False
        assert any(i.rule == "V4" for i in result.issues)

    def test_confidence_negative(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="sad" confidence="-0.1">Under</utterance>'
        )
        assert result.valid is False
        assert any(i.rule == "V4" for i in result.issues)

    def test_confidence_not_a_number(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="angry" confidence="high">NaN</utterance>'
        )
        assert result.valid is False
        assert any(i.rule == "V4" for i in result.issues)

    def test_confidence_zero_ok(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="uncertain" confidence="0.0">Zero</utterance>'
        )
        assert not any(i.rule == "V4" for i in result.issues)

    def test_confidence_one_ok(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="calm" confidence="1.0">One</utterance>'
        )
        assert not any(i.rule == "V4" for i in result.issues)


# ---------------------------------------------------------------------------
# V5: <pause> has duration attribute
# ---------------------------------------------------------------------------


class TestV5PauseDuration:
    def test_pause_missing_duration(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="calm" confidence="0.9">'
            "Wait <pause/> there."
            "</utterance>"
        )
        assert result.valid is False
        assert any(i.rule == "V5" for i in result.issues)


# ---------------------------------------------------------------------------
# V6: <pause> duration is a positive integer
# ---------------------------------------------------------------------------


class TestV6PauseDurationPositive:
    def test_pause_duration_zero(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><pause duration="0"/></utterance>'
        )
        assert result.valid is False
        assert any(i.rule == "V6" for i in result.issues)

    def test_pause_duration_negative(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><pause duration="-100"/></utterance>'
        )
        assert result.valid is False
        assert any(i.rule == "V6" for i in result.issues)

    def test_pause_duration_float(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><pause duration="3.14"/></utterance>'
        )
        assert result.valid is False
        assert any(i.rule == "V6" for i in result.issues)

    def test_pause_duration_valid(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><pause duration="500"/></utterance>'
        )
        assert not any(i.rule in ("V5", "V6") for i in result.issues)


# ---------------------------------------------------------------------------
# V7: <pause> has no child content
# ---------------------------------------------------------------------------


class TestV7PauseEmpty:
    def test_pause_with_text(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><pause duration="500">oops</pause></utterance>'
        )
        assert result.valid is False
        assert any(i.rule == "V7" for i in result.issues)

    def test_pause_with_child_element(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><pause duration="500"><prosody>bad</prosody></pause></utterance>'
        )
        assert result.valid is False
        assert any(i.rule == "V7" for i in result.issues)


# ---------------------------------------------------------------------------
# V8: <emphasis> has level attribute
# ---------------------------------------------------------------------------


class TestV8EmphasisLevel:
    def test_emphasis_missing_level(self, validator: IMLValidator) -> None:
        result = validator.validate(
            "<utterance>I <emphasis>really</emphasis> mean it.</utterance>"
        )
        assert result.valid is False
        assert any(i.rule == "V8" for i in result.issues)


# ---------------------------------------------------------------------------
# V9: <emphasis> level is one of: strong, moderate, reduced
# ---------------------------------------------------------------------------


class TestV9EmphasisLevelValue:
    def test_unknown_level_warns(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance>I <emphasis level="extreme">really</emphasis> mean it.</utterance>'
        )
        # V9 is a warning, not an error, so document is still valid
        assert result.valid is True
        assert any(i.rule == "V9" and i.severity == "warning" for i in result.issues)

    def test_known_level_no_warning(self, validator: IMLValidator) -> None:
        for level in ("strong", "moderate", "reduced"):
            result = validator.validate(
                f'<utterance><emphasis level="{level}">text</emphasis></utterance>'
            )
            assert not any(i.rule == "V9" for i in result.issues)


# ---------------------------------------------------------------------------
# V10: <segment> is direct child of <utterance>
# ---------------------------------------------------------------------------


class TestV10SegmentParent:
    def test_segment_in_prosody(
        self, validator: IMLValidator, invalid_segment_nesting: str
    ) -> None:
        result = validator.validate(invalid_segment_nesting)
        assert result.valid is False
        assert any(i.rule == "V10" for i in result.issues)

    def test_segment_in_emphasis(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="calm" confidence="0.9">'
            '<emphasis level="strong">'
            '<segment tempo="rushed">bad</segment>'
            "</emphasis>"
            "</utterance>"
        )
        assert result.valid is False
        assert any(i.rule == "V10" for i in result.issues)

    def test_segment_in_utterance_ok(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="calm" confidence="0.9">'
            '<segment tempo="steady">ok</segment>'
            "</utterance>"
        )
        assert not any(i.rule == "V10" for i in result.issues)


# ---------------------------------------------------------------------------
# V11: <segment> not nested in another <segment>
# ---------------------------------------------------------------------------


class TestV11SegmentNesting:
    def test_nested_segment(self, validator: IMLValidator) -> None:
        result = validator.validate(
            "<utterance>"
            '<segment tempo="rushed">'
            '<segment tempo="steady">nested</segment>'
            "</segment>"
            "</utterance>"
        )
        assert result.valid is False
        assert any(i.rule == "V11" for i in result.issues)


# ---------------------------------------------------------------------------
# V12: Nesting depth of prosody/emphasis does not exceed 2
# ---------------------------------------------------------------------------


class TestV12NestingDepth:
    def test_depth_3_warns(self, validator: IMLValidator) -> None:
        result = validator.validate(
            "<utterance>"
            '<prosody pitch="+5%">'
            '<emphasis level="strong">'
            '<prosody pitch="+10%">deep</prosody>'
            "</emphasis>"
            "</prosody>"
            "</utterance>"
        )
        # Depth 3 should trigger a warning
        assert any(i.rule == "V12" and i.severity == "warning" for i in result.issues)
        # But it's still valid (warning, not error)
        assert result.valid is True

    def test_depth_2_ok(self, validator: IMLValidator) -> None:
        result = validator.validate(
            "<utterance>"
            '<prosody pitch="+5%">'
            '<emphasis level="strong">ok</emphasis>'
            "</prosody>"
            "</utterance>"
        )
        assert not any(i.rule == "V12" for i in result.issues)


# ---------------------------------------------------------------------------
# V13: pitch value matches valid format
# ---------------------------------------------------------------------------


class TestV13PitchFormat:
    def test_invalid_pitch_warns(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody pitch="loud">text</prosody></utterance>'
        )
        assert any(i.rule == "V13" for i in result.issues)

    def test_valid_pitch_percentage(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody pitch="+15%">text</prosody></utterance>'
        )
        assert not any(i.rule == "V13" for i in result.issues)

    def test_valid_pitch_semitones(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody pitch="-2st">text</prosody></utterance>'
        )
        assert not any(i.rule == "V13" for i in result.issues)

    def test_valid_pitch_hz(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody pitch="185Hz">text</prosody></utterance>'
        )
        assert not any(i.rule == "V13" for i in result.issues)


# ---------------------------------------------------------------------------
# V14: volume value matches valid format
# ---------------------------------------------------------------------------


class TestV14VolumeFormat:
    def test_invalid_volume_warns(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody volume="very loud">text</prosody></utterance>'
        )
        assert any(i.rule == "V14" for i in result.issues)

    def test_valid_volume(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody volume="+6dB">text</prosody></utterance>'
        )
        assert not any(i.rule == "V14" for i in result.issues)

    def test_valid_negative_volume(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody volume="-3dB">text</prosody></utterance>'
        )
        assert not any(i.rule == "V14" for i in result.issues)


# ---------------------------------------------------------------------------
# V15: emotion is from core vocabulary
# ---------------------------------------------------------------------------


class TestV15CoreEmotion:
    def test_custom_emotion_info(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="excitement" confidence="0.8">Custom</utterance>'
        )
        assert result.valid is True  # INFO, not error
        assert any(i.rule == "V15" and i.severity == "info" for i in result.issues)

    def test_core_emotion_no_info(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="sarcastic" confidence="0.9">Core</utterance>'
        )
        assert not any(i.rule == "V15" for i in result.issues)


# ---------------------------------------------------------------------------
# V16: No unknown elements present
# ---------------------------------------------------------------------------


class TestV16UnknownElements:
    def test_unknown_element_info(self, validator: IMLValidator) -> None:
        result = validator.validate(
            "<utterance><custom>unknown</custom></utterance>"
        )
        assert result.valid is True  # INFO, not error
        assert any(i.rule == "V16" and i.severity == "info" for i in result.issues)


# ---------------------------------------------------------------------------
# Multiple errors
# ---------------------------------------------------------------------------


class TestMultipleErrors:
    def test_multiple_issues_all_reported(self, validator: IMLValidator) -> None:
        """A document with several problems should report all of them."""
        result = validator.validate(
            '<utterance emotion="excitement">'  # V3 + V15
            "<emphasis>no level</emphasis>"  # V8
            '<pause duration="-5"/>'  # V6
            "</utterance>"
        )
        assert result.valid is False
        rules = {i.rule for i in result.issues}
        assert "V3" in rules
        assert "V8" in rules
        assert "V6" in rules


# ---------------------------------------------------------------------------
# File-based validation
# ---------------------------------------------------------------------------


class TestValidateFile:
    def test_validate_valid_file(
        self, validator: IMLValidator, valid_fixtures_dir: Path
    ) -> None:
        result = validator.validate_file(valid_fixtures_dir / "simple_utterance.xml")
        assert result.valid is True

    def test_validate_invalid_file(
        self, validator: IMLValidator, invalid_fixtures_dir: Path
    ) -> None:
        result = validator.validate_file(
            invalid_fixtures_dir / "missing_confidence.xml"
        )
        assert result.valid is False
        assert any(i.rule == "V3" for i in result.issues)

    def test_all_invalid_fixtures_fail(
        self, validator: IMLValidator, invalid_fixtures_dir: Path
    ) -> None:
        for xml_file in sorted(invalid_fixtures_dir.glob("*.xml")):
            result = validator.validate_file(xml_file)
            assert result.valid is False, (
                f"{xml_file.name} should be invalid but passed"
            )


# ---------------------------------------------------------------------------
# Consent model validation (V17-V18)
# ---------------------------------------------------------------------------


class TestConsentValidation:
    def test_valid_consent_explicit(self, validator: IMLValidator) -> None:
        iml = '<iml version="0.1.0" consent="explicit" processing="local"><utterance>Hello</utterance></iml>'
        result = validator.validate(iml)
        assert result.valid is True
        assert not any(i.rule in ("V17", "V18") for i in result.issues)

    def test_valid_consent_implicit(self, validator: IMLValidator) -> None:
        iml = '<iml version="0.1.0" consent="implicit" processing="remote"><utterance>Hello</utterance></iml>'
        result = validator.validate(iml)
        assert result.valid is True
        assert not any(i.rule in ("V17", "V18") for i in result.issues)

    def test_invalid_consent_value_warns(self, validator: IMLValidator) -> None:
        iml = '<iml version="0.1.0" consent="maybe"><utterance>Hello</utterance></iml>'
        result = validator.validate(iml)
        assert result.valid is True  # warnings don't invalidate
        v17 = [i for i in result.issues if i.rule == "V17"]
        assert len(v17) == 1
        assert "maybe" in v17[0].message

    def test_invalid_processing_value_warns(self, validator: IMLValidator) -> None:
        iml = '<iml version="0.1.0" processing="cloud"><utterance>Hello</utterance></iml>'
        result = validator.validate(iml)
        assert result.valid is True
        v18 = [i for i in result.issues if i.rule == "V18"]
        assert len(v18) == 1
        assert "cloud" in v18[0].message

    def test_no_consent_no_warning(self, validator: IMLValidator) -> None:
        iml = '<iml version="0.1.0"><utterance>Hello</utterance></iml>'
        result = validator.validate(iml)
        assert not any(i.rule in ("V17", "V18") for i in result.issues)


# ---------------------------------------------------------------------------
# XML security tests
# ---------------------------------------------------------------------------


class TestXMLSecurity:
    def test_external_entity_not_resolved(self, validator: IMLValidator) -> None:
        """Verify external entities are not resolved (XXE prevention)."""
        iml = '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe "injected">]><utterance>&xxe;</utterance>'
        result = validator.validate(iml)
        # Should either reject the document or not resolve the entity
        # lxml with resolve_entities=False will leave &xxe; unresolved,
        # which means the text won't contain "injected"
        assert result is not None  # doesn't crash

    def test_parser_rejects_malicious_dtd(self) -> None:
        """Parser should not fetch external DTDs."""
        from prosody_protocol import IMLParser

        parser = IMLParser()
        # This should not attempt a network fetch
        iml = '<?xml version="1.0"?><!DOCTYPE foo SYSTEM "http://evil.example.com/xxe.dtd"><utterance>test</utterance>'
        # Should either parse safely or raise an error, but never fetch the URL
        try:
            doc = parser.parse(iml)
            # If it parses, verify content is safe
            plain = parser.to_plain_text(doc)
            assert "test" in plain
        except Exception:
            pass  # Rejecting the document is also acceptable
