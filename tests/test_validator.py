"""Tests for prosody_protocol.validator.

One test per validation rule (V1-V33), plus tests for:
- Valid documents passing cleanly
- Documents with multiple errors returning all of them
- File-based validation, including non-UTF-8 files
- Consent model validation (V17-V18)
- Comments, processing instructions, unknown elements and namespaces
- XML security (DOCTYPE, external entities)
- ValidationResult.errors / .warnings / raise_for_errors
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prosody_protocol.exceptions import IMLValidationError
from prosody_protocol.validator import IMLValidator, ValidationIssue, ValidationResult

# The error rules each invalid fixture is expected to break -- exactly these.
INVALID_FIXTURE_RULES: dict[str, set[str]] = {
    "confidence_nan.xml": {"V4"},
    "confidence_out_of_range.xml": {"V4"},
    "content_outside_utterance.xml": {"V19"},
    "doctype_entity.xml": {"V31"},
    "emphasis_in_emphasis.xml": {"V21"},
    "emphasis_missing_level.xml": {"V8"},
    "encoding_latin1.xml": {"V30"},
    "invalid_consent.xml": {"V17", "V18"},
    "invalid_emphasis_level.xml": {"V9"},
    "invalid_enum_values.xml": {"V22", "V23", "V24", "V25", "V26"},
    "invalid_extended_attributes.xml": {"V27"},
    "invalid_pitch_volume.xml": {"V13", "V14"},
    "invalid_version_language.xml": {"V28", "V29"},
    "missing_confidence.xml": {"V3"},
    "missing_pause_duration.xml": {"V5"},
    "nested_utterance.xml": {"V20"},
    "pause_duration_not_integer.xml": {"V6"},
    "pause_duration_too_large.xml": {"V6"},
    "pause_with_content.xml": {"V7"},
    "pause_with_unknown_element.xml": {"V7"},
    "segment_in_prosody.xml": {"V10"},
    "segment_in_segment.xml": {"V10", "V11"},
}


def _error_rules(result: ValidationResult) -> set[str]:
    return {i.rule for i in result.issues if i.severity == "error"}


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

    def test_foreign_default_namespace_root_is_named_by_namespace(
        self, validator: IMLValidator
    ) -> None:
        result = validator.validate('<iml xmlns="urn:other"><utterance>x</utterance></iml>')
        assert _error_rules(result) == {"V2"}
        assert "got <{urn:other}iml>" in result.errors[0].message


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

    def test_pause_with_whitespace_is_empty(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance>a<pause duration="500">\n  </pause>b</utterance>'
        )
        assert result.issues == []

    def test_unknown_element_in_pause_is_content(self, validator: IMLValidator) -> None:
        """Spec 3.3: a pause contains no elements, unknown ones included."""
        result = validator.validate(
            '<utterance>a<pause duration="500"><f:x xmlns:f="urn:x"/></pause>b</utterance>'
        )
        assert _error_rules(result) == {"V7"}
        assert "child elements" in result.errors[0].message


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
    def test_unknown_level_is_error(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance>I <emphasis level="extreme">really</emphasis> mean it.</utterance>'
        )
        # Spec 6.3(3): an attribute outside its enumeration makes the document invalid.
        assert result.valid is False
        assert any(i.rule == "V9" and i.severity == "error" for i in result.issues)

    def test_level_is_case_sensitive(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><emphasis level="Strong">x</emphasis></utterance>'
        )
        assert _error_rules(result) == {"V9"}

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

    def test_invalid_fixtures_break_exactly_their_rules(
        self, validator: IMLValidator, invalid_fixtures_dir: Path
    ) -> None:
        names = {f.name for f in invalid_fixtures_dir.glob("*.xml")}
        assert names == set(INVALID_FIXTURE_RULES), "update INVALID_FIXTURE_RULES"
        for name, rules in INVALID_FIXTURE_RULES.items():
            result = validator.validate_file(invalid_fixtures_dir / name)
            assert _error_rules(result) == rules, name

    def test_latin1_file_is_v30_not_an_exception(
        self, validator: IMLValidator, invalid_fixtures_dir: Path
    ) -> None:
        result = validator.validate_file(invalid_fixtures_dir / "encoding_latin1.xml")
        assert [i.rule for i in result.issues] == ["V30"]
        assert result.issues[0].line == 4  # the line with the first non-UTF-8 byte
        assert "0xe9" in result.issues[0].message

    def test_utf16_file_is_v30(self, validator: IMLValidator, tmp_path: Path) -> None:
        path = tmp_path / "utf16.xml"
        path.write_text("<utterance>Hello</utterance>", encoding="utf-16")
        result = validator.validate_file(path)
        assert result.valid is False
        assert _error_rules(result) == {"V30"}

    def test_utf8_bom_file_is_valid(self, validator: IMLValidator, tmp_path: Path) -> None:
        path = tmp_path / "bom.xml"
        path.write_bytes(b"\xef\xbb\xbf<utterance>caf\xc3\xa9</utterance>")
        assert validator.validate_file(path).valid is True

    def test_missing_file_raises_oserror(
        self, validator: IMLValidator, tmp_path: Path
    ) -> None:
        with pytest.raises(OSError):
            validator.validate_file(tmp_path / "missing.xml")


# ---------------------------------------------------------------------------
# Consent model validation (V17-V18)
# ---------------------------------------------------------------------------


class TestConsentValidation:
    def test_valid_consent_explicit(self, validator: IMLValidator) -> None:
        iml = (
            '<iml version="0.1.0" consent="explicit" processing="local">'
            "<utterance>Hello</utterance></iml>"
        )
        result = validator.validate(iml)
        assert result.valid is True
        assert not any(i.rule in ("V17", "V18") for i in result.issues)

    def test_valid_consent_implicit(self, validator: IMLValidator) -> None:
        iml = (
            '<iml version="0.1.0" consent="implicit" processing="remote">'
            "<utterance>Hello</utterance></iml>"
        )
        result = validator.validate(iml)
        assert result.valid is True
        assert not any(i.rule in ("V17", "V18") for i in result.issues)

    def test_consent_none_and_processing_hybrid_ok(self, validator: IMLValidator) -> None:
        iml = '<iml consent="none" processing="hybrid"><utterance>Hello</utterance></iml>'
        result = validator.validate(iml)
        assert result.valid is True
        assert result.issues == []

    def test_invalid_consent_value_is_error(self, validator: IMLValidator) -> None:
        iml = '<iml version="0.1.0" consent="maybe"><utterance>Hello</utterance></iml>'
        result = validator.validate(iml)
        assert result.valid is False
        v17 = [i for i in result.issues if i.rule == "V17"]
        assert len(v17) == 1
        assert v17[0].severity == "error"
        assert "maybe" in v17[0].message

    def test_invalid_processing_value_is_error(self, validator: IMLValidator) -> None:
        iml = '<iml version="0.1.0" processing="cloud"><utterance>Hello</utterance></iml>'
        result = validator.validate(iml)
        assert result.valid is False
        v18 = [i for i in result.issues if i.rule == "V18"]
        assert len(v18) == 1
        assert v18[0].severity == "error"
        assert "cloud" in v18[0].message

    def test_no_consent_no_warning(self, validator: IMLValidator) -> None:
        iml = '<iml version="0.1.0"><utterance>Hello</utterance></iml>'
        result = validator.validate(iml)
        assert not any(i.rule in ("V17", "V18") for i in result.issues)




# ---------------------------------------------------------------------------
# Numeric syntax (V4, V6, spec 2.6)
# ---------------------------------------------------------------------------


class TestNumericSyntax:
    @pytest.mark.parametrize("raw", ["NaN", "nan", "inf", "-inf", "1_0", "0.\u0665", "0,5", ""])
    def test_confidence_not_a_float(self, validator: IMLValidator, raw: str) -> None:
        result = validator.validate(f'<utterance emotion="calm" confidence="{raw}">x</utterance>')
        assert _error_rules(result) == {"V4"}

    @pytest.mark.parametrize("raw", ["0.5", "1e-1", "5E-1", "+0.25", ".75", "1.", " 0.5 ", "1"])
    def test_confidence_float_spellings_ok(self, validator: IMLValidator, raw: str) -> None:
        result = validator.validate(f'<utterance emotion="calm" confidence="{raw}">x</utterance>')
        assert result.valid is True, result.issues

    @pytest.mark.parametrize(
        "raw", ["8_00", "\u0668\u0660\u0660", "3.14", "1e3", "0", "-100", "abc"]
    )
    def test_pause_duration_not_a_positive_integer(
        self, validator: IMLValidator, raw: str
    ) -> None:
        result = validator.validate(f'<utterance>a<pause duration="{raw}"/>b</utterance>')
        assert _error_rules(result) == {"V6"}

    @pytest.mark.parametrize(
        "raw",
        ["1" * 5000, "9" * 100_000, "2147483648", "99999999999999999999"],
        ids=["5000-digits", "100000-digits", "max-plus-1", "20-digits"],
    )
    def test_pause_duration_too_large_is_v6_not_a_crash(
        self, validator: IMLValidator, raw: str
    ) -> None:
        # int() refuses strings of more than 4300 digits with ValueError.
        result = validator.validate(f'<utterance>a<pause duration="{raw}"/>b</utterance>')
        assert _error_rules(result) == {"V6"}
        assert "2147483647" in result.errors[0].message
        assert len(result.errors[0].message) < 200

    @pytest.mark.parametrize(
        "raw",
        ["800", "+800", "0800", " 800 ", "2147483647", "0" * 5000 + "1"],
        ids=["plain", "plus", "leading-zero", "spaces", "max", "5000-leading-zeros"],
    )
    def test_pause_duration_integer_spellings_ok(
        self, validator: IMLValidator, raw: str
    ) -> None:
        result = validator.validate(f'<utterance>a<pause duration="{raw}"/>b</utterance>')
        assert result.valid is True, result.issues

    def test_non_ascii_digits_in_pitch_rejected(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody pitch="+\u0661\u0665%">x</prosody></utterance>'
        )
        assert _error_rules(result) == {"V13"}


# ---------------------------------------------------------------------------
# Attribute vocabularies and formats (V13, V14, V22-V29)
# ---------------------------------------------------------------------------


class TestAttributeValues:
    def test_invalid_pitch_and_volume_are_errors(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody pitch="high" volume="loud">x</prosody></utterance>'
        )
        assert result.valid is False
        assert _error_rules(result) == {"V13", "V14"}

    @pytest.mark.parametrize("pitch", ["15%", "+15", "+15 %", "-185Hz", "+15%\n"])
    def test_pitch_format_is_strict(self, validator: IMLValidator, pitch: str) -> None:
        # "+15%&#10;" puts a real newline in the value (a trailing-newline regex bug).
        value = pitch.replace("\n", "&#10;")
        result = validator.validate(f'<utterance><prosody pitch="{value}">x</prosody></utterance>')
        assert _error_rules(result) == {"V13"}

    @pytest.mark.parametrize("rate", ["fast", "slow", "medium", "150%", "80.5%"])
    def test_valid_rates(self, validator: IMLValidator, rate: str) -> None:
        result = validator.validate(f'<utterance><prosody rate="{rate}">x</prosody></utterance>')
        assert result.valid is True

    @pytest.mark.parametrize("rate", ["ludicrous", "Fast", "+150%", "x-fast", "150"])
    def test_invalid_rate(self, validator: IMLValidator, rate: str) -> None:
        result = validator.validate(f'<utterance><prosody rate="{rate}">x</prosody></utterance>')
        assert _error_rules(result) == {"V22"}

    def test_invalid_pitch_contour_and_quality(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody pitch_contour="wobble" quality="robotic">x</prosody></utterance>'
        )
        assert _error_rules(result) == {"V23", "V24"}

    def test_invalid_tempo_and_rhythm(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><segment tempo="glacial" rhythm="jazzy">x</segment></utterance>'
        )
        assert _error_rules(result) == {"V25", "V26"}

    def test_spec_vocabularies_accepted(self, validator: IMLValidator) -> None:
        contours = ["rise", "fall", "rise-fall", "fall-rise", "fall-sharp", "rise-sharp", "flat"]
        qualities = ["modal", "breathy", "tense", "creaky", "whispery", "harsh"]
        body = "".join(
            f'<prosody pitch_contour="{c}" quality="{q}">w</prosody> '
            for c, q in zip(contours, qualities + ["modal"], strict=True)
        )
        body += "".join(
            f'<segment tempo="{t}" rhythm="{r}">s</segment>'
            for t, r in zip(
                ["rushed", "steady", "drawn-out"], ["staccato", "legato", "syncopated"], strict=True
            )
        )
        result = validator.validate(f"<utterance>{body}</utterance>")
        assert result.issues == []

    @pytest.mark.parametrize(
        ("attr", "value"),
        [
            ("f0_mean", "abc"),
            ("f0_mean", "-120"),
            ("f0_range", "high"),
            ("f0_range", "120 - 240"),
            ("f0_contour", "150, 160"),
            ("intensity_mean", "NaN"),
            ("intensity_range", "-3"),
            ("speech_rate", "fast"),
            ("duration_ms", "-1"),
            ("duration_ms", "12.5"),
            pytest.param("duration_ms", "1" * 5000, id="duration_ms-5000-digits"),
            ("duration_ms", "2147483648"),
            ("f0_mean", "1e400"),
            ("intensity_mean", "-1e400"),
            ("jitter", "-0.5"),
            ("shimmer", "1%"),
            ("hnr", "inf"),
        ],
    )
    def test_invalid_extended_attribute(
        self, validator: IMLValidator, attr: str, value: str
    ) -> None:
        result = validator.validate(
            f'<utterance><prosody {attr}="{value}">x</prosody></utterance>'
        )
        assert _error_rules(result) == {"V27"}
        assert attr in result.errors[0].message

    def test_valid_extended_attributes(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody f0_mean="185" f0_range="120-240" f0_contour="150,165.5,180"'
            ' intensity_mean="-12.5" intensity_range="15" speech_rate="4.2" duration_ms="840"'
            ' jitter="1.2e-1" shimmer="3.8" hnr="-2">x</prosody></utterance>'
        )
        assert result.issues == []

    @pytest.mark.parametrize("version", ["0.1.0", "0.1.0-alpha", "1.2.3-rc.1+build.5"])
    def test_valid_versions(self, validator: IMLValidator, version: str) -> None:
        result = validator.validate(f'<iml version="{version}"><utterance>x</utterance></iml>')
        assert result.valid is True

    def test_invalid_version_and_language(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<iml version="0.1" language="en_US"><utterance>x</utterance></iml>'
        )
        assert _error_rules(result) == {"V28", "V29"}


# ---------------------------------------------------------------------------
# Document structure (V19, V20, V21)
# ---------------------------------------------------------------------------


class TestStructure:
    def test_markup_directly_in_iml_is_error(self, validator: IMLValidator) -> None:
        result = validator.validate(
            "<iml><utterance>Hello</utterance>"
            '<prosody pitch="+20%">IMPORTANT WARNING</prosody></iml>'
        )
        assert result.valid is False
        assert _error_rules(result) == {"V19"}

    def test_text_directly_in_iml_is_error(self, validator: IMLValidator) -> None:
        result = validator.validate("<iml>stray text<utterance>x</utterance></iml>")
        assert _error_rules(result) == {"V19"}
        assert "stray text" in result.errors[0].message

    def test_whitespace_and_comments_in_iml_ok(self, validator: IMLValidator) -> None:
        result = validator.validate(
            "<iml>\n  <!-- first -->\n  <utterance>x</utterance>\n  <?pi y?>\n</iml>"
        )
        assert result.issues == []

    def test_nested_utterance_is_error(self, validator: IMLValidator) -> None:
        result = validator.validate("<utterance>a<utterance>b</utterance></utterance>")
        assert _error_rules(result) == {"V20"}

    def test_nested_utterance_is_still_checked(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance>a<utterance emotion="sad">b</utterance></utterance>'
        )
        assert _error_rules(result) == {"V20", "V3"}

    def test_iml_inside_utterance_is_error(self, validator: IMLValidator) -> None:
        result = validator.validate("<utterance><iml><utterance>x</utterance></iml></utterance>")
        assert _error_rules(result) == {"V20"}

    def test_iml_inside_iml_is_error(self, validator: IMLValidator) -> None:
        result = validator.validate(
            "<iml><utterance>x</utterance><iml><utterance>y</utterance></iml></iml>"
        )
        assert _error_rules(result) == {"V20"}

    def test_emphasis_in_emphasis_is_error(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><emphasis level="strong">'
            '<emphasis level="moderate">x</emphasis></emphasis></utterance>'
        )
        assert _error_rules(result) == {"V21"}

    def test_emphasis_in_prosody_in_emphasis_ok(self, validator: IMLValidator) -> None:
        # Only a direct child is forbidden; depth 3 is a V12 warning.
        result = validator.validate(
            '<utterance><emphasis level="strong"><prosody pitch="+5%">'
            '<emphasis level="moderate">x</emphasis></prosody></emphasis></utterance>'
        )
        assert result.valid is True
        assert {i.rule for i in result.warnings} == {"V12"}

    def test_pause_in_emphasis_and_prosody_in_prosody_ok(
        self, validator: IMLValidator
    ) -> None:
        result = validator.validate(
            '<utterance><emphasis level="strong">never<pause duration="300"/> again</emphasis>'
            '<prosody pitch="+5%"><prosody volume="+3dB">x</prosody></prosody></utterance>'
        )
        assert result.issues == []

    def test_pause_comment_is_not_content(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance>a<pause duration="5"><!-- c --></pause></utterance>'
        )
        assert result.issues == []

    def test_utterance_root_in_iml_namespace_ok(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<iml:utterance xmlns:iml="http://prosody-protocol.org/iml/0.1"'
            ' emotion="calm" confidence="0.94">Hello'
            ' <iml:emphasis level="strong">world</iml:emphasis>.</iml:utterance>'
        )
        assert result.issues == []

    def test_foreign_namespace_root_is_not_iml(self, validator: IMLValidator) -> None:
        result = validator.validate('<x:utterance xmlns:x="urn:other">Hello</x:utterance>')
        assert _error_rules(result) == {"V2"}


# ---------------------------------------------------------------------------
# Unknown elements and attributes (V16, V32, spec 6.2, 9.2)
# ---------------------------------------------------------------------------


class TestUnknownContent:
    def test_errors_inside_unknown_element_are_found(self, validator: IMLValidator) -> None:
        """Unknown elements are transparent: they must not hide invalid IML."""
        result = validator.validate(
            '<utterance><prosody pitch="+5%"><span><segment>x</segment><pause/></span>'
            "</prosody></utterance>"
        )
        assert result.valid is False
        assert _error_rules(result) == {"V10", "V5"}
        assert any(i.rule == "V16" and i.severity == "info" for i in result.issues)

    def test_segment_through_unknown_wrapper_is_direct_child(
        self, validator: IMLValidator
    ) -> None:
        result = validator.validate("<utterance><span><segment>x</segment></span></utterance>")
        assert result.valid is True

    def test_foreign_namespace_element_is_unknown_not_iml(
        self, validator: IMLValidator
    ) -> None:
        result = validator.validate(
            '<utterance><prosody pitch="+5%"><f:segment xmlns:f="urn:other">x</f:segment>'
            "</prosody></utterance>"
        )
        assert result.valid is True
        v16 = [i for i in result.issues if i.rule == "V16"]
        assert len(v16) == 1
        assert "f:segment" in v16[0].message

    def test_utterance_inside_unknown_wrapper_in_iml_counts(
        self, validator: IMLValidator
    ) -> None:
        result = validator.validate(
            '<iml><f:turn xmlns:f="urn:x">note<utterance>x</utterance></f:turn></iml>'
        )
        assert result.valid is True
        assert not any(i.rule == "V2" for i in result.issues)

    def test_markup_inside_unknown_wrapper_in_iml_is_error(
        self, validator: IMLValidator
    ) -> None:
        result = validator.validate(
            '<iml><utterance>x</utterance><meta><prosody pitch="+5%">y</prosody></meta></iml>'
        )
        assert _error_rules(result) == {"V19"}

    def test_x_and_namespaced_attributes_are_silent(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance xmlns:a="urn:a" a:note="n" x-id="7">'
            '<prosody x-formant-shift="+200Hz" x-nasality="0.7">x</prosody></utterance>'
        )
        assert result.issues == []

    def test_unknown_unprefixed_attribute_warns(self, validator: IMLValidator) -> None:
        result = validator.validate('<utterance><prosody pich="+5%">x</prosody></utterance>')
        assert result.valid is True
        assert [(i.severity, i.rule) for i in result.issues] == [("warning", "V32")]
        assert "pich" in result.issues[0].message

    def test_attribute_in_iml_namespace_warns(self, validator: IMLValidator) -> None:
        """Spec 2.3: iml:emotion is not the emotion attribute, so it would be lost."""
        result = validator.validate(
            '<iml:utterance xmlns:iml="http://prosody-protocol.org/iml/0.1"'
            ' iml:emotion="angry" iml:confidence="0.9">x</iml:utterance>'
        )
        assert result.valid is True
        assert [(i.severity, i.rule) for i in result.issues] == [
            ("warning", "V32"),
            ("warning", "V32"),
        ]
        assert "'iml:emotion'" in result.issues[0].message
        assert "never namespace-qualified" in result.issues[0].message

    def test_extended_attribute_on_wrong_element_warns(self, validator: IMLValidator) -> None:
        result = validator.validate('<utterance f0_mean="120">x</utterance>')
        assert [i.rule for i in result.warnings] == ["V32"]


# ---------------------------------------------------------------------------
# Comments, processing instructions, entities (spec 2.5)
# ---------------------------------------------------------------------------


class TestNonElementNodes:
    @pytest.mark.parametrize(
        "iml",
        [
            "<iml><?pi x?><utterance>hi</utterance></iml>",
            "<iml><utterance>hi</utterance><?pi x?></iml>",
            "<utterance>hi<?pi x?> there</utterance>",
            "<utterance>hi<!-- note --> there</utterance>",
            "<iml><!-- note --><utterance>hi</utterance></iml>",
        ],
    )
    def test_pi_and_comments_are_ignored(self, validator: IMLValidator, iml: str) -> None:
        result = validator.validate(iml)
        assert result.valid is True
        assert result.issues == []

    def test_doctype_is_error(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<!DOCTYPE utterance [<!ENTITY who "Bob">]><utterance>Hi &who; there</utterance>'
        )
        assert result.valid is False
        assert _error_rules(result) == {"V31"}

    def test_billion_laughs_is_rejected(self, validator: IMLValidator) -> None:
        entities = '<!ENTITY lol "lol">' + "".join(
            f'<!ENTITY lol{i} "{("&lol%s;" % (i - 1 or "")) * 10}">' for i in range(1, 10)
        )
        result = validator.validate(
            f"<!DOCTYPE utterance [{entities}]><utterance>&lol9;</utterance>"
        )
        assert result.valid is False
        assert _error_rules(result) & {"V1", "V31"}


# ---------------------------------------------------------------------------
# Encoding (V30, spec 2.2)
# ---------------------------------------------------------------------------


class TestEncoding:
    def test_non_utf8_declaration_is_error(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<?xml version="1.0" encoding="ISO-8859-1"?><utterance>caf\u00e9</utterance>'
        )
        assert result.valid is False
        assert _error_rules(result) == {"V30"}
        assert "ISO-8859-1" in result.errors[0].message

    @pytest.mark.parametrize("declared", ["UTF-8", "utf-8", "UTF8"])
    def test_utf8_declaration_ok(self, validator: IMLValidator, declared: str) -> None:
        result = validator.validate(
            f"<?xml version='1.0' encoding='{declared}'?><utterance>caf\u00e9</utterance>"
        )
        assert result.issues == []

    def test_lone_surrogate_is_error_not_exception(self, validator: IMLValidator) -> None:
        result = validator.validate("<utterance>\ud800</utterance>")
        assert [i.rule for i in result.issues] == ["V30"]


# ---------------------------------------------------------------------------
# V33: plausible values (spec 6.4)
# ---------------------------------------------------------------------------


def _v33(result: ValidationResult) -> list[ValidationIssue]:
    return [i for i in result.issues if i.rule == "V33"]


class TestV33Plausibility:
    @pytest.mark.parametrize(
        ("markup", "attr"),
        [
            ('<prosody pitch="+24.5st">x</prosody>', "pitch"),
            ('<prosody pitch="-30st">x</prosody>', "pitch"),
            ('<prosody pitch="+301%">x</prosody>', "pitch"),
            ('<prosody pitch="-76%">x</prosody>', "pitch"),
            ('<prosody pitch="-150%">x</prosody>', "pitch"),  # a negative frequency
            ('<prosody pitch="39Hz">x</prosody>', "pitch"),
            ('<prosody pitch="0Hz">x</prosody>', "pitch"),
            ('<prosody pitch="1500Hz">x</prosody>', "pitch"),
            ('<prosody volume="+41dB">x</prosody>', "volume"),
            ('<prosody volume="-80dB">x</prosody>', "volume"),
            ('<prosody rate="1000%">x</prosody>', "rate"),
            ('<prosody rate="10%">x</prosody>', "rate"),
            ('a<pause duration="60001"/>b', "duration"),
            ('<prosody f0_mean="0">x</prosody>', "f0_mean"),
            ('<prosody f0_mean="2400">x</prosody>', "f0_mean"),
            ('<prosody f0_range="20-240">x</prosody>', "f0_range"),
            ('<prosody f0_range="240-120">x</prosody>', "f0_range"),
            ('<prosody f0_contour="150,165,3000">x</prosody>', "f0_contour"),
            ('<prosody speech_rate="25">x</prosody>', "speech_rate"),
            ('<prosody duration_ms="3600001">x</prosody>', "duration_ms"),
        ],
    )
    def test_implausible_value_warns(
        self, validator: IMLValidator, markup: str, attr: str
    ) -> None:
        result = validator.validate(f"<utterance>{markup}</utterance>")
        assert result.valid is True
        assert result.errors == []
        (issue,) = _v33(result)
        assert issue.severity == "warning"
        assert issue.message.startswith(f"{attr}=")
        assert "spec 6.4" in issue.message and issue.line == 1

    @pytest.mark.parametrize(
        "markup",
        [
            '<prosody pitch="+24st">x</prosody>',
            '<prosody pitch="-24st">x</prosody>',
            '<prosody pitch="+300%">x</prosody>',
            '<prosody pitch="-75%">x</prosody>',
            '<prosody pitch="40Hz">x</prosody>',
            '<prosody pitch="1200Hz">x</prosody>',
            '<prosody volume="+40dB">x</prosody>',
            '<prosody volume="-40dB">x</prosody>',
            '<prosody rate="25%">x</prosody>',
            '<prosody rate="400%">x</prosody>',
            'a<pause duration="60000"/>b',
            '<prosody f0_mean="40" f0_range="40-1200" f0_contour="40,1200">x</prosody>',
            '<prosody f0_range="120-120">x</prosody>',
            '<prosody speech_rate="20" duration_ms="3600000">x</prosody>',
            '<prosody speech_rate="0">x</prosody>',
        ],
    )
    def test_limits_are_inclusive(self, validator: IMLValidator, markup: str) -> None:
        assert validator.validate(f"<utterance>{markup}</utterance>").issues == []

    def test_each_implausible_attribute_is_reported(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance><prosody pitch="+30st" volume="+50dB" speech_rate="30">x'
            '<pause duration="90000"/></prosody></utterance>'
        )
        assert result.valid is True
        assert sorted(i.message.split("=")[0] for i in _v33(result)) == [
            "duration", "pitch", "speech_rate", "volume",
        ]

    @pytest.mark.parametrize(
        ("markup", "rule"),
        [
            ('<prosody pitch="+300">x</prosody>', "V13"),
            ('<prosody volume="+99">x</prosody>', "V14"),
            ('<prosody rate="-900%">x</prosody>', "V22"),
            ('<prosody f0_mean="-9000">x</prosody>', "V27"),
            ('a<pause duration="99999999999"/>b', "V6"),
        ],
    )
    def test_invalid_values_are_errors_not_warnings(
        self, validator: IMLValidator, markup: str, rule: str
    ) -> None:
        """A value that breaks its type is left to its error rule."""
        result = validator.validate(f"<utterance>{markup}</utterance>")
        assert [i.rule for i in result.issues] == [rule]

    @pytest.mark.parametrize(
        ("markup", "attr"),
        [
            (f'<prosody f0_range="{"9" * 400}-240">x</prosody>', "f0_range"),
            (f'<prosody f0_range="120-{"9" * 400}">x</prosody>', "f0_range"),
            (f'<prosody f0_contour="120,{"9" * 400}">x</prosody>', "f0_contour"),
            (f'<prosody f0_contour="{"9" * 400}">x</prosody>', "f0_contour"),
            (f'<prosody pitch="{"9" * 400}Hz">x</prosody>', "pitch"),
            (f'<prosody pitch="+{"9" * 400}st">x</prosody>', "pitch"),
            (f'<prosody pitch="-{"9" * 400}%">x</prosody>', "pitch"),
            (f'<prosody volume="+{"9" * 400}dB">x</prosody>', "volume"),
            (f'<prosody rate="{"9" * 400}%">x</prosody>', "rate"),
        ],
        ids=["range-low", "range-high", "contour", "contour-one", "hz", "st", "percent",
             "volume", "rate"],
    )
    def test_numbers_too_long_for_a_float_warn(
        self, validator: IMLValidator, markup: str, attr: str
    ) -> None:
        """A value of 309 or more digits is an infinite float; formatting
        it for the f0_range message raised OverflowError (HTTP 500 from
        /v1/validate) where the document used to validate cleanly."""
        result = validator.validate(f"<utterance>{markup}</utterance>")
        assert result.valid is True
        issues = _v33(result)  # f0_range="999...-240" also has its low value above its high
        assert issues and all(i.message.startswith(f"{attr}=") for i in issues)

    def test_long_value_is_quoted_as_written(self, validator: IMLValidator) -> None:
        result = validator.validate(
            f'<utterance><prosody f0_contour="120,{"9" * 400}">x</prosody></utterance>'
        )
        (issue,) = _v33(result)
        assert f"has {'9' * 17}... Hz, outside 40-1200 Hz" in issue.message
        assert "inf" not in issue.message

    def test_long_pause_is_not_called_an_error(self, validator: IMLValidator) -> None:
        """A minute of silence is real (a voicemail, an interview), not a
        measurement error; the warning says what to write instead."""
        result = validator.validate('<utterance>a<pause duration="62000"/>b</utterance>')
        (issue,) = _v33(result)
        assert "should end the utterance, or be written as a pause of at most 60000 ms" in (
            issue.message
        )
        assert "error" not in issue.message

    def test_realistic_research_annotation_is_silent(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="frustrated" confidence="0.92"><prosody f0_mean="220"'
            ' f0_range="180-310" f0_contour="190,240,310,260" intensity_mean="72"'
            ' speech_rate="5.1" duration_ms="1800" jitter="2.1" shimmer="4.5" hnr="12">I'
            ' <prosody pitch="+18%" volume="+8dB" rate="120%">again</prosody>'
            '<pause duration="2500"/></prosody></utterance>'
        )
        assert result.issues == []


# ---------------------------------------------------------------------------
# ValidationResult helpers
# ---------------------------------------------------------------------------


class TestValidationResultHelpers:
    def test_errors_and_warnings(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="excited"><prosody pich="+5%">x</prosody></utterance>'
        )
        assert [i.rule for i in result.errors] == ["V3"]
        assert [i.rule for i in result.warnings] == ["V32"]
        assert any(i.severity == "info" for i in result.issues)  # V15 is neither

    def test_readme_pattern(self, validator: IMLValidator) -> None:
        result = validator.validate(
            '<utterance emotion="happy"><prosody pitch="+20%">This is great!</prosody></utterance>'
        )
        report = "Valid" if result.valid else f"Errors: {result.errors}"
        assert report.startswith("Errors:")
        assert "V3" in report

    def test_raise_for_errors_valid(self, validator: IMLValidator) -> None:
        validator.validate("<utterance>fine</utterance>").raise_for_errors()

    def test_raise_for_errors_invalid(self, validator: IMLValidator) -> None:
        result = validator.validate('<utterance emotion="angry"><pause/></utterance>')
        with pytest.raises(IMLValidationError) as info:
            result.raise_for_errors()
        assert [i.rule for i in info.value.issues] == ["V3", "V5"]
        assert "V3" in str(info.value)
        assert "V5" in str(info.value)

    def test_raise_for_errors_ignores_warnings(self, validator: IMLValidator) -> None:
        result = validator.validate('<utterance><prosody pich="+5%">x</prosody></utterance>')
        assert result.warnings
        result.raise_for_errors()

    def test_raise_for_errors_truncates_long_lists(self, validator: IMLValidator) -> None:
        result = validator.validate("<utterance>" + "<pause/>" * 8 + "</utterance>")
        with pytest.raises(IMLValidationError, match="and 3 more") as info:
            result.raise_for_errors()
        assert len(info.value.issues) == 8

    def test_validation_error_default_issues(self) -> None:
        assert IMLValidationError("bad").issues == ()
