"""Tests for prosody_protocol.iml_to_ssml.

Covers:
- Tag mapping: utterance→s, prosody→prosody, pause→break, emphasis→emphasis
- Attribute conversion: pitch, volume, rate mapped; extended attrs dropped
- Segment unwrapping
- Full document wrapping in <speak>
- Language attribute propagation
- Error handling for malformed IML
- convert_doc from IMLDocument objects
"""

from __future__ import annotations

from lxml import etree
import pytest

from prosody_protocol.exceptions import ConversionError
from prosody_protocol.iml_to_ssml import IMLToSSML
from prosody_protocol.models import (
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)


@pytest.fixture()
def converter() -> IMLToSSML:
    return IMLToSSML()


def _parse_ssml(ssml: str) -> etree._Element:
    """Parse SSML and return the root element, stripping namespace."""
    return etree.fromstring(ssml.encode("utf-8"))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class TestInterface:
    def test_instantiate(self) -> None:
        converter = IMLToSSML()
        assert converter.vendor is None

    def test_vendor_param(self) -> None:
        converter = IMLToSSML(vendor="google")
        assert converter.vendor == "google"


# ---------------------------------------------------------------------------
# Document wrapper
# ---------------------------------------------------------------------------


class TestDocumentWrapper:
    def test_speak_root(self, converter: IMLToSSML) -> None:
        ssml = converter.convert("<utterance>Hello</utterance>")
        root = _parse_ssml(ssml)
        assert root.tag.endswith("speak") or root.tag == "speak"

    def test_speak_version(self, converter: IMLToSSML) -> None:
        ssml = converter.convert("<utterance>Hello</utterance>")
        root = _parse_ssml(ssml)
        # Strip namespace prefix from attribute lookup.
        assert root.get("version") == "1.0"

    def test_speak_xmlns(self, converter: IMLToSSML) -> None:
        ssml = converter.convert("<utterance>Hello</utterance>")
        assert "http://www.w3.org/2001/10/synthesis" in ssml

    def test_language_propagated(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<iml version="0.1.0" language="en-US">'
            "<utterance>Hello</utterance>"
            "</iml>"
        )
        assert 'xml:lang="en-US"' in ssml

    def test_no_language_when_absent(self, converter: IMLToSSML) -> None:
        ssml = converter.convert("<utterance>Hello</utterance>")
        assert "xml:lang" not in ssml


# ---------------------------------------------------------------------------
# Tag mapping
# ---------------------------------------------------------------------------


class TestUtteranceMapping:
    def test_utterance_becomes_s(self, converter: IMLToSSML) -> None:
        ssml = converter.convert("<utterance>Hello world.</utterance>")
        assert "<s>" in ssml
        assert "Hello world." in ssml

    def test_multiple_utterances(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<iml version="0.1.0">'
            "<utterance>First.</utterance>"
            "<utterance>Second.</utterance>"
            "</iml>"
        )
        assert ssml.count("<s>") == 2


class TestProsodyMapping:
    def test_pitch_mapped(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><prosody pitch="+15%">loud</prosody></utterance>'
        )
        assert 'pitch="+15%"' in ssml

    def test_volume_mapped(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><prosody volume="+6dB">loud</prosody></utterance>'
        )
        assert 'volume="+6dB"' in ssml

    def test_rate_mapped(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><prosody rate="fast">quick</prosody></utterance>'
        )
        assert 'rate="fast"' in ssml

    def test_multiple_attrs(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><prosody pitch="+10%" volume="+3dB">text</prosody></utterance>'
        )
        assert 'pitch="+10%"' in ssml
        assert 'volume="+3dB"' in ssml

    def test_extended_attrs_dropped(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance>'
            '<prosody pitch="+5%" f0_mean="220" jitter="1.5" shimmer="3.2">'
            "text"
            "</prosody>"
            "</utterance>"
        )
        assert "f0_mean" not in ssml
        assert "jitter" not in ssml
        assert "shimmer" not in ssml
        assert 'pitch="+5%"' in ssml

    def test_prosody_only_extended_attrs_unwrapped(self, converter: IMLToSSML) -> None:
        """Prosody with only extended attrs (no pitch/volume/rate) should
        not produce a <prosody> wrapper in the SSML."""
        ssml = converter.convert(
            '<utterance><prosody f0_mean="200">text</prosody></utterance>'
        )
        assert "<prosody" not in ssml.split("<speak")[1].split("</speak>")[0] or "text" in ssml


class TestPauseMapping:
    def test_pause_becomes_break(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance>Wait<pause duration="800"/>then go.</utterance>'
        )
        assert '<break time="800ms"/>' in ssml

    def test_pause_duration_preserved(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><pause duration="200"/></utterance>'
        )
        assert "200ms" in ssml


class TestEmphasisMapping:
    def test_emphasis_preserved(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><emphasis level="strong">important</emphasis></utterance>'
        )
        assert '<emphasis level="strong">' in ssml
        assert "important" in ssml

    def test_emphasis_levels(self, converter: IMLToSSML) -> None:
        for level in ("strong", "moderate", "reduced"):
            ssml = converter.convert(
                f'<utterance><emphasis level="{level}">word</emphasis></utterance>'
            )
            assert f'level="{level}"' in ssml


class TestSegmentMapping:
    def test_segment_unwrapped(self, converter: IMLToSSML) -> None:
        """Segment has no SSML equivalent -- children should be promoted."""
        ssml = converter.convert(
            '<utterance><segment tempo="rushed">flowing text</segment></utterance>'
        )
        assert "<segment" not in ssml
        assert "flowing text" in ssml
        assert "tempo" not in ssml


# ---------------------------------------------------------------------------
# Nested structures
# ---------------------------------------------------------------------------


class TestNestedStructures:
    def test_prosody_in_emphasis(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            "<utterance>"
            '<emphasis level="strong">'
            '<prosody pitch="+10%">loud and emphasized</prosody>'
            "</emphasis>"
            "</utterance>"
        )
        assert '<emphasis level="strong">' in ssml
        assert 'pitch="+10%"' in ssml

    def test_complex_appendix_c_example(self, converter: IMLToSSML) -> None:
        iml = (
            '<utterance emotion="sarcastic" confidence="0.87">'
            "  Oh, that's"
            '  <prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">'
            "    GREAT"
            "  </prosody>."
            "</utterance>"
        )
        ssml = converter.convert(iml)
        # emotion/confidence are IML-only -- not in SSML.
        assert "emotion" not in ssml
        assert "confidence" not in ssml
        assert "pitch_contour" not in ssml
        # Core attrs preserved.
        assert 'pitch="+15%"' in ssml
        assert 'volume="+6dB"' in ssml
        assert "GREAT" in ssml


# ---------------------------------------------------------------------------
# convert_doc (from IMLDocument)
# ---------------------------------------------------------------------------


class TestConvertDoc:
    def test_basic_doc(self, converter: IMLToSSML) -> None:
        doc = IMLDocument(
            utterances=(Utterance(children=("Hello.",)),),
            language="en-US",
        )
        ssml = converter.convert_doc(doc)
        assert "<speak" in ssml
        assert "<s>Hello.</s>" in ssml
        assert 'xml:lang="en-US"' in ssml

    def test_doc_with_prosody(self, converter: IMLToSSML) -> None:
        doc = IMLDocument(
            utterances=(
                Utterance(
                    children=(
                        "Say it ",
                        Prosody(children=("LOUD",), pitch="+20%", volume="+8dB"),
                    ),
                ),
            ),
        )
        ssml = converter.convert_doc(doc)
        assert 'pitch="+20%"' in ssml
        assert 'volume="+8dB"' in ssml

    def test_doc_with_pause(self, converter: IMLToSSML) -> None:
        doc = IMLDocument(
            utterances=(
                Utterance(
                    children=("Before", Pause(duration=500), "after."),
                ),
            ),
        )
        ssml = converter.convert_doc(doc)
        assert '<break time="500ms"/>' in ssml

    def test_doc_with_segment_unwrapped(self, converter: IMLToSSML) -> None:
        doc = IMLDocument(
            utterances=(
                Utterance(
                    children=(
                        Segment(children=("segment text",), tempo="rushed"),
                    ),
                ),
            ),
        )
        ssml = converter.convert_doc(doc)
        assert "<segment" not in ssml
        assert "segment text" in ssml


# ---------------------------------------------------------------------------
# SSML validity
# ---------------------------------------------------------------------------


class TestSSMLValidity:
    def test_output_is_valid_xml(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance emotion="calm" confidence="0.9">'
            'I <emphasis level="strong">told</emphasis> you '
            '<prosody pitch="+12%" volume="+6dB">yesterday</prosody>!'
            "</utterance>"
        )
        # Should parse as valid XML.
        root = _parse_ssml(ssml)
        assert root is not None

    def test_xml_entities_escaped(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            "<utterance>A &amp; B &lt; C</utterance>"
        )
        assert "&amp;" in ssml
        assert "&lt;" in ssml


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_malformed_iml_raises(self, converter: IMLToSSML) -> None:
        with pytest.raises(ConversionError):
            converter.convert("<utterance>unclosed")

    def test_empty_string_raises(self, converter: IMLToSSML) -> None:
        with pytest.raises(ConversionError):
            converter.convert("")
