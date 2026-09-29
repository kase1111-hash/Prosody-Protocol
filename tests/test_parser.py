"""Tests for prosody_protocol.parser.

Covers:
- Parsing all Appendix C examples from the spec
- Round-trip: parse -> serialize -> parse produces identical documents,
  including consent/processing, extension attributes and invalid values
- Malformed XML raises IMLParseError
- Mixed content ordering (text + child interleaving)
- Extended attributes on <prosody>
- File-based parsing and UTF-8 enforcement
- Plain text extraction and whitespace normalization
- Serialization always gives well-formed XML, or raises ConversionError
- Comments, processing instructions, CDATA, unknown elements, namespaces
- XML security: DOCTYPE, external entities (file and network), billion laughs
"""

from __future__ import annotations

import http.server
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from lxml import etree

from prosody_protocol import parser as parser_module
from prosody_protocol import validator as validator_module
from prosody_protocol.exceptions import ConversionError, IMLParseError, IMLValidationError
from prosody_protocol.models import (
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)
from prosody_protocol.parser import MAX_INTEGER, IMLParser
from prosody_protocol.validator import IMLValidator


@pytest.fixture()
def parser() -> IMLParser:
    return IMLParser()


# ---------------------------------------------------------------------------
# Appendix C examples
# ---------------------------------------------------------------------------


class TestAppendixCExamples:
    """Parse every example from spec.md Appendix C."""

    def test_c1_simple_sarcasm(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<utterance emotion="sarcastic" confidence="0.87">'
            "  Oh, that's"
            '  <prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">'
            "    GREAT"
            "  </prosody>."
            "</utterance>"
        )
        assert len(doc.utterances) == 1
        utt = doc.utterances[0]
        assert utt.emotion == "sarcastic"
        assert utt.confidence == 0.87

        prosody_nodes = [c for c in utt.children if isinstance(c, Prosody)]
        assert len(prosody_nodes) == 1
        assert prosody_nodes[0].pitch == "+15%"
        assert prosody_nodes[0].volume == "+6dB"
        assert prosody_nodes[0].pitch_contour == "fall-sharp"

    def test_c2_multi_speaker(self, parser: IMLParser, multi_speaker: str) -> None:
        doc = parser.parse(multi_speaker)
        assert doc.version == "0.1.0"
        assert doc.language == "en-US"
        assert len(doc.utterances) == 2
        assert doc.utterances[0].emotion == "neutral"
        assert doc.utterances[0].speaker_id == "agent"
        assert doc.utterances[1].emotion == "frustrated"
        assert doc.utterances[1].speaker_id == "caller"

        caller_children_types = [
            type(c).__name__
            for c in doc.utterances[1].children
            if not isinstance(c, str)
        ]
        assert "Prosody" in caller_children_types
        assert "Emphasis" in caller_children_types

    def test_c3_accessibility(self, parser: IMLParser, segment_example: str) -> None:
        doc = parser.parse(segment_example)
        assert len(doc.utterances) == 1
        utt = doc.utterances[0]
        assert utt.emotion == "excitement"
        assert utt.confidence == 0.78
        assert utt.speaker_id == "user_789"

        segments = [c for c in utt.children if isinstance(c, Segment)]
        assert len(segments) == 1
        assert segments[0].tempo == "rushed"
        assert segments[0].rhythm == "legato"

    def test_c4_research_grade(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<utterance emotion="frustrated" confidence="0.92">'
            '  <prosody f0_mean="220" f0_range="180-310" intensity_mean="72"'
            '           intensity_range="15" speech_rate="5.1" jitter="2.1" shimmer="4.5">'
            '    I <emphasis level="strong">can\'t believe</emphasis>'
            '    <pause duration="350"/>'
            "    this happened"
            '    <prosody pitch="+18%" volume="+8dB" pitch_contour="rise-fall">again</prosody>!'
            "  </prosody>"
            "</utterance>"
        )
        utt = doc.utterances[0]
        outer_prosody = [c for c in utt.children if isinstance(c, Prosody)]
        assert len(outer_prosody) == 1
        p = outer_prosody[0]
        assert p.f0_mean == 220.0
        assert p.f0_range == "180-310"
        assert p.intensity_mean == 72.0
        assert p.intensity_range == 15.0
        assert p.speech_rate == 5.1
        assert p.jitter == 2.1
        assert p.shimmer == 4.5

        emphasis_nodes = [c for c in p.children if isinstance(c, Emphasis)]
        pause_nodes = [c for c in p.children if isinstance(c, Pause)]
        inner_prosody = [c for c in p.children if isinstance(c, Prosody)]
        assert len(emphasis_nodes) == 1
        assert emphasis_nodes[0].level == "strong"
        assert len(pause_nodes) == 1
        assert pause_nodes[0].duration == 350
        assert len(inner_prosody) == 1
        assert inner_prosody[0].pitch == "+18%"


# ---------------------------------------------------------------------------
# Standalone <utterance> parsing
# ---------------------------------------------------------------------------


class TestStandaloneUtterance:
    def test_simple_text_only(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance>Hello world.</utterance>")
        assert len(doc.utterances) == 1
        assert doc.utterances[0].children == ("Hello world.",)
        assert doc.utterances[0].emotion is None
        assert doc.utterances[0].confidence is None

    def test_with_emotion_and_confidence(
        self, parser: IMLParser, simple_utterance: str
    ) -> None:
        doc = parser.parse(simple_utterance)
        utt = doc.utterances[0]
        assert utt.emotion == "frustrated"
        assert utt.confidence == 0.92

    def test_document_has_no_wrapper_metadata(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance>hi</utterance>")
        assert doc.version is None
        assert doc.language is None


# ---------------------------------------------------------------------------
# Mixed content ordering
# ---------------------------------------------------------------------------


class TestMixedContent:
    def test_text_prosody_text(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<utterance>Before <prosody pitch="+5%">middle</prosody> after.</utterance>'
        )
        children = doc.utterances[0].children
        assert children[0] == "Before "
        assert isinstance(children[1], Prosody)
        assert children[1].children == ("middle",)
        assert children[2] == " after."

    def test_adjacent_elements(self, parser: IMLParser) -> None:
        doc = parser.parse(
            "<utterance>"
            '<emphasis level="strong">word1</emphasis>'
            '<pause duration="200"/>'
            '<prosody pitch="+3%">word2</prosody>'
            "</utterance>"
        )
        children = doc.utterances[0].children
        assert isinstance(children[0], Emphasis)
        assert isinstance(children[1], Pause)
        assert isinstance(children[2], Prosody)

    def test_pause_inserts_space_in_plain_text(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<utterance>Well<pause duration="800"/>I suppose.</utterance>'
        )
        text = parser.to_plain_text(doc)
        assert text == "Well I suppose."


# ---------------------------------------------------------------------------
# Round-trip: parse -> serialize -> parse
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_simple_utterance_round_trip(self, parser: IMLParser) -> None:
        original = '<utterance emotion="calm" confidence="0.9">Hello.</utterance>'
        doc1 = parser.parse(original)
        xml = parser.to_iml_string(doc1)
        doc2 = parser.parse(xml)
        assert doc1 == doc2

    def test_multi_utterance_round_trip(
        self, parser: IMLParser, multi_speaker: str
    ) -> None:
        doc1 = parser.parse(multi_speaker)
        xml = parser.to_iml_string(doc1)
        doc2 = parser.parse(xml)
        assert doc1.version == doc2.version
        assert doc1.language == doc2.language
        assert len(doc1.utterances) == len(doc2.utterances)
        for u1, u2 in zip(doc1.utterances, doc2.utterances, strict=True):
            assert u1.emotion == u2.emotion
            assert u1.confidence == u2.confidence
            assert u1.speaker_id == u2.speaker_id

    def test_segment_round_trip(
        self, parser: IMLParser, segment_example: str
    ) -> None:
        doc1 = parser.parse(segment_example)
        xml = parser.to_iml_string(doc1)
        doc2 = parser.parse(xml)
        seg1 = [c for c in doc1.utterances[0].children if isinstance(c, Segment)][0]
        seg2 = [c for c in doc2.utterances[0].children if isinstance(c, Segment)][0]
        assert seg1.tempo == seg2.tempo
        assert seg1.rhythm == seg2.rhythm

    def test_extended_attrs_round_trip(self, parser: IMLParser) -> None:
        original = (
            '<utterance emotion="angry" confidence="0.8">'
            '<prosody f0_mean="200" jitter="1.5" shimmer="3.2" hnr="15.0">'
            "loud words"
            "</prosody>"
            "</utterance>"
        )
        doc1 = parser.parse(original)
        xml = parser.to_iml_string(doc1)
        doc2 = parser.parse(xml)
        p1 = [c for c in doc1.utterances[0].children if isinstance(c, Prosody)][0]
        p2 = [c for c in doc2.utterances[0].children if isinstance(c, Prosody)][0]
        assert p1.f0_mean == p2.f0_mean
        assert p1.jitter == p2.jitter
        assert p1.shimmer == p2.shimmer
        assert p1.hnr == p2.hnr


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_malformed_xml(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError):
            parser.parse("<utterance>unclosed")

    def test_wrong_root_element(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="Expected root element"):
            parser.parse("<div>not iml</div>")

    def test_empty_string(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError):
            parser.parse("")

    def test_iml_parse_error_has_line_info(self, parser: IMLParser) -> None:
        try:
            parser.parse("<utterance>\n<unclosed>")
        except IMLParseError as exc:
            assert exc.line is not None


# ---------------------------------------------------------------------------
# Plain text extraction
# ---------------------------------------------------------------------------


class TestToPlainText:
    def test_strips_all_markup(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<utterance emotion="frustrated" confidence="0.9">'
            'I <emphasis level="strong">told</emphasis> you '
            '<prosody pitch="+12%" volume="+6dB">yesterday</prosody>!'
            "</utterance>"
        )
        text = parser.to_plain_text(doc)
        assert "told" in text
        assert "yesterday" in text
        assert "<" not in text

    def test_multi_utterance_plain_text(
        self, parser: IMLParser, multi_speaker: str
    ) -> None:
        doc = parser.parse(multi_speaker)
        text = parser.to_plain_text(doc)
        assert "How can I help you today?" in text
        assert "thirty minutes" in text

    def test_segment_text_extracted(
        self, parser: IMLParser, segment_example: str
    ) -> None:
        doc = parser.parse(segment_example)
        text = parser.to_plain_text(doc)
        assert "keyboard" in text


# ---------------------------------------------------------------------------
# File-based parsing
# ---------------------------------------------------------------------------


class TestParseFile:
    def test_parse_valid_fixture(
        self, parser: IMLParser, valid_fixtures_dir: Path
    ) -> None:
        doc = parser.parse_file(valid_fixtures_dir / "sarcasm.xml")
        assert len(doc.utterances) == 1
        assert doc.utterances[0].emotion == "sarcastic"

    def test_parse_multi_speaker_fixture(
        self, parser: IMLParser, valid_fixtures_dir: Path
    ) -> None:
        doc = parser.parse_file(valid_fixtures_dir / "multi_speaker.xml")
        assert len(doc.utterances) == 3
        assert doc.version == "0.1.0"

    def test_parse_all_valid_fixtures(
        self, parser: IMLParser, valid_fixtures_dir: Path
    ) -> None:
        for xml_file in sorted(valid_fixtures_dir.glob("*.xml")):
            doc = parser.parse_file(xml_file)
            assert len(doc.utterances) >= 1, f"No utterances in {xml_file.name}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestToIMLString:
    def test_single_utterance_no_wrapper(self, parser: IMLParser) -> None:
        doc = IMLDocument(utterances=(Utterance(children=("hi",)),))
        xml = parser.to_iml_string(doc)
        assert xml.startswith("<utterance>")
        assert "<iml" not in xml

    def test_multi_utterance_gets_wrapper(self, parser: IMLParser) -> None:
        doc = IMLDocument(
            utterances=(Utterance(children=("a",)), Utterance(children=("b",))),
            version="1.0.0",
            language="en-US",
        )
        xml = parser.to_iml_string(doc)
        assert xml.startswith('<iml version="1.0.0"')
        assert "en-US" in xml

    def test_serialized_xml_is_parseable(self, parser: IMLParser) -> None:
        doc = IMLDocument(
            utterances=(
                Utterance(
                    children=(
                        "Hello ",
                        Emphasis(level="strong", children=("world",)),
                        Pause(duration=500),
                        " end.",
                    ),
                    emotion="joyful",
                    confidence=0.85,
                ),
            ),
        )
        xml = parser.to_iml_string(doc)
        reparsed = parser.parse(xml)
        assert len(reparsed.utterances) == 1
        assert reparsed.utterances[0].emotion == "joyful"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_xml_comment_ignored(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance><!-- comment -->Hello</utterance>")
        assert doc.utterances[0].children == ("Hello",)

    def test_empty_utterance(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance></utterance>")
        assert doc.utterances[0].children == ()

    def test_prosody_with_no_attributes(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance><prosody>text</prosody></utterance>")
        p = doc.utterances[0].children[0]
        assert isinstance(p, Prosody)
        assert p.pitch is None
        assert p.volume is None

    def test_xml_entities_in_text(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance>A &amp; B &lt; C</utterance>")
        text = parser.to_plain_text(doc)
        assert "A & B < C" in text

    def test_cdata_is_text(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance>a <![CDATA[<b> & c]]> d</utterance>")
        assert doc.utterances[0].children == ("a <b> & c d",)
        assert parser.to_iml_string(doc) == "<utterance>a &lt;b&gt; &amp; c d</utterance>"


# ---------------------------------------------------------------------------
# Comments, processing instructions and unknown elements (spec 2.5, 6.2)
# ---------------------------------------------------------------------------


class TestNonIMLContent:
    def test_comment_text_never_becomes_content(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance>Hello <!-- secret note --> world</utterance>")
        assert doc.utterances[0].children == ("Hello  world",)
        assert "secret" not in parser.to_iml_string(doc)
        assert parser.to_plain_text(doc) == "Hello world"

    @pytest.mark.parametrize(
        "iml",
        [
            "<iml><?pi x?><utterance>hi there</utterance></iml>",
            "<iml><utterance>hi there</utterance><?pi x?></iml>",
            "<utterance>hi<?pi x?> there</utterance>",
            "<utterance><?pi x?>hi there</utterance>",
        ],
    )
    def test_processing_instructions_are_skipped(self, parser: IMLParser, iml: str) -> None:
        doc = parser.parse(iml)
        assert parser.to_plain_text(doc) == "hi there"

    def test_comment_and_pi_inside_unknown_element(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance><span>a<!-- secret -->b<?foo bar?>c</span></utterance>")
        assert doc.utterances[0].children == ("abc",)

    def test_unknown_element_keeps_iml_markup(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<utterance>I <span>really <emphasis level="strong">mean</emphasis></span>'
            " it</utterance>"
        )
        assert doc.utterances[0].children == (
            "I really ",
            Emphasis(level="strong", children=("mean",)),
            " it",
        )

    def test_foreign_namespace_element_is_unknown(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<utterance><prosody pitch="+5%"><f:segment xmlns:f="urn:other">x</f:segment>'
            "</prosody></utterance>"
        )
        assert doc.utterances[0].children == (Prosody(pitch="+5%", children=("x",)),)

    def test_iml_namespace_is_iml(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<iml:iml xmlns:iml="http://prosody-protocol.org/iml/0.1" version="0.1.0">'
            '<iml:utterance emotion="calm" confidence="0.94">Hello'
            ' <iml:emphasis level="strong">world</iml:emphasis>.</iml:utterance></iml:iml>'
        )
        assert doc.version == "0.1.0"
        utt = doc.utterances[0]
        assert utt.emotion == "calm"
        assert utt.children == ("Hello ", Emphasis(level="strong", children=("world",)), ".")

    def test_utterances_inside_unknown_wrapper_in_iml(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<iml><f:meta xmlns:f="urn:x">recorded on device 42</f:meta>'
            "<turn><utterance>x</utterance></turn><utterance>y</utterance></iml>"
        )
        assert [u.children for u in doc.utterances] == [("x",), ("y",)]

    def test_foreign_root_is_rejected(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="x:utterance"):
            parser.parse('<x:utterance xmlns:x="urn:other">Hello</x:utterance>')

    def test_foreign_default_namespace_root_is_named_by_namespace(
        self, parser: IMLParser
    ) -> None:
        # Without a prefix, "got <iml>" would read as if <iml> were refused.
        with pytest.raises(IMLParseError, match=r"got <\{urn:other\}iml>"):
            parser.parse('<iml xmlns="urn:other"><utterance>x</utterance></iml>')

    def test_long_text_split_by_comments_parses_in_linear_time(
        self, parser: IMLParser
    ) -> None:
        """Text runs split by comments or unknown tags are joined once, not repeatedly."""

        def best_time(n: int) -> float:
            iml = "<utterance>" + "word <!----><span/>" * n + "</utterance>"
            times = []
            for _ in range(3):
                start = time.perf_counter()
                doc = parser.parse(iml)
                times.append(time.perf_counter() - start)
            assert doc.utterances[0].children == ("word " * n,)
            return min(times)

        # Linear work grows 4x from n to 4n; the old repeated concatenation
        # grew about 16x.
        assert best_time(160_000) < 8 * best_time(40_000)


# ---------------------------------------------------------------------------
# Content the model cannot hold raises instead of being dropped
# ---------------------------------------------------------------------------


class TestUnrepresentableContent:
    def test_markup_directly_in_iml(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="<prosody> must be inside an <utterance>"):
            parser.parse(
                "<iml><utterance>Hello</utterance>"
                '<prosody pitch="+20%">IMPORTANT WARNING</prosody></iml>'
            )

    def test_text_directly_in_iml(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="stray text"):
            parser.parse("<iml>stray text<utterance>x</utterance></iml>")

    def test_markup_in_unknown_wrapper_in_iml(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="<pause>"):
            parser.parse('<iml><utterance>x</utterance><meta><pause duration="5"/></meta></iml>')

    def test_nested_utterance(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="<utterance> cannot be nested"):
            parser.parse("<iml><utterance>Hi <utterance>nested</utterance></utterance></iml>")

    def test_iml_inside_utterance(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="<iml> cannot be nested"):
            parser.parse("<utterance><iml><utterance>x</utterance></iml></utterance>")

    def test_pause_with_content(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="<pause> must be an empty element"):
            parser.parse('<utterance>Wait <pause duration="500">oops</pause> there.</utterance>')

    def test_pause_with_whitespace_or_comment_is_empty(self, parser: IMLParser) -> None:
        doc = parser.parse('<utterance>a<pause duration="5"> <!-- c --> </pause>b</utterance>')
        assert doc.utterances[0].children == ("a", Pause(duration=5), "b")

    def test_unknown_element_in_pause_is_content(self, parser: IMLParser) -> None:
        # Spec 3.3; dropping it would turn this V7-invalid document into a valid one.
        with pytest.raises(IMLParseError, match="<pause> must be an empty element"):
            parser.parse(
                '<utterance>a<pause duration="5"><f:x xmlns:f="urn:x"/></pause>b</utterance>'
            )


# ---------------------------------------------------------------------------
# Plain text (spec M9)
# ---------------------------------------------------------------------------


class TestPlainTextNormalization:
    def test_quickstart_output(self, parser: IMLParser, valid_fixtures_dir: Path) -> None:
        doc = parser.parse_file(valid_fixtures_dir / "sarcasm.xml")
        assert parser.to_plain_text(doc) == "Oh, that's GREAT."

    def test_utterances_joined_with_space(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<iml><utterance speaker_id="a">Is it?</utterance>'
            '<utterance speaker_id="b">No.</utterance><utterance/>'
            "<utterance>Yes!</utterance></iml>"
        )
        assert parser.to_plain_text(doc) == "Is it? No. Yes!"

    def test_pretty_printed_multi_speaker(
        self, parser: IMLParser, valid_fixtures_dir: Path
    ) -> None:
        doc = parser.parse_file(valid_fixtures_dir / "multi_speaker.xml")
        assert parser.to_plain_text(doc) == (
            "How can I help you today? "
            "I've been on hold for thirty minutes and my account is still locked. "
            "I'm so sorry about that. Let me fix this right away."
        )

    def test_space_inside_element_still_separates_words(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<utterance>I <emphasis level="strong">told </emphasis>you</utterance>'
        )
        assert parser.to_plain_text(doc) == "I told you"

    def test_source_text_spacing_is_kept(self, parser: IMLParser) -> None:
        # Only XML whitespace at an element boundary is treated as formatting.
        doc = parser.parse("<utterance>Bonjour ! Caf\u00e9\u00a0cr\u00e8me.</utterance>")
        assert parser.to_plain_text(doc) == "Bonjour ! Caf\u00e9\u00a0cr\u00e8me."

    def test_pause_between_spaces(self, parser: IMLParser) -> None:
        doc = parser.parse('<utterance>Well <pause duration="800"/> I suppose.</utterance>')
        assert parser.to_plain_text(doc) == "Well I suppose."

    @pytest.mark.parametrize(
        ("iml", "text"),
        [
            ('<utterance>Wait<pause duration="300"/>.</utterance>', "Wait."),
            ('<utterance>Wait<pause duration="300"/>, then go.</utterance>', "Wait, then go."),
            (
                '<utterance><prosody rate="slow">Wait<pause duration="300"/></prosody>!'
                "</utterance>",
                "Wait!",
            ),
        ],
    )
    def test_pause_before_punctuation_adds_no_space(
        self, parser: IMLParser, iml: str, text: str
    ) -> None:
        assert parser.to_plain_text(parser.parse(iml)) == text

    def test_empty_document(self, parser: IMLParser) -> None:
        assert parser.to_plain_text(IMLDocument()) == ""


# ---------------------------------------------------------------------------
# Numeric attributes (spec 2.6)
# ---------------------------------------------------------------------------


class TestNumericAttributes:
    @pytest.mark.parametrize("raw", ["NaN", "nan", "inf", "1_0", "1.5", "-0.1", "high"])
    def test_invalid_confidence_is_not_a_number_in_the_model(
        self, parser: IMLParser, raw: str
    ) -> None:
        doc = parser.parse(f'<utterance emotion="calm" confidence="{raw}">x</utterance>')
        utt = doc.utterances[0]
        assert utt.confidence is None
        assert utt.extra_attributes == (("confidence", raw),)

    def test_exponent_confidence(self, parser: IMLParser) -> None:
        doc = parser.parse('<utterance emotion="calm" confidence="1e-1">x</utterance>')
        assert doc.utterances[0].confidence == pytest.approx(0.1)

    @pytest.mark.parametrize("raw", ["8_00", "\u0668\u0660\u0660", "0", "-5", "3.5"])
    def test_invalid_pause_duration(self, parser: IMLParser, raw: str) -> None:
        doc = parser.parse(f'<utterance><pause duration="{raw}"/></utterance>')
        pause = doc.utterances[0].children[0]
        assert pause == Pause(duration=0, extra_attributes=(("duration", raw),))

    @pytest.mark.parametrize(
        "raw",
        ["1" * 5000, "9" * 100_000, str(MAX_INTEGER + 1), "99999999999999999999"],
        ids=["5000-digits", "100000-digits", "max-plus-1", "20-digits"],
    )
    def test_too_large_pause_duration_is_invalid_not_a_crash(
        self, parser: IMLParser, raw: str
    ) -> None:
        # int() refuses strings of more than 4300 digits with ValueError.
        doc = parser.parse(f'<utterance>a<pause duration="{raw}"/>b</utterance>')
        pause = doc.utterances[0].children[1]
        assert pause == Pause(duration=0, extra_attributes=(("duration", raw),))

    def test_too_large_duration_ms_is_invalid_not_a_crash(self, parser: IMLParser) -> None:
        raw = "1" * 5000
        p = parser.parse(
            f'<utterance><prosody duration_ms="{raw}">x</prosody></utterance>'
        ).utterances[0].children[0]
        assert isinstance(p, Prosody)
        assert p.duration_ms is None
        assert p.extra_attributes == (("duration_ms", raw),)

    @pytest.mark.parametrize(
        ("raw", "value"),
        [(str(MAX_INTEGER), MAX_INTEGER), ("0" * 5000 + "800", 800), ("+0800", 800)],
        ids=["max", "5000-leading-zeros", "plus-sign"],
    )
    def test_integer_limits(self, parser: IMLParser, raw: str, value: int) -> None:
        doc = parser.parse(f'<utterance><pause duration="{raw}"/></utterance>')
        assert doc.utterances[0].children == (Pause(duration=value),)

    def test_invalid_extended_values_are_kept_raw(self, parser: IMLParser) -> None:
        p = parser.parse(
            '<utterance><prosody f0_mean="abc" jitter="-1" duration_ms="12.5" hnr="-3">x'
            "</prosody></utterance>"
        ).utterances[0].children[0]
        assert isinstance(p, Prosody)
        assert (p.f0_mean, p.jitter, p.duration_ms, p.hnr) == (None, None, None, -3.0)
        assert p.extra_attributes == (("f0_mean", "abc"), ("jitter", "-1"), ("duration_ms", "12.5"))


# ---------------------------------------------------------------------------
# Lossless round trip
# ---------------------------------------------------------------------------


def _round_trip(parser: IMLParser, iml: str) -> str:
    return parser.to_iml_string(parser.parse(iml))


class TestLosslessRoundTrip:
    def test_consent_and_processing_kept_on_single_utterance(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<iml consent="explicit" processing="local"><utterance emotion="sad"'
            ' confidence="0.8">I miss her.</utterance></iml>'
        )
        out = parser.to_iml_string(doc)
        assert out.startswith('<iml consent="explicit" processing="local">')
        again = parser.parse(out)
        assert (again.consent, again.processing) == ("explicit", "local")

    def test_extension_and_unknown_attributes_kept(self, parser: IMLParser) -> None:
        iml = (
            '<iml version="0.1.0" x-session="7"><utterance x-annotator="human" foo="bar">'
            '<prosody pitch="+5%" x-formant-shift="+200Hz" x-nasality="0.7">x</prosody>'
            '<pause duration="300" x-kind="breath"/>'
            '<emphasis level="strong" x-w="1">y</emphasis>'
            '<segment tempo="steady" x-s="2">z</segment></utterance></iml>'
        )
        out = _round_trip(parser, iml)
        assert out == iml
        assert IMLValidator().validate(out).valid is True

    def test_namespaced_attributes_kept(self, parser: IMLParser) -> None:
        iml = (
            '<utterance xmlns:a="urn:a" xml:lang="fr" a:note="n&#10;2">'
            "Bonjour</utterance>"
        )
        doc = parser.parse(iml)
        assert doc.utterances[0].extra_attributes == (
            ("{http://www.w3.org/XML/1998/namespace}lang", "fr"),
            ("{urn:a}note", "n\n2"),
        )
        assert parser.parse(parser.to_iml_string(doc)) == doc

    def test_empty_level_is_kept_apart_from_missing_level(self, parser: IMLParser) -> None:
        empty = parser.parse('<utterance><emphasis level="">x</emphasis></utterance>')
        missing = parser.parse("<utterance><emphasis>x</emphasis></utterance>")
        assert empty.utterances[0].children == (
            Emphasis(level="", children=("x",), extra_attributes=(("level", ""),)),
        )
        assert empty != missing

    def test_xml_declaration_is_not_written(self, parser: IMLParser) -> None:
        # The output is a str with no declaration; a non-UTF-8 one (V30) is not kept.
        out = _round_trip(
            parser, '<?xml version="1.0" encoding="ISO-8859-1"?><utterance>caf\u00e9</utterance>'
        )
        assert out == "<utterance>caf\u00e9</utterance>"

    def test_carriage_return_survives(self, parser: IMLParser) -> None:
        doc = parser.parse("<utterance>a&#13;b</utterance>")
        assert parser.parse(parser.to_iml_string(doc)) == doc

    @pytest.mark.parametrize(
        ("iml", "rules"),
        [
            ('<utterance emotion="calm" confidence="NaN">x</utterance>', {"V4"}),
            ("<utterance>Wait <pause/> there.</utterance>", {"V5"}),
            ('<utterance>Wait <pause duration="-5"/> there.</utterance>', {"V6"}),
            ('<utterance>Wait <pause duration="0"/> there.</utterance>', {"V6"}),
            ("<utterance>I <emphasis>really</emphasis> mean it.</utterance>", {"V8"}),
            ('<utterance>I <emphasis level="">really</emphasis> mean it.</utterance>', {"V9"}),
            pytest.param(
                f'<utterance>Wait <pause duration="{"1" * 5000}"/> there.</utterance>',
                {"V6"},
                id="pause-duration-5000-digits",
            ),
            ('<utterance><prosody f0_mean="abc">x</prosody></utterance>', {"V27"}),
        ],
    )
    def test_invalid_input_stays_invalid(
        self, parser: IMLParser, iml: str, rules: set[str]
    ) -> None:
        out = _round_trip(parser, iml)
        assert out == iml
        errors = {i.rule for i in IMLValidator().validate(out).errors}
        assert errors == rules

    def test_invalid_fixtures_keep_their_errors(
        self, parser: IMLParser, invalid_fixtures_dir: Path
    ) -> None:
        validator = IMLValidator()
        checked = 0
        for xml_file in sorted(invalid_fixtures_dir.glob("*.xml")):
            try:
                doc = parser.parse_file(xml_file)
            except IMLParseError:
                continue  # content the model cannot hold -- covered elsewhere
            before = {i.rule for i in validator.validate_file(xml_file).errors}
            after = {i.rule for i in validator.validate(parser.to_iml_string(doc)).errors}
            assert after == before, xml_file.name
            checked += 1
        assert checked >= 10

    def test_valid_fixtures_round_trip(
        self, parser: IMLParser, valid_fixtures_dir: Path
    ) -> None:
        validator = IMLValidator()
        for xml_file in sorted(valid_fixtures_dir.glob("*.xml")):
            doc = parser.parse_file(xml_file)
            out = parser.to_iml_string(doc)
            assert parser.parse(out) == doc, xml_file.name
            assert validator.validate(out).errors == [], xml_file.name

    def test_programmatic_duplicate_extra_attribute_not_written_twice(
        self, parser: IMLParser
    ) -> None:
        doc = IMLDocument(
            utterances=(
                Utterance(children=("x",), confidence=0.5, extra_attributes=(("confidence", "?"),)),
            )
        )
        assert parser.to_iml_string(doc) == '<utterance confidence="0.5">x</utterance>'


# ---------------------------------------------------------------------------
# Serialization is always well-formed (spec 6.1)
# ---------------------------------------------------------------------------

# Characters XML 1.0 does not allow at all, not even as character references.
FORBIDDEN_CHARS = ["\x00", "\x08", "\x0b", "\x0c", "\x1b", "\x1f", "\ud800", "\udfff",
                   "\ufffe", "\uffff"]
# Unusual characters XML does allow.
ALLOWED_CHARS = ["\t", "\n", "\r", "\x7f", "\x85", "\x9f", "\u2028", "\ue000", "\ufffd",
                 "\U00010000", "\U0010ffff"]


def _with_text(text: str) -> IMLDocument:
    return IMLDocument(utterances=(Utterance(children=(f"a{text}b",)),))


def _with_attributes(value: str) -> list[IMLDocument]:
    """Documents with *value* in each kind of attribute the serializer writes."""
    extra = (("x-note", value),)
    return [
        IMLDocument(utterances=(Utterance(children=("x",), emotion=value, confidence=0.5),)),
        IMLDocument(utterances=(Utterance(children=("x",), speaker_id=value),)),
        IMLDocument(utterances=(Utterance(children=("x",), extra_attributes=extra),)),
        IMLDocument(utterances=(Utterance(children=("x",)),), language=value),
        IMLDocument(utterances=(Utterance(children=("x",)),), extra_attributes=extra),
        IMLDocument(utterances=(Utterance(children=(
            Prosody(children=("x",), pitch=value, f0_contour=value, extra_attributes=extra),
            Pause(duration=300, extra_attributes=extra),
            Emphasis(level=value, children=("y",)),
            Segment(children=("z",), tempo=value),
        )),)),
    ]


class TestWellFormedOutput:
    @pytest.mark.parametrize("char", FORBIDDEN_CHARS, ids=lambda c: f"U+{ord(c):04X}")
    def test_forbidden_character_in_text_raises(self, parser: IMLParser, char: str) -> None:
        """'a\\x1bb' used to be written as is: IML that parse() rejects."""
        message = f"text contains the character U\\+{ord(char):04X}"
        with pytest.raises(ConversionError, match=message):
            parser.to_iml_string(_with_text(char))

    @pytest.mark.parametrize("char", FORBIDDEN_CHARS, ids=lambda c: f"U+{ord(c):04X}")
    def test_forbidden_character_in_attributes_raises(self, parser: IMLParser, char: str) -> None:
        for doc in _with_attributes(f"v{char}"):
            with pytest.raises(ConversionError, match="the value of attribute"):
                parser.to_iml_string(doc)

    @pytest.mark.parametrize("char", ALLOWED_CHARS, ids=lambda c: f"U+{ord(c):04X}")
    def test_allowed_characters_round_trip(self, parser: IMLParser, char: str) -> None:
        for doc in [_with_text(char), *_with_attributes(f"v{char}")]:
            assert parser.parse(parser.to_iml_string(doc)) == doc

    def test_docstrings_hold_no_control_characters(self) -> None:
        """The to_iml_string docstring gave its example as ``"\\x1b"`` in a
        non-raw string, which put a real ESC into help() and the API docs."""
        docs = {"module": parser_module.__doc__ or ""}
        for owner in (parser_module, IMLParser):
            for name, member in vars(owner).items():
                docs[name] = getattr(member, "__doc__", None) or ""
        for name, doc in docs.items():
            assert all(c in "\t\n" or " " <= c != "\x7f" for c in doc), name

    def test_error_names_the_problem(self, parser: IMLParser) -> None:
        with pytest.raises(ConversionError) as info:
            parser.to_iml_string(_with_attributes("v\x1b")[1])
        assert "'speaker_id'" in str(info.value) and "U+001B" in str(info.value)

    @pytest.mark.parametrize("name", [
        "a b", "", "1a", "-a", "a:b", "xml:lang", "xmlns", "xmlns:a", "a\x1b", "{}a",
        "{urn:x}", "{urn:x}a:b", "{http://www.w3.org/2000/xmlns/}a", "{a b}c", "{urn:\x1b}c",
    ])
    def test_invalid_extra_attribute_name_raises(self, parser: IMLParser, name: str) -> None:
        """'a b' was written as <utterance a b="v">; 'xmlns' as a namespace
        declaration that moved the utterance out of IML."""
        doc = IMLDocument(utterances=(Utterance(children=("x",), extra_attributes=((name, "v"),)),))
        with pytest.raises(ConversionError, match="attribute"):
            parser.to_iml_string(doc)

    @pytest.mark.parametrize("name", [
        "x-note", "caf\u00e9", "a\u00b7b", "_a.b-c", "{urn:x}a",
        "{http://www.w3.org/XML/1998/namespace}lang", "{http://example.com/?a=1&b=2}c",
    ])
    def test_unusual_valid_attribute_names_round_trip(self, parser: IMLParser, name: str) -> None:
        doc = IMLDocument(utterances=(Utterance(children=("x",), extra_attributes=((name, "v"),)),))
        assert parser.parse(parser.to_iml_string(doc)) == doc

    def test_random_documents_serialize_to_parseable_iml_or_raise(
        self, parser: IMLParser
    ) -> None:
        """Whatever the text and attribute values hold, the output parses back
        to the same document, or serialization raises ConversionError."""
        import random

        rng = random.Random(1234)
        alphabet = [*FORBIDDEN_CHARS, *ALLOWED_CHARS, "a", " ", "<", "&", '"', "'", ">", "]]>"]
        raised = parsed = 0
        for _ in range(400):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))
            value = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 4)))
            doc = IMLDocument(utterances=(Utterance(
                children=(text, Prosody(children=(text,), pitch=value)),
                speaker_id=value,
                extra_attributes=(("x-v", value),),
            ),))
            try:
                out = parser.to_iml_string(doc)
            except ConversionError:
                raised += 1
                continue
            assert parser.parse(out) == doc
            parsed += 1
        assert raised > 50 and parsed > 50


# ---------------------------------------------------------------------------
# Encoding (spec 2.2)
# ---------------------------------------------------------------------------


class TestEncoding:
    def test_declared_latin1_string_is_not_double_decoded(self, parser: IMLParser) -> None:
        doc = parser.parse(
            '<?xml version="1.0" encoding="ISO-8859-1"?><utterance>caf\u00e9</utterance>'
        )
        assert doc.utterances[0].children == ("caf\u00e9",)

    def test_latin1_file_raises_parse_error(
        self, parser: IMLParser, invalid_fixtures_dir: Path
    ) -> None:
        with pytest.raises(IMLParseError, match="UTF-8") as info:
            parser.parse_file(invalid_fixtures_dir / "encoding_latin1.xml")
        assert info.value.line == 4

    def test_utf16_file_raises_parse_error(self, parser: IMLParser, tmp_path: Path) -> None:
        path = tmp_path / "utf16.xml"
        path.write_text("<utterance>Hello</utterance>", encoding="utf-16")
        with pytest.raises(IMLParseError, match="UTF-8"):
            parser.parse_file(path)

    def test_utf8_bom_file(self, parser: IMLParser, tmp_path: Path) -> None:
        path = tmp_path / "bom.xml"
        path.write_bytes(b"\xef\xbb\xbf<utterance>caf\xc3\xa9</utterance>")
        assert parser.parse_file(path).utterances[0].children == ("caf\u00e9",)

    def test_lone_surrogate_raises_parse_error(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="UTF-8"):
            parser.parse("<utterance>\ud800</utterance>")


# ---------------------------------------------------------------------------
# XML security (spec 2.5)
# ---------------------------------------------------------------------------

SECRET = "xxe-secret-7f3a"
# What a parser that resolves entities would do; used as a positive control so
# the tests below cannot pass vacuously.
_INSECURE_PARSER = etree.XMLParser(resolve_entities=True, no_network=False, load_dtd=True)


@pytest.fixture()
def secret_file(tmp_path: Path) -> Path:
    path = tmp_path / "secret.txt"
    path.write_text(SECRET)
    return path


def _xxe_documents(secret_file: Path) -> Iterator[str]:
    """Documents that pull *secret_file* in through an external entity."""
    yield (
        f'<!DOCTYPE utterance [<!ENTITY xxe SYSTEM "{secret_file.as_uri()}">]>'
        "<utterance>Hi &xxe;</utterance>"
    )
    dtd = secret_file.with_name("evil.dtd")
    dtd.write_text(f'<!ENTITY xxe SYSTEM "{secret_file.as_uri()}">')
    yield f'<!DOCTYPE utterance SYSTEM "{dtd.as_uri()}"><utterance>Hi &xxe;</utterance>'


class TestXMLSecurity:
    def test_positive_control_insecure_parser_leaks(self, secret_file: Path) -> None:
        for doc in _xxe_documents(secret_file):
            root = etree.fromstring(doc.encode(), parser=_INSECURE_PARSER)
            assert SECRET in "".join(root.itertext())

    @pytest.mark.parametrize("module", [parser_module, validator_module])
    def test_hardened_parser_never_reads_entities(self, secret_file: Path, module: object) -> None:
        secure = module._SECURE_PARSER  # type: ignore[attr-defined]
        for doc in _xxe_documents(secret_file):
            root = etree.fromstring(doc.encode(), parser=secure)
            assert SECRET not in etree.tostring(root, encoding="unicode")

    def test_parser_rejects_external_entities(self, parser: IMLParser, secret_file: Path) -> None:
        for doc in _xxe_documents(secret_file):
            with pytest.raises(IMLParseError, match="DOCTYPE") as info:
                parser.parse(doc)
            assert SECRET not in str(info.value)

    def test_validator_rejects_external_entities(self, secret_file: Path) -> None:
        for doc in _xxe_documents(secret_file):
            result = IMLValidator().validate(doc)
            assert result.valid is False
            assert {i.rule for i in result.errors} == {"V31"}
            assert all(SECRET not in i.message for i in result.issues)

    def test_internal_entity_rejected(self, parser: IMLParser) -> None:
        with pytest.raises(IMLParseError, match="DOCTYPE"):
            parser.parse(
                '<!DOCTYPE utterance [<!ENTITY who "Bob">]><utterance>Hi &who;</utterance>'
            )

    def test_billion_laughs_rejected(self, parser: IMLParser) -> None:
        entities = '<!ENTITY lol "lol">' + "".join(
            f'<!ENTITY lol{i} "{("&lol%s;" % (i - 1 or "")) * 10}">' for i in range(1, 10)
        )
        with pytest.raises(IMLParseError):
            parser.parse(f"<!DOCTYPE utterance [{entities}]><utterance>&lol9;</utterance>")

    def test_external_dtd_not_fetched(self, parser: IMLParser) -> None:
        """A DOCTYPE pointing at a URL is rejected without being fetched.

        (libxml2 builds without HTTP support could not fetch it anyway; the
        request counter guards builds that can.)
        """
        hits: list[str] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                hits.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'<!ENTITY xxe "fetched">')

            def log_message(self, *args: object) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/xxe.dtd"
            with pytest.raises(IMLParseError, match="DOCTYPE"):
                parser.parse(f'<!DOCTYPE utterance SYSTEM "{url}"><utterance>&xxe;</utterance>')
        finally:
            server.shutdown()
            server.server_close()
        assert hits == []


# ---------------------------------------------------------------------------
# Serialization: numbers as written, utterances apart, invalid numbers refused
# ---------------------------------------------------------------------------


class TestNumbersKeepTheirForm:
    """A parse/serialize round trip is byte-stable for valid numbers."""

    @pytest.mark.parametrize(
        "iml",
        [
            '<utterance emotion="calm" confidence="1">x</utterance>',
            '<utterance emotion="calm" confidence="0.870">x</utterance>',
            '<utterance emotion="calm" confidence="1e-1">x</utterance>',
            '<utterance><prosody f0_mean="220" intensity_mean="72" intensity_range="15" '
            'speech_rate="5" jitter="1.20" shimmer="4.50" hnr="15">x</prosody></utterance>',
            '<utterance><prosody f0_mean="220.12345678901234567890" hnr="-0">x</prosody>'
            "</utterance>",
        ],
    )
    def test_round_trip_is_byte_stable(self, parser: IMLParser, iml: str) -> None:
        # f0_mean="220" used to come back as "220.0", confidence="1" as "1.0".
        assert _round_trip(parser, iml) == iml

    def test_surrounding_whitespace_is_dropped(self, parser: IMLParser) -> None:
        iml = '<utterance emotion="calm" confidence=" 0.5 ">x</utterance>'
        assert _round_trip(parser, iml) == (
            '<utterance emotion="calm" confidence="0.5">x</utterance>'
        )

    def test_parsed_numbers_are_ordinary_floats(self, parser: IMLParser) -> None:
        import copy
        import pickle

        doc = parser.parse('<utterance emotion="calm" confidence="1">x</utterance>')
        confidence = doc.utterances[0].confidence
        assert confidence == 1.0 and isinstance(confidence, float)
        assert doc == IMLDocument(utterances=(
            Utterance(children=("x",), emotion="calm", confidence=1.0),
        ))
        for copied in (pickle.loads(pickle.dumps(doc)), copy.deepcopy(doc), copy.copy(doc)):
            assert copied == doc
            assert parser.to_iml_string(copied) == parser.to_iml_string(doc)

    def test_built_floats_are_written_as_before(self, parser: IMLParser) -> None:
        doc = IMLDocument(utterances=(
            Utterance(children=(Prosody(children=("x",), f0_mean=220.0),), emotion="calm",
                      confidence=0.87),
        ))
        assert parser.to_iml_string(doc) == (
            '<utterance emotion="calm" confidence="0.87"><prosody f0_mean="220.0">x</prosody>'
            "</utterance>"
        )


class TestUtterancesStayApart:
    def test_tag_stripped_text_keeps_sentences_apart(self, parser: IMLParser) -> None:
        # "First one.Second one!" once a consumer stripped the tags (spec 6.2).
        from prosody_protocol.text_to_iml import TextToIML

        doc = TextToIML().predict_document("First one. Second one!")
        xml = parser.to_iml_string(doc)
        assert "".join(etree.fromstring(xml.encode()).itertext()) == "First one. Second one!"
        assert parser.parse(xml) == doc

    def test_written_with_a_space(self, parser: IMLParser) -> None:
        iml = "<iml><utterance>One.</utterance><utterance>Two.</utterance></iml>"
        assert _round_trip(parser, iml) == (
            "<iml><utterance>One.</utterance> <utterance>Two.</utterance></iml>"
        )
        assert IMLValidator().validate(_round_trip(parser, iml)).issues == []


def _one(node: Utterance) -> IMLDocument:
    return IMLDocument(utterances=(node,))


class TestInvalidNumbersAreNotWritten:
    """Numeric fields built in code are checked when the document is written."""

    @pytest.mark.parametrize(
        ("doc", "rule", "field"),
        [
            (_one(Utterance(children=("hi",), emotion="calm", confidence=float("nan"))), "V4",
             "Utterance.confidence"),
            (_one(Utterance(children=("hi",), emotion="calm", confidence=5.0)), "V4",
             "Utterance.confidence"),
            (_one(Utterance(children=("hi",), emotion="calm", confidence=-1.0)), "V4",
             "Utterance.confidence"),
            (_one(Utterance(children=("a", Pause(duration=-5), "b"))), "V6", "Pause.duration"),
            (_one(Utterance(children=("a", Pause(duration=MAX_INTEGER + 1)))), "V6",
             "Pause.duration"),
            (_one(Utterance(children=("a", Pause(duration=2.5)))), "V6",  # type: ignore[arg-type]
             "Pause.duration"),
            (_one(Utterance(children=(Prosody(children=("x",), f0_mean=-1.0),))), "V27",
             "Prosody.f0_mean"),
            (_one(Utterance(children=(Prosody(children=("x",), hnr=float("inf")),))), "V27",
             "Prosody.hnr"),
            (_one(Utterance(children=(Prosody(children=("x",), duration_ms=0),))), "V27",
             "Prosody.duration_ms"),
            (_one(Utterance(children=(Prosody(children=("x",), jitter=-0.5),))), "V27",
             "Prosody.jitter"),
        ],
    )
    def test_raises_instead_of_writing_invalid_iml(
        self, parser: IMLParser, doc: IMLDocument, rule: str, field: str
    ) -> None:
        # to_iml_string used to write confidence="nan", duration="-5", ...
        with pytest.raises(IMLValidationError, match=field) as info:
            parser.to_iml_string(doc)
        assert [issue.rule for issue in info.value.issues] == [rule]

    def test_valid_built_values_are_written(self, parser: IMLParser) -> None:
        doc = _one(Utterance(
            children=("a", Pause(duration=MAX_INTEGER), Prosody(children=("x",), hnr=-3.0)),
            emotion="calm", confidence=0.0,
        ))
        xml = parser.to_iml_string(doc)
        assert IMLValidator().validate(xml).valid
        assert parser.parse(xml) == doc

    def test_zero_duration_is_a_missing_duration(self, parser: IMLParser) -> None:
        # The parser's value for <pause/>, so that invalid input stays invalid.
        doc = _one(Utterance(children=("a", Pause(duration=0), "b")))
        assert parser.to_iml_string(doc) == "<utterance>a<pause/>b</utterance>"

    def test_numpy_numbers(self, parser: IMLParser) -> None:
        np = pytest.importorskip("numpy")
        doc = _one(Utterance(
            children=("a", Pause(duration=np.int64(300))), emotion="calm",
            confidence=np.float64(0.5),
        ))
        assert parser.to_iml_string(doc) == (
            '<utterance emotion="calm" confidence="0.5">a<pause duration="300"/></utterance>'
        )
