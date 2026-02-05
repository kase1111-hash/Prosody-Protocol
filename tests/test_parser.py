"""Tests for prosody_protocol.parser.

Covers:
- Parsing all Appendix C examples from the spec
- Round-trip: parse -> serialize -> parse produces identical documents
- Malformed XML raises IMLParseError
- Mixed content ordering (text + child interleaving)
- Extended attributes on <prosody>
- File-based parsing
- Plain text extraction
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prosody_protocol.exceptions import IMLParseError
from prosody_protocol.models import (
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)
from prosody_protocol.parser import IMLParser


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
        for u1, u2 in zip(doc1.utterances, doc2.utterances):
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
        for child in doc.utterances[0].children:
            assert isinstance(child, str)

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
