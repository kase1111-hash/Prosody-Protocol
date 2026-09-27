"""Tests for prosody_protocol.iml_to_ssml.

Covers:
- Tag mapping: utterance→s, prosody→prosody, pause→break, emphasis→emphasis
- Attribute conversion: pitch, volume, rate mapped; extended attrs dropped
- pitch_contour → SSML contour (flat → range), segment tempo → rate
- SSML 1.1 <speak> root with xml:lang always present
- Text order, escaping, comments never spoken
- Validation of input documents (strict mode) and unrenderable values
- speaker_voices → <voice name>
- The espeak-ng adaptation (vendor="espeak-ng")
- convert_doc from IMLDocument objects
"""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import pytest
from lxml import etree

from prosody_protocol.exceptions import ConversionError, IMLValidationError
from prosody_protocol.iml_to_ssml import IMLToSSML
from prosody_protocol.models import (
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)
from prosody_protocol.parser import IMLParser
from prosody_protocol.text_to_iml import TextToIML

SSML_NS = "http://www.w3.org/2001/10/synthesis"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
VALID_FIXTURES = sorted((Path(__file__).parent / "fixtures" / "valid").glob("*.xml"))


@pytest.fixture()
def converter() -> IMLToSSML:
    return IMLToSSML()


@pytest.fixture()
def espeak() -> IMLToSSML:
    return IMLToSSML(vendor="espeak-ng")


def _parse_ssml(ssml: str) -> etree._Element:
    """Parse SSML and return the root element."""
    return etree.fromstring(ssml.encode("utf-8"))


def _body(ssml: str) -> str:
    """The markup inside <speak>...</speak>."""
    return ssml.split(">", 1)[1].rsplit("</speak>", 1)[0]


def _spoken_text(ssml: str) -> str:
    """All text an engine would speak, with whitespace removed."""
    return re.sub(r"\s+", "", "".join(_parse_ssml(ssml).itertext()))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class TestInterface:
    def test_instantiate(self) -> None:
        converter = IMLToSSML()
        assert converter.vendor is None
        assert converter.default_language == "en-US"
        assert converter.strict is True

    def test_vendor_param(self) -> None:
        with pytest.warns(UserWarning, match="not yet implemented"):
            converter = IMLToSSML(vendor="google")
        assert converter.vendor == "google"

    def test_espeak_vendor_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            IMLToSSML(vendor="espeak-ng")
            IMLToSSML(vendor="espeak")

    def test_invalid_default_language_rejected(self) -> None:
        with pytest.raises(ValueError, match="BCP 47"):
            IMLToSSML(default_language="not a tag")


# ---------------------------------------------------------------------------
# Document wrapper
# ---------------------------------------------------------------------------


class TestDocumentWrapper:
    def test_speak_root(self, converter: IMLToSSML) -> None:
        ssml = converter.convert("<utterance>Hello</utterance>")
        root = _parse_ssml(ssml)
        assert root.tag == f"{{{SSML_NS}}}speak"
        assert ssml.startswith("<speak")

    def test_speak_version_is_1_1(self, converter: IMLToSSML) -> None:
        """The output uses SSML 1.1 syntax (dB volume, unsigned % rate)."""
        root = _parse_ssml(converter.convert("<utterance>Hello</utterance>"))
        assert root.get("version") == "1.1"

    def test_speak_xmlns(self, converter: IMLToSSML) -> None:
        ssml = converter.convert("<utterance>Hello</utterance>")
        assert f'xmlns="{SSML_NS}"' in ssml

    def test_language_propagated(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<iml version="0.1.0" language="de-DE">'
            "<utterance>Hallo</utterance>"
            "</iml>"
        )
        assert _parse_ssml(ssml).get(XML_LANG) == "de-DE"

    def test_default_language_when_absent(self, converter: IMLToSSML) -> None:
        """SSML requires xml:lang on <speak>; a bare <utterance> gets the default."""
        ssml = converter.convert("<utterance>Hello</utterance>")
        assert _parse_ssml(ssml).get(XML_LANG) == "en-US"

    def test_default_language_parameter(self) -> None:
        ssml = IMLToSSML(default_language="fr-FR").convert("<utterance>Bonjour</utterance>")
        assert _parse_ssml(ssml).get(XML_LANG) == "fr-FR"

    def test_document_language_beats_default(self) -> None:
        ssml = IMLToSSML(default_language="fr-FR").convert(
            '<iml version="0.1.0" language="en-GB"><utterance>Hello</utterance></iml>'
        )
        assert _parse_ssml(ssml).get(XML_LANG) == "en-GB"


# ---------------------------------------------------------------------------
# Tag mapping
# ---------------------------------------------------------------------------


class TestUtteranceMapping:
    def test_utterance_becomes_s(self, converter: IMLToSSML) -> None:
        ssml = converter.convert("<utterance>Hello world.</utterance>")
        assert "<s>Hello world.</s>" in ssml

    def test_multiple_utterances(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<iml version="0.1.0">'
            "<utterance>First.</utterance>"
            "<utterance>Second.</utterance>"
            "</iml>"
        )
        assert ssml.count("<s>") == 2
        assert ssml.index("First.") < ssml.index("Second.")

    def test_emotion_and_confidence_not_written(self, converter: IMLToSSML) -> None:
        """SSML has no vendor-neutral emotion markup (see module docstring)."""
        ssml = converter.convert(
            '<utterance emotion="calm" confidence="0.9" speaker_id="agent">Hi.</utterance>'
        )
        assert "emotion" not in ssml
        assert "confidence" not in ssml
        assert "agent" not in ssml


class TestProsodyMapping:
    def test_pitch_mapped(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><prosody pitch="+15%">loud</prosody></utterance>'
        )
        assert 'pitch="+15%"' in ssml

    def test_pitch_units_mapped(self, converter: IMLToSSML) -> None:
        for pitch in ("+3st", "-2.5st", "185Hz", "-10%"):
            ssml = converter.convert(
                f'<utterance><prosody pitch="{pitch}">word</prosody></utterance>'
            )
            assert f'pitch="{pitch}"' in ssml

    def test_volume_mapped(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><prosody volume="+6dB">loud</prosody></utterance>'
        )
        assert 'volume="+6dB"' in ssml

    def test_rate_mapped(self, converter: IMLToSSML) -> None:
        for rate in ("fast", "slow", "medium", "150%"):
            ssml = converter.convert(
                f'<utterance><prosody rate="{rate}">quick</prosody></utterance>'
            )
            assert f'rate="{rate}"' in ssml

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
        """Prosody with only extended attrs (no pitch/volume/rate/contour) should
        not produce a <prosody> wrapper in the SSML."""
        ssml = converter.convert(
            '<utterance><prosody f0_mean="200" quality="breathy">text</prosody></utterance>'
        )
        assert _body(ssml) == "<s>text</s>"


class TestContourMapping:
    """pitch_contour must reach the SSML: it carries sarcasm vs. sincerity."""

    @pytest.mark.parametrize(
        ("contour", "expected"),
        [
            ("rise", "(0%,+0%) (100%,+20%)"),
            ("fall", "(0%,+0%) (100%,-20%)"),
            ("rise-fall", "(0%,+0%) (50%,+20%) (100%,-10%)"),
            ("fall-rise", "(0%,+0%) (50%,-15%) (100%,+15%)"),
            ("rise-sharp", "(0%,+0%) (70%,+5%) (100%,+40%)"),
            ("fall-sharp", "(0%,+10%) (70%,+5%) (100%,-30%)"),
        ],
    )
    def test_contour_values(self, converter: IMLToSSML, contour: str, expected: str) -> None:
        ssml = converter.convert(
            f'<utterance><prosody pitch_contour="{contour}">Are you sure</prosody></utterance>'
        )
        assert _body(ssml) == f'<s><prosody contour="{expected}">Are you sure</prosody></s>'

    def test_flat_contour_narrows_range(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><prosody pitch_contour="flat">monotone</prosody></utterance>'
        )
        assert '<prosody range="x-low">monotone</prosody>' in ssml

    def test_contour_with_pitch_is_nested(self, converter: IMLToSSML) -> None:
        """Contour targets are relative to the shifted pitch, so they are nested."""
        ssml = converter.convert(
            '<utterance><prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">'
            "GREAT</prosody></utterance>"
        )
        assert _body(ssml) == (
            '<s><prosody pitch="+15%" volume="+6dB">'
            '<prosody contour="(0%,+10%) (70%,+5%) (100%,-30%)">GREAT</prosody>'
            "</prosody></s>"
        )

    def test_sarcastic_and_sincere_differ(self, converter: IMLToSSML) -> None:
        sarcastic = (Path(__file__).parent / "fixtures" / "valid" / "sarcasm.xml").read_text()
        sincere = sarcastic.replace("sarcastic", "sincere").replace(
            ' pitch_contour="fall-sharp"', ""
        )
        assert converter.convert(sarcastic) != converter.convert(sincere)

    def test_question_differs_from_statement(self, converter: IMLToSSML) -> None:
        """TextToIML marks questions with a rising contour; it must survive."""
        predictor = TextToIML()
        question = converter.convert(predictor.predict("You are coming tonight?"))
        statement = converter.convert(predictor.predict("You are coming tonight."))
        assert "contour=" in question
        assert "contour=" not in statement


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
        assert '<break time="200ms"/>' in ssml


class TestEmphasisMapping:
    def test_emphasis_preserved(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><emphasis level="strong">important</emphasis></utterance>'
        )
        assert '<emphasis level="strong">important</emphasis>' in ssml

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
            '<utterance><segment rhythm="staccato">flowing text</segment></utterance>'
        )
        assert _body(ssml) == "<s>flowing text</s>"

    @pytest.mark.parametrize(
        ("tempo", "rate"), [("rushed", "fast"), ("drawn-out", "slow")]
    )
    def test_segment_tempo_becomes_rate(
        self, converter: IMLToSSML, tempo: str, rate: str
    ) -> None:
        ssml = converter.convert(
            f'<utterance><segment tempo="{tempo}">flowing text</segment></utterance>'
        )
        assert "<segment" not in ssml
        assert "tempo" not in ssml
        assert f'<prosody rate="{rate}">flowing text</prosody>' in ssml

    def test_steady_tempo_unchanged(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance><segment tempo="steady">even text</segment></utterance>'
        )
        assert _body(ssml) == "<s>even text</s>"


class TestSpeakerVoices:
    def test_mapped_speaker_gets_voice(self) -> None:
        converter = IMLToSSML(speaker_voices={"agent": "en-US-Wavenet-D"})
        ssml = converter.convert(
            '<iml version="0.1.0">'
            '<utterance speaker_id="agent">How can I help?</utterance>'
            '<utterance speaker_id="caller">My account is locked.</utterance>'
            "</iml>"
        )
        assert _body(ssml) == (
            '<voice name="en-US-Wavenet-D"><s>How can I help?</s></voice>'
            "<s>My account is locked.</s>"
        )


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
        assert '<emphasis level="strong"><prosody pitch="+10%">' in ssml

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
        # Core attrs preserved, contour mapped.
        assert 'pitch="+15%"' in ssml
        assert 'volume="+6dB"' in ssml
        assert 'contour="(0%,+10%) (70%,+5%) (100%,-30%)"' in ssml
        assert "GREAT" in ssml


# ---------------------------------------------------------------------------
# Text: order, escaping, comments
# ---------------------------------------------------------------------------


class TestText:
    def test_text_order_preserved(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            '<utterance>I <emphasis level="strong">told</emphasis> you '
            '<prosody pitch="+12%">yesterday</prosody>!</utterance>'
        )
        assert "".join(_parse_ssml(ssml).itertext()) == "I told you yesterday!"

    def test_xml_entities_escaped(self, converter: IMLToSSML) -> None:
        ssml = converter.convert(
            "<utterance>A &amp; B &lt; C</utterance>"
        )
        assert "<s>A &amp; B &lt; C</s>" in ssml

    def test_comments_not_spoken(self, converter: IMLToSSML, espeak: IMLToSSML) -> None:
        iml = "<utterance>Hello <!-- secret note --> world</utterance>"
        for conv in (converter, espeak):
            ssml = conv.convert(iml)
            assert "secret" not in ssml
            assert _spoken_text(ssml) == "Helloworld"

    def test_processing_instruction_not_spoken(self, converter: IMLToSSML) -> None:
        ssml = converter.convert("<utterance>Hello <?note internal?> world</utterance>")
        assert "internal" not in ssml

    def test_control_characters_replaced(self) -> None:
        """Characters XML cannot carry never reach the SSML (lenient mode)."""
        doc = IMLDocument(utterances=(Utterance(children=("Ctrl\x01char",)),))
        ssml = IMLToSSML(strict=False).convert_doc(doc)
        assert "".join(_parse_ssml(ssml).itertext()) == "Ctrl char"

    @pytest.mark.parametrize("path", VALID_FIXTURES, ids=lambda p: p.name)
    @pytest.mark.parametrize("vendor", [None, "espeak-ng"])
    def test_valid_fixtures_keep_all_text(self, path: Path, vendor: str | None) -> None:
        """Every word of every valid fixture is spoken, in order, as valid XML."""
        iml = path.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssml = IMLToSSML(vendor=vendor).convert(iml)
        expected = re.sub(r"\s+", "", IMLParser().to_plain_text(IMLParser().parse(iml)))
        assert _spoken_text(ssml) == expected


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
        assert '<prosody pitch="+20%" volume="+8dB">LOUD</prosody>' in ssml

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
                        Segment(children=("segment text",), rhythm="legato"),
                    ),
                ),
            ),
        )
        ssml = converter.convert_doc(doc)
        assert "<segment" not in ssml
        assert "<s>segment text</s>" in ssml


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
        root = _parse_ssml(ssml)
        assert [etree.QName(el).localname for el in root.iter()] == [
            "speak", "s", "emphasis", "prosody"
        ]


# ---------------------------------------------------------------------------
# Validation and error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_malformed_iml_raises(self, converter: IMLToSSML) -> None:
        with pytest.raises(ConversionError):
            converter.convert("<utterance>unclosed")

    def test_empty_string_raises(self, converter: IMLToSSML) -> None:
        with pytest.raises(ConversionError):
            converter.convert("")

    @pytest.mark.parametrize(
        ("iml", "rule"),
        [
            ('<utterance emotion="angry">No confidence</utterance>', "V3"),
            ("<utterance>a<pause/>b</utterance>", "V5"),
            ('<utterance>a<pause duration="-500"/>b</utterance>', "V6"),
            ("<utterance><emphasis>no level</emphasis></utterance>", "V8"),
            (
                '<utterance><prosody pitch="+5%"><segment>x</segment></prosody></utterance>',
                "V10",
            ),
        ],
    )
    def test_invalid_document_rejected(self, converter: IMLToSSML, iml: str, rule: str) -> None:
        with pytest.raises(IMLValidationError) as info:
            converter.convert(iml)
        assert rule in {issue.rule for issue in info.value.issues}

    def test_invalid_doc_object_rejected(self, converter: IMLToSSML) -> None:
        doc = IMLDocument(utterances=(Utterance(children=("a", Pause(duration=-500), "b")),))
        with pytest.raises(IMLValidationError):
            converter.convert_doc(doc)

    def test_lenient_mode_converts_imperfect_documents(self) -> None:
        ssml = IMLToSSML(strict=False).convert(
            '<utterance emotion="angry"><prosody pitch="loud" rate="+50%">Hi</prosody>'
            "<pause/></utterance>"
        )
        # The invalid values are ignored rather than copied into the SSML.
        assert _body(ssml) == "<s>Hi</s>"

    def test_negative_pause_cannot_be_rendered(self) -> None:
        doc = IMLDocument(utterances=(Utterance(children=("a", Pause(duration=-500), "b")),))
        for vendor in (None, "espeak-ng"):
            with pytest.raises(ConversionError, match="duration=-500"):
                IMLToSSML(vendor=vendor, strict=False).convert_doc(doc)


# ---------------------------------------------------------------------------
# espeak-ng adaptation
# ---------------------------------------------------------------------------


def _pitch_steps(ssml: str) -> list[int]:
    return [int(v) for v in re.findall(r'pitch="([+-]\d+)%"', ssml)]


def _effective(ssml: str, word: str, attr: str) -> float:
    """Product of the nested percentage ``attr`` values that apply to ``word``.

    For volume ("40%") this is the amplitude relative to espeak-ng's default;
    for pitch ("+29%") the factor applied to espeak-ng's pitch parameter.
    """
    root = _parse_ssml(ssml)
    for el in root.iter():
        for text, owner in ((el.text, el), (el.tail, el.getparent())):
            if text and word in text and owner is not None:
                factor = 1.0
                for node in [owner, *owner.iterancestors()]:
                    value = node.get(attr)
                    if value:
                        number = float(value.rstrip("%"))
                        factor *= 1 + number / 100 if value[0] in "+-" else number / 100
                return factor
    raise AssertionError(f"{word!r} not in {ssml}")


class TestExtremeValues:
    """Values the validator accepts but no engine can render (were OverflowError)."""

    @pytest.mark.parametrize(
        "prosody",
        [
            'volume="+7000dB"',
            'volume="-7000dB"',
            'pitch="+20000st"',
            'pitch="-20000st"',
            f'pitch="{"9" * 400}Hz"',
            f'pitch="+{"9" * 400}%"',
            f'volume="+{"9" * 400}dB"',
            'pitch="0Hz"',
            f'rate="{"9" * 400}%"',
        ],
    )
    @pytest.mark.parametrize("vendor", [None, "espeak-ng"])
    def test_converted_not_raised(self, prosody: str, vendor: str | None) -> None:
        iml = f"<utterance>a <prosody {prosody}>hi</prosody> b</utterance>"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # espeak-ng mode warns that it clamps
            ssml = IMLToSSML(vendor=vendor).convert(iml)
        assert _spoken_text(ssml) == "ahib"

    def test_standard_ssml_maps_extreme_volume_to_names(self, converter: IMLToSSML) -> None:
        ssml = converter.convert('<utterance><prosody volume="+7000dB">hi</prosody></utterance>')
        assert '<prosody volume="x-loud">hi</prosody>' in ssml

    def test_espeak_clamps_extreme_pitch_with_warning(self, espeak: IMLToSSML) -> None:
        for pitch in ("+20000st", "-20000st"):
            with pytest.warns(UserWarning, match="clamped"):
                ssml = espeak.convert(
                    f'<utterance><prosody pitch="{pitch}">hi</prosody></utterance>'
                )
            # The curve's ends: parameter 100 (x2) and 10 (x0.2).
            assert _effective(ssml, "hi", "pitch") == pytest.approx(
                2.0 if pitch[0] == "+" else 0.2
            )


class TestEspeakVendor:
    def test_pitch_rescaled_for_espeak(self, espeak: IMLToSSML) -> None:
        """espeak-ng applies relative pitch to a 0-100 parameter, compressing it
        (more so downwards); values are rescaled along its measured curve."""
        ssml = espeak.convert('<utterance><prosody pitch="+15%">word</prosody></utterance>')
        assert _pitch_steps(ssml) == [29]
        ssml = espeak.convert('<utterance><prosody pitch="-20%">word</prosody></utterance>')
        assert _pitch_steps(ssml) == [-53]

    def test_semitones_and_hz_become_percentages(self, espeak: IMLToSSML) -> None:
        for pitch in ("+3st", "120Hz"):
            ssml = espeak.convert(
                f'<utterance><prosody pitch="{pitch}">word</prosody></utterance>'
            )
            assert "st" not in re.findall(r'pitch="([^"]+)"', ssml)[0]
            assert "Hz" not in ssml

    def test_out_of_range_pitch_warns(self, espeak: IMLToSSML) -> None:
        with pytest.warns(UserWarning, match="beyond what espeak-ng can render"):
            espeak.convert('<utterance><prosody pitch="+12st">word</prosody></utterance>')

    def test_volume_as_percentage(self, espeak: IMLToSSML) -> None:
        """espeak-ng misreads dB; +6 dB is written as twice the amplitude."""
        ssml = espeak.convert(
            '<utterance>a <prosody volume="+6dB">b</prosody> c</utterance>'
        )
        assert "dB" not in ssml
        assert '<prosody volume="200%">b</prosody>' in ssml

    def test_headroom_wrapper(self, espeak: IMLToSSML) -> None:
        plain = espeak.convert("<utterance>Hello there.</utterance>")
        assert _body(plain) == '<s><prosody volume="40%">Hello there.</prosody></s>'
        loud = espeak.convert(
            '<utterance>a <prosody volume="+20dB">b</prosody></utterance>'
        )
        # 10x louder span: the sentence volume drops so the span stays unclipped.
        assert '<s><prosody volume="7%">' in loud
        assert _effective(loud, "b", "volume") == pytest.approx(0.70)

    @pytest.mark.parametrize(
        ("body", "loud"),
        [
            # The peak estimate clamped the cumulative gain but the writer
            # clamped each level: this came out at 253%, not the planned 120%.
            ('<prosody volume="+6dB"><prosody volume="+40dB">loud</prosody></prosody>', "loud"),
            (
                '<prosody volume="+12dB"><emphasis level="strong">loud</emphasis> x</prosody>',
                "loud",
            ),
            (
                '<emphasis level="strong"><emphasis level="strong">loud</emphasis></emphasis>',
                "loud",
            ),
            # espeak-ng speaks symbols and emoji ("grinning face", "and"), so
            # they need headroom like words.
            ('<prosody volume="+20dB">\U0001F600</prosody>', "\U0001F600"),
            ('<prosody volume="+20dB">&amp;</prosody>', "&"),
        ],
    )
    def test_loudest_span_stays_within_headroom(self, body: str, loud: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the +46 dB case warns (clamped)
            # Lenient: nested <emphasis> is a validation error (V21), but renderable.
            ssml = IMLToSSML(vendor="espeak-ng", strict=False).convert(
                f"<utterance>{body} quiet</utterance>"
            )
        assert 0.25 < _effective(ssml, loud, "volume") <= 0.705
        assert _effective(ssml, "quiet", "volume") < _effective(ssml, loud, "volume")

    def test_cumulative_volume_clamped_with_warning(self, espeak: IMLToSSML) -> None:
        with pytest.warns(UserWarning, match="clamped to \\+30 dB"):
            ssml = espeak.convert(
                '<utterance>quiet <prosody volume="+6dB"><prosody volume="+40dB">loud'
                "</prosody></prosody></utterance>"
            )
        ratio = _effective(ssml, "loud", "volume") / _effective(ssml, "quiet", "volume")
        assert ratio == pytest.approx(10 ** (30 / 20), rel=0.02)

    def test_symbol_only_span_does_not_lower_the_sentence(self, espeak: IMLToSSML) -> None:
        """A final '!' moves out of the span, so it does not need headroom."""
        ssml = espeak.convert('<utterance>Stop it <prosody volume="+20dB">!</prosody></utterance>')
        assert '<s><prosody volume="40%">' in ssml

    def test_absolute_hz_ignores_enclosing_shift(self, espeak: IMLToSSML) -> None:
        """"150Hz" inside a strong emphasis used to be raised by the emphasis too."""
        ssml = espeak.convert(
            '<utterance><emphasis level="strong"><prosody pitch="150Hz">inside</prosody>'
            '</emphasis> <prosody pitch="150Hz">outside</prosody></utterance>'
        )
        inside = _effective(ssml, "inside", "pitch")
        assert inside == pytest.approx(_effective(ssml, "outside", "pitch"), rel=0.01)
        assert inside > 1.0  # 150 Hz is above the nominal 100 Hz baseline

    def test_contour_becomes_word_steps(self, espeak: IMLToSSML) -> None:
        rise = espeak.convert(
            '<utterance><prosody pitch_contour="rise">one two three four</prosody></utterance>'
        )
        fall = espeak.convert(
            '<utterance><prosody pitch_contour="fall">one two three four</prosody></utterance>'
        )
        assert "contour" not in rise
        steps = _pitch_steps(rise)
        assert len(steps) == 4 and steps == sorted(steps) and steps[-1] > 0
        steps = _pitch_steps(fall)
        assert len(steps) == 4 and steps == sorted(steps, reverse=True) and steps[-1] < 0

    def test_single_word_contour_moves_the_word(self, espeak: IMLToSSML) -> None:
        up = espeak.convert('<utterance><prosody pitch_contour="rise">yes</prosody></utterance>')
        down = espeak.convert('<utterance><prosody pitch_contour="fall">yes</prosody></utterance>')
        assert _pitch_steps(up)[0] > 0 > _pitch_steps(down)[0]

    def test_emphasis_becomes_pitch_and_volume(self, espeak: IMLToSSML) -> None:
        ssml = espeak.convert(
            '<utterance>I <emphasis level="strong">told</emphasis> '
            '<emphasis level="reduced">you</emphasis></utterance>'
        )
        assert "<emphasis" not in ssml
        assert '<prosody pitch="+29%" volume="150%">told</prosody>' in ssml
        assert '<prosody pitch="-7%" volume="80%">you</prosody>' in ssml

    def test_final_period_moves_into_last_element(self, espeak: IMLToSSML) -> None:
        """A '.' after a closing tag makes espeak-ng add ~0.3 s of silence."""
        ssml = espeak.convert(
            "<utterance>Oh, that's <prosody pitch=\"+15%\">GREAT</prosody>.</utterance>"
        )
        assert '<prosody pitch="+29%">GREAT.</prosody></prosody></s>' in ssml

    def test_long_punctuation_run_takes_linear_time(self, espeak: IMLToSSML) -> None:
        """Finding the final punctuation was quadratic (minutes for this input)."""
        start = time.perf_counter()
        ssml = espeak.convert("<utterance>" + "!" * 200_000 + "x</utterance>")
        assert time.perf_counter() - start < 5.0
        assert _spoken_text(ssml) == "!" * 200_000 + "x"

    def test_final_question_mark_moves_after_elements(self, espeak: IMLToSSML) -> None:
        """A '?' before a closing tag makes espeak-ng add ~0.3 s of silence."""
        ssml = espeak.convert(
            '<utterance><prosody pitch="+5%">Are you coming?</prosody>'
            '<pause duration="300"/></utterance>'
        )
        assert ssml.endswith('coming</prosody></prosody>?<break time="300ms"/></s></speak>')
