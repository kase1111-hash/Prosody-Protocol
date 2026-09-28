"""Tests for schemas/iml-1.0.xsd and its agreement with spec.md and IMLValidator.

Covers:
- The schema compiles, also when included into the IML namespace (spec 2.3)
- Every valid fixture passes; every invalid fixture fails unless its rule is
  one XSD 1.0 cannot express (listed explicitly)
- Every IML example in spec.md (extracted from the ```xml blocks, so the spec
  cannot drift) passes the schema and the validator, and round-trips
- The spec's value vocabularies equal the schema's and the validator's
- Appendix A's content models match what the schema and validator accept
- The validator and the schema agree on a corpus of edge cases, including
  implausible values, which only draw validator warnings (V33, spec 6.4)
- Spec 6.4's plausible ranges are the limits the validator warns at
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from lxml import etree

from prosody_protocol import validator as validator_module
from prosody_protocol.parser import IML_NAMESPACE, IMLParser
from prosody_protocol.validator import IMLValidator

REPO_ROOT = Path(__file__).resolve().parent.parent
XSD_PATH = REPO_ROOT / "schemas" / "iml-1.0.xsd"
SPEC_PATH = REPO_ROOT / "spec.md"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
XS = "{http://www.w3.org/2001/XMLSchema}"

_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)

# Invalid fixtures whose rule XML Schema 1.0 cannot express; the schema
# accepts them and only IMLValidator rejects them.
XSD_CANNOT_EXPRESS = {
    "missing_confidence.xml": "V3: attribute co-occurrence (emotion requires confidence)",
    "encoding_latin1.xml": "V30: the schema sees decoded text, not the file's encoding",
}
# libxml2 cannot schema-validate a tree holding unresolved entity references.
XSD_NOT_APPLICABLE = {
    "doctype_entity.xml": "V31: a DOCTYPE is outside what a schema validates",
}


@pytest.fixture(scope="module")
def xsd() -> etree.XMLSchema:
    return etree.XMLSchema(etree.parse(str(XSD_PATH)))


@pytest.fixture(scope="module")
def ns_xsd() -> etree.XMLSchema:
    """The schema included into the IML namespace, as spec 2.3 describes."""
    wrapper = (
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"'
        f' targetNamespace="{IML_NAMESPACE}" elementFormDefault="qualified">'
        f'<xs:include schemaLocation="{XSD_PATH.as_uri()}"/></xs:schema>'
    )
    return etree.XMLSchema(etree.fromstring(wrapper.encode()))


def _xsd_valid(root: etree._Element, xsd: etree.XMLSchema, ns_xsd: etree.XMLSchema) -> bool:
    schema = ns_xsd if root.tag.startswith(f"{{{IML_NAMESPACE}}}") else xsd
    return bool(schema.validate(root))


def _fixtures(kind: str) -> list[Path]:
    return sorted((FIXTURES_DIR / kind).glob("*.xml"))


# ---------------------------------------------------------------------------
# spec.md examples
# ---------------------------------------------------------------------------


def _as_document(block: str) -> str:
    """Turn a spec ```xml block into a complete IML document."""
    if re.fullmatch(r"<\?xml[^>]*\?>", block):
        return block + "\n<utterance>Hello.</utterance>"  # the declaration alone
    body = re.sub(r"\A(?:<\?xml[^>]*\?>|<!--.*?-->|\s)+", "", block, flags=re.S)
    if re.match(r"<(?:\w+:)?(?:iml|utterance)\b", body):
        return block
    return f"<utterance>{block}</utterance>"  # an inline fragment


def _spec_examples() -> list[tuple[str, str]]:
    text = SPEC_PATH.read_text(encoding="utf-8")
    examples: list[tuple[str, str]] = []
    for match in re.finditer(r"```xml\n(.*?)```", text, re.S):
        heading = re.findall(r"^#{2,3} (.+)$", text[: match.start()], re.M)[-1]
        examples.append((heading, _as_document(match.group(1).strip())))
    return examples


SPEC_EXAMPLES = _spec_examples()


def test_spec_examples_found() -> None:
    headings = [h for h, _ in SPEC_EXAMPLES]
    assert len(SPEC_EXAMPLES) >= 16
    assert any(h.startswith("C.4") for h in headings)
    assert any(h.startswith("2.3") for h in headings)


@pytest.mark.parametrize(
    "document", [d for _, d in SPEC_EXAMPLES], ids=[h for h, _ in SPEC_EXAMPLES]
)
class TestSpecExamples:
    def test_schema_accepts(
        self, document: str, xsd: etree.XMLSchema, ns_xsd: etree.XMLSchema
    ) -> None:
        root = etree.fromstring(document.encode(), parser=_PARSER)
        assert _xsd_valid(root, xsd, ns_xsd)

    def test_validator_accepts_without_warnings(self, document: str) -> None:
        result = IMLValidator().validate(document)
        assert result.errors == []
        assert result.warnings == []

    def test_round_trips(self, document: str) -> None:
        parser = IMLParser()
        doc = parser.parse(document)
        assert parser.parse(parser.to_iml_string(doc)) == doc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def test_schema_compiles(xsd: etree.XMLSchema, ns_xsd: etree.XMLSchema) -> None:
    assert isinstance(xsd, etree.XMLSchema)
    assert isinstance(ns_xsd, etree.XMLSchema)


@pytest.mark.parametrize("path", _fixtures("valid"), ids=lambda p: p.name)
def test_valid_fixture_passes_schema(
    path: Path, xsd: etree.XMLSchema, ns_xsd: etree.XMLSchema
) -> None:
    root = etree.parse(str(path), _PARSER).getroot()
    assert _xsd_valid(root, xsd, ns_xsd), path.name


@pytest.mark.parametrize("path", _fixtures("invalid"), ids=lambda p: p.name)
def test_invalid_fixture_fails_schema(
    path: Path, xsd: etree.XMLSchema, ns_xsd: etree.XMLSchema
) -> None:
    if path.name in XSD_NOT_APPLICABLE:
        pytest.skip(XSD_NOT_APPLICABLE[path.name])
    root = etree.parse(str(path), _PARSER).getroot()
    # An exempt fixture must really be schema-valid, so the list stays honest.
    expected = path.name in XSD_CANNOT_EXPRESS
    assert _xsd_valid(root, xsd, ns_xsd) is expected, path.name
    assert IMLValidator().validate_file(path).valid is False


def test_namespaced_schema_rejects_bad_values(ns_xsd: etree.XMLSchema) -> None:
    root = etree.fromstring(
        f'<iml:utterance xmlns:iml="{IML_NAMESPACE}"><iml:emphasis level="loud">x'
        "</iml:emphasis></iml:utterance>".encode()
    )
    assert not ns_xsd.validate(root)


# ---------------------------------------------------------------------------
# Vocabularies: spec.md == XSD == validator
# ---------------------------------------------------------------------------


def _spec_bullet_values(label: str) -> set[str]:
    """Backticked values of the bullet list under a ``**<label>:**`` line."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    block = text.split(f"**{label}:**", 1)[1].split("\n\n", 2)[1]
    return set(re.findall(r"^- `([^`]+)`", block, re.M))


def _spec_named_rates() -> set[str]:
    text = SPEC_PATH.read_text(encoding="utf-8")
    named = re.search(r"^- Named: (.*)$", text, re.M)
    assert named is not None
    return set(re.findall(r"`([^`]+)`", named.group(1)))


def _xsd_enumeration(type_name: str) -> set[str]:
    schema = etree.parse(str(XSD_PATH))
    found = schema.xpath(
        "//xs:simpleType[@name=$name]//xs:enumeration/@value",
        namespaces={"xs": XS[1:-1]},
        name=type_name,
    )
    assert isinstance(found, list)
    return {str(v) for v in found}


VOCABULARIES = [
    ("Emotion values (core set)", "coreEmotionType", "_CORE_EMOTIONS"),
    ("Pitch contour values", "pitchContourType", "_VALID_PITCH_CONTOURS"),
    ("Quality values", "qualityType", "_VALID_QUALITIES"),
    ("Level values", "emphasisLevelType", "_VALID_EMPHASIS_LEVELS"),
    ("Tempo values", "tempoType", "_VALID_TEMPOS"),
    ("Rhythm values", "rhythmType", "_VALID_RHYTHMS"),
    ("Consent values", "consentType", "_VALID_CONSENT_VALUES"),
    ("Processing values", "processingType", "_VALID_PROCESSING_VALUES"),
]


@pytest.mark.parametrize(("label", "xsd_type", "constant"), VOCABULARIES)
def test_vocabulary_agrees(label: str, xsd_type: str, constant: str) -> None:
    spec_values = _spec_bullet_values(label)
    assert spec_values, label
    assert _xsd_enumeration(xsd_type) == spec_values
    assert set(getattr(validator_module, constant)) == spec_values


def test_named_rates_agree() -> None:
    spec_values = _spec_named_rates()
    assert spec_values == {"fast", "slow", "medium"}
    assert _xsd_enumeration("rateNamedType") == spec_values
    assert set(validator_module._VALID_NAMED_RATES) == spec_values


def _spec_section(heading: str) -> str:
    text = SPEC_PATH.read_text(encoding="utf-8")
    return text.split(f"### {heading}", 1)[1].split("\n---", 1)[0]


def test_plausible_ranges_agree() -> None:
    """Spec 6.4's table states the limits the validator warns at (V33)."""
    table = _spec_section("6.4 Plausible Values")
    v = validator_module
    rows = {
        "`pitch` (relative)": f"`-{v.MAX_PITCH_SEMITONES:g}st` to `+{v.MAX_PITCH_SEMITONES:g}st`,"
        f" `{v.MIN_PITCH_PERCENT:g}%` to `+{v.MAX_PITCH_PERCENT:g}%`",
        "`pitch` (absolute)": f"`{v.MIN_F0_HZ:g}Hz` to `{v.MAX_F0_HZ:g}Hz`",
        "`volume`": f"`-{v.MAX_VOLUME_DB:g}dB` to `+{v.MAX_VOLUME_DB:g}dB`",
        "`rate` (percentage)": f"`{v.MIN_RATE_PERCENT:g}%` to `{v.MAX_RATE_PERCENT:g}%`",
        "`duration`": f"At most {v.MAX_PAUSE_MS} ms",
        "`f0_mean`, `f0_range`, `f0_contour`": f"Every value {v.MIN_F0_HZ:g} to {v.MAX_F0_HZ:g} Hz",
        "`speech_rate`": f"At most {v.MAX_SPEECH_RATE:g} syllables/second",
        "`duration_ms`": f"At most {v.MAX_DURATION_MS} ms",
    }
    for attribute, limits in rows.items():
        row = re.search(rf"^\| {re.escape(attribute)} \|[^|]*\|([^|]*)\|$", table, re.M)
        assert row is not None, attribute
        assert limits in row.group(1), attribute
    # Two octaves, whichever the unit.
    highest, lowest = (1 + v.MAX_PITCH_PERCENT / 100, 1 + v.MIN_PITCH_PERCENT / 100)
    assert highest == pytest.approx(2 ** (v.MAX_PITCH_SEMITONES / 12))
    assert lowest == pytest.approx(2 ** (-v.MAX_PITCH_SEMITONES / 12))


def test_plausibility_is_a_should() -> None:
    table = _spec_section("6.4 Plausible Values")
    assert "Producers SHOULD NOT emit values outside the ranges below" in table
    assert "validators SHOULD report them as warnings" in table
    appendix = SPEC_PATH.read_text(encoding="utf-8").split("### D.2", 1)[1].split("### D.3")[0]
    assert re.search(r"^\| S17 \| .*plausible ranges.* \| 6\.4 \|$", appendix, re.M)
    # A long silence is real; the validator's advice for it is the spec's.
    assert (
        "producers SHOULD end the utterance at such a silence, or write it as a `<pause>` of "
        f"at most {validator_module.MAX_PAUSE_MS} ms" in table
    )


# ---------------------------------------------------------------------------
# Content models: spec Appendix A == XSD == validator
# ---------------------------------------------------------------------------

_CHILD_MARKUP = {
    "prosody": '<prosody pitch="+5%">c</prosody>',
    "emphasis": '<emphasis level="strong">c</emphasis>',
    "pause": '<pause duration="100"/>',
    "segment": "<segment>c</segment>",
    "utterance": "<utterance>c</utterance>",
    "iml": "<iml><utterance>c</utterance></iml>",
}
_PARENT_OPEN = {
    "prosody": '<prosody pitch="+5%">',
    "emphasis": '<emphasis level="strong">',
    "segment": "<segment>",
    "pause": '<pause duration="100">',
}


def _appendix_a_content() -> dict[str, set[str]]:
    text = SPEC_PATH.read_text(encoding="utf-8")
    table = text.split("## Appendix A", 1)[1].split("## Appendix B", 1)[0]
    content: dict[str, set[str]] = {}
    for row in re.findall(r"^\| `<(\w+)>` \|[^|]*\|([^|]*)\|", table, re.M):
        tag, cell = row
        content[tag] = set(re.findall(r"<(\w+)>", cell))
    return content


def _nest(parent: str, child: str) -> str:
    if parent == "iml":
        return f"<iml><utterance>x</utterance>{_CHILD_MARKUP[child]}</iml>"
    if parent == "utterance":
        return f"<utterance>{_CHILD_MARKUP[child]}</utterance>"
    return f"<utterance>{_PARENT_OPEN[parent]}{_CHILD_MARKUP[child]}</{parent}></utterance>"


def test_appendix_a_parsed() -> None:
    content = _appendix_a_content()
    assert set(content) == {"iml", "utterance", "prosody", "pause", "emphasis", "segment"}
    assert content["pause"] == set()


@pytest.mark.parametrize("parent", ["iml", "utterance", "prosody", "emphasis", "segment", "pause"])
@pytest.mark.parametrize("child", list(_CHILD_MARKUP))
def test_content_model_agrees(
    parent: str, child: str, xsd: etree.XMLSchema, ns_xsd: etree.XMLSchema
) -> None:
    allowed = child in _appendix_a_content()[parent]
    document = _nest(parent, child)
    assert IMLValidator().validate(document).valid is allowed, document
    root = etree.fromstring(document.encode(), parser=_PARSER)
    assert _xsd_valid(root, xsd, ns_xsd) is allowed, document


# ---------------------------------------------------------------------------
# Validator/XSD agreement corpus
# ---------------------------------------------------------------------------

AGREEMENT_CORPUS = [
    # attribute values
    '<utterance><prosody rate="ludicrous">x</prosody></utterance>',
    '<utterance><prosody rate="150%">x</prosody></utterance>',
    '<utterance><prosody pitch_contour="wobble" quality="robotic">x</prosody></utterance>',
    '<utterance><segment tempo="glacial" rhythm="jazzy">x</segment></utterance>',
    '<utterance><prosody pitch="high" volume="loud">x</prosody></utterance>',
    '<utterance><prosody pitch="+\u0661\u0665%">x</prosody></utterance>',
    '<utterance><emphasis level="extreme">x</emphasis></utterance>',
    '<utterance><prosody f0_mean="abc" duration_ms="-1" f0_range="high">x</prosody></utterance>',
    '<utterance><prosody f0_mean="-3">x</prosody></utterance>',
    '<utterance><prosody jitter="1e-05" hnr="-3" intensity_mean="-12">x</prosody></utterance>',
    # finite floats and bounded integers (spec 2.6)
    '<utterance><prosody hnr="1e400">x</prosody></utterance>',
    '<utterance><prosody f0_mean="1e400">x</prosody></utterance>',
    '<utterance><prosody intensity_mean="-1e400">x</prosody></utterance>',
    '<utterance><prosody hnr="1e308" jitter="1e-400">x</prosody></utterance>',
    f'<utterance><prosody f0_mean="{"9" * 400}">x</prosody></utterance>',
    f'<utterance>a<pause duration="{"1" * 5000}"/>b</utterance>',
    f'<utterance><prosody duration_ms="{"1" * 5000}">x</prosody></utterance>',
    '<utterance>a<pause duration="2147483647"/>b</utterance>',
    '<utterance>a<pause duration="2147483648"/>b</utterance>',
    f'<utterance>a<pause duration="{"0" * 5000}5"/>b</utterance>',
    '<utterance emotion="calm" confidence="NaN">x</utterance>',
    '<utterance emotion="calm" confidence="1e-1">x</utterance>',
    '<utterance emotion="calm" confidence=" 0.5 ">x</utterance>',
    '<utterance emotion="calm" confidence="1.5">x</utterance>',
    '<utterance>a<pause duration="8_00"/>b</utterance>',
    '<utterance>a<pause duration="\u0668\u0660\u0660"/>b</utterance>',
    '<utterance>a<pause duration="+800"/>b</utterance>',
    '<utterance>a<pause duration="0"/>b</utterance>',
    '<iml consent="none" processing="hybrid"><utterance>x</utterance></iml>',
    '<iml consent="maybe"><utterance>x</utterance></iml>',
    '<iml version="0.1.0-alpha" language="en-US"><utterance>x</utterance></iml>',
    '<iml version="0.1" language="en_US"><utterance>x</utterance></iml>',
    # extensions (spec 9.2)
    '<utterance><prosody x-formant-shift="+200Hz" x-nasality="0.7">x</prosody></utterance>',
    '<utterance foo="bar">x</utterance>',
    '<utterance xmlns:a="urn:a" a:note="n">x</utterance>',
    # implausible but valid values (V33 warnings, spec 6.4)
    '<utterance><prosody volume="+80dB" pitch="+40st" rate="1000%">x</prosody></utterance>',
    '<utterance><prosody pitch="-150%">x</prosody><prosody pitch="0Hz">y</prosody></utterance>',
    '<utterance>a<pause duration="600000"/>b</utterance>',
    '<utterance><prosody f0_mean="0" f0_range="900-20" f0_contour="5,5000" speech_rate="50"'
    ' duration_ms="999999999">x</prosody></utterance>',
    # structure
    "<iml><utterance>Hello</utterance><prosody>IMPORTANT</prosody></iml>",
    "<iml>stray text<utterance>x</utterance></iml>",
    "<iml></iml>",
    "<utterance>a<utterance>b</utterance></utterance>",
    '<utterance><emphasis level="strong"><emphasis level="moderate">x</emphasis></emphasis>'
    "</utterance>",
    '<utterance><emphasis level="strong">a<pause duration="300"/>b</emphasis></utterance>',
    '<utterance><prosody pitch="+5%"><prosody volume="+3dB">x</prosody></prosody></utterance>',
    '<utterance><segment>a<segment>b</segment></segment></utterance>',
    '<utterance><pause duration="5">oops</pause></utterance>',
    '<utterance>a<pause duration="300">\n  </pause>b</utterance>',
    '<utterance>a<pause duration="300"><!-- c --><?pi x?></pause>b</utterance>',
    '<utterance>a<pause duration="300"><f:x xmlns:f="urn:x"/></pause>b</utterance>',
    # namespaces and foreign content
    f'<iml:utterance xmlns:iml="{IML_NAMESPACE}" emotion="calm" confidence="0.9">x'
    "</iml:utterance>",
    '<utterance>Hi <f:b xmlns:f="urn:x">there</f:b></utterance>',
    '<iml><f:meta xmlns:f="urn:x">recorded</f:meta><utterance>x</utterance></iml>',
    '<iml><f:turn xmlns:f="urn:x"><utterance emotion="calm" confidence="5">y</utterance>'
    "</f:turn><utterance>z</utterance></iml>",
    f'<iml:utterance xmlns:iml="{IML_NAMESPACE}" iml:emotion="angry" iml:confidence="9">x'
    "</iml:utterance>",
    # comments and processing instructions
    "<iml><?pi x?><utterance>hi<!-- c --> there</utterance></iml>",
]


def _case_id(document: str) -> str:
    """Test id for a corpus document, with long runs of one character abbreviated."""
    return re.sub(r"(.)\1{19,}", lambda m: f"{m.group(1)}{{{len(m.group(0))}}}", document)


@pytest.mark.parametrize("document", AGREEMENT_CORPUS, ids=_case_id)
def test_validator_agrees_with_schema(
    document: str, xsd: etree.XMLSchema, ns_xsd: etree.XMLSchema
) -> None:
    root = etree.fromstring(document.encode(), parser=_PARSER)
    assert IMLValidator().validate(document).valid is _xsd_valid(root, xsd, ns_xsd)


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        ('<utterance emotion="angry">x</utterance>', "V3 co-occurrence"),
        ("<utterance>Hi <span>there</span></utterance>", "unqualified unknown element"),
        (
            '<utterance><f:w xmlns:f="urn:x"><segment>a<segment>b</segment></segment></f:w>'
            "</utterance>",
            "schema does not look inside foreign elements",
        ),
        (
            '<iml><f:turn xmlns:f="urn:x"><utterance>y</utterance></f:turn></iml>',
            "V2 counts utterances in foreign wrappers; the schema wants a direct one",
        ),
    ],
)
def test_known_disagreements(
    document: str, reason: str, xsd: etree.XMLSchema, ns_xsd: etree.XMLSchema
) -> None:
    """Cases XSD 1.0 cannot express; documented in the schema header."""
    root = etree.fromstring(document.encode(), parser=_PARSER)
    assert IMLValidator().validate(document).valid is not _xsd_valid(root, xsd, ns_xsd), reason
