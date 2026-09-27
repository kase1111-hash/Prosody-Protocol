"""IML parser -- converts IML XML strings into ``models.IMLDocument`` objects.

Uses ``lxml.etree`` for XML parsing with recursive descent through the
element tree, constructing immutable model objects. Handles mixed content
(text interleaved with child elements) via ``element.text`` / ``child.tail``.

As a consumer (spec Section 6.2) the parser is lenient about attribute values
and nesting, which :class:`~prosody_protocol.validator.IMLValidator` reports on:

- Elements in no namespace or in the IML namespace (:data:`IML_NAMESPACE`)
  are IML elements; elements in any other namespace are unknown (Section 2.3).
- Unknown elements are transparent: their tags are dropped and their content,
  text and IML markup alike, is kept in place (Section 6.2).
- Comments and processing instructions are dropped; they never become text.
  CDATA sections are ordinary text.
- Attributes without a typed model field (``x-`` extensions, unknown
  attributes, invalid values of known ones) are kept in ``extra_attributes``.

It raises :class:`~prosody_protocol.exceptions.IMLParseError` instead of
silently losing content: for malformed XML, a DOCTYPE (Section 2.5), a file
that is not UTF-8 (Section 2.2), and content the model has no place for --
text or markup outside any ``<utterance>``, an ``<utterance>`` or ``<iml>``
nested inside another element, or a ``<pause>`` with content.

Spec reference: Sections 2-3, 6.2.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar, cast

from lxml import etree

from .exceptions import IMLParseError
from .models import (
    ChildNode,
    Emphasis,
    ExtraAttributes,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)

#: Namespace for IML embedded in other XML formats (spec Section 2.3).
IML_NAMESPACE = "http://prosody-protocol.org/iml/0.1"

_XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"

# Hardened XML parser, shared with the validator. It never resolves entities,
# loads a DTD or touches the network (defense-in-depth against XXE), and it
# always decodes as UTF-8 (spec M1): callers hand it text that is already
# decoded, so an encoding declaration must not make lxml decode it twice.
_SECURE_PARSER = etree.XMLParser(
    resolve_entities=False, no_network=True, load_dtd=False, encoding="utf-8"
)

_IML_ELEMENTS = frozenset({"iml", "utterance", "prosody", "pause", "emphasis", "segment"})

# The whitespace characters of XML (S production); nothing else is formatting.
_XML_WHITESPACE = " \t\r\n"
_XML_WHITESPACE_RUN = re.compile(r"[ \t\r\n]+")

# Lexical forms of IML numbers (spec Section 2.6): ASCII digits only, no NaN,
# infinity or digit-group underscores (which Python's float()/int() accept).
_FLOAT_RE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_POSITIVE_INT_RE = re.compile(r"\+?[0-9]+")

#: Largest IML Integer value (spec Section 2.6): 2**31 - 1.
MAX_INTEGER = 2_147_483_647

_XML_DECLARATION_RE = re.compile(
    r"""\ufeff?<\?xml\s[^>]*?\bencoding\s*=\s*(?:"([^"]*)"|'([^']*)')"""
)
_UTF8_NAMES = frozenset({"utf-8", "utf8"})

# Plain-text extraction drops XML indentation between an element's last word
# and closing punctuation that follows the element ("GREAT\n  </prosody>.").
_CLOSING_PUNCTUATION = frozenset(".,;:!?)]}\u2026")

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Lexical helpers (shared with the validator)
# ---------------------------------------------------------------------------


def _parse_float(raw: str) -> float | None:
    """Parse an IML Float; ``None`` unless it is a finite ASCII decimal number."""
    text = raw.strip(_XML_WHITESPACE)
    if not _FLOAT_RE.fullmatch(text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def _parse_non_negative_float(raw: str) -> float | None:
    value = _parse_float(raw)
    return value if value is not None and value >= 0.0 else None


def _parse_confidence(raw: str) -> float | None:
    value = _parse_float(raw)
    return value if value is not None and 0.0 <= value <= 1.0 else None


def _parse_positive_int(raw: str) -> int | None:
    """Parse a positive IML Integer (ASCII digits, at most :data:`MAX_INTEGER`).

    Returns ``None`` for anything else. The digit count is checked before
    converting, so an over-long value never reaches ``int()``, which refuses
    strings of more than 4300 digits.
    """
    text = raw.strip(_XML_WHITESPACE)
    if not _POSITIVE_INT_RE.fullmatch(text):
        return None
    digits = text.lstrip("+").lstrip("0")
    if not digits or len(digits) > len(str(MAX_INTEGER)):
        return None
    value = int(digits)
    return value if value <= MAX_INTEGER else None


def _iml_name(node: etree._Element) -> str | None:
    """Return the name to dispatch *node* on, or ``None`` if it is not an element.

    Comments, processing instructions and entity references give ``None``.
    Elements in no namespace or the IML namespace give their local name;
    elements in any other namespace keep their Clark name (``{uri}name``),
    which never matches an IML tag, so they are handled as unknown elements.
    """
    tag = node.tag
    if not isinstance(tag, str):
        return None
    if tag.startswith("{"):
        uri, local = tag[1:].split("}", 1)
        return local if uri == IML_NAMESPACE else tag
    return tag


def _display_name(node: etree._Element) -> str:
    """Return the element name as written in the document (with its prefix).

    An element in a foreign default namespace has no prefix to show, so it
    is named by its Clark name (``{uri}name``) to tell it apart from IML.
    """
    qname = etree.QName(node)
    if node.prefix:
        return f"{node.prefix}:{qname.localname}"
    if qname.namespace not in (None, IML_NAMESPACE):
        return qname.text
    return qname.localname


def _iter_content(
    element: etree._Element, *, text: bool = True
) -> Iterator[str | etree._Element]:
    """Yield the content of *element* in document order.

    Yields text runs (only when *text* is true) and IML child elements.
    Comments, processing instructions and entity references are skipped,
    but the text after them is kept. Unknown elements are transparent: their
    content is yielded in their place (spec Section 6.2).
    """
    if text and element.text:
        yield element.text
    for child in element:
        name = _iml_name(child)
        if name in _IML_ELEMENTS:
            yield child
        elif name is not None:
            yield from _iter_content(child, text=text)
        if text and child.tail:
            yield child.tail


def _iter_iml_content(root: etree._Element) -> Iterator[str | etree._Element]:
    """Yield the content of an ``<iml>`` element in document order.

    Like :func:`_iter_content`, except that text inside unknown elements is
    skipped: outside an ``<utterance>`` it is not IML text content, so only
    the IML elements inside unknown elements count (spec Section 6.2).
    """
    if root.text:
        yield root.text
    for child in root:
        name = _iml_name(child)
        if name in _IML_ELEMENTS:
            yield child
        elif name is not None:
            yield from _iter_content(child, text=False)
        if child.tail:
            yield child.tail


def _is_blank(text: str) -> bool:
    return not text.strip(_XML_WHITESPACE)


def _declared_encoding(iml_string: str) -> str | None:
    """Return the encoding named in the XML declaration, if there is one."""
    match = _XML_DECLARATION_RE.match(iml_string)
    if match is None:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def _is_utf8_name(encoding: str) -> bool:
    return encoding.strip(_XML_WHITESPACE).lower() in _UTF8_NAMES


def _has_doctype(root: etree._Element) -> bool:
    doctype: str = getattr(root.getroottree().docinfo, "doctype", "")  # missing from lxml-stubs
    return bool(doctype)


def _line(element: etree._Element) -> int | None:
    """Return the source line of *element* (lxml-stubs mistypes ``sourceline``)."""
    return cast("int | None", element.sourceline)


def _syntax_error_location(exc: etree.XMLSyntaxError) -> tuple[int | None, int | None]:
    position = getattr(exc, "position", None)
    if position is None:
        return getattr(exc, "lineno", None), None
    line, column = position
    return line, column


def _utf8_error(data: bytes, exc: UnicodeDecodeError) -> tuple[str, int, int]:
    """Describe the first non-UTF-8 byte in *data* as (message, line, column)."""
    line_start = data.rfind(b"\n", 0, exc.start) + 1
    line = data.count(b"\n", 0, exc.start) + 1
    column = len(data[line_start:exc.start].decode("utf-8", errors="replace")) + 1
    message = (
        f"IML documents must be encoded in UTF-8 (spec 2.2); "
        f"byte 0x{data[exc.start]:02x} is not valid UTF-8"
    )
    return message, line, column


def _decode_utf8(data: bytes) -> str:
    """Decode IML file contents, which must be UTF-8 (a byte order mark is allowed)."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        message, line, column = _utf8_error(data, exc)
        raise IMLParseError(message, line=line, column=column) from exc


def _parse_xml(iml_string: str) -> etree._Element:
    """Parse *iml_string* with the hardened parser, raising IMLParseError."""
    try:
        data = iml_string.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise IMLParseError(
            f"IML text cannot be encoded as UTF-8 (spec 2.2): {exc.reason}"
        ) from exc
    try:
        root = etree.fromstring(data, parser=_SECURE_PARSER)  # noqa: S320
    except etree.XMLSyntaxError as exc:
        line, column = _syntax_error_location(exc)
        raise IMLParseError(str(exc), line=line, column=column) from exc
    if _has_doctype(root):
        raise IMLParseError("IML documents must not contain a DOCTYPE declaration (spec 2.5)")
    return root


def _snippet(text: str, limit: int = 40) -> str:
    text = _XML_WHITESPACE_RUN.sub(" ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Element parsers
# ---------------------------------------------------------------------------


class _AttributeReader:
    """Read an element's attributes into typed fields.

    Every attribute that is not consumed by :meth:`text` or a successful
    :meth:`convert` ends up in :meth:`extra`, in document order.
    """

    def __init__(self, element: etree._Element) -> None:
        self._items = [(str(name), str(value)) for name, value in element.attrib.items()]
        self._attrib = dict(self._items)
        self._typed: set[str] = set()

    def text(self, name: str) -> str | None:
        self._typed.add(name)
        return self._attrib.get(name)

    def convert(self, name: str, parse: Callable[[str], _T | None]) -> _T | None:
        """Return ``parse(value)``; a value it rejects (``None``) is left for :meth:`extra`."""
        raw = self._attrib.get(name)
        if raw is None:
            return None
        value = parse(raw)
        if value is not None:
            self._typed.add(name)
        return value

    def extra(self) -> ExtraAttributes:
        return tuple((name, value) for name, value in self._items if name not in self._typed)


def _collect_children(element: etree._Element) -> tuple[ChildNode, ...]:
    """Walk mixed content of *element*, returning an ordered tuple of
    text strings and parsed child model objects.
    """
    children: list[ChildNode] = []
    # Consecutive text runs (split by comments, processing instructions or
    # unknown element tags) are joined once, keeping the walk linear.
    text: list[str] = []
    for item in _iter_content(element):
        if isinstance(item, str):
            text.append(item)
            continue
        if text:
            children.append("".join(text))
            text.clear()
        children.append(_parse_inline(item))
    if text:
        children.append("".join(text))
    return tuple(children)


def _parse_inline(element: etree._Element) -> ChildNode:
    """Parse an IML element that appears inside an utterance."""
    tag = _iml_name(element)
    if tag == "prosody":
        return _parse_prosody(element)
    if tag == "pause":
        return _parse_pause(element)
    if tag == "emphasis":
        return _parse_emphasis(element)
    if tag == "segment":
        return _parse_segment(element)
    raise IMLParseError(
        f"<{tag}> cannot be nested inside another IML element (spec 5.2)",
        line=_line(element),
    )


def _parse_pause(element: etree._Element) -> Pause:
    # Spec 3.3: no elements, not even unknown ones, and only whitespace text.
    has_element = any(_iml_name(child) is not None for child in element)
    if has_element or not all(
        isinstance(item, str) and _is_blank(item) for item in _iter_content(element)
    ):
        raise IMLParseError(
            "<pause> must be an empty element but has content (spec 3.3)",
            line=_line(element),
        )
    attrs = _AttributeReader(element)
    # Parser is lenient -- the validator reports a missing or invalid
    # duration. The model holds 0 and an invalid value stays in extra_attributes.
    duration = attrs.convert("duration", _parse_positive_int)
    return Pause(duration=duration or 0, extra_attributes=attrs.extra())


def _parse_prosody(element: etree._Element) -> Prosody:
    attrs = _AttributeReader(element)
    return Prosody(
        children=_collect_children(element),
        pitch=attrs.text("pitch"),
        pitch_contour=attrs.text("pitch_contour"),
        volume=attrs.text("volume"),
        rate=attrs.text("rate"),
        quality=attrs.text("quality"),
        # Extended attributes (Section 4).
        f0_mean=attrs.convert("f0_mean", _parse_non_negative_float),
        f0_range=attrs.text("f0_range"),
        f0_contour=attrs.text("f0_contour"),
        intensity_mean=attrs.convert("intensity_mean", _parse_float),
        intensity_range=attrs.convert("intensity_range", _parse_non_negative_float),
        speech_rate=attrs.convert("speech_rate", _parse_non_negative_float),
        duration_ms=attrs.convert("duration_ms", _parse_positive_int),
        jitter=attrs.convert("jitter", _parse_non_negative_float),
        shimmer=attrs.convert("shimmer", _parse_non_negative_float),
        hnr=attrs.convert("hnr", _parse_float),
        extra_attributes=attrs.extra(),
    )


def _non_empty(raw: str) -> str | None:
    return raw or None


def _parse_emphasis(element: etree._Element) -> Emphasis:
    attrs = _AttributeReader(element)
    return Emphasis(
        # The model holds "" for a missing level; level="" stays in
        # extra_attributes so that it is written back as level="".
        level=attrs.convert("level", _non_empty) or "",
        children=_collect_children(element),
        extra_attributes=attrs.extra(),
    )


def _parse_segment(element: etree._Element) -> Segment:
    attrs = _AttributeReader(element)
    return Segment(
        children=_collect_children(element),
        tempo=attrs.text("tempo"),
        rhythm=attrs.text("rhythm"),
        extra_attributes=attrs.extra(),
    )


def _parse_utterance(element: etree._Element) -> Utterance:
    attrs = _AttributeReader(element)
    return Utterance(
        children=_collect_children(element),
        emotion=attrs.text("emotion"),
        confidence=attrs.convert("confidence", _parse_confidence),
        speaker_id=attrs.text("speaker_id"),
        extra_attributes=attrs.extra(),
    )


def _parse_iml(root: etree._Element) -> IMLDocument:
    utterances: list[Utterance] = []
    for item in _iter_iml_content(root):
        if isinstance(item, str):
            if not _is_blank(item):
                raise IMLParseError(
                    f"Text outside any <utterance> in <iml>: {_snippet(item)!r} (spec 5.2)"
                )
        elif _iml_name(item) == "utterance":
            utterances.append(_parse_utterance(item))
        elif _iml_name(item) == "iml":
            raise IMLParseError(
                "<iml> cannot be nested inside another IML element (spec 5.2)",
                line=_line(item),
            )
        else:
            raise IMLParseError(
                f"<{_iml_name(item)}> must be inside an <utterance>, "
                "not directly inside <iml> (spec 5.2)",
                line=_line(item),
            )
    attrs = _AttributeReader(root)
    return IMLDocument(
        utterances=tuple(utterances),
        version=attrs.text("version"),
        language=attrs.text("language"),
        consent=attrs.text("consent"),
        processing=attrs.text("processing"),
        extra_attributes=attrs.extra(),
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _children_to_plain_text(children: tuple[ChildNode, ...]) -> str:
    parts: list[str] = []
    for index, child in enumerate(children):
        following = children[index + 1] if index + 1 < len(children) else None
        closes = isinstance(following, str) and following[:1] in _CLOSING_PUNCTUATION
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, Pause):
            # A pause separates words, but not a word from its punctuation.
            if not closes:
                parts.append(" ")
        else:
            inner = _children_to_plain_text(child.children)
            parts.append(inner.rstrip(_XML_WHITESPACE) if closes else inner)
    return "".join(parts)


def _serialize_children(children: tuple[ChildNode, ...]) -> str:
    parts: list[str] = []
    for child in children:
        if isinstance(child, str):
            parts.append(_escape_xml(child))
        elif isinstance(child, Pause):
            attrs = _attrs([("duration", child.duration or None)], child.extra_attributes)
            parts.append(f"<pause{attrs}/>")
        elif isinstance(child, Prosody):
            parts.append(_serialize_prosody(child))
        elif isinstance(child, Emphasis):
            parts.append(_serialize_emphasis(child))
        elif isinstance(child, Segment):
            parts.append(_serialize_segment(child))
    return "".join(parts)


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        # A literal CR would be normalized to LF when the output is re-parsed.
        .replace("\r", "&#13;")
    )


def _escape_attr(value: str) -> str:
    # Attribute-value normalization turns literal tabs and newlines into spaces.
    return _escape_xml(value).replace("\t", "&#9;").replace("\n", "&#10;")


def _attrs(
    typed: list[tuple[str, str | float | int | None]], extra: ExtraAttributes = ()
) -> str:
    """Serialize typed attributes (skipping ``None``) followed by extra attributes."""
    parts: list[str] = []
    written: set[str] = set()
    for name, value in typed:
        if value is not None:
            parts.append(f' {name}="{_escape_attr(str(value))}"')
            written.add(name)
    prefixes: dict[str, str] = {}
    for name, value in extra:
        if name in written:
            continue
        qname = name
        if name.startswith("{"):
            uri, local = name[1:].split("}", 1)
            if uri == _XML_NAMESPACE:
                qname = f"xml:{local}"
            else:
                prefix = prefixes.get(uri)
                if prefix is None:
                    prefix = prefixes[uri] = f"ns{len(prefixes)}"
                    parts.append(f' xmlns:{prefix}="{_escape_attr(uri)}"')
                qname = f"{prefix}:{local}"
        parts.append(f' {qname}="{_escape_attr(value)}"')
        written.add(name)
    return "".join(parts)


def _serialize_prosody(p: Prosody) -> str:
    attrs = _attrs(
        [
            ("pitch", p.pitch),
            ("pitch_contour", p.pitch_contour),
            ("volume", p.volume),
            ("rate", p.rate),
            ("quality", p.quality),
            ("f0_mean", p.f0_mean),
            ("f0_range", p.f0_range),
            ("f0_contour", p.f0_contour),
            ("intensity_mean", p.intensity_mean),
            ("intensity_range", p.intensity_range),
            ("speech_rate", p.speech_rate),
            ("duration_ms", p.duration_ms),
            ("jitter", p.jitter),
            ("shimmer", p.shimmer),
            ("hnr", p.hnr),
        ],
        p.extra_attributes,
    )
    inner = _serialize_children(p.children)
    return f"<prosody{attrs}>{inner}</prosody>"


def _serialize_emphasis(e: Emphasis) -> str:
    # A missing level is written as missing, not as level="".
    attrs = _attrs([("level", e.level or None)], e.extra_attributes)
    inner = _serialize_children(e.children)
    return f"<emphasis{attrs}>{inner}</emphasis>"


def _serialize_segment(s: Segment) -> str:
    attrs = _attrs([("tempo", s.tempo), ("rhythm", s.rhythm)], s.extra_attributes)
    inner = _serialize_children(s.children)
    return f"<segment{attrs}>{inner}</segment>"


def _serialize_utterance(u: Utterance) -> str:
    attrs = _attrs(
        [("emotion", u.emotion), ("confidence", u.confidence), ("speaker_id", u.speaker_id)],
        u.extra_attributes,
    )
    inner = _serialize_children(u.children)
    return f"<utterance{attrs}>{inner}</utterance>"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class IMLParser:
    """Parse IML XML into structured :class:`IMLDocument` objects."""

    def parse(self, iml_string: str) -> IMLDocument:
        """Parse an IML XML string.

        Accepts both standalone ``<utterance>`` elements and ``<iml>``-wrapped
        documents, in no namespace or in the IML namespace.

        Raises :class:`~prosody_protocol.exceptions.IMLParseError` on
        malformed XML, a DOCTYPE declaration, or content the document model
        cannot hold without losing it (see the module docstring).
        """
        root = _parse_xml(iml_string)
        root_tag = _iml_name(root)
        try:
            if root_tag == "iml":
                return _parse_iml(root)
            if root_tag == "utterance":
                return IMLDocument(utterances=(_parse_utterance(root),))
        except RecursionError as exc:
            raise IMLParseError("IML document is nested too deeply to parse") from exc

        raise IMLParseError(
            f"Expected root element <iml> or <utterance>, got <{_display_name(root)}>"
        )

    def parse_file(self, path: str | Path) -> IMLDocument:
        """Parse an IML XML file from disk.

        The file must be UTF-8 (spec 2.2; a byte order mark is allowed).
        Raises :class:`~prosody_protocol.exceptions.IMLParseError` when it is
        not, and :class:`OSError` when it cannot be read.
        """
        return self.parse(_decode_utf8(Path(path).read_bytes()))

    def to_plain_text(self, doc: IMLDocument) -> str:
        """Extract plain text from an IML document, stripping all markup.

        Whitespace is normalized for reading: runs of XML whitespace collapse
        to one space, indentation between a marked-up word and the closing
        punctuation after it is dropped, and utterances are joined with a
        single space. The result has no leading or trailing whitespace.
        """
        texts = (
            _XML_WHITESPACE_RUN.sub(" ", _children_to_plain_text(utt.children)).strip(" ")
            for utt in doc.utterances
        )
        return " ".join(text for text in texts if text)

    def to_iml_string(self, doc: IMLDocument) -> str:
        """Serialize an IML document back to an XML string.

        A single utterance with no document-level attributes is written as a
        bare ``<utterance>``; anything else gets an ``<iml>`` wrapper. The
        output is un-namespaced IML with comments, processing instructions
        and unknown element tags removed, and without an XML declaration (so
        a non-UTF-8 encoding declaration, V30, is not carried over). All
        attributes are kept, including ``extra_attributes``, so a document
        with invalid attribute values stays invalid.
        """
        doc_attrs = _attrs(
            [
                ("version", doc.version),
                ("language", doc.language),
                ("consent", doc.consent),
                ("processing", doc.processing),
            ],
            doc.extra_attributes,
        )
        if len(doc.utterances) == 1 and not doc_attrs:
            return _serialize_utterance(doc.utterances[0])

        inner = "".join(_serialize_utterance(u) for u in doc.utterances)
        return f"<iml{doc_attrs}>{inner}</iml>"
