"""IML parser -- converts IML XML strings into ``models.IMLDocument`` objects.

Uses ``lxml.etree`` for XML parsing with recursive descent through the
element tree, constructing immutable model objects. Handles mixed content
(text interleaved with child elements) via ``element.text`` / ``child.tail``.

Spec reference: Sections 2-3.
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from .exceptions import IMLParseError
from .models import (
    ChildNode,
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)

# Extended attributes that live on <prosody> (Section 4).
_PROSODY_EXTENDED_ATTRS: dict[str, type] = {
    "f0_mean": float,
    "f0_range": str,
    "f0_contour": str,
    "intensity_mean": float,
    "intensity_range": float,
    "speech_rate": float,
    "duration_ms": int,
    "jitter": float,
    "shimmer": float,
    "hnr": float,
}


def _strip_ns(tag: str) -> str:
    """Remove namespace prefix from an element tag if present."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _collect_children(element: etree._Element) -> tuple[ChildNode, ...]:
    """Walk mixed content of *element*, returning an ordered tuple of
    text strings and parsed child model objects.
    """
    children: list[ChildNode] = []

    # Leading text before the first child element.
    if element.text:
        children.append(element.text)

    for child_el in element:
        node = _parse_element(child_el)
        if node is not None:
            children.append(node)
        # Tail text following the child element.
        if child_el.tail:
            children.append(child_el.tail)

    return tuple(children)


def _parse_pause(element: etree._Element) -> Pause:
    raw = element.get("duration")
    if raw is None:
        # Parser is lenient -- validator catches this.  Default to 0.
        return Pause(duration=0)
    try:
        return Pause(duration=int(raw))
    except ValueError:
        return Pause(duration=0)


def _parse_prosody(element: etree._Element) -> Prosody:
    kwargs: dict[str, object] = {
        "children": _collect_children(element),
        "pitch": element.get("pitch"),
        "pitch_contour": element.get("pitch_contour"),
        "volume": element.get("volume"),
        "rate": element.get("rate"),
        "quality": element.get("quality"),
    }
    for attr, typ in _PROSODY_EXTENDED_ATTRS.items():
        raw = element.get(attr)
        if raw is not None:
            try:
                kwargs[attr] = typ(raw)
            except (ValueError, TypeError):
                kwargs[attr] = None
    return Prosody(**kwargs)  # type: ignore[arg-type]


def _parse_emphasis(element: etree._Element) -> Emphasis:
    return Emphasis(
        level=element.get("level", ""),
        children=_collect_children(element),
    )


def _parse_segment(element: etree._Element) -> Segment:
    return Segment(
        children=_collect_children(element),
        tempo=element.get("tempo"),
        rhythm=element.get("rhythm"),
    )


def _parse_utterance(element: etree._Element) -> Utterance:
    confidence_raw = element.get("confidence")
    confidence: float | None = None
    if confidence_raw is not None:
        try:
            confidence = float(confidence_raw)
        except ValueError:
            confidence = None

    return Utterance(
        children=_collect_children(element),
        emotion=element.get("emotion"),
        confidence=confidence,
        speaker_id=element.get("speaker_id"),
    )


def _parse_element(element: etree._Element) -> ChildNode | None:
    """Dispatch parsing based on element tag name."""
    if isinstance(element, etree._Comment):
        return None
    tag = _strip_ns(element.tag)  # type: ignore[arg-type]
    if tag == "utterance":
        return _parse_utterance(element)
    if tag == "prosody":
        return _parse_prosody(element)
    if tag == "pause":
        return _parse_pause(element)
    if tag == "emphasis":
        return _parse_emphasis(element)
    if tag == "segment":
        return _parse_segment(element)
    # Unknown element -- skip it but preserve text content.
    return None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _children_to_plain_text(children: tuple[ChildNode, ...]) -> str:
    parts: list[str] = []
    for child in children:
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, Pause):
            parts.append(" ")
        elif isinstance(child, (Prosody, Emphasis, Segment, Utterance)):
            parts.append(_children_to_plain_text(child.children))
    return "".join(parts)


def _serialize_children(children: tuple[ChildNode, ...]) -> str:
    parts: list[str] = []
    for child in children:
        if isinstance(child, str):
            parts.append(_escape_xml(child))
        elif isinstance(child, Pause):
            parts.append(f'<pause duration="{child.duration}"/>')
        elif isinstance(child, Prosody):
            parts.append(_serialize_prosody(child))
        elif isinstance(child, Emphasis):
            parts.append(_serialize_emphasis(child))
        elif isinstance(child, Segment):
            parts.append(_serialize_segment(child))
        elif isinstance(child, Utterance):
            parts.append(_serialize_utterance(child))
    return "".join(parts)


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _attr(name: str, value: str | float | int | None) -> str:
    if value is None:
        return ""
    return f' {name}="{_escape_xml(str(value))}"'


def _serialize_prosody(p: Prosody) -> str:
    attrs = (
        _attr("pitch", p.pitch)
        + _attr("pitch_contour", p.pitch_contour)
        + _attr("volume", p.volume)
        + _attr("rate", p.rate)
        + _attr("quality", p.quality)
        + _attr("f0_mean", p.f0_mean)
        + _attr("f0_range", p.f0_range)
        + _attr("f0_contour", p.f0_contour)
        + _attr("intensity_mean", p.intensity_mean)
        + _attr("intensity_range", p.intensity_range)
        + _attr("speech_rate", p.speech_rate)
        + _attr("duration_ms", p.duration_ms)
        + _attr("jitter", p.jitter)
        + _attr("shimmer", p.shimmer)
        + _attr("hnr", p.hnr)
    )
    inner = _serialize_children(p.children)
    return f"<prosody{attrs}>{inner}</prosody>"


def _serialize_emphasis(e: Emphasis) -> str:
    inner = _serialize_children(e.children)
    return f'<emphasis level="{_escape_xml(e.level)}">{inner}</emphasis>'


def _serialize_segment(s: Segment) -> str:
    attrs = _attr("tempo", s.tempo) + _attr("rhythm", s.rhythm)
    inner = _serialize_children(s.children)
    return f"<segment{attrs}>{inner}</segment>"


def _serialize_utterance(u: Utterance) -> str:
    attrs = (
        _attr("emotion", u.emotion)
        + _attr("confidence", u.confidence)
        + _attr("speaker_id", u.speaker_id)
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
        documents.

        Raises :class:`~prosody_protocol.exceptions.IMLParseError` on
        malformed XML.
        """
        try:
            root = etree.fromstring(iml_string.encode("utf-8"))  # noqa: S320
        except etree.XMLSyntaxError as exc:
            raise IMLParseError(
                str(exc),
                line=getattr(exc, "lineno", None),
                column=getattr(exc, "position", (None, None))[1] if hasattr(exc, "position") else None,
            ) from exc

        root_tag = _strip_ns(root.tag)

        if root_tag == "iml":
            utterances = tuple(
                _parse_utterance(child)
                for child in root
                if not isinstance(child, etree._Comment)
                and _strip_ns(child.tag) == "utterance"  # type: ignore[arg-type]
            )
            return IMLDocument(
                utterances=utterances,
                version=root.get("version"),
                language=root.get("language"),
                consent=root.get("consent"),
                processing=root.get("processing"),
            )

        if root_tag == "utterance":
            return IMLDocument(utterances=(_parse_utterance(root),))

        raise IMLParseError(
            f"Expected root element <iml> or <utterance>, got <{root_tag}>"
        )

    def parse_file(self, path: str | Path) -> IMLDocument:
        """Parse an IML XML file from disk."""
        p = Path(path)
        return self.parse(p.read_text(encoding="utf-8"))

    def to_plain_text(self, doc: IMLDocument) -> str:
        """Extract plain text from an IML document, stripping all markup."""
        parts: list[str] = []
        for utt in doc.utterances:
            parts.append(_children_to_plain_text(utt.children))
        return "".join(parts)

    def to_iml_string(self, doc: IMLDocument) -> str:
        """Serialize an IML document back to an XML string."""
        if len(doc.utterances) == 1 and doc.version is None and doc.language is None:
            return _serialize_utterance(doc.utterances[0])

        attrs = _attr("version", doc.version) + _attr("language", doc.language)
        if doc.consent is not None:
            attrs += _attr("consent", doc.consent)
        if doc.processing is not None:
            attrs += _attr("processing", doc.processing)
        inner = "".join(_serialize_utterance(u) for u in doc.utterances)
        return f"<iml{attrs}>{inner}</iml>"
