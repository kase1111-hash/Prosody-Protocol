"""IMLToSSML -- convert IML markup to SSML for TTS engines.

Maps IML tags to their SSML equivalents:

  IML              SSML
  ─────────────    ──────────────
  <iml>            <speak>
  <utterance>      <s>
  <prosody>        <prosody>      (core attrs only; extended attrs dropped)
  <pause>          <break>
  <emphasis>       <emphasis>
  <segment>        (unwrapped -- children promoted to parent)

Attributes without SSML counterparts (f0_mean, jitter, shimmer, etc.)
are silently dropped.

Spec reference: Section 1.5 (relationship to SSML), Appendix A.
"""

from __future__ import annotations

import re

from .exceptions import ConversionError
from .models import (
    ChildNode,
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)
from .parser import IMLParser


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _ssml_attr(name: str, value: str | None) -> str:
    if value is None:
        return ""
    return f' {name}="{_escape_xml(value)}"'


# ---------------------------------------------------------------------------
# Prosody attribute mapping
# ---------------------------------------------------------------------------

# IML prosody attrs that have direct SSML equivalents.
# IML name → SSML name
_PROSODY_ATTR_MAP: dict[str, str] = {
    "pitch": "pitch",
    "volume": "volume",
    "rate": "rate",
}


_DB_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)dB$", re.IGNORECASE)


def _iml_volume_to_ssml(volume: str) -> str:
    """Convert IML volume (relative dB) to an SSML-compatible volume string.

    SSML prosody volume accepts ``+XdB`` / ``-XdB`` per the W3C spec, so
    in most cases the IML value passes through directly.  However, some
    engines only support named values, so map extreme dB values to names
    as a fallback.
    """
    m = _DB_RE.match(volume)
    if m is None:
        return volume  # unrecognised format -- pass through unchanged
    db = float(m.group(1))
    # The W3C SSML spec defines these named levels:
    #   silent, x-soft, soft, medium, loud, x-loud
    # Map extreme values so engines without dB support still behave sensibly.
    if db <= -20:
        return "x-soft"
    if db >= 20:
        return "x-loud"
    # Standard +/-NdB is valid SSML; normalise the suffix to "dB".
    sign = "+" if db >= 0 else ""
    return f"{sign}{db:g}dB"


def _map_prosody_attrs(p: Prosody) -> str:
    """Build SSML attribute string from IML Prosody core attributes."""
    parts: list[str] = []

    if p.pitch is not None:
        parts.append(_ssml_attr("pitch", p.pitch))
    if p.volume is not None:
        parts.append(_ssml_attr("volume", _iml_volume_to_ssml(p.volume)))
    if p.rate is not None:
        parts.append(_ssml_attr("rate", p.rate))

    return "".join(parts)


# ---------------------------------------------------------------------------
# Node serializers
# ---------------------------------------------------------------------------


def _convert_children(children: tuple[ChildNode, ...]) -> str:
    """Recursively convert mixed-content children to SSML."""
    parts: list[str] = []
    for child in children:
        if isinstance(child, str):
            parts.append(_escape_xml(child))
        elif isinstance(child, Pause):
            parts.append(f'<break time="{child.duration}ms"/>')
        elif isinstance(child, Prosody):
            attrs = _map_prosody_attrs(child)
            inner = _convert_children(child.children)
            if attrs:
                parts.append(f"<prosody{attrs}>{inner}</prosody>")
            else:
                # No mappable attrs -- emit content without wrapper.
                parts.append(inner)
        elif isinstance(child, Emphasis):
            level_attr = _ssml_attr("level", child.level) if child.level else ""
            inner = _convert_children(child.children)
            parts.append(f"<emphasis{level_attr}>{inner}</emphasis>")
        elif isinstance(child, Segment):
            # No SSML equivalent -- unwrap children.
            parts.append(_convert_children(child.children))
        elif isinstance(child, Utterance):
            # Nested utterance (unusual) -- treat as <s>.
            inner = _convert_children(child.children)
            parts.append(f"<s>{inner}</s>")
    return "".join(parts)


def _convert_utterance(utt: Utterance) -> str:
    """Convert a single Utterance to an SSML <s> element."""
    inner = _convert_children(utt.children)
    return f"<s>{inner}</s>"


def _convert_document(doc: IMLDocument) -> str:
    """Convert an IMLDocument to a full SSML document."""
    lang_attr = ""
    if doc.language:
        lang_attr = f' xml:lang="{_escape_xml(doc.language)}"'

    body = "".join(_convert_utterance(u) for u in doc.utterances)
    return f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"{lang_attr}>{body}</speak>'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class IMLToSSML:
    """Convert IML documents to SSML.

    Parameters
    ----------
    vendor:
        Optional TTS vendor hint.  Currently unused but reserved for
        vendor-specific SSML quirks (e.g. Google, Amazon, Microsoft).
    """

    def __init__(self, vendor: str | None = None) -> None:
        self.vendor = vendor
        self._parser = IMLParser()
        if vendor is not None:
            import warnings

            warnings.warn(
                f"Vendor-specific SSML adaptation for {vendor!r} is not yet "
                f"implemented. Output will use standard SSML 1.0 without "
                f"vendor extensions.",
                stacklevel=2,
            )

    def convert(self, iml_string: str) -> str:
        """Convert an IML XML string to an SSML XML string.

        Raises :class:`~prosody_protocol.exceptions.ConversionError`
        if the IML cannot be parsed.
        """
        try:
            doc = self._parser.parse(iml_string)
        except Exception as exc:
            raise ConversionError(f"Cannot parse IML for SSML conversion: {exc}") from exc
        return _convert_document(doc)

    def convert_doc(self, doc: IMLDocument) -> str:
        """Convert an :class:`IMLDocument` to an SSML XML string."""
        return _convert_document(doc)
