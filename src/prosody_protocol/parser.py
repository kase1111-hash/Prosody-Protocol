"""IML parser -- converts IML XML strings into ``models.IMLDocument`` objects.

Implementation will use ``lxml.etree`` for XML parsing.
Spec reference: Sections 2-3.
"""

from __future__ import annotations

from pathlib import Path

from .models import IMLDocument


class IMLParser:
    """Parse IML XML into structured :class:`IMLDocument` objects."""

    def parse(self, iml_string: str) -> IMLDocument:
        """Parse an IML XML string.

        Raises :class:`~prosody_protocol.exceptions.IMLParseError` on malformed XML.
        """
        raise NotImplementedError

    def parse_file(self, path: str | Path) -> IMLDocument:
        """Parse an IML XML file from disk."""
        raise NotImplementedError

    def to_plain_text(self, doc: IMLDocument) -> str:
        """Extract plain text from an IML document, stripping all markup."""
        raise NotImplementedError

    def to_iml_string(self, doc: IMLDocument) -> str:
        """Serialize an IML document back to an XML string."""
        raise NotImplementedError
