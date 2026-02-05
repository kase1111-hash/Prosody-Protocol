"""IMLToSSML -- convert IML markup to SSML for TTS engines.

Maps IML tags to their SSML equivalents. Attributes without SSML
counterparts are silently dropped.
Spec reference: Appendix A (tag mapping).
"""

from __future__ import annotations

from .models import IMLDocument


class IMLToSSML:
    """Convert IML documents to SSML."""

    def __init__(self, vendor: str | None = None) -> None:
        self.vendor = vendor

    def convert(self, iml_string: str) -> str:
        """Convert an IML XML string to an SSML XML string."""
        raise NotImplementedError

    def convert_doc(self, doc: IMLDocument) -> str:
        """Convert an :class:`IMLDocument` to an SSML XML string."""
        raise NotImplementedError
