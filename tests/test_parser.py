"""Tests for prosody_protocol.parser.

These tests are stubs -- they verify the interface exists and will be
fleshed out when the parser is implemented in Phase 3.
"""

from __future__ import annotations

import pytest

from prosody_protocol.parser import IMLParser


class TestIMLParserInterface:
    def test_instantiate(self) -> None:
        parser = IMLParser()
        assert parser is not None

    def test_parse_not_implemented(self, simple_utterance: str) -> None:
        parser = IMLParser()
        with pytest.raises(NotImplementedError):
            parser.parse(simple_utterance)

    def test_to_plain_text_not_implemented(self) -> None:
        from prosody_protocol.models import IMLDocument

        parser = IMLParser()
        with pytest.raises(NotImplementedError):
            parser.to_plain_text(IMLDocument())

    def test_to_iml_string_not_implemented(self) -> None:
        from prosody_protocol.models import IMLDocument

        parser = IMLParser()
        with pytest.raises(NotImplementedError):
            parser.to_iml_string(IMLDocument())
