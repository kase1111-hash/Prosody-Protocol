"""Tests for prosody_protocol.iml_to_ssml -- interface stubs."""

from __future__ import annotations

import pytest

from prosody_protocol.iml_to_ssml import IMLToSSML


class TestIMLToSSMLInterface:
    def test_instantiate(self) -> None:
        converter = IMLToSSML()
        assert converter.vendor is None

    def test_convert_not_implemented(self) -> None:
        converter = IMLToSSML()
        with pytest.raises(NotImplementedError):
            converter.convert("<utterance>hello</utterance>")
