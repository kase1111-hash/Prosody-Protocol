"""Tests for prosody_protocol.text_to_iml -- interface stubs."""

from __future__ import annotations

import pytest

from prosody_protocol.text_to_iml import TextToIML


class TestTextToIMLInterface:
    def test_instantiate(self) -> None:
        predictor = TextToIML()
        assert predictor.model == "rule-based"
        assert predictor.default_confidence == 0.6

    def test_predict_not_implemented(self) -> None:
        predictor = TextToIML()
        with pytest.raises(NotImplementedError):
            predictor.predict("Hello world")
