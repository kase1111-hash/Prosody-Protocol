"""Tests for prosody_protocol.prosody_analyzer -- interface stubs."""

from __future__ import annotations

import pytest

from prosody_protocol.prosody_analyzer import (
    ProsodyAnalyzer,
    SpanFeatures,
    WordAlignment,
)


class TestDataModels:
    def test_word_alignment(self) -> None:
        wa = WordAlignment(word="hello", start_ms=0, end_ms=500)
        assert wa.word == "hello"
        assert wa.start_ms == 0

    def test_span_features_defaults(self) -> None:
        sf = SpanFeatures(start_ms=0, end_ms=500, text="hello")
        assert sf.f0_mean is None
        assert sf.quality is None


class TestProsodyAnalyzerInterface:
    def test_instantiate(self) -> None:
        analyzer = ProsodyAnalyzer()
        assert analyzer is not None

    def test_analyze_not_implemented(self) -> None:
        analyzer = ProsodyAnalyzer()
        with pytest.raises(NotImplementedError):
            analyzer.analyze("nonexistent.wav", [])
