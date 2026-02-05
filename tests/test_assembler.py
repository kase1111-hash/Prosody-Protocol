"""Tests for prosody_protocol.assembler.

Tests utterance grouping, prosody wrapping, emphasis detection,
and pause insertion.
"""

from __future__ import annotations

import pytest

from prosody_protocol.assembler import IMLAssembler
from prosody_protocol.models import (
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Utterance,
)
from prosody_protocol.prosody_analyzer import PauseInterval, SpanFeatures, WordAlignment
from prosody_protocol.validator import IMLValidator


@pytest.fixture()
def assembler() -> IMLAssembler:
    return IMLAssembler(include_extended=False)


@pytest.fixture()
def assembler_extended() -> IMLAssembler:
    return IMLAssembler(include_extended=True)


@pytest.fixture()
def validator() -> IMLValidator:
    return IMLValidator()


def _make_alignment(word: str, start_ms: int, end_ms: int) -> WordAlignment:
    return WordAlignment(word=word, start_ms=start_ms, end_ms=end_ms)


def _make_features(
    word: str,
    start_ms: int,
    end_ms: int,
    f0_mean: float = 180.0,
    intensity_mean: float = 65.0,
    **kwargs: object,
) -> SpanFeatures:
    return SpanFeatures(
        start_ms=start_ms,
        end_ms=end_ms,
        text=word,
        f0_mean=f0_mean,
        intensity_mean=intensity_mean,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Basic assembly
# ---------------------------------------------------------------------------


class TestBasicAssembly:
    def test_single_word(self, assembler: IMLAssembler) -> None:
        alignments = [_make_alignment("Hello.", 0, 500)]
        features = [_make_features("Hello.", 0, 500)]
        doc = assembler.assemble(alignments, features, pauses=[])
        assert isinstance(doc, IMLDocument)
        assert len(doc.utterances) >= 1

    def test_utterance_has_emotion(self, assembler: IMLAssembler) -> None:
        alignments = [_make_alignment("Hi.", 0, 500)]
        features = [_make_features("Hi.", 0, 500)]
        doc = assembler.assemble(alignments, features, pauses=[])
        utt = doc.utterances[0]
        assert utt.emotion is not None
        assert utt.confidence is not None

    def test_version_set(self, assembler: IMLAssembler) -> None:
        alignments = [_make_alignment("Hi.", 0, 500)]
        features = [_make_features("Hi.", 0, 500)]
        doc = assembler.assemble(alignments, features, pauses=[])
        assert doc.version == "0.1.0"

    def test_language_passed_through(self, assembler: IMLAssembler) -> None:
        alignments = [_make_alignment("Hi.", 0, 500)]
        features = [_make_features("Hi.", 0, 500)]
        doc = assembler.assemble(alignments, features, pauses=[], language="en-US")
        assert doc.language == "en-US"


# ---------------------------------------------------------------------------
# Utterance grouping
# ---------------------------------------------------------------------------


class TestUtteranceGrouping:
    def test_sentence_boundary_splits_utterances(
        self, assembler: IMLAssembler
    ) -> None:
        """Words ending with '.' should trigger a new utterance."""
        alignments = [
            _make_alignment("Hello.", 0, 500),
            _make_alignment("World.", 600, 1100),
        ]
        features = [
            _make_features("Hello.", 0, 500),
            _make_features("World.", 600, 1100),
        ]
        doc = assembler.assemble(alignments, features, pauses=[])
        assert len(doc.utterances) == 2

    def test_long_pause_splits_utterances(self, assembler: IMLAssembler) -> None:
        """A pause >= 1000ms should split utterances."""
        alignments = [
            _make_alignment("Before", 0, 400),
            _make_alignment("after", 1500, 1900),
        ]
        features = [
            _make_features("Before", 0, 400),
            _make_features("after", 1500, 1900),
        ]
        pauses = [PauseInterval(start_ms=400, end_ms=1500)]
        doc = assembler.assemble(alignments, features, pauses=pauses)
        # The 1100ms gap should split into two utterances.
        assert len(doc.utterances) >= 2


# ---------------------------------------------------------------------------
# Prosody wrapping
# ---------------------------------------------------------------------------


class TestProsodyWrapping:
    def test_high_f0_gets_prosody_tag(self, assembler: IMLAssembler) -> None:
        """A word with F0 significantly above baseline should get a <prosody> tag."""
        # Baseline will be ~180 Hz. Make one word at 250 Hz.
        alignments = [
            _make_alignment("normal", 0, 500),
            _make_alignment("high.", 600, 1100),
        ]
        features = [
            _make_features("normal", 0, 500, f0_mean=180),
            _make_features("high.", 600, 1100, f0_mean=250),
        ]
        doc = assembler.assemble(alignments, features, pauses=[])
        # Find prosody nodes across all utterances.
        prosody_nodes = []
        for utt in doc.utterances:
            for child in utt.children:
                if isinstance(child, Prosody):
                    prosody_nodes.append(child)
                elif isinstance(child, Emphasis):
                    for inner in child.children:
                        if isinstance(inner, Prosody):
                            prosody_nodes.append(inner)
        assert len(prosody_nodes) >= 1

    def test_normal_features_no_prosody_tag(self, assembler: IMLAssembler) -> None:
        """Words near baseline should not get prosody wrapping."""
        alignments = [_make_alignment("normal.", 0, 500)]
        features = [_make_features("normal.", 0, 500, f0_mean=180, intensity_mean=65)]
        doc = assembler.assemble(alignments, features, pauses=[])
        prosody_nodes = [
            c for c in doc.utterances[0].children if isinstance(c, Prosody)
        ]
        assert len(prosody_nodes) == 0


# ---------------------------------------------------------------------------
# Pause insertion
# ---------------------------------------------------------------------------


class TestPauseInsertion:
    def test_gap_produces_pause_element(self, assembler: IMLAssembler) -> None:
        """A 500ms gap between words should produce a <pause> element."""
        alignments = [
            _make_alignment("before", 0, 300),
            _make_alignment("after.", 800, 1100),
        ]
        features = [
            _make_features("before", 0, 300),
            _make_features("after.", 800, 1100),
        ]
        doc = assembler.assemble(alignments, features, pauses=[])
        pause_nodes = [
            c for c in doc.utterances[0].children if isinstance(c, Pause)
        ]
        assert len(pause_nodes) >= 1
        assert pause_nodes[0].duration == 500


# ---------------------------------------------------------------------------
# Extended attributes
# ---------------------------------------------------------------------------


class TestExtendedAttrs:
    def test_extended_attrs_present_when_enabled(
        self, assembler_extended: IMLAssembler
    ) -> None:
        alignments = [_make_alignment("loud.", 0, 500)]
        features = [
            _make_features(
                "loud.", 0, 500,
                f0_mean=250,
                intensity_mean=80,
                jitter=0.01,
                shimmer=0.05,
                hnr=15.0,
            )
        ]
        doc = assembler_extended.assemble(alignments, features, pauses=[])
        # Find prosody nodes.
        for utt in doc.utterances:
            for child in utt.children:
                if isinstance(child, Prosody):
                    assert child.f0_mean is not None
                elif isinstance(child, Emphasis):
                    for inner in child.children:
                        if isinstance(inner, Prosody):
                            assert inner.f0_mean is not None


# ---------------------------------------------------------------------------
# Validation of output
# ---------------------------------------------------------------------------


class TestOutputValidation:
    def test_assembled_doc_validates(
        self, assembler: IMLAssembler, validator: IMLValidator
    ) -> None:
        """Assembled documents should always pass validation."""
        from prosody_protocol.parser import IMLParser

        parser = IMLParser()
        alignments = [
            _make_alignment("Hello", 0, 300),
            _make_alignment("world.", 400, 700),
        ]
        features = [
            _make_features("Hello", 0, 300),
            _make_features("world.", 400, 700),
        ]
        doc = assembler.assemble(alignments, features, pauses=[])
        xml = parser.to_iml_string(doc)
        result = validator.validate(xml)
        errors = [i for i in result.issues if i.severity == "error"]
        assert result.valid, f"Errors: {[i.message for i in errors]}"
