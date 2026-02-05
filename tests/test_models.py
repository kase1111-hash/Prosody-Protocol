"""Tests for prosody_protocol.models."""

from __future__ import annotations

from prosody_protocol.models import (
    Emphasis,
    IMLDocument,
    Pause,
    Prosody,
    Segment,
    Utterance,
)


class TestPause:
    def test_create(self) -> None:
        p = Pause(duration=800)
        assert p.duration == 800

    def test_immutable(self) -> None:
        p = Pause(duration=800)
        try:
            p.duration = 400  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass


class TestProsody:
    def test_create_minimal(self) -> None:
        pr = Prosody(children=("hello",), pitch="+10%")
        assert pr.pitch == "+10%"
        assert pr.children == ("hello",)

    def test_defaults_none(self) -> None:
        pr = Prosody()
        assert pr.volume is None
        assert pr.f0_mean is None


class TestEmphasis:
    def test_create(self) -> None:
        e = Emphasis(level="strong", children=("important",))
        assert e.level == "strong"


class TestSegment:
    def test_create(self) -> None:
        s = Segment(children=("text",), tempo="rushed", rhythm="staccato")
        assert s.tempo == "rushed"
        assert s.rhythm == "staccato"


class TestUtterance:
    def test_create_with_emotion(self) -> None:
        u = Utterance(
            children=("hello",),
            emotion="frustrated",
            confidence=0.92,
            speaker_id="user_001",
        )
        assert u.emotion == "frustrated"
        assert u.confidence == 0.92

    def test_create_without_emotion(self) -> None:
        u = Utterance(children=("hello",))
        assert u.emotion is None
        assert u.confidence is None


class TestIMLDocument:
    def test_create(self) -> None:
        u = Utterance(children=("hi",))
        doc = IMLDocument(utterances=(u,), version="0.1.0", language="en-US")
        assert len(doc.utterances) == 1
        assert doc.version == "0.1.0"

    def test_empty_document(self) -> None:
        doc = IMLDocument()
        assert doc.utterances == ()
