"""Shared test fixtures for the prosody_protocol test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VALID_DIR = FIXTURES_DIR / "valid"
INVALID_DIR = FIXTURES_DIR / "invalid"
AUDIO_DIR = FIXTURES_DIR / "audio"


# ---------------------------------------------------------------------------
# Sample IML strings
# ---------------------------------------------------------------------------

SIMPLE_UTTERANCE = (
    '<utterance emotion="frustrated" confidence="0.92">'
    "I've been waiting for hours!"
    "</utterance>"
)

SARCASM_EXAMPLE = (
    '<utterance emotion="sarcastic" confidence="0.87">'
    "  Oh, that's"
    '  <prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">'
    "    GREAT"
    "  </prosody>."
    "</utterance>"
)

MULTI_SPEAKER = (
    '<iml version="0.1.0" language="en-US">'
    '  <utterance emotion="neutral" confidence="0.95" speaker_id="agent">'
    "    How can I help you today?"
    "  </utterance>"
    '  <utterance emotion="frustrated" confidence="0.89" speaker_id="caller">'
    "    I've been on hold for"
    '    <prosody pitch="+8%" volume="+5dB">thirty minutes</prosody>'
    "    and my"
    '    <emphasis level="strong">account is still locked</emphasis>.'
    "  </utterance>"
    "</iml>"
)

SEGMENT_EXAMPLE = (
    '<iml version="0.1.0" language="en-US">'
    '  <utterance emotion="excitement" confidence="0.78" speaker_id="user_789">'
    '    <segment tempo="rushed" rhythm="legato">'
    "      I just got the new keyboard and it works perfectly with the setup."
    "    </segment>"
    "  </utterance>"
    "</iml>"
)

MISSING_CONFIDENCE = (
    '<utterance emotion="angry">'
    "This should fail validation."
    "</utterance>"
)

INVALID_SEGMENT_NESTING = (
    '<utterance emotion="calm" confidence="0.9">'
    '  <prosody pitch="+5%">'
    '    <segment tempo="rushed">Invalid nesting</segment>'
    "  </prosody>"
    "</utterance>"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_utterance() -> str:
    return SIMPLE_UTTERANCE


@pytest.fixture()
def sarcasm_example() -> str:
    return SARCASM_EXAMPLE


@pytest.fixture()
def multi_speaker() -> str:
    return MULTI_SPEAKER


@pytest.fixture()
def segment_example() -> str:
    return SEGMENT_EXAMPLE


@pytest.fixture()
def missing_confidence() -> str:
    return MISSING_CONFIDENCE


@pytest.fixture()
def invalid_segment_nesting() -> str:
    return INVALID_SEGMENT_NESTING


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def valid_fixtures_dir() -> Path:
    return VALID_DIR


@pytest.fixture()
def invalid_fixtures_dir() -> Path:
    return INVALID_DIR
