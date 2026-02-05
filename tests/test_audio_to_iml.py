"""Tests for prosody_protocol.audio_to_iml -- interface stubs."""

from __future__ import annotations

import pytest

from prosody_protocol.audio_to_iml import AudioToIML


class TestAudioToIMLInterface:
    def test_instantiate(self) -> None:
        converter = AudioToIML()
        assert converter.stt_model == "base"
        assert converter.include_extended is False

    def test_convert_not_implemented(self) -> None:
        converter = AudioToIML()
        with pytest.raises(NotImplementedError):
            converter.convert("nonexistent.wav")
