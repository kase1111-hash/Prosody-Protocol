"""Tests for prosody_protocol.iml_to_audio -- interface stubs."""

from __future__ import annotations

import pytest

from prosody_protocol.iml_to_audio import IMLToAudio


class TestIMLToAudioInterface:
    def test_instantiate(self) -> None:
        synth = IMLToAudio()
        assert synth.voice == "en_US-female-medium"
        assert synth.engine == "coqui"

    def test_synthesize_not_implemented(self) -> None:
        synth = IMLToAudio()
        with pytest.raises(NotImplementedError):
            synth.synthesize("<utterance>hello</utterance>")
