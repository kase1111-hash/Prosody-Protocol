"""Tests for prosody_protocol.profiles -- interface stubs."""

from __future__ import annotations

import pytest

from prosody_protocol.profiles import (
    ProfileApplier,
    ProfileLoader,
    ProsodyMapping,
    ProsodyProfile,
)


class TestProfileModels:
    def test_prosody_mapping(self) -> None:
        m = ProsodyMapping(
            pattern={"pitch_contour": "flat", "rate": "fast"},
            interpretation_emotion="excitement",
            confidence_boost=0.15,
        )
        assert m.interpretation_emotion == "excitement"

    def test_prosody_profile(self) -> None:
        p = ProsodyProfile(
            profile_version="0.1.0",
            user_id="user_789",
            description="Test profile",
            mappings=[],
        )
        assert p.user_id == "user_789"


class TestProfileLoaderInterface:
    def test_instantiate(self) -> None:
        loader = ProfileLoader()
        assert loader is not None

    def test_load_not_implemented(self) -> None:
        loader = ProfileLoader()
        with pytest.raises(NotImplementedError):
            loader.load("nonexistent.json")


class TestProfileApplierInterface:
    def test_instantiate(self) -> None:
        applier = ProfileApplier()
        assert applier is not None

    def test_apply_not_implemented(self) -> None:
        profile = ProsodyProfile("0.1.0", "u1", None, [])
        applier = ProfileApplier()
        with pytest.raises(NotImplementedError):
            applier.apply(profile, {}, "neutral", 0.5)
