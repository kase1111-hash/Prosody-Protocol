"""Tests for prosody_protocol.profiles (Phase 8 -- Prosody Profiles).

Covers:
- Acceptance criteria from the execution guide
- ProfileLoader: load from file, load from dict, error handling
- ProfileLoader.validate: version, user_id, mappings, pattern keys/values
- ProfileApplier: pattern matching (match), specificity, confidence capping
- categorize_features: measured features to pattern vocabulary
- Data model frozen semantics
- Edge cases: empty features, no match, multiple matches
"""

from __future__ import annotations

import ast
import inspect
import math
import shutil
import statistics
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from prosody_protocol._types import PauseInterval, SpanFeatures
from prosody_protocol.exceptions import ProfileError
from prosody_protocol.profiles import (
    RATE_FAST_SYLLABLES_PER_S,
    RATE_SLOW_SYLLABLES_PER_S,
    ProfileApplier,
    ProfileLoader,
    ProsodyMapping,
    ProsodyProfile,
    categorize_features,
)

PROFILES_DIR = Path(__file__).parent / "fixtures" / "profiles"


@pytest.fixture()
def loader() -> ProfileLoader:
    return ProfileLoader()


@pytest.fixture()
def applier() -> ProfileApplier:
    return ProfileApplier()


@pytest.fixture()
def autism_profile(loader: ProfileLoader) -> ProsodyProfile:
    return loader.load(PROFILES_DIR / "autism_spectrum.json")


@pytest.fixture()
def spec_profile_json() -> dict[str, Any]:
    """The example profile from spec Section 7.1."""
    return {
        "profile_version": "0.1.0",
        "user_id": "user_789",
        "description": "Autism spectrum - monotone speech with rate-based expression",
        "prosody_mappings": [
            {
                "pattern": {"pitch_contour": "flat", "rate": "fast"},
                "interpretation": {"emotion": "excitement", "confidence_boost": 0.15},
            },
            {
                "pattern": {"pitch_contour": "flat", "pause_frequency": "high"},
                "interpretation": {"emotion": "thinking_carefully", "confidence_boost": 0.10},
            },
            {
                "pattern": {"volume": "spike"},
                "interpretation": {"emotion": "emphasis_not_anger", "confidence_boost": 0.20},
            },
        ],
    }


# ---------------------------------------------------------------------------
# Acceptance criteria (from EXECUTION_GUIDE.md Phase 8.3)
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    def test_spec_profile_loads_and_validates(
        self, loader: ProfileLoader, spec_profile_json: dict[str, object]
    ) -> None:
        """Profile JSON from spec Section 7.1 loads and validates successfully."""
        profile = loader.load_json(spec_profile_json)
        result = loader.validate(profile)
        assert result.valid, f"Validation failed: {result.issues}"

    def test_autism_profile_flat_pitch_fast_rate_produces_excitement(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        """Applying the autism-spectrum profile to flat-pitch + fast-rate
        features produces emotion='excitement'."""
        features = {"pitch_contour": "flat", "rate": "fast"}
        emotion, confidence = applier.apply(
            autism_profile, features, base_emotion="neutral", base_confidence=0.5
        )
        assert emotion == "excitement"

    def test_invalid_profiles_fail_with_clear_errors(self, loader: ProfileLoader) -> None:
        """Invalid profiles (missing fields, bad version) fail validation
        with clear errors."""
        # Missing user_id.
        with pytest.raises(ProfileError, match="user_id"):
            loader.load(PROFILES_DIR / "invalid_missing_user.json")

    def test_confidence_boost_never_exceeds_one(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        """confidence_boost never produces confidence > 1.0."""
        features = {"volume": "spike"}
        emotion, confidence = applier.apply(
            autism_profile, features, base_emotion="angry", base_confidence=0.95
        )
        assert confidence <= 1.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_prosody_mapping_creation(self) -> None:
        m = ProsodyMapping(
            pattern={"pitch_contour": "flat", "rate": "fast"},
            interpretation_emotion="excitement",
            confidence_boost=0.15,
        )
        assert m.pattern == {"pitch_contour": "flat", "rate": "fast"}
        assert m.interpretation_emotion == "excitement"
        assert m.confidence_boost == 0.15

    def test_prosody_mapping_default_boost(self) -> None:
        m = ProsodyMapping(
            pattern={"pitch": "high"},
            interpretation_emotion="excited",
        )
        assert m.confidence_boost == 0.0

    def test_prosody_profile_creation(self) -> None:
        p = ProsodyProfile(
            profile_version="0.1.0",
            user_id="user_789",
            description="Test profile",
            mappings=(),
        )
        assert p.profile_version == "0.1.0"
        assert p.user_id == "user_789"
        assert p.description == "Test profile"
        assert p.mappings == ()

    def test_prosody_profile_none_description(self) -> None:
        p = ProsodyProfile("1.0.0", "u1", None, ())
        assert p.description is None


# ---------------------------------------------------------------------------
# ProfileLoader.load (file)
# ---------------------------------------------------------------------------


class TestProfileLoaderFile:
    def test_load_autism_spectrum(self, loader: ProfileLoader) -> None:
        profile = loader.load(PROFILES_DIR / "autism_spectrum.json")
        assert profile.profile_version == "0.1.0"
        assert profile.user_id == "user_789"
        assert len(profile.mappings) == 3

    def test_load_minimal_valid(self, loader: ProfileLoader) -> None:
        profile = loader.load(PROFILES_DIR / "minimal_valid.json")
        assert profile.profile_version == "1.0.0"
        assert profile.user_id == "user_001"
        assert len(profile.mappings) == 1
        assert profile.mappings[0].interpretation_emotion == "excited"

    def test_load_nonexistent_file_raises(self, loader: ProfileLoader) -> None:
        with pytest.raises(ProfileError, match="Cannot read"):
            loader.load("/nonexistent/path.json")

    def test_load_invalid_json_raises(self, loader: ProfileLoader, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(ProfileError, match="Invalid JSON"):
            loader.load(bad_file)

    def test_load_missing_user_raises(self, loader: ProfileLoader) -> None:
        with pytest.raises(ProfileError, match="user_id"):
            loader.load(PROFILES_DIR / "invalid_missing_user.json")

    @pytest.mark.parametrize(
        "text",
        ["[" * 50_000, '{"a": ' * 50_000, "1" * 5_000],
        ids=["deep array", "deep object", "integer too long to convert"],
    )
    def test_json_python_cannot_parse_raises_profile_error(
        self, loader: ProfileLoader, tmp_path: Path, text: str
    ) -> None:
        """RecursionError and the int-conversion ValueError used to escape."""
        path = tmp_path / "hostile.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ProfileError, match="Invalid JSON in profile file"):
            loader.load(path)

    def test_non_utf8_file_raises_profile_error(
        self, loader: ProfileLoader, tmp_path: Path
    ) -> None:
        path = tmp_path / "latin1.json"
        path.write_bytes('{"user_id": "Jos\u00e9"}'.encode("latin-1"))
        with pytest.raises(ProfileError, match="not UTF-8"):
            loader.load(path)

    def test_path_with_nul_raises_profile_error(self, loader: ProfileLoader) -> None:
        """Path.read_text raises a bare ValueError for an embedded NUL."""
        with pytest.raises(ProfileError, match="Cannot read profile file"):
            loader.load("profile\x00.json")

    def test_byte_order_mark_is_allowed(self, loader: ProfileLoader, tmp_path: Path) -> None:
        path = tmp_path / "bom.json"
        path.write_bytes(b"\xef\xbb\xbf" + (PROFILES_DIR / "minimal_valid.json").read_bytes())
        assert loader.load(path).user_id == "user_001"


# ---------------------------------------------------------------------------
# ProfileLoader.load_json (dict)
# ---------------------------------------------------------------------------


class TestProfileLoaderJSON:
    def test_load_spec_example(
        self, loader: ProfileLoader, spec_profile_json: dict[str, object]
    ) -> None:
        profile = loader.load_json(spec_profile_json)
        assert profile.user_id == "user_789"
        assert profile.description is not None
        assert "Autism" in profile.description
        assert len(profile.mappings) == 3
        assert profile.mappings[0].interpretation_emotion == "excitement"
        assert profile.mappings[0].confidence_boost == 0.15

    def test_missing_profile_version(self, loader: ProfileLoader) -> None:
        with pytest.raises(ProfileError, match="profile_version"):
            loader.load_json({"user_id": "u1", "prosody_mappings": []})

    def test_missing_user_id(self, loader: ProfileLoader) -> None:
        with pytest.raises(ProfileError, match="user_id"):
            loader.load_json({"profile_version": "1.0.0", "prosody_mappings": []})

    def test_missing_prosody_mappings(self, loader: ProfileLoader) -> None:
        with pytest.raises(ProfileError, match="prosody_mappings"):
            loader.load_json({"profile_version": "1.0.0", "user_id": "u1"})

    def test_non_dict_raises(self, loader: ProfileLoader) -> None:
        with pytest.raises(ProfileError, match="JSON object"):
            loader.load_json("not a dict")  # type: ignore[arg-type]

    def test_mapping_missing_pattern(self, loader: ProfileLoader) -> None:
        with pytest.raises(ProfileError, match="pattern"):
            loader.load_json({
                "profile_version": "1.0.0",
                "user_id": "u1",
                "prosody_mappings": [{"interpretation": {"emotion": "happy"}}],
            })

    def test_mapping_missing_interpretation(self, loader: ProfileLoader) -> None:
        with pytest.raises(ProfileError, match="interpretation"):
            loader.load_json({
                "profile_version": "1.0.0",
                "user_id": "u1",
                "prosody_mappings": [{"pattern": {"pitch": "high"}}],
            })

    def test_mapping_missing_emotion(self, loader: ProfileLoader) -> None:
        with pytest.raises(ProfileError, match="emotion"):
            loader.load_json({
                "profile_version": "1.0.0",
                "user_id": "u1",
                "prosody_mappings": [{
                    "pattern": {"pitch": "high"},
                    "interpretation": {"confidence_boost": 0.1},
                }],
            })

    def test_no_description_allowed(self, loader: ProfileLoader) -> None:
        profile = loader.load_json({
            "profile_version": "1.0.0",
            "user_id": "u1",
            "prosody_mappings": [{
                "pattern": {"pitch": "high"},
                "interpretation": {"emotion": "excited"},
            }],
        })
        assert profile.description is None

    def test_default_confidence_boost_zero(self, loader: ProfileLoader) -> None:
        profile = loader.load_json({
            "profile_version": "1.0.0",
            "user_id": "u1",
            "prosody_mappings": [{
                "pattern": {"pitch": "high"},
                "interpretation": {"emotion": "excited"},
            }],
        })
        assert profile.mappings[0].confidence_boost == 0.0

    @pytest.mark.parametrize("boost", [True, float("nan"), float("inf"), "0.1"])
    def test_non_numeric_confidence_boost_rejected(
        self, loader: ProfileLoader, boost: object
    ) -> None:
        """A bool or NaN boost used to pin every matched confidence to 1.0."""
        with pytest.raises(ProfileError, match="confidence_boost must be a finite number"):
            loader.load_json({
                "profile_version": "1.0.0",
                "user_id": "u1",
                "prosody_mappings": [{
                    "pattern": {"volume": "spike"},
                    "interpretation": {"emotion": "x", "confidence_boost": boost},
                }],
            })

    def test_nan_in_file_rejected(self, loader: ProfileLoader, tmp_path: Path) -> None:
        path = tmp_path / "nan.json"
        path.write_text(
            '{"profile_version": "1.0.0", "user_id": "u1", "prosody_mappings": [{"pattern": '
            '{"volume": "spike"}, "interpretation": {"emotion": "x", "confidence_boost": NaN}}]}',
            encoding="utf-8",
        )
        with pytest.raises(ProfileError, match="NaN is not a JSON number"):
            loader.load(path)

    @pytest.mark.parametrize(
        ("where", "match"),
        [("profile", r"profile has unknown key\(s\) \['evil'\]"),
         ("mapping", r"prosody_mappings\[0\] has unknown key\(s\) \['weight'\]"),
         ("interpretation", r"interpretation has unknown key\(s\) \['note'\]")],
    )
    def test_unknown_keys_rejected(
        self, loader: ProfileLoader, spec_profile_json: dict[str, Any], where: str, match: str
    ) -> None:
        if where == "profile":
            spec_profile_json["evil"] = 1
        elif where == "mapping":
            spec_profile_json["prosody_mappings"][0]["weight"] = 2
        else:
            spec_profile_json["prosody_mappings"][0]["interpretation"]["note"] = "x"
        with pytest.raises(ProfileError, match=match):
            loader.load_json(spec_profile_json)

    def test_metadata_accepted_when_an_object(
        self, loader: ProfileLoader, spec_profile_json: dict[str, Any]
    ) -> None:
        spec_profile_json["metadata"] = {"author": "clinic", "created_at": "2025-01-01T00:00:00Z"}
        assert loader.validate(loader.load_json(spec_profile_json)).valid
        spec_profile_json["metadata"] = "clinic"
        with pytest.raises(ProfileError, match="'metadata' must be an object"):
            loader.load_json(spec_profile_json)

    @pytest.mark.parametrize("digits", [400, 5_000])
    def test_integer_too_large_for_a_float_rejected(
        self, loader: ProfileLoader, digits: int
    ) -> None:
        """OverflowError used to escape (and a 5000-digit int cannot even be printed)."""
        with pytest.raises(ProfileError, match="confidence_boost must be a finite number"):
            loader.load_json({
                "profile_version": "1.0.0",
                "user_id": "u1",
                "prosody_mappings": [{
                    "pattern": {"pitch": "high"},
                    "interpretation": {"emotion": "excited", "confidence_boost": 10**digits},
                }],
            })

    def test_integer_too_large_in_a_file_rejected(
        self, loader: ProfileLoader, tmp_path: Path
    ) -> None:
        path = tmp_path / "huge.json"
        path.write_text(
            '{"profile_version": "1.0.0", "user_id": "u1", "prosody_mappings": [{"pattern": '
            '{"pitch": "high"}, "interpretation": {"emotion": "x", "confidence_boost": 1'
            + "0" * 400 + "}}]}",
            encoding="utf-8",
        )
        with pytest.raises(ProfileError, match=r"finite number, got 1000.*\.\.\.$"):
            loader.load(path)

    def test_integer_confidence_boost_accepted(self, loader: ProfileLoader) -> None:
        profile = loader.load_json({
            "profile_version": "1.0.0",
            "user_id": "u1",
            "prosody_mappings": [{
                "pattern": {"pitch": "high"},
                "interpretation": {"emotion": "excited", "confidence_boost": 1},
            }],
        })
        assert profile.mappings[0].confidence_boost == 1.0


# ---------------------------------------------------------------------------
# ProfileLoader.validate
# ---------------------------------------------------------------------------


class TestProfileValidation:
    def test_valid_spec_profile(
        self, loader: ProfileLoader, spec_profile_json: dict[str, object]
    ) -> None:
        profile = loader.load_json(spec_profile_json)
        result = loader.validate(profile)
        assert result.valid
        assert len(result.issues) == 0

    def test_bad_version_format(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            profile_version="not-a-version",
            user_id="u1",
            description=None,
            mappings=[ProsodyMapping({"pitch": "high"}, "excited", 0.1)],
        )
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P1" for i in result.issues)

    def test_valid_semver_with_prerelease(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            profile_version="1.0.0-alpha",
            user_id="u1",
            description=None,
            mappings=[ProsodyMapping({"pitch": "high"}, "excited", 0.1)],
        )
        result = loader.validate(profile)
        assert result.valid

    def test_empty_mappings_invalid(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile("1.0.0", "u1", None, [])
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P3" for i in result.issues)

    def test_empty_pattern_invalid(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            "1.0.0", "u1", None,
            [ProsodyMapping({}, "excited", 0.1)],
        )
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P4" for i in result.issues)

    def test_unknown_pattern_key_invalid(self, loader: ProfileLoader) -> None:
        """A key categorize_features never produces could never match."""
        profile = ProsodyProfile(
            "1.0.0", "u1", None,
            [ProsodyMapping({"unknown_key": "value"}, "excited", 0.1)],
        )
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P5" and i.severity == "error" for i in result.issues)

    def test_invalid_pattern_value_invalid(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            "1.0.0", "u1", None,
            [ProsodyMapping({"pitch": "very_high"}, "excited", 0.1)],
        )
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P6" and i.severity == "error" for i in result.issues)

    def test_version_with_trailing_newline_invalid(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            "1.0.0\n", "u1", None, [ProsodyMapping({"pitch": "high"}, "excited", 0.1)]
        )
        assert any(i.rule == "P1" for i in loader.validate(profile).errors)

    def test_nan_boost_invalid(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            "1.0.0", "u1", None, [ProsodyMapping({"pitch": "high"}, "excited", float("nan"))]
        )
        assert any(i.rule == "P8" for i in loader.validate(profile).errors)

    def test_empty_emotion_invalid(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            "1.0.0", "u1", None,
            [ProsodyMapping({"pitch": "high"}, "", 0.1)],
        )
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P7" for i in result.issues)

    def test_confidence_boost_out_of_range(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            "1.0.0", "u1", None,
            [ProsodyMapping({"pitch": "high"}, "excited", 1.5)],
        )
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P8" for i in result.issues)

    def test_negative_confidence_boost_invalid(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            "1.0.0", "u1", None,
            [ProsodyMapping({"pitch": "high"}, "excited", -0.1)],
        )
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P8" for i in result.issues)

    def test_all_valid_pattern_keys(self, loader: ProfileLoader) -> None:
        """All recognized pattern keys should validate without warnings."""
        profile = ProsodyProfile(
            "1.0.0", "u1", None,
            [ProsodyMapping(
                {
                    "pitch": "high",
                    "pitch_contour": "rise",
                    "volume": "loud",
                    "rate": "fast",
                    "quality": "breathy",
                    "pause_frequency": "high",
                    "emphasis_frequency": "low",
                },
                "excited",
                0.1,
            )],
        )
        result = loader.validate(profile)
        assert result.valid
        assert len(result.issues) == 0


# ---------------------------------------------------------------------------
# ProfileApplier
# ---------------------------------------------------------------------------


class TestProfileApplierMatching:
    def test_single_match(self, applier: ProfileApplier) -> None:
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high"}, "excited", 0.1),
        ])
        emotion, conf = applier.apply(profile, {"pitch": "high"}, "neutral", 0.5)
        assert emotion == "excited"
        assert conf == pytest.approx(0.6)

    def test_no_match_returns_base(self, applier: ProfileApplier) -> None:
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high"}, "excited", 0.1),
        ])
        emotion, conf = applier.apply(profile, {"pitch": "low"}, "neutral", 0.5)
        assert emotion == "neutral"
        assert conf == 0.5

    def test_partial_pattern_no_match(self, applier: ProfileApplier) -> None:
        """All pattern keys must match -- partial match is not enough."""
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high", "rate": "fast"}, "excited", 0.1),
        ])
        emotion, conf = applier.apply(profile, {"pitch": "high"}, "neutral", 0.5)
        assert emotion == "neutral"

    def test_extra_features_ok(self, applier: ProfileApplier) -> None:
        """Extra features in the input don't prevent a match."""
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high"}, "excited", 0.1),
        ])
        features = {"pitch": "high", "rate": "fast", "volume": "loud"}
        emotion, conf = applier.apply(profile, features, "neutral", 0.5)
        assert emotion == "excited"


class TestProfileApplierSpecificity:
    def test_more_specific_wins(self, applier: ProfileApplier) -> None:
        """Among multiple matches, the most specific (most keys) wins."""
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high"}, "excited", 0.1),
            ProsodyMapping({"pitch": "high", "rate": "fast"}, "very_excited", 0.2),
        ])
        features = {"pitch": "high", "rate": "fast"}
        emotion, conf = applier.apply(profile, features, "neutral", 0.5)
        assert emotion == "very_excited"
        assert conf == pytest.approx(0.7)

    def test_tie_broken_by_order(self, applier: ProfileApplier) -> None:
        """Equal specificity: first match in profile order wins."""
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high"}, "first", 0.1),
            ProsodyMapping({"rate": "fast"}, "second", 0.2),
        ])
        features = {"pitch": "high", "rate": "fast"}
        emotion, _ = applier.apply(profile, features, "neutral", 0.5)
        # Both match with specificity 1 -- first encountered wins.
        assert emotion == "first"


class TestProfileApplierMatch:
    """ProfileApplier.match: the mapping apply() uses, and the assembler too."""

    def test_most_specific_matching_mapping(self, applier: ProfileApplier) -> None:
        general = ProsodyMapping({"pitch_contour": "flat"}, "calm", 0.15)
        specific = ProsodyMapping({"pitch_contour": "flat", "rate": "fast"}, "joyful", 0.3)
        profile = ProsodyProfile("1.0.0", "u1", None, (general, specific))
        assert applier.match(profile, {"pitch_contour": "flat", "rate": "fast"}) is specific
        assert applier.match(profile, {"pitch_contour": "flat", "rate": "slow"}) is general
        assert applier.match(profile, {"rate": "fast"}) is None
        assert applier.match(profile, {}) is None

    def test_first_wins_a_tie(self, applier: ProfileApplier) -> None:
        first = ProsodyMapping({"pitch": "high", "rate": "fast"}, "first", 0.1)
        second = ProsodyMapping({"rate": "fast", "volume": "loud"}, "second", 0.1)
        profile = ProsodyProfile("1.0.0", "u1", None, (first, second))
        features = {"pitch": "high", "rate": "fast", "volume": "loud"}
        assert applier.match(profile, features) is first

    def test_empty_pattern_never_matches(self, applier: ProfileApplier) -> None:
        """An empty pattern (invalid, rule P4) would otherwise match everything."""
        profile = ProsodyProfile("1.0.0", "u1", None, (ProsodyMapping({}, "calm", 0.5),))
        assert applier.match(profile, {"pitch": "high"}) is None
        assert applier.apply(profile, {"pitch": "high"}, "neutral", 0.4) == ("neutral", 0.4)

    def test_any_mapping_of_features(self, applier: ProfileApplier) -> None:
        from types import MappingProxyType

        mapping = ProsodyMapping({"volume": "spike"}, "sincere", 0.2)
        profile = ProsodyProfile("1.0.0", "u1", None, (mapping,))
        assert applier.match(profile, MappingProxyType({"volume": "spike"})) is mapping

    def test_apply_uses_match(self) -> None:
        chosen = ProsodyMapping({"rate": "fast"}, "chosen", 0.25)

        class Custom(ProfileApplier):
            def match(
                self, profile: ProsodyProfile, features: Any
            ) -> ProsodyMapping | None:
                return chosen

        profile = ProsodyProfile("1.0.0", "u1", None, (
            ProsodyMapping({"pitch": "high"}, "excited", 0.1),
        ))
        assert Custom().apply(profile, {"pitch": "high"}, "neutral", 0.5) == ("chosen", 0.75)


class TestProfileApplierConfidence:
    def test_confidence_boost_applied(self, applier: ProfileApplier) -> None:
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high"}, "excited", 0.15),
        ])
        _, conf = applier.apply(profile, {"pitch": "high"}, "neutral", 0.7)
        assert conf == pytest.approx(0.85)

    def test_confidence_capped_at_one(self, applier: ProfileApplier) -> None:
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high"}, "excited", 0.5),
        ])
        _, conf = applier.apply(profile, {"pitch": "high"}, "neutral", 0.8)
        assert conf == 1.0

    def test_zero_boost_preserves_base(self, applier: ProfileApplier) -> None:
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high"}, "excited", 0.0),
        ])
        _, conf = applier.apply(profile, {"pitch": "high"}, "neutral", 0.6)
        assert conf == pytest.approx(0.6)


class TestProfileApplierAutismSpectrum:
    """End-to-end tests with the autism-spectrum profile from the spec."""

    def test_flat_pitch_fast_rate(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        features = {"pitch_contour": "flat", "rate": "fast"}
        emotion, conf = applier.apply(autism_profile, features, "neutral", 0.5)
        assert emotion == "excitement"
        assert conf == pytest.approx(0.65)

    def test_flat_pitch_high_pauses(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        features = {"pitch_contour": "flat", "pause_frequency": "high"}
        emotion, conf = applier.apply(autism_profile, features, "neutral", 0.5)
        assert emotion == "thinking_carefully"
        assert conf == pytest.approx(0.6)

    def test_volume_spike(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        features = {"volume": "spike"}
        emotion, conf = applier.apply(autism_profile, features, "angry", 0.7)
        assert emotion == "emphasis_not_anger"
        assert conf == pytest.approx(0.9)

    def test_flat_pitch_fast_rate_beats_volume_spike(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        """When both flat+fast (2 keys) and volume spike (1 key) match,
        the more specific pattern wins."""
        features = {"pitch_contour": "flat", "rate": "fast", "volume": "spike"}
        emotion, _ = applier.apply(autism_profile, features, "neutral", 0.5)
        assert emotion == "excitement"  # 2-key pattern wins

    def test_unrecognized_features_no_match(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        features = {"pitch_contour": "rise", "rate": "slow"}
        emotion, conf = applier.apply(autism_profile, features, "neutral", 0.5)
        assert emotion == "neutral"
        assert conf == 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_features_dict(self, applier: ProfileApplier) -> None:
        profile = ProsodyProfile("1.0.0", "u1", None, [
            ProsodyMapping({"pitch": "high"}, "excited", 0.1),
        ])
        emotion, conf = applier.apply(profile, {}, "neutral", 0.5)
        assert emotion == "neutral"
        assert conf == 0.5

    def test_empty_mappings_no_match(self, applier: ProfileApplier) -> None:
        profile = ProsodyProfile("1.0.0", "u1", None, [])
        emotion, conf = applier.apply(profile, {"pitch": "high"}, "neutral", 0.5)
        assert emotion == "neutral"

    def test_load_file_validates_successfully(self, loader: ProfileLoader) -> None:
        """Load from file and validate in sequence."""
        profile = loader.load(PROFILES_DIR / "autism_spectrum.json")
        result = loader.validate(profile)
        assert result.valid

    def test_bad_version_file_loads_but_fails_validation(
        self, loader: ProfileLoader
    ) -> None:
        """File with bad version can be loaded but validation catches it."""
        profile = loader.load(PROFILES_DIR / "invalid_bad_version.json")
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P1" for i in result.issues)

    def test_empty_mappings_file_loads_but_fails_validation(
        self, loader: ProfileLoader
    ) -> None:
        profile = loader.load(PROFILES_DIR / "invalid_empty_mappings.json")
        result = loader.validate(profile)
        assert not result.valid
        assert any(i.rule == "P3" for i in result.issues)


# ---------------------------------------------------------------------------
# SpanFeatures, the input of categorize_features
# ---------------------------------------------------------------------------


def test_span_features_quality_doc_matches_the_analyzer() -> None:
    """SpanFeatures.quality documents the labels ProsodyAnalyzer can give
    (read from its source, so this runs without numpy or parselmouth)."""
    import prosody_protocol

    source = Path(prosody_protocol.__file__).with_name("prosody_analyzer.py").read_text(
        encoding="utf-8"
    )
    classify = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_classify_quality"
    )
    labels = {
        constant.value
        for ret in ast.walk(classify) if isinstance(ret, ast.Return) and ret.value is not None
        for constant in ast.walk(ret.value)
        if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
    }
    assert labels == {"creaky", "harsh", "breathy", "modal"}

    doc = inspect.getdoc(SpanFeatures) or ""
    section = " ".join(doc.split("\nquality:\n", 1)[1].split())
    assert all(f"``{label}``" in section for label in labels)
    assert "relative to the speaker's usual voice" in section
    assert "never yields ``tense`` or ``whispery``" in section
    assert "``None`` when it cannot tell" in section


# ---------------------------------------------------------------------------
# categorize_features
# ---------------------------------------------------------------------------


def _utterance(
    semitones: Callable[[float], float] = lambda x: 0.0,
    n: int = 8,
    word_ms: int = 250,
    gap_ms: int = 50,
    **per_span: Any,
) -> list[SpanFeatures]:
    """*n* word spans whose F0 follows ``semitones(x)`` relative to 200 Hz,
    with x running from 0 to 1 over the utterance (4 samples per span).
    Other SpanFeatures fields are given as a value or a per-span list."""
    values: dict[str, Any] = {"intensity_mean": 65.0, "speech_rate": 4.5}
    values.update(per_span)
    total = 4 * n
    spans = []
    for i in range(n):
        start = i * (word_ms + gap_ms)
        contour = [200.0 * 2 ** (semitones((4 * i + k) / (total - 1)) / 12) for k in range(4)]
        fields = {k: v[i] if isinstance(v, list) else v for k, v in values.items()}
        fields.setdefault("f0_mean", statistics.fmean(contour))
        spans.append(SpanFeatures(
            start_ms=start, end_ms=start + word_ms, text=f"w{i}", f0_contour=contour, **fields
        ))
    return spans


def _monotone(x: float) -> float:
    return 0.4 * math.sin(40 * x)


class TestCategorizeFeatures:
    def test_empty_input(self) -> None:
        assert categorize_features([]) == {}

    def test_spec_example_flat_and_fast_means_excitement(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        """Measured monotone, fast speech drives the spec 7.1 profile."""
        observed = categorize_features(_utterance(_monotone, speech_rate=6.8))
        assert observed["pitch_contour"] == "flat"
        assert observed["rate"] == "fast"
        assert applier.apply(autism_profile, observed, "neutral", 0.5) == (
            "excitement", pytest.approx(0.65)
        )

    def test_flat_with_many_pauses_means_thinking(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        observed = categorize_features(_utterance(_monotone, gap_ms=400))
        assert observed["pause_frequency"] == "high"
        assert applier.apply(autism_profile, observed, "neutral", 0.5)[0] == "thinking_carefully"

    def test_volume_spike_means_emphasis_not_anger(
        self, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        levels = [64.0, 66.0, 65.0, 77.5, 65.0, 64.0, 66.0, 65.0]
        observed = categorize_features(_utterance(lambda x: 6 * x, intensity_mean=levels))
        assert observed["volume"] == "spike"
        assert applier.apply(autism_profile, observed, "angry", 0.7)[0] == "emphasis_not_anger"

    def test_emphasis_is_not_a_spike(self) -> None:
        levels = [64.0, 66.0, 65.0, 73.0, 65.0, 64.0, 66.0, 65.0]
        assert "volume" not in categorize_features(_utterance(intensity_mean=levels))

    @pytest.mark.parametrize(
        ("semitones", "expected"),
        [
            (lambda x: 6 * x, "rise"),
            (lambda x: -6 * x, "fall"),
            (lambda x: 6 * math.sin(math.pi * x), "rise-fall"),
            (lambda x: -6 * math.sin(math.pi * x), "fall-rise"),
        ],
        ids=["rise", "fall", "rise-fall", "fall-rise"],
    )
    def test_contour_shapes(self, semitones: Callable[[float], float], expected: str) -> None:
        assert categorize_features(_utterance(semitones))["pitch_contour"] == expected

    def test_fast_large_movement_is_sharp(self) -> None:
        short = _utterance(lambda x: 8 * x, n=4, word_ms=60, gap_ms=0)
        assert categorize_features(short)["pitch_contour"] == "rise-sharp"
        falling = _utterance(lambda x: -8 * x, n=4, word_ms=60, gap_ms=0)
        assert categorize_features(falling)["pitch_contour"] == "fall-sharp"

    def test_varied_pitch_without_shape_has_no_contour(self) -> None:
        zigzag = _utterance(lambda x: 3.0 if round(x * 31) % 2 else -3.0)
        assert "pitch_contour" not in categorize_features(zigzag)

    def test_octave_errors_do_not_break_flatness(self) -> None:
        spans = _utterance(_monotone)
        doubled = [v * 2 for v in spans[2].f0_contour or []]
        spans[2] = SpanFeatures(**{**spans[2].__dict__, "f0_contour": doubled})
        assert categorize_features(spans)["pitch_contour"] == "flat"

    def test_too_few_f0_samples_give_no_contour(self) -> None:
        spans = [
            SpanFeatures(start_ms=0, end_ms=200, text="a", f0_mean=200.0),
            SpanFeatures(start_ms=250, end_ms=450, text="b", f0_mean=260.0),
        ]
        assert "pitch_contour" not in categorize_features(spans)

    @pytest.mark.parametrize(
        ("rate", "expected"),
        [
            (RATE_FAST_SYLLABLES_PER_S, "fast"),
            (RATE_FAST_SYLLABLES_PER_S - 0.1, "normal"),
            (4.5, "normal"),
            (RATE_SLOW_SYLLABLES_PER_S + 0.1, "normal"),
            # Without a baseline a low reading may be missed syllables.
            (RATE_SLOW_SYLLABLES_PER_S, None),
            (1.5, None),
        ],
    )
    def test_absolute_rate(self, rate: float, expected: str | None) -> None:
        assert categorize_features(_utterance(speech_rate=rate)).get("rate") == expected

    def test_unmeasured_rate_is_left_out(self) -> None:
        assert "rate" not in categorize_features(_utterance(speech_rate=None))

    def test_sparse_rate_estimates_are_left_out(self) -> None:
        """One voiced word in a whispered sentence is no rate for the sentence."""
        sparse = [None, None, 1.0, None, None, None, None, None]
        assert "rate" not in categorize_features(
            _utterance(speech_rate=sparse), baseline=_utterance()
        )
        half = [7.0, None, 7.0, None, 7.0, None, 7.0, None]
        assert categorize_features(_utterance(speech_rate=half))["rate"] == "fast"

    def test_rate_relative_to_baseline(self) -> None:
        """A slow talker speaking at an ordinary rate is fast for them."""
        baseline = _utterance(speech_rate=3.0)
        assert categorize_features(_utterance(speech_rate=4.5))["rate"] == "normal"
        assert categorize_features(_utterance(speech_rate=4.5), baseline=baseline)["rate"] == (
            "fast"
        )
        assert categorize_features(_utterance(speech_rate=2.2), baseline=baseline)["rate"] == (
            "slow"
        )

    @pytest.mark.parametrize(
        ("offset", "expected"), [(3.0, "high"), (-3.0, "low"), (1.0, "normal")]
    )
    def test_pitch_level_needs_baseline(self, offset: float, expected: str) -> None:
        spans = _utterance(lambda x: offset + _monotone(x))
        assert "pitch" not in categorize_features(spans)
        assert categorize_features(spans, baseline=_utterance(_monotone))["pitch"] == expected

    @pytest.mark.parametrize(
        ("level", "expected"), [(70.0, "loud"), (60.0, "quiet"), (66.0, "normal")]
    )
    def test_volume_level_needs_baseline(self, level: float, expected: str) -> None:
        spans = _utterance(intensity_mean=level)
        assert "volume" not in categorize_features(spans)
        assert categorize_features(spans, baseline=_utterance())["volume"] == expected

    def test_dominant_quality(self) -> None:
        mostly_breathy = ["breathy"] * 5 + ["modal"] * 3
        assert categorize_features(_utterance(quality=mostly_breathy))["quality"] == "breathy"
        split = ["breathy", "modal", "tense", "creaky"] * 2
        assert "quality" not in categorize_features(_utterance(quality=split))
        assert "quality" not in categorize_features(_utterance(quality="unknown"))

    def test_pause_frequency_from_detected_pauses(self) -> None:
        spans = _utterance(n=8, gap_ms=20)
        # Three pauses between words, plus ones that do not count: before the
        # first word, too short, and after the last word.
        pauses = [
            PauseInterval(start_ms=-400, end_ms=0),
            PauseInterval(start_ms=250, end_ms=520),
            PauseInterval(start_ms=790, end_ms=1100),
            PauseInterval(start_ms=1330, end_ms=1400),
            PauseInterval(start_ms=1600, end_ms=1900),
            PauseInterval(start_ms=2150, end_ms=2600),
        ]
        assert categorize_features(spans)["pause_frequency"] == "normal"
        assert categorize_features(spans, pauses)["pause_frequency"] == "high"

    def test_pause_frequency_low_needs_a_long_utterance(self) -> None:
        assert categorize_features(_utterance(n=12))["pause_frequency"] == "low"
        assert categorize_features(_utterance(n=6))["pause_frequency"] == "normal"
        assert "pause_frequency" not in categorize_features(_utterance(n=3))

    def test_emphasis_frequency(self) -> None:
        many = [72.0 if i % 3 == 0 else 65.0 for i in range(10)]
        assert categorize_features(_utterance(n=10, intensity_mean=many))[
            "emphasis_frequency"
        ] == "high"
        one = [72.0 if i == 4 else 65.0 for i in range(10)]
        assert categorize_features(_utterance(n=10, intensity_mean=one))[
            "emphasis_frequency"
        ] == "normal"
        assert categorize_features(_utterance(n=10))["emphasis_frequency"] == "low"
        # Pitch alone also marks emphasis: two of five words 8 semitones up.
        high_words = [320.0 if i in (1, 3) else 200.0 for i in range(5)]
        assert categorize_features(_utterance(n=5, f0_mean=high_words))[
            "emphasis_frequency"
        ] == "high"

    def test_span_order_does_not_matter(self) -> None:
        spans = _utterance(lambda x: 6 * x)
        assert categorize_features(list(reversed(spans))) == categorize_features(spans)


_ESPEAK = shutil.which("espeak-ng")
_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the children watch from the window"
)


@pytest.mark.skipif(_ESPEAK is None, reason="espeak-ng is not installed")
class TestCategorizeRealSpeech:
    """categorize_features on speech synthesised by espeak-ng and measured
    by ProsodyAnalyzer (word times spread evenly over the recording)."""

    @staticmethod
    def _measure(
        tmp_path: Path, ssml: str, pitch: int = 50
    ) -> tuple[list[SpanFeatures], list[PauseInterval]]:
        """Speak *ssml* with espeak-ng (base *pitch* 0-99; 50 is about 100 Hz,
        20 about 80 Hz) and measure it."""
        pytest.importorskip("parselmouth")
        from prosody_protocol._types import WordAlignment
        from prosody_protocol.prosody_analyzer import ProsodyAnalyzer

        wav = tmp_path / f"speech-{len(list(tmp_path.glob('*.wav')))}.wav"
        subprocess.run(
            [str(_ESPEAK), "-m", "-p", str(pitch), "-w", str(wav), f"<speak>{ssml}</speak>"],
            check=True, capture_output=True,
        )
        analyzer = ProsodyAnalyzer()
        pauses = analyzer.detect_pauses(wav)
        words = _SENTENCE.split()
        with wave.open(str(wav)) as audio:
            step = 1000 * audio.getnframes() / audio.getframerate() / len(words)
        alignments = [
            WordAlignment(word, int(i * step), int((i + 1) * step)) for i, word in enumerate(words)
        ]
        return analyzer.analyze(wav, alignments), pauses

    def test_monotone_fast_speech_reads_as_excitement(
        self, tmp_path: Path, autism_profile: ProsodyProfile, applier: ProfileApplier
    ) -> None:
        spans, pauses = self._measure(
            tmp_path, f'<prosody rate="x-fast" range="x-low">{_SENTENCE}</prosody>'
        )
        observed = categorize_features(spans, pauses)
        assert observed["pitch_contour"] == "flat"
        assert observed["rate"] == "fast"
        assert applier.apply(autism_profile, observed, "neutral", 0.5)[0] == "excitement"

    def test_slow_lively_speech_is_neither(self, tmp_path: Path) -> None:
        baseline, _ = self._measure(tmp_path, _SENTENCE)
        spans, pauses = self._measure(
            tmp_path, f'<prosody rate="x-slow" range="x-high">{_SENTENCE}</prosody>'
        )
        observed = categorize_features(spans, pauses, baseline=baseline)
        assert observed.get("pitch_contour") != "flat"
        assert observed["rate"] == "slow"

    def test_low_voice_at_ordinary_speed_is_not_slow(self, tmp_path: Path) -> None:
        """An 80 Hz voice at espeak's default speed measures about 3
        syllables/s (the analyzer misses syllables), which read as slow."""
        spans, pauses = self._measure(tmp_path, _SENTENCE, pitch=20)
        assert categorize_features(spans, pauses).get("rate") != "slow"

    def test_low_voice_rate_against_its_own_baseline(self, tmp_path: Path) -> None:
        baseline, _ = self._measure(tmp_path, _SENTENCE, pitch=20)
        for rate, expected in (("x-slow", "slow"), ("default", "normal"), ("x-fast", "fast")):
            spans, pauses = self._measure(
                tmp_path, f'<prosody rate="{rate}">{_SENTENCE}</prosody>', pitch=20
            )
            assert categorize_features(spans, pauses, baseline=baseline)["rate"] == expected

    def test_pauses_between_words_are_frequent(self, tmp_path: Path) -> None:
        words = _SENTENCE.split()
        ssml = " ".join(
            word + (' <break time="600ms"/>' if i % 3 == 2 else "") for i, word in enumerate(words)
        )
        spans, pauses = self._measure(tmp_path, ssml)
        assert len(pauses) >= 4
        assert categorize_features(spans, pauses)["pause_frequency"] == "high"
