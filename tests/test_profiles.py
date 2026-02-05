"""Tests for prosody_protocol.profiles (Phase 8 -- Prosody Profiles).

Covers:
- Acceptance criteria from the execution guide
- ProfileLoader: load from file, load from dict, error handling
- ProfileLoader.validate: version, user_id, mappings, pattern keys/values
- ProfileApplier: pattern matching, specificity, confidence capping
- Data model frozen semantics
- Edge cases: empty features, no match, multiple matches
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prosody_protocol.exceptions import ProfileError
from prosody_protocol.profiles import (
    ProfileApplier,
    ProfileLoader,
    ProsodyMapping,
    ProsodyProfile,
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
def spec_profile_json() -> dict[str, object]:
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
            mappings=[],
        )
        assert p.profile_version == "0.1.0"
        assert p.user_id == "user_789"
        assert p.description == "Test profile"
        assert p.mappings == []

    def test_prosody_profile_none_description(self) -> None:
        p = ProsodyProfile("1.0.0", "u1", None, [])
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

    def test_unknown_pattern_key_warns(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            "1.0.0", "u1", None,
            [ProsodyMapping({"unknown_key": "value"}, "excited", 0.1)],
        )
        result = loader.validate(profile)
        assert result.valid  # Warnings don't invalidate.
        assert any(i.rule == "P5" and i.severity == "warning" for i in result.issues)

    def test_invalid_pattern_value_warns(self, loader: ProfileLoader) -> None:
        profile = ProsodyProfile(
            "1.0.0", "u1", None,
            [ProsodyMapping({"pitch": "very_high"}, "excited", 0.1)],
        )
        result = loader.validate(profile)
        assert result.valid  # Warnings don't invalidate.
        assert any(i.rule == "P6" and i.severity == "warning" for i in result.issues)

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
