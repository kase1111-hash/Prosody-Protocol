"""Prosody profile loader and applier for accessibility support.

Loads JSON prosody profiles (spec Section 7) and applies pattern-based
emotion re-mapping to adjust classifications for atypical speakers.

Profile matching logic (Section 8.2 of the execution guide):
  1. For each mapping, check if *all* pattern keys match observed features.
  2. Matching is categorical (exact string match on observed feature labels).
  3. If multiple mappings match, use the most specific (most pattern keys).
  4. Apply ``confidence_boost`` (capped at 1.0).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ProfileError
from .validator import ValidationIssue, ValidationResult


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProsodyMapping:
    """A single pattern-to-interpretation mapping."""

    pattern: dict[str, str]
    interpretation_emotion: str
    confidence_boost: float = 0.0


@dataclass(frozen=True)
class ProsodyProfile:
    """A user's prosody profile."""

    profile_version: str
    user_id: str
    description: str | None
    mappings: tuple[ProsodyMapping, ...]


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$")

_VALID_PATTERN_KEYS = frozenset({
    "pitch", "pitch_contour", "volume", "rate",
    "quality", "pause_frequency", "emphasis_frequency",
})

_PITCH_VALUES = frozenset({"high", "low", "normal"})
_PITCH_CONTOUR_VALUES = frozenset({
    "rise", "fall", "rise-fall", "fall-rise",
    "fall-sharp", "rise-sharp", "flat",
})
_VOLUME_VALUES = frozenset({"loud", "quiet", "normal", "spike"})
_RATE_VALUES = frozenset({"fast", "slow", "normal"})
_QUALITY_VALUES = frozenset({
    "modal", "breathy", "tense", "creaky", "whispery", "harsh",
})
_FREQUENCY_VALUES = frozenset({"high", "low", "normal"})

_PATTERN_VALUE_MAP: dict[str, frozenset[str]] = {
    "pitch": _PITCH_VALUES,
    "pitch_contour": _PITCH_CONTOUR_VALUES,
    "volume": _VOLUME_VALUES,
    "rate": _RATE_VALUES,
    "quality": _QUALITY_VALUES,
    "pause_frequency": _FREQUENCY_VALUES,
    "emphasis_frequency": _FREQUENCY_VALUES,
}


# ---------------------------------------------------------------------------
# ProfileLoader
# ---------------------------------------------------------------------------


class ProfileLoader:
    """Load and validate prosody profile JSON files."""

    def load(self, path: str | Path) -> ProsodyProfile:
        """Load a profile from a JSON file.

        Raises :class:`~prosody_protocol.exceptions.ProfileError`
        if the file cannot be read or the JSON structure is invalid.
        """
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProfileError(f"Cannot read profile file: {exc}") from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProfileError(f"Invalid JSON in profile file: {exc}") from exc

        return self.load_json(data)

    def load_json(self, data: dict[str, object]) -> ProsodyProfile:
        """Load a profile from a parsed JSON dict.

        Raises :class:`~prosody_protocol.exceptions.ProfileError`
        if required fields are missing or have wrong types.
        """
        if not isinstance(data, dict):
            raise ProfileError("Profile must be a JSON object")

        # Required fields.
        profile_version = data.get("profile_version")
        if not isinstance(profile_version, str) or not profile_version:
            raise ProfileError("Missing or invalid 'profile_version' (must be a non-empty string)")

        user_id = data.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            raise ProfileError("Missing or invalid 'user_id' (must be a non-empty string)")

        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise ProfileError("'description' must be a string or null")

        raw_mappings = data.get("prosody_mappings")
        if not isinstance(raw_mappings, list):
            raise ProfileError("Missing or invalid 'prosody_mappings' (must be an array)")

        mappings: list[ProsodyMapping] = []
        for i, raw in enumerate(raw_mappings):
            if not isinstance(raw, dict):
                raise ProfileError(f"prosody_mappings[{i}] must be an object")
            mappings.append(self._parse_mapping(raw, i))

        return ProsodyProfile(
            profile_version=profile_version,
            user_id=user_id,
            description=description,
            mappings=tuple(mappings),
        )

    def validate(self, profile: ProsodyProfile) -> ValidationResult:
        """Validate a loaded profile against spec rules.

        Returns a :class:`ValidationResult` with ``valid=True`` when
        no errors are found.
        """
        issues: list[ValidationIssue] = []

        # Version format.
        if not _VERSION_RE.match(profile.profile_version):
            issues.append(ValidationIssue(
                severity="error",
                rule="P1",
                message=(
                    f"profile_version '{profile.profile_version}' "
                    "does not match semver format (X.Y.Z)"
                ),
            ))

        # User ID non-empty.
        if not profile.user_id:
            issues.append(ValidationIssue(
                severity="error",
                rule="P2",
                message="user_id must be a non-empty string",
            ))

        # At least one mapping.
        if not profile.mappings:
            issues.append(ValidationIssue(
                severity="error",
                rule="P3",
                message="prosody_mappings must contain at least one mapping",
            ))

        # Validate each mapping.
        for i, m in enumerate(profile.mappings):
            if not m.pattern:
                issues.append(ValidationIssue(
                    severity="error",
                    rule="P4",
                    message=f"prosody_mappings[{i}].pattern must have at least one key",
                ))

            for key, value in m.pattern.items():
                if key not in _VALID_PATTERN_KEYS:
                    issues.append(ValidationIssue(
                        severity="warning",
                        rule="P5",
                        message=f"prosody_mappings[{i}].pattern has unknown key '{key}'",
                    ))
                elif value not in _PATTERN_VALUE_MAP[key]:
                    issues.append(ValidationIssue(
                        severity="warning",
                        rule="P6",
                        message=(
                            f"prosody_mappings[{i}].pattern.{key}='{value}' "
                            f"is not a recognized value"
                        ),
                    ))

            if not m.interpretation_emotion:
                issues.append(ValidationIssue(
                    severity="error",
                    rule="P7",
                    message=f"prosody_mappings[{i}].interpretation.emotion must be non-empty",
                ))

            if m.confidence_boost < 0.0 or m.confidence_boost > 1.0:
                issues.append(ValidationIssue(
                    severity="error",
                    rule="P8",
                    message=(
                        f"prosody_mappings[{i}].interpretation.confidence_boost="
                        f"{m.confidence_boost} must be between 0.0 and 1.0"
                    ),
                ))

        valid = not any(issue.severity == "error" for issue in issues)
        return ValidationResult(valid=valid, issues=issues)

    @staticmethod
    def _parse_mapping(raw: dict[str, object], index: int) -> ProsodyMapping:
        """Parse a single mapping entry from a JSON dict."""
        pattern_raw = raw.get("pattern")
        if not isinstance(pattern_raw, dict):
            raise ProfileError(
                f"prosody_mappings[{index}].pattern must be an object"
            )

        pattern: dict[str, str] = {}
        for k, v in pattern_raw.items():
            if not isinstance(v, str):
                raise ProfileError(
                    f"prosody_mappings[{index}].pattern.{k} must be a string"
                )
            pattern[k] = v

        interpretation = raw.get("interpretation")
        if not isinstance(interpretation, dict):
            raise ProfileError(
                f"prosody_mappings[{index}].interpretation must be an object"
            )

        emotion = interpretation.get("emotion")
        if not isinstance(emotion, str) or not emotion:
            raise ProfileError(
                f"prosody_mappings[{index}].interpretation.emotion must be a non-empty string"
            )

        confidence_boost = interpretation.get("confidence_boost", 0.0)
        if not isinstance(confidence_boost, (int, float)):
            raise ProfileError(
                f"prosody_mappings[{index}].interpretation.confidence_boost must be a number"
            )

        return ProsodyMapping(
            pattern=pattern,
            interpretation_emotion=emotion,
            confidence_boost=float(confidence_boost),
        )


# ---------------------------------------------------------------------------
# ProfileApplier
# ---------------------------------------------------------------------------


class ProfileApplier:
    """Apply prosody profiles to adjust emotion classification.

    Matching logic:
      1. Check each mapping's pattern against observed features.
      2. A mapping matches only if *all* its pattern keys are present
         in the features dict and their values match.
      3. Among all matching mappings, select the most specific one
         (most pattern keys).  Ties are broken by order in the profile
         (first match wins).
      4. Apply ``confidence_boost`` (capped at 1.0).
    """

    def apply(
        self,
        profile: ProsodyProfile,
        features: dict[str, str],
        base_emotion: str,
        base_confidence: float,
    ) -> tuple[str, float]:
        """Return ``(adjusted_emotion, adjusted_confidence)``.

        If no mapping matches, returns the base values unchanged.
        """
        best_match: ProsodyMapping | None = None
        best_specificity = 0

        for mapping in profile.mappings:
            if self._matches(mapping.pattern, features):
                specificity = len(mapping.pattern)
                if specificity > best_specificity:
                    best_match = mapping
                    best_specificity = specificity

        if best_match is None:
            return (base_emotion, base_confidence)

        adjusted_emotion = best_match.interpretation_emotion
        adjusted_confidence = min(1.0, base_confidence + best_match.confidence_boost)

        return (adjusted_emotion, adjusted_confidence)

    @staticmethod
    def _matches(pattern: dict[str, str], features: dict[str, str]) -> bool:
        """Check if all pattern entries match observed features."""
        for key, expected in pattern.items():
            if features.get(key) != expected:
                return False
        return True
