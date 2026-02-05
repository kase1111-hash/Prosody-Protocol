"""Prosody profile loader and applier for accessibility support.

Loads JSON prosody profiles (spec Section 7) and applies pattern-based
emotion re-mapping to adjust classifications for atypical speakers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .validator import ValidationResult


@dataclass(frozen=True)
class ProsodyMapping:
    """A single pattern-to-interpretation mapping."""

    pattern: dict[str, str]
    interpretation_emotion: str
    confidence_boost: float


@dataclass(frozen=True)
class ProsodyProfile:
    """A user's prosody profile."""

    profile_version: str
    user_id: str
    description: str | None
    mappings: list[ProsodyMapping]


class ProfileLoader:
    """Load and validate prosody profile JSON files."""

    def load(self, path: str | Path) -> ProsodyProfile:
        """Load a profile from a JSON file."""
        raise NotImplementedError

    def load_json(self, data: dict[str, object]) -> ProsodyProfile:
        """Load a profile from a parsed JSON dict."""
        raise NotImplementedError

    def validate(self, profile: ProsodyProfile) -> ValidationResult:
        """Validate a loaded profile."""
        raise NotImplementedError


class ProfileApplier:
    """Apply prosody profiles to adjust emotion classification."""

    def apply(
        self,
        profile: ProsodyProfile,
        features: dict[str, str],
        base_emotion: str,
        base_confidence: float,
    ) -> tuple[str, float]:
        """Return ``(adjusted_emotion, adjusted_confidence)``."""
        raise NotImplementedError
