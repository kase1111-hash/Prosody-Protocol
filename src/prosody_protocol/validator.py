"""IML validator -- checks IML documents against the spec rules.

Implements validation rules V1-V16 from the execution guide (Phase 3).
Spec reference: Sections 5-6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding."""

    severity: Literal["error", "warning", "info"]
    rule: str
    message: str
    line: int | None = None
    column: int | None = None


@dataclass
class ValidationResult:
    """Outcome of validating an IML document."""

    valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)


class IMLValidator:
    """Validate IML documents against the specification."""

    def validate(self, iml_string: str) -> ValidationResult:
        """Validate an IML XML string.

        Returns a :class:`ValidationResult` with ``valid=True`` when no
        errors are found (warnings and info issues are allowed).
        """
        raise NotImplementedError

    def validate_file(self, path: str | Path) -> ValidationResult:
        """Validate an IML XML file from disk."""
        raise NotImplementedError
