"""Tests for prosody_protocol.validator.

These tests are stubs -- they verify the interface exists and will be
fleshed out when the validator is implemented in Phase 3.
"""

from __future__ import annotations

import pytest

from prosody_protocol.validator import IMLValidator, ValidationIssue, ValidationResult


class TestValidationModels:
    def test_validation_issue(self) -> None:
        issue = ValidationIssue(
            severity="error",
            rule="V3",
            message="Missing confidence",
            line=1,
            column=15,
        )
        assert issue.severity == "error"
        assert issue.rule == "V3"

    def test_validation_result_default(self) -> None:
        result = ValidationResult()
        assert result.valid is True
        assert result.issues == []


class TestIMLValidatorInterface:
    def test_instantiate(self) -> None:
        validator = IMLValidator()
        assert validator is not None

    def test_validate_not_implemented(self, simple_utterance: str) -> None:
        validator = IMLValidator()
        with pytest.raises(NotImplementedError):
            validator.validate(simple_utterance)
