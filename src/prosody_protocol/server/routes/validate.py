"""Validation endpoint: validate IML documents."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from prosody_protocol import IMLValidator

router = APIRouter()


class ValidateRequest(BaseModel):
    iml: str


class ValidationIssueResponse(BaseModel):
    severity: str
    rule: str
    message: str
    line: int | None = None
    column: int | None = None


class ValidateResponse(BaseModel):
    valid: bool
    issues: list[ValidationIssueResponse]


@router.post("/validate", response_model=ValidateResponse)
async def validate_iml(request: ValidateRequest) -> ValidateResponse:
    validator = IMLValidator()
    result = validator.validate(request.iml)

    issues = [
        ValidationIssueResponse(
            severity=issue.severity,
            rule=issue.rule,
            message=issue.message,
            line=getattr(issue, "line", None),
            column=getattr(issue, "column", None),
        )
        for issue in result.issues
    ]

    return ValidateResponse(valid=result.valid, issues=issues)
