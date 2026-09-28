"""Validation endpoint: validate IML documents."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from prosody_protocol import IMLValidator

from ..deps import SettingsDep, check_text_length
from ..errors import ERROR_RESPONSES, ValidationIssueResponse, issue_fields

__all__ = ["ValidateRequest", "ValidateResponse", "ValidationIssueResponse", "router"]

router = APIRouter()


class ValidateRequest(BaseModel):
    iml: str


class ValidateResponse(BaseModel):
    valid: bool
    issues: list[ValidationIssueResponse]


# A plain ``def`` runs in the threadpool, off the event loop.
@router.post("/validate", response_model=ValidateResponse, responses=ERROR_RESPONSES)
def validate_iml(request: ValidateRequest, settings: SettingsDep) -> ValidateResponse:
    """Validate an IML document against the spec rules.

    An invalid document is still a 200 response, with ``valid: false`` and
    the rule violations in ``issues``.
    """
    check_text_length(settings, iml=request.iml)
    result = IMLValidator().validate(request.iml)
    issues = [ValidationIssueResponse(**issue_fields(issue)) for issue in result.issues]
    return ValidateResponse(valid=result.valid, issues=issues)
