"""Validation endpoint: validate IML documents."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

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
    raise NotImplementedError
