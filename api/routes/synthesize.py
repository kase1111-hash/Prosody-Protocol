"""Synthesis endpoint: IML to audio."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter()


class SynthesizeRequest(BaseModel):
    iml: str
    voice: str = "en_US-female-medium"


@router.post("/synthesize")
async def synthesize(request: SynthesizeRequest) -> Response:
    raise NotImplementedError
