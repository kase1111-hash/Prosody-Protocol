"""Conversion endpoints: audio-to-iml, text-to-iml, iml-to-ssml."""

from __future__ import annotations

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

router = APIRouter()


class TextToIMLRequest(BaseModel):
    text: str
    context: str | None = None


class IMLToSSMLRequest(BaseModel):
    iml: str


class ConvertResponse(BaseModel):
    iml: str
    plain_text: str | None = None


class SSMLResponse(BaseModel):
    ssml: str


@router.post("/audio-to-iml", response_model=ConvertResponse)
async def audio_to_iml(audio: UploadFile, language: str | None = None) -> ConvertResponse:
    raise NotImplementedError


@router.post("/text-to-iml", response_model=ConvertResponse)
async def text_to_iml(request: TextToIMLRequest) -> ConvertResponse:
    raise NotImplementedError


@router.post("/iml-to-ssml", response_model=SSMLResponse)
async def iml_to_ssml(request: IMLToSSMLRequest) -> SSMLResponse:
    raise NotImplementedError
