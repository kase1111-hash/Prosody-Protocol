"""Conversion endpoints: audio-to-iml, text-to-iml, iml-to-ssml."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

from prosody_protocol import AudioToIML, IMLParser, IMLToSSML, TextToIML

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
    converter = AudioToIML(language=language or "en-US")
    parser = IMLParser()

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        doc = converter.convert_to_doc(tmp_path)
        iml_string = parser.to_iml_string(doc)
        plain_text = parser.to_plain_text(doc)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return ConvertResponse(iml=iml_string, plain_text=plain_text)


@router.post("/text-to-iml", response_model=ConvertResponse)
async def text_to_iml(request: TextToIMLRequest) -> ConvertResponse:
    predictor = TextToIML()
    iml_string = predictor.predict(request.text, context=request.context)

    parser = IMLParser()
    doc = parser.parse(iml_string)
    plain_text = parser.to_plain_text(doc)

    return ConvertResponse(iml=iml_string, plain_text=plain_text)


@router.post("/iml-to-ssml", response_model=SSMLResponse)
async def iml_to_ssml(request: IMLToSSMLRequest) -> SSMLResponse:
    converter = IMLToSSML()
    ssml = converter.convert(request.iml)
    return SSMLResponse(ssml=ssml)
