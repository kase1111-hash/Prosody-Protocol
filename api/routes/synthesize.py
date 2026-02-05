"""Synthesis endpoint: IML to audio."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

from prosody_protocol import IMLToAudio

router = APIRouter()


class SynthesizeRequest(BaseModel):
    iml: str
    voice: str = "en_US-female-medium"


@router.post("/synthesize")
async def synthesize(request: SynthesizeRequest) -> Response:
    synth = IMLToAudio()
    wav_bytes = synth.synthesize(request.iml)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )
