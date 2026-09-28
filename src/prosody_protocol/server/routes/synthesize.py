"""Synthesis endpoint: IML to audio."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .. import _worker
from ..deps import JobsDep, SettingsDep, check_text_length
from ..errors import BUSY_RESPONSES, ERROR_RESPONSES

router = APIRouter()


class SynthesizeRequest(BaseModel):
    iml: str
    voice: str | None = Field(
        default=None,
        description=(
            "None picks a voice for the document's language. Otherwise a language tag "
            '("en-US"), optionally with gender and pitch level ("en_US-female-medium", '
            '"fr-male-low", "male"), or an espeak-ng voice with a variant ("en-us+f3"). '
            "An unusable voice is a 400 conversion_error."
        ),
    )
    engine: Literal["auto", "espeak", "tones"] = Field(
        default="auto",
        description=(
            '"espeak": speech from espeak-ng (400 if the server does not have it). '
            '"tones": a prosody preview, one sine tone per word, not speech. '
            '"auto": espeak-ng when installed, otherwise tones. The engine used is '
            "returned in the X-Prosody-Engine header."
        ),
    )
    strict: bool = Field(
        default=False,
        description=(
            "Reject documents with validation errors (400 validation_error, listing the "
            "issues). By default the document is synthesized as spec section 6.2 asks of "
            "consumers: invalid attribute values are ignored."
        ),
    )


_WAV_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "A mono 16-bit PCM WAV file.",
        "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
        "headers": {
            "X-Prosody-Engine": {
                "description": '"espeak" (speech) or "tones" (prosody preview).',
                "schema": {"type": "string", "enum": ["espeak", "tones"]},
            }
        },
    },
}


# ``Response`` has no media type of its own, so the OpenAPI schema takes the
# 200 content from _WAV_RESPONSE and documents the errors as JSON.
@router.post(
    "/synthesize",
    response_class=Response,
    responses={**_WAV_RESPONSE, **ERROR_RESPONSES, **BUSY_RESPONSES},
)
async def synthesize(
    request: SynthesizeRequest, settings: SettingsDep, jobs: JobsDep
) -> Response:
    """Synthesize IML markup to a WAV file.

    Audio longer than the server's PP_MAX_SYNTH_SECONDS is rejected (400
    conversion_error) before any of it is rendered. With ``strict``, a
    document with validation errors is a 400 validation_error.
    """
    check_text_length(settings, iml=request.iml)

    with jobs.admit():
        wav_bytes, backend = await jobs.run(
            _worker.synthesize,
            request.iml,
            request.voice,
            request.engine,
            settings.max_synth_seconds,
            request.strict,
        )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=output.wav",
            "X-Prosody-Engine": backend,
        },
    )
