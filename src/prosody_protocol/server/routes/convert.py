"""Conversion endpoints: audio-to-iml, text-to-iml, iml-to-ssml."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path, PurePath
from typing import Annotated, Any, BinaryIO, Literal

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from prosody_protocol import IMLParser, IMLToSSML, TextToIML

from .. import _worker
from ..deps import JobsDep, SettingsDep, check_text_length
from ..errors import BUSY_RESPONSES, ERROR_RESPONSES, APIError, ErrorResponse

router = APIRouter()

# BCP 47 language tag, as validator rule V29 checks it. The query parameter
# may also be empty, which means no language (an empty form field already
# arrives as None).
_LANGUAGE_PATTERN = r"^[A-Za-z]{1,8}(-[A-Za-z0-9]{1,8})*$"
_LANGUAGE_QUERY_PATTERN = r"^([A-Za-z]{1,8}(-[A-Za-z0-9]{1,8})*)?$"
_LANGUAGE_DESCRIPTION = (
    'BCP 47 language tag of the speech ("fr-FR"). It labels the output and is passed '
    "to speech recognition. When omitted, the language speech recognition detects "
    "labels the output (none without speech recognition)."
)
# File name suffixes kept on the temporary copy of an upload.
_SUFFIX_RE = re.compile(r"\.[A-Za-z0-9]{1,10}")
_COPY_CHUNK_BYTES = 1024 * 1024


class TextToIMLRequest(BaseModel):
    text: str
    context: str | None = Field(
        default=None,
        description=(
            "Optional surrounding text (e.g. the previous turn). Emotion words in it, "
            'such as "frustrated", act as a weak prior for the predicted emotion.'
        ),
    )


class IMLToSSMLRequest(BaseModel):
    iml: str
    strict: bool = Field(
        default=False,
        description=(
            "Reject documents with validation errors (400 validation_error, listing the "
            "issues). By default the document is converted as spec section 6.2 asks of "
            "consumers: invalid attribute values are ignored."
        ),
    )


class ConvertResponse(BaseModel):
    iml: str
    plain_text: str | None = None


class AudioToIMLResponse(ConvertResponse):
    transcript_source: Literal["words", "transcript", "whisper", "none"] = Field(
        description=(
            'Where the words came from: "whisper" (built-in speech recognition) or '
            '"none" (no speech recognition on this server: each stretch of speech is a '
            '"[speech]" placeholder).'
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Anything that degraded the output, such as placeholder text.",
    )


class SSMLResponse(BaseModel):
    ssml: str


_UNSUPPORTED_MEDIA: dict[int | str, dict[str, Any]] = {
    415: {"model": ErrorResponse, "description": "The request is not multipart/form-data."},
}


def _require_multipart(request: Request) -> None:
    content_type = request.headers.get("content-type")
    if content_type is not None and not content_type.lower().startswith("multipart/form-data"):
        raise APIError(
            415,
            "unsupported_media_type",
            f"Expected multipart/form-data with an 'audio' file field, got {content_type!r} "
            "(e.g. curl -F audio=@speech.wav).",
        )


def _save_upload(source: BinaryIO, dest: Path, max_bytes: int) -> None:
    """Copy *source* to *dest* in chunks, refusing more than *max_bytes*."""
    copied = 0
    with dest.open("wb") as out:
        while chunk := source.read(_COPY_CHUNK_BYTES):
            copied += len(chunk)
            if copied > max_bytes:
                raise APIError(
                    413,
                    "payload_too_large",
                    f"Uploaded file exceeds maximum allowed size ({max_bytes} bytes).",
                )
            out.write(chunk)


def _upload_suffix(filename: str | None) -> str:
    """The upload's file name suffix (``.ogg``), so decoders can use it."""
    suffix = PurePath(filename or "").suffix
    return suffix.lower() if _SUFFIX_RE.fullmatch(suffix) else ""


@router.post(
    "/audio-to-iml",
    response_model=AudioToIMLResponse,
    responses={**ERROR_RESPONSES, **_UNSUPPORTED_MEDIA, **BUSY_RESPONSES},
    dependencies=[Depends(_require_multipart)],
)
async def audio_to_iml(
    audio: Annotated[
        UploadFile,
        File(
            description=(
                "Audio file. WAV, AIFF, FLAC and MP3 are always read; OGG/Opus, WebM, "
                "M4A and other formats need ffmpeg on the server (see /v1/health)."
            )
        ),
    ],
    settings: SettingsDep,
    jobs: JobsDep,
    language: Annotated[
        str | None,
        Form(pattern=_LANGUAGE_PATTERN, max_length=35, description=_LANGUAGE_DESCRIPTION),
    ] = None,
    language_query: Annotated[
        str | None,
        Query(
            alias="language",
            pattern=_LANGUAGE_QUERY_PATTERN,
            max_length=35,
            description="Same as the language form field (kept for older clients).",
        ),
    ] = None,
) -> AudioToIMLResponse:
    """Convert an uploaded audio file to IML.

    Send multipart/form-data with the file in ``audio`` and, optionally, a
    ``language`` field. Unreadable audio, or audio longer than the server's
    PP_MAX_AUDIO_SECONDS, is a 400 audio_processing_error.
    """
    # BCP 47 tags are case-insensitive: "FR-fr" and "fr-FR" agree.
    if language and language_query and language.lower() != language_query.lower():
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("query", "language"),
                    "msg": (
                        f"language is {language!r} in the form but {language_query!r} "
                        "in the query string"
                    ),
                    "input": language_query,
                }
            ]
        )
    language = language or language_query or None
    display_name = repr(PurePath(audio.filename).name) if audio.filename else "upload"

    with jobs.admit(), tempfile.TemporaryDirectory(prefix="prosody-protocol-upload-") as tmp:
        path = Path(tmp) / f"upload{_upload_suffix(audio.filename)}"
        await run_in_threadpool(_save_upload, audio.file, path, settings.max_upload_bytes)
        # Drop the server's first copy of the upload while the job waits for a worker.
        await audio.close()
        result = await jobs.run(
            _worker.convert_audio, str(path), language, display_name, settings.max_audio_seconds
        )
    return AudioToIMLResponse(
        iml=result.iml,
        plain_text=IMLParser().to_plain_text(result.document),
        transcript_source=result.transcript_source,
        warnings=list(result.warnings),
    )


@router.post("/text-to-iml", response_model=ConvertResponse, responses=ERROR_RESPONSES)
def text_to_iml(request: TextToIMLRequest, settings: SettingsDep) -> ConvertResponse:
    """Predict IML prosody markup for plain text (rule-based)."""
    check_text_length(settings, text=request.text, context=request.context)
    iml_string = TextToIML().predict(request.text, context=request.context)
    parser = IMLParser()
    plain_text = parser.to_plain_text(parser.parse(iml_string))
    return ConvertResponse(iml=iml_string, plain_text=plain_text)


@router.post("/iml-to-ssml", response_model=SSMLResponse, responses=ERROR_RESPONSES)
def iml_to_ssml(request: IMLToSSMLRequest, settings: SettingsDep) -> SSMLResponse:
    """Convert IML to SSML 1.1 for text-to-speech engines.

    With ``strict``, a document with validation errors is a 400 validation_error.
    """
    check_text_length(settings, iml=request.iml)
    converter = IMLToSSML(strict=request.strict)
    return SSMLResponse(ssml=converter.convert(request.iml))
