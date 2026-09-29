"""Conversion endpoints: audio-to-iml, text-to-iml, iml-to-ssml, iml-to-prompt."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path, PurePath
from typing import Annotated, Any, BinaryIO, Literal

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from prosody_protocol import IMLParser, IMLToSSML, IMLValidator, TextToIML
from prosody_protocol._types import WordAlignment
from prosody_protocol.alignment import parse_word_timings
from prosody_protocol.assembler import _checked_profile
from prosody_protocol.audio_to_iml import MAX_WORD_OVERLAP_MS, _checked_text, _checked_words
from prosody_protocol.exceptions import ConversionError, ProfileError
from prosody_protocol.llm import (
    DEFAULT_MIN_CONFIDENCE,
    SYSTEM_PROMPT,
    build_messages,
    to_llm_context,
)
from prosody_protocol.profiles import ProfileLoader, ProsodyProfile

from .. import _worker
from ..config import Settings
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

#: The most calibration recordings one audio-to-iml request may send. Each
#: is analysed like the upload itself (up to PP_MAX_AUDIO_SECONDS).
MAX_CALIBRATION_FILES = 5


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


class IMLToPromptRequest(BaseModel):
    iml: str
    min_confidence: float = Field(
        default=DEFAULT_MIN_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description=(
            "Emotions with lower confidence are described as \"emotion not reliably "
            'detected" instead of being named (spec 6.2 treats confidence below 0.5 as low).'
        ),
    )
    include_numbers: bool = Field(
        default=False,
        description=(
            "Also give measured values: pitch and volume offsets, rate percentages, "
            "extended attributes, exact pause lengths."
        ),
    )
    instruction: str | None = Field(
        default=None,
        description=(
            'Optional task for the model, appended to the user message after the '
            'transcript (e.g. "Summarize the caller\'s problem."). Without it the model '
            "answers the speaker."
        ),
    )
    strict: bool = Field(
        default=False,
        description=(
            "Reject documents with validation errors (400 validation_error, listing the "
            "issues). By default the document is described as spec section 6.2 asks of "
            "consumers: invalid attribute values are ignored."
        ),
    )


class ConvertResponse(BaseModel):
    iml: str
    plain_text: str | None = None


class ProfileMatchResponse(BaseModel):
    """A prosody profile mapping that matched an utterance (spec Section 7)."""

    utterance: int = Field(description="Index of the utterance in the document (0-based).")
    observed: dict[str, str] = Field(
        description="The utterance's prosody in the profile vocabulary (pitch, rate, ...)."
    )
    pattern: dict[str, str] = Field(description="The pattern of the mapping that matched.")
    emotion: str = Field(description="The mapping's interpretation.")
    confidence: float = Field(
        description="The classifier's confidence plus the mapping's confidence_boost (<= 1)."
    )
    applied: bool = Field(
        description=(
            "Whether the utterance carries this emotion (with x-profile). False when the "
            "confidence is below 0.5: the classifier had too little evidence."
        )
    )


class AudioToIMLResponse(ConvertResponse):
    transcript_source: Literal["words", "transcript", "whisper", "none"] = Field(
        description=(
            'Where the words came from: "words" (the words field), "transcript" (the '
            'transcript field: prosody is described for the whole utterance only), '
            '"whisper" (built-in speech recognition) or "none" (no speech recognition on '
            'this server: each stretch of speech is a "[speech]" placeholder).'
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Anything that degraded the output, such as placeholder text.",
    )
    profile_matches: list[ProfileMatchResponse] = Field(
        default_factory=list,
        description=(
            "With a profile: the utterances one of its mappings matched (spec 7.2 asks "
            "that profile usage be reported). Utterances whose emotion a mapping set "
            'also carry x-profile in the IML, with the pattern that matched (e.g. '
            'x-profile="pitch_contour=flat rate=fast"); the user_id is not written.'
        ),
    )


class SSMLResponse(BaseModel):
    ssml: str


class PromptResponse(BaseModel):
    context: str = Field(
        description=(
            "The document as an annotated transcript: *stressed* words, prosody notes in "
            "parentheses, [pause 0.8s] markers and a Delivery: line per utterance."
        )
    )
    system_prompt: str = Field(
        description="Explains the notation, and the limits of prosodic cues, to the model."
    )
    messages: list[dict[str, str]] = Field(
        description=(
            "Chat messages for any chat-completion API: the system prompt, then a user "
            "message with the transcript in <transcript> tags (and the instruction)."
        )
    )


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


def _field_error(name: str, message: str) -> RequestValidationError:
    """A 422 response naming the form field *name*."""
    return RequestValidationError(
        [{"type": "value_error", "loc": ("body", name), "msg": message}]
    )


def _too_large(name: str, size: str, limit: int, setting: str) -> APIError:
    return APIError(
        413,
        "text_too_large",
        f"Field {name!r} has {size}; the maximum is {limit} ({setting}).",
    )


async def _field_bytes(value: UploadFile, name: str, limit: int, setting: str) -> bytes:
    """The contents of a form field sent as a file, refusing more than *limit* bytes."""
    data = await value.read(limit + 1)
    await value.close()
    if len(data) > limit:
        raise _too_large(name, f"more than {limit} bytes", limit, setting)
    return data


async def _field_text(
    value: UploadFile | str | None, name: str, limit: int, setting: str
) -> str | None:
    """A text form field, sent as a field or as a file (UTF-8), within *limit*."""
    if value is None or isinstance(value, str):
        if value is not None and len(value) > limit:
            raise _too_large(name, f"{len(value)} characters", limit, setting)
        return value
    data = await _field_bytes(value, name, limit, setting)
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _field_error(name, f"{name} must be UTF-8 text: {exc.reason}") from None


async def _words_field(
    value: UploadFile | str | None, settings: Settings
) -> tuple[WordAlignment, ...] | None:
    """The ``words`` field parsed as word timings (never read as a file name)."""
    if value is None:
        return None
    limit, setting = settings.max_words_chars, "PP_MAX_WORDS_CHARS"
    if isinstance(value, str):
        if len(value) > limit:
            raise _too_large("words", f"{len(value)} characters", limit, setting)
        raw: str | bytes = value
    else:
        # parse_word_timings detects UTF-8, UTF-16 and UTF-32 in bytes.
        raw = await _field_bytes(value, "words", limit, setting)
    try:
        # Up to PP_MAX_WORDS_CHARS of JSON: parsed off the event loop.
        return await run_in_threadpool(_parse_words, raw)
    except (ConversionError, ValueError, TypeError) as exc:
        raise _field_error("words", f"Invalid word timings: {exc}") from None


def _parse_words(raw: str | bytes) -> tuple[WordAlignment, ...]:
    return tuple(_checked_words(parse_word_timings(raw)))


def _calibration_files(values: list[UploadFile | str] | None) -> list[UploadFile]:
    """The ``calibration`` recordings; empty fields count as absent, text is a 422."""
    files = []
    for value in values or ():
        if isinstance(value, str):
            if value:
                raise _field_error(
                    "calibration",
                    "calibration must be sent as a file (e.g. curl -F calibration=@calm.wav), "
                    "not as text.",
                )
        elif value.filename or value.size:
            files.append(value)
    return files


def _reject_constant(name: str) -> float:
    raise ProfileError(f"Invalid JSON in profile: {name} is not a JSON number")


def _profile_field(text: str | None) -> ProsodyProfile | None:
    """The ``profile`` field loaded and validated; errors are ProfileError (400)."""
    if text is None:
        return None
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError, or nested too deeply
        raise ProfileError(f"Invalid JSON in profile: {exc}") from None
    return _checked_profile(ProfileLoader().load_json(data))


_WORDS_DESCRIPTION = (
    "Optional word timings from any speech recogniser, as JSON text or a JSON file: "
    "openai-whisper, faster-whisper or WhisperX output, the OpenAI transcription API's "
    "verbose_json, Deepgram, AssemblyAI, Google Cloud Speech-to-Text, or a list of "
    '{"word", "start_ms", "end_ms"} records. The words become the transcript and no '
    "speech recognition runs. Invalid timings, including words that overlap by more "
    f"than {MAX_WORD_OVERLAP_MS} ms, are a 422."
)
_TRANSCRIPT_DESCRIPTION = (
    "Optional plain text of the speech, without timings (not together with words). "
    "No speech recognition runs; prosody is described for the utterance as a whole."
)
_PROFILE_DESCRIPTION = (
    "Optional prosody profile of the speaker (spec Section 7), as JSON text or a JSON "
    "file. A mapping that matches an utterance sets its emotion (see profile_matches). "
    "An invalid profile is a 400 profile_error."
)
_CALIBRATION_DESCRIPTION = (
    "Optional recording of the same speaker talking neutrally, as a file (repeat the "
    f"field for up to {MAX_CALIBRATION_FILES} recordings, e.g. earlier turns of a "
    "conversation). It is the speaker baseline that pitch, loudness, rate and emotion "
    "are measured against. Without it the baseline is the recording's typical "
    "utterances, so a single utterance gets no overall delivery and no emotion. Each "
    "recording is limited like the audio (PP_MAX_AUDIO_SECONDS); an unreadable one, "
    "or one without voiced speech, is a 400 audio_processing_error."
)

#: OpenAPI ``responses`` of audio-to-iml beyond the shared ones.
_AUDIO_RESPONSES: dict[int | str, dict[str, Any]] = {
    500: {
        "model": ErrorResponse,
        "description": (
            "`speech_recognition_failed`: the server's Whisper failed on audio it could "
            "read; `internal_error`: a bug, or the worker running the request exited."
        ),
    },
    503: {
        "model": ErrorResponse,
        "description": (
            "`server_busy`: every worker is busy and the queue is full. "
            "`speech_recognition_unavailable`: the server's Whisper model cannot be "
            "loaded; send words or transcript instead. Retry after `Retry-After` seconds."
        ),
        "headers": BUSY_RESPONSES[503]["headers"],
    },
}


@router.post(
    "/audio-to-iml",
    response_model=AudioToIMLResponse,
    responses={**ERROR_RESPONSES, **_UNSUPPORTED_MEDIA, **_AUDIO_RESPONSES},
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
    words: Annotated[UploadFile | str | None, Form(description=_WORDS_DESCRIPTION)] = None,
    transcript: Annotated[
        UploadFile | str | None, Form(description=_TRANSCRIPT_DESCRIPTION)
    ] = None,
    profile: Annotated[UploadFile | str | None, Form(description=_PROFILE_DESCRIPTION)] = None,
    calibration: Annotated[
        list[UploadFile | str] | None, File(description=_CALIBRATION_DESCRIPTION)
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

    Send multipart/form-data with the file in ``audio`` and, optionally,
    ``language``, ``words`` (word timings from any speech recogniser) or
    ``transcript``, a prosody ``profile``, and ``calibration`` recordings of
    the speaker. ``words``, ``transcript`` and ``profile`` may be sent as
    text fields or as files. Unreadable audio, or audio longer than the
    server's PP_MAX_AUDIO_SECONDS, is a 400 audio_processing_error.
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

    # Check the other fields before taking a worker's place.
    transcript_text = await _field_text(
        transcript, "transcript", settings.max_text_chars, "PP_MAX_TEXT_CHARS"
    )
    profile_text = await _field_text(
        profile, "profile", settings.max_text_chars, "PP_MAX_TEXT_CHARS"
    )
    alignments = await _words_field(words, settings)
    if alignments is not None and transcript_text is not None:
        raise _field_error("transcript", "Send either words or transcript, not both.")
    if transcript_text is not None:
        try:
            _checked_text(transcript_text, "transcript")
        except ValueError as exc:
            raise _field_error("transcript", str(exc)) from None
    calibration_files = _calibration_files(calibration)
    if len(calibration_files) > MAX_CALIBRATION_FILES:
        raise _field_error(
            "calibration",
            f"At most {MAX_CALIBRATION_FILES} calibration recordings may be sent; "
            f"got {len(calibration_files)}.",
        )
    options = _worker.ConvertOptions(
        max_duration_s=settings.max_audio_seconds,
        language=language,
        words=alignments,
        transcript=transcript_text,
        profile=_profile_field(profile_text),
        stt_model=settings.stt_model,
    )

    with jobs.admit(), tempfile.TemporaryDirectory(prefix="prosody-protocol-upload-") as tmp:
        path = Path(tmp) / f"upload{_upload_suffix(audio.filename)}"
        await run_in_threadpool(_save_upload, audio.file, path, settings.max_upload_bytes)
        # Drop the server's first copies of the uploads while the job waits for a worker.
        await audio.close()
        saved: list[tuple[str, str]] = []
        for number, upload in enumerate(calibration_files, start=1):
            copy = Path(tmp) / f"calibration{number}{_upload_suffix(upload.filename)}"
            await run_in_threadpool(_save_upload, upload.file, copy, settings.max_upload_bytes)
            await upload.close()
            name = (
                f"calibration {PurePath(upload.filename).name!r}"
                if upload.filename
                else f"calibration recording {number}"
            )
            saved.append((str(copy), name))
        result = await jobs.run(
            _worker.convert_audio, str(path), display_name, options, tuple(saved)
        )
    return AudioToIMLResponse(
        iml=result.iml,
        plain_text=IMLParser().to_plain_text(result.document),
        transcript_source=result.transcript_source,
        warnings=list(result.warnings),
        profile_matches=[
            ProfileMatchResponse(
                utterance=m.utterance,
                observed=m.observed,
                pattern=m.pattern,
                emotion=m.emotion,
                confidence=m.confidence,
                applied=m.applied,
            )
            for m in result.profile_matches
        ],
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


@router.post("/iml-to-prompt", response_model=PromptResponse, responses=ERROR_RESPONSES)
def iml_to_prompt(request: IMLToPromptRequest, settings: SettingsDep) -> PromptResponse:
    """Describe IML for a large language model.

    Returns the document as an annotated transcript (``context``), the
    system prompt that explains its notation, and both as chat
    ``messages`` for any chat-completion API. Malformed IML is a 400
    iml_parse_error; with ``strict``, a document with validation errors is a
    400 validation_error.
    """
    check_text_length(settings, iml=request.iml, instruction=request.instruction)
    doc = IMLParser().parse(request.iml)
    if request.strict:
        IMLValidator().validate(request.iml).raise_for_errors()
    options: dict[str, Any] = {
        "min_confidence": request.min_confidence,
        "include_numbers": request.include_numbers,
    }
    return PromptResponse(
        context=to_llm_context(doc, **options),
        system_prompt=SYSTEM_PROMPT,
        messages=build_messages(doc, request.instruction, **options),
    )
