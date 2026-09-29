"""Tests for prosody_protocol.alignment.

Covers:
- Each adapter on a payload shaped like the service's documented output:
  openai-whisper, the OpenAI transcription API (verbose_json), faster-whisper
  and WhisperX segments, Deepgram, AssemblyAI and Google Cloud STT
- SDK response objects with the same field names as the JSON
- Word cleanup: whitespace stripped, punctuated forms preferred, empty tokens dropped
- Time validation (negative, reversed, out of order, non-finite, wrong unit)
  and rounding to whole milliseconds; numpy scalar times
- load_word_timings shape detection from dicts, lists and files (UTF-8,
  UTF-8 with BOM, UTF-16, UTF-32)
- parse_word_timings parses JSON text and never reads a file, in time
  linear in its size (punctuation restoration included)
- Speaker labels from diarization are kept (Deepgram, AssemblyAI, Google,
  WhisperX, records)
- The adapters' output drives the assembler: punctuation splits utterances
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import time
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from prosody_protocol._types import SpanFeatures, WordAlignment
from prosody_protocol.alignment import (
    from_assemblyai,
    from_deepgram,
    from_google,
    from_records,
    from_seconds,
    from_whisper,
    load_word_timings,
    parse_word_timings,
)
from prosody_protocol.assembler import IMLAssembler
from prosody_protocol.exceptions import ConversionError, ProsodyProtocolError
from prosody_protocol.parser import IMLParser

AUDIO_DIR = Path(__file__).parent / "fixtures" / "audio"

# The JFK inaugural sentence, with plausible word times (seconds).
JFK_TEXT = (
    "And so, my fellow Americans, ask not what your country can do for you. "
    "Ask what you can do for your country."
)
JFK_WORDS = [
    ("And", 0.30, 0.52), ("so,", 0.52, 0.92), ("my", 1.26, 1.48), ("fellow", 1.48, 1.84),
    ("Americans,", 1.84, 2.70), ("ask", 3.24, 3.62), ("not", 3.96, 4.36), ("what", 5.16, 5.44),
    ("your", 5.44, 5.64), ("country", 5.64, 6.10), ("can", 6.10, 6.38), ("do", 6.38, 6.58),
    ("for", 6.58, 6.78), ("you.", 6.78, 7.12), ("Ask", 7.74, 8.04), ("what", 8.04, 8.28),
    ("you", 8.28, 8.46), ("can", 8.46, 8.66), ("do", 8.66, 8.84), ("for", 8.84, 9.02),
    ("your", 9.02, 9.20), ("country.", 9.20, 10.04),
]
JFK_EXPECTED = [WordAlignment(w, round(s * 1000), round(e * 1000)) for w, s, e in JFK_WORDS]


def _bare(word: str) -> str:
    """A word as APIs without punctuation give it."""
    return word.strip(".,")


# ---------------------------------------------------------------------------
# Vendor payloads
# ---------------------------------------------------------------------------


def openai_whisper_result() -> dict[str, Any]:
    """``whisper.load_model(...).transcribe(path, word_timestamps=True)``."""
    segments = []
    for seg_id, (lo, hi) in enumerate([(0, 14), (14, 22)]):
        words = JFK_WORDS[lo:hi]
        segments.append(
            {
                "id": seg_id,
                "seek": 0,
                "start": words[0][1],
                "end": words[-1][2],
                "text": "".join(f" {w}" for w, _, _ in words),
                "tokens": [50364 + i for i in range(len(words))],
                "temperature": 0.0,
                "avg_logprob": -0.21,
                "compression_ratio": 1.32,
                "no_speech_prob": 0.012,
                "words": [
                    {"word": f" {w}", "start": s, "end": e, "probability": 0.93}
                    for w, s, e in words
                ],
            }
        )
    return {"text": f" {JFK_TEXT}", "segments": segments, "language": "en"}


def openai_api_verbose_json() -> dict[str, Any]:
    """``POST /v1/audio/transcriptions`` with ``response_format=verbose_json``
    and ``timestamp_granularities[]=word`` -- words carry no punctuation."""
    return {
        "task": "transcribe",
        "language": "english",
        "duration": 10.5,
        "text": JFK_TEXT,
        "words": [{"word": _bare(w), "start": s, "end": e} for w, s, e in JFK_WORDS],
    }


def deepgram_response(*, diarize: bool = False) -> dict[str, Any]:
    """``POST /v1/listen?punctuate=true`` (pre-recorded)."""
    words = []
    for w, s, e in JFK_WORDS:
        word: dict[str, Any] = {
            "word": _bare(w).lower(),
            "start": s,
            "end": e,
            "confidence": 0.98,
            "punctuated_word": w,
        }
        if diarize:
            word.update({"speaker": 0, "speaker_confidence": 0.61})
        words.append(word)
    return {
        "metadata": {
            "transaction_key": "deprecated",
            "request_id": "5f3c7a1e-2d1b-4c55-9b11-0d6c3f3e8a01",
            "sha256": "154e291ecfa8be6ab8343560bcc109008fa7853eb5372533e8efdefc9b504c33",
            "created": "2024-05-02T10:14:21.911Z",
            "duration": 10.5,
            "channels": 1,
            "models": ["30089e05-99d1-4376-b32e-c263170674af"],
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {"transcript": JFK_TEXT, "confidence": 0.99, "words": words}
                    ]
                }
            ]
        },
    }


def assemblyai_transcript(status: str = "completed") -> dict[str, Any]:
    """``GET /v2/transcript/{id}`` -- word times in milliseconds."""
    done = status == "completed"
    return {
        "id": "6rlr37h6jk-7ab0-4d45-9e36-0c0a4dcd6d11",
        "language_model": "assemblyai_default",
        "acoustic_model": "assemblyai_default",
        "language_code": "en_us",
        "status": status,
        "audio_url": "https://example.org/jfk.wav",
        "text": JFK_TEXT if done else None,
        "words": [
            {"text": w, "start": round(s * 1000), "end": round(e * 1000),
             "confidence": 0.97, "speaker": None}
            for w, s, e in JFK_WORDS
        ] if done else None,
        "utterances": None,
        "confidence": 0.96 if done else None,
        "audio_duration": 11 if done else None,
        "punctuate": True,
        "format_text": True,
        "error": "Download error, unable to download the audio" if status == "error" else None,
    }


def _transcript(words: list[tuple[str, float, float]]) -> str:
    return " ".join(w for w, _, _ in words)


def _google_duration(seconds: float) -> str:
    return f"{seconds:.3f}s"


def google_response(*, diarize: bool = False) -> dict[str, Any]:
    """``POST /v1/speech:recognize`` with ``enableWordTimeOffsets`` and
    automatic punctuation. Two results; the first word has no ``startTime``
    because proto3 JSON leaves out zero durations."""

    def words(chunk: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
        out = []
        for w, s, e in chunk:
            word: dict[str, Any] = {"endTime": _google_duration(e), "word": w}
            if s > 0:
                word["startTime"] = _google_duration(s)
            out.append(word)
        return out

    shifted = [(w, s - 0.30, e - 0.30) for w, s, e in JFK_WORDS]  # first word at 0 s
    first, second = shifted[:14], shifted[14:]
    results = [
        {
            "alternatives": [
                {"transcript": _transcript(first), "confidence": 0.94, "words": words(first)}
            ],
            "resultEndTime": "6.900s",
            "languageCode": "en-us",
        },
        {
            "alternatives": [
                {"transcript": _transcript(second), "confidence": 0.95, "words": words(second)}
            ],
            "resultEndTime": "9.800s",
            "languageCode": "en-us",
        },
    ]
    if diarize:
        tagged = [{**word, "speakerTag": 1} for word in words(shifted)]
        results.append(
            {"alternatives": [{"transcript": "", "words": tagged}], "languageCode": "en-us"}
        )
    return {"results": results, "totalBilledTime": "11s", "requestId": "7294625153912946427"}


GOOGLE_EXPECTED = [
    WordAlignment(a.word, a.start_ms - 300, a.end_ms - 300) for a in JFK_EXPECTED
]


def _spoken_by(speaker: str, words: list[WordAlignment]) -> list[WordAlignment]:
    return [dataclasses.replace(w, speaker=speaker) for w in words]


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class TestFromSeconds:
    def test_converts_to_whole_milliseconds(self) -> None:
        assert from_seconds(" country.", 5.64, 6.1) == WordAlignment("country.", 5640, 6100)

    def test_rounds_to_nearest_millisecond(self) -> None:
        assert from_seconds("a", 1.2344, 1.2346) == WordAlignment("a", 1234, 1235)

    def test_empty_word_is_an_error(self) -> None:
        with pytest.raises(ConversionError, match="empty"):
            from_seconds("  ", 0.0, 0.5)


class TestFromWhisper:
    def test_openai_whisper_transcribe_result(self) -> None:
        # Leading spaces stripped, words of both segments in order.
        assert from_whisper(openai_whisper_result()) == JFK_EXPECTED

    def test_openai_api_verbose_json_restores_punctuation(self) -> None:
        # The API's words have no punctuation; it is taken from "text".
        result = from_whisper(openai_api_verbose_json())
        assert result == JFK_EXPECTED

    def test_api_punctuation_survives_a_mismatch(self) -> None:
        payload = openai_api_verbose_json()
        payload["words"][3]["word"] = "yellow"  # misheard in the word list only
        del payload["words"][9]  # a word missing from the word list
        words = [a.word for a in from_whisper(payload)]
        assert words[:5] == ["And", "so,", "my", "yellow", "Americans,"]
        assert words[12] == "you."
        assert words[-1] == "country."

    def test_api_text_is_optional(self) -> None:
        payload = openai_api_verbose_json()
        del payload["text"]
        assert [a.word for a in from_whisper(payload)][:2] == ["And", "so"]

    def test_segments_without_word_timestamps_are_an_error(self) -> None:
        result = openai_whisper_result()
        for segment in result["segments"]:
            del segment["words"]
        with pytest.raises(ConversionError, match="word_timestamps=True"):
            from_whisper(result)

    def test_silence_gives_no_words(self) -> None:
        assert from_whisper({"text": "", "segments": [], "language": "en"}) == []

    def test_faster_whisper_segment_objects(self) -> None:
        @dataclasses.dataclass
        class Word:  # faster_whisper.transcribe.Word
            start: float
            end: float
            word: str
            probability: float

        @dataclasses.dataclass
        class Segment:  # faster_whisper.transcribe.Segment (abridged)
            id: int
            start: float
            end: float
            text: str
            words: list[Word] | None

        segments = (
            Segment(i, 0.0, 0.0, "", [Word(s, e, f" {w}", 0.9) for w, s, e in chunk])
            for i, chunk in enumerate([JFK_WORDS[:14], JFK_WORDS[14:]])
        )
        # model.transcribe() returns a generator of segments.
        assert from_whisper(segments) == JFK_EXPECTED

    def test_whisperx_unaligned_words_join_a_neighbour(self) -> None:
        result = {
            "segments": [
                {
                    "start": 0.5,
                    "end": 2.9,
                    "text": " 3 people came in 1990.",
                    "words": [
                        {"word": "3"},  # WhisperX could not align these
                        {"word": "people", "start": 0.5, "end": 0.9, "score": 0.8},
                        {"word": "came", "start": 1.0, "end": 1.3, "score": 0.9},
                        {"word": "in", "start": 1.4, "end": 1.5, "score": 0.7},
                        {"word": "1990."},
                    ],
                }
            ],
            "word_segments": [],
        }
        assert from_whisper(result) == [
            WordAlignment("3 people", 500, 900),
            WordAlignment("came", 1000, 1300),
            WordAlignment("in 1990.", 1400, 1500),
        ]

    def test_whisperx_unaligned_word_opening_a_sentence_joins_the_next_word(self) -> None:
        # Gluing "1990" onto "came." would hide the sentence end from the assembler.
        result = {
            "segments": [
                {
                    "text": " Three people came.",
                    "words": [
                        {"word": "Three", "start": 0.2, "end": 0.5},
                        {"word": "people", "start": 0.5, "end": 0.9},
                        {"word": "came.", "start": 1.0, "end": 1.3},
                    ],
                },
                {
                    "text": " 1990 was good. 2000 was not.",
                    "words": [
                        {"word": "1990"},  # opens the segment
                        {"word": "was", "start": 2.1, "end": 2.3},
                        {"word": "good.", "start": 2.3, "end": 2.7},
                        {"word": "2000"},  # opens a sentence mid-segment
                        {"word": "was", "start": 3.4, "end": 3.6},
                        {"word": "not."},  # ends it: joins the word before
                    ],
                },
            ]
        }
        assert [a.word for a in from_whisper(result)] == [
            "Three", "people", "came.", "1990 was", "good.", "2000 was not.",
        ]
        assert from_whisper(result)[3] == WordAlignment("1990 was", 2100, 2300)

    def test_whisperx_unaligned_words_at_the_very_end_are_kept(self) -> None:
        result = {"segments": [{"words": [
            {"word": "Good.", "start": 0.1, "end": 0.4}, {"word": "2024"},
        ]}]}
        assert from_whisper(result) == [WordAlignment("Good. 2024", 100, 400)]

    def test_whisperx_without_any_times_is_an_error(self) -> None:
        with pytest.raises(ConversionError, match="no word has start and end times"):
            from_whisper({"segments": [{"words": [{"word": "1990"}]}]})

    def test_openai_sdk_response_object(self) -> None:
        @dataclasses.dataclass
        class TranscriptionWord:
            word: str
            start: float
            end: float

        @dataclasses.dataclass
        class TranscriptionVerbose:
            text: str
            language: str
            duration: float
            words: list[TranscriptionWord] | None
            segments: list[Any] | None = None

        payload = openai_api_verbose_json()
        response = TranscriptionVerbose(
            text=payload["text"],
            language="english",
            duration=10.5,
            words=[TranscriptionWord(**w) for w in payload["words"]],
        )
        assert from_whisper(response) == JFK_EXPECTED


class TestFromDeepgram:
    def test_prefers_punctuated_word(self) -> None:
        assert from_deepgram(deepgram_response()) == JFK_EXPECTED

    def test_falls_back_to_word(self) -> None:
        response = deepgram_response()
        for word in response["results"]["channels"][0]["alternatives"][0]["words"]:
            del word["punctuated_word"]
        assert [a.word for a in from_deepgram(response)][:2] == ["and", "so"]

    def test_second_channel(self) -> None:
        response = deepgram_response()
        channels = response["results"]["channels"]
        channels.append(json.loads(json.dumps(channels[0])))
        channels[1]["alternatives"][0]["words"] = channels[1]["alternatives"][0]["words"][:3]
        assert len(from_deepgram(response, channel=1)) == 3

    def test_missing_channel_is_an_error(self) -> None:
        with pytest.raises(ConversionError, match="no channel 2"):
            from_deepgram(deepgram_response(), channel=2)

    def test_not_a_deepgram_response(self) -> None:
        with pytest.raises(ConversionError, match="results.channels"):
            from_deepgram({"text": "hello"})


class TestFromAssemblyAI:
    def test_completed_transcript_in_milliseconds(self) -> None:
        assert from_assemblyai(assemblyai_transcript()) == JFK_EXPECTED

    @pytest.mark.parametrize("status", ["queued", "processing"])
    def test_unfinished_transcript_is_an_error(self, status: str) -> None:
        with pytest.raises(ConversionError, match=f"status '{status}'"):
            from_assemblyai(assemblyai_transcript(status))

    def test_failed_transcript_reports_its_error(self) -> None:
        with pytest.raises(ConversionError, match="unable to download"):
            from_assemblyai(assemblyai_transcript("error"))

    def test_sdk_transcript_object(self) -> None:
        class TranscriptStatus(str, Enum):  # assemblyai.TranscriptStatus
            queued = "queued"
            completed = "completed"

        @dataclasses.dataclass
        class Word:  # assemblyai.Word
            text: str
            start: int
            end: int
            confidence: float

        @dataclasses.dataclass
        class Transcript:  # assemblyai.Transcript (abridged)
            id: str
            status: TranscriptStatus
            words: list[Word] | None

        payload = assemblyai_transcript()
        transcript = Transcript(
            id=payload["id"],
            status=TranscriptStatus.completed,
            words=[Word(w["text"], w["start"], w["end"], 0.9) for w in payload["words"]],
        )
        assert from_assemblyai(transcript) == JFK_EXPECTED
        queued = Transcript(id="x", status=TranscriptStatus.queued, words=None)
        with pytest.raises(ConversionError, match="'queued'"):
            from_assemblyai(queued)


class TestFromGoogle:
    def test_v1_json_duration_strings(self) -> None:
        assert from_google(google_response()) == GOOGLE_EXPECTED

    def test_diarization_summary_result_is_not_duplicated(self) -> None:
        # The summary's speaker tags are kept as speaker labels.
        assert from_google(google_response(diarize=True)) == _spoken_by("1", GOOGLE_EXPECTED)

    def test_v2_offsets(self) -> None:
        response = {
            "results": [
                {
                    "alternatives": [
                        {
                            "transcript": "hello world",
                            "words": [
                                {"startOffset": "0.100s", "endOffset": "0.500s", "word": "Hello"},
                                {"startOffset": "0.600s", "endOffset": "1.100s", "word": "world."},
                            ],
                        }
                    ],
                    "resultEndOffset": "1.200s",
                }
            ]
        }
        assert from_google(response) == [
            WordAlignment("Hello", 100, 500),
            WordAlignment("world.", 600, 1100),
        ]

    def test_diarization_with_v2_speaker_labels(self) -> None:
        response = google_response(diarize=True)
        for word in response["results"][-1]["alternatives"][0]["words"]:
            word["speakerLabel"] = str(word.pop("speakerTag"))
        assert from_google(response) == _spoken_by("1", GOOGLE_EXPECTED)

    def test_summary_rule_needs_speaker_labels(self) -> None:
        # Without speaker labels the repeated words are not taken for a
        # diarization summary; nothing is dropped silently.
        response = google_response(diarize=True)
        for word in response["results"][-1]["alternatives"][0]["words"]:
            del word["speakerTag"]
        with pytest.raises(ConversionError, match="time order"):
            from_google(response)

    def test_transcripts_without_word_offsets_are_an_error(self) -> None:
        # enableWordTimeOffsets is off by default: the words would be lost.
        response = {
            "results": [
                {"alternatives": [{"transcript": "And so my fellow Americans",
                                   "confidence": 0.9}]}
            ]
        }
        with pytest.raises(ConversionError, match="enableWordTimeOffsets"):
            from_google(response)

    @pytest.mark.parametrize(
        "response",
        [
            {"totalBilledTime": "3s", "requestId": "7294625153912946427"},
            {"metadata": {"totalBilledDuration": "3s"}},
            {"results": []},
            {"results": [{"alternatives": [{"transcript": ""}]}]},
        ],
        ids=["v1-silence", "v2-silence", "empty-results", "empty-transcript"],
    )
    def test_no_speech_gives_no_words(self, response: dict[str, Any]) -> None:
        assert from_google(response) == []

    def test_separate_channels(self) -> None:
        # enableSeparateRecognitionPerChannel on a stereo call: both channels
        # start near 0 s, and neither may be dropped or interleaved silently.
        def result(tag: int, words: list[tuple[str, float, float]]) -> dict[str, Any]:
            return {
                "alternatives": [{
                    "transcript": _transcript(words),
                    "words": [
                        {"startTime": f"{s:.3f}s", "endTime": f"{e:.3f}s", "word": w}
                        for w, s, e in words
                    ],
                }],
                "channelTag": tag,
            }

        agent = [("Hello,", 0.2, 0.5), ("how", 0.5, 0.7), ("can", 0.7, 0.9), ("I", 0.9, 1.0),
                 ("help?", 1.0, 1.5)]
        caller = [("My", 0.1, 0.3), ("account", 0.3, 0.8), ("is", 0.8, 0.9),
                  ("locked.", 0.9, 1.4)]
        response = {"results": [result(1, agent), result(2, caller)]}
        with pytest.raises(ConversionError, match=r"channels \[1, 2\].*channel_tag"):
            from_google(response)
        assert [a.word for a in from_google(response, channel_tag=1)] == [w for w, _, _ in agent]
        assert [a.word for a in from_google(response, channel_tag=2)] == [w for w, _, _ in caller]
        with pytest.raises(ConversionError, match="no results for channel_tag 3"):
            from_google(response, channel_tag=3)

    def test_python_client_timedeltas(self) -> None:
        @dataclasses.dataclass
        class WordInfo:  # google.cloud.speech.WordInfo as the client returns it
            word: str
            start_time: timedelta
            end_time: timedelta

        response = {
            "results": [
                {"alternatives": [{"words": [
                    WordInfo("how", timedelta(0), timedelta(seconds=0.3)),
                    WordInfo("old", timedelta(seconds=0.3), timedelta(seconds=0.6)),
                    WordInfo("is", timedelta(seconds=0.6), timedelta(seconds=0.8)),
                ]}]}
            ]
        }
        assert from_google(response) == [
            WordAlignment("how", 0, 300),
            WordAlignment("old", 300, 600),
            WordAlignment("is", 600, 800),
        ]


class TestFromRecords:
    def test_seconds_by_default(self) -> None:
        records = [{"word": w, "start": s, "end": e} for w, s, e in JFK_WORDS]
        assert from_records(records) == JFK_EXPECTED

    def test_custom_keys_and_milliseconds(self) -> None:
        records = [{"token": " hi", "t0": 120, "t1": 480}, {"token": "there", "t0": 500, "t1": 900}]
        assert from_records(records, word_key="token", start_key="t0", end_key="t1", unit="ms") == [
            WordAlignment("hi", 120, 480),
            WordAlignment("there", 500, 900),
        ]

    def test_csv_rows_with_string_times(self) -> None:
        text = 'word,start,end\nAnd,0.30,0.52\n"so,",0.52,0.92\nmy,1.26,1.48\n'
        assert from_records(csv.DictReader(io.StringIO(text))) == JFK_EXPECTED[:3]

    def test_missing_key_is_an_error(self) -> None:
        with pytest.raises(ConversionError, match="record 1 has no 'end'"):
            from_records([{"word": "a", "start": 0, "end": 1}, {"word": "b", "start": 1}])

    def test_unknown_unit(self) -> None:
        with pytest.raises(ValueError, match="unit"):
            from_records([], unit="min")  # type: ignore[arg-type]

    def test_speaker_field(self) -> None:
        records = [
            {"word": "Hi.", "start": 0.1, "end": 0.4, "speaker": "agent"},
            {"word": "Hello.", "start": 0.9, "end": 1.3, "speaker": 2},
            {"word": "Bye.", "start": 1.5, "end": 1.9},
        ]
        assert [w.speaker for w in from_records(records)] == ["agent", "2", None]
        assert [w.speaker for w in from_records(records, speaker_key=None)] == [None] * 3


# ---------------------------------------------------------------------------
# Speaker labels
# ---------------------------------------------------------------------------


class TestSpeakerLabels:
    """Diarization labels are kept, so each speaker gets their own baseline."""

    def test_deepgram_speakers(self) -> None:
        response = deepgram_response(diarize=True)
        words = response["results"]["channels"][0]["alternatives"][0]["words"]
        for word in words[14:]:
            word["speaker"] = 1
        result = from_deepgram(response)
        assert [w.speaker for w in result] == ["0"] * 14 + ["1"] * 8
        assert [(w.word, w.start_ms, w.end_ms) for w in result] == [
            (w.word, w.start_ms, w.end_ms) for w in JFK_EXPECTED
        ]

    def test_assemblyai_speakers(self) -> None:
        transcript = assemblyai_transcript()
        for index, word in enumerate(transcript["words"]):
            word["speaker"] = "A" if index < 14 else "B"
        assert [w.speaker for w in from_assemblyai(transcript)] == ["A"] * 14 + ["B"] * 8

    def test_google_v1_speaker_tags(self) -> None:
        response = google_response(diarize=True)
        for word in response["results"][-1]["alternatives"][0]["words"][14:]:
            word["speakerTag"] = 2
        assert [w.speaker for w in from_google(response)] == ["1"] * 14 + ["2"] * 8

    def test_google_python_client_unset_tag_is_no_speaker(self) -> None:
        # The Python client reports speaker_tag=0 on words without diarization.
        words = [
            {"word": "Hello", "start_time": timedelta(seconds=0.1),
             "end_time": timedelta(seconds=0.5), "speaker_tag": 0},
        ]
        response = {"results": [{"alternatives": [{"transcript": "Hello", "words": words}]}]}
        assert from_google(response) == [WordAlignment("Hello", 100, 500)]

    def test_whisperx_word_and_segment_speakers(self) -> None:
        result = {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "words": [
                        {"word": "Hi,", "start": 0.1, "end": 0.4, "speaker": "SPEAKER_00"},
                        {"word": "there.", "start": 0.45, "end": 0.8},  # no word label
                    ],
                },
                {
                    "speaker": "SPEAKER_01",
                    "words": [{"word": "Hello.", "start": 1.2, "end": 1.6,
                               "speaker": "SPEAKER_01"}],
                },
            ]
        }
        assert [w.speaker for w in from_whisper(result)] == [
            "SPEAKER_00", "SPEAKER_00", "SPEAKER_01"
        ]

    def test_word_alignment_round_trip(self) -> None:
        words = [WordAlignment("Hi.", 100, 400, "A"), WordAlignment("Yes.", 900, 1200)]
        assert load_word_timings([dataclasses.asdict(w) for w in words]) == words
        assert load_word_timings(words) == words

    def test_repr_shows_a_speaker_only_when_there_is_one(self) -> None:
        assert repr(WordAlignment("Hi.", 100, 400)) == (
            "WordAlignment(word='Hi.', start_ms=100, end_ms=400)"
        )
        assert repr(WordAlignment("Hi.", 100, 400, "A")) == (
            "WordAlignment(word='Hi.', start_ms=100, end_ms=400, speaker='A')"
        )

    @pytest.mark.parametrize("speaker", [1.5, True, ["A"], {"id": 1}])
    def test_invalid_speaker_is_an_error(self, speaker: Any) -> None:
        with pytest.raises(ConversionError, match="speaker must be a string or an integer"):
            from_records([{"word": "Hi", "start": 0.1, "end": 0.4, "speaker": speaker}])


# ---------------------------------------------------------------------------
# Cleanup and validation
# ---------------------------------------------------------------------------


class TestWordCleanup:
    def test_empty_and_whitespace_tokens_are_dropped(self) -> None:
        records = [
            {"word": " Hello", "start": 0.0, "end": 0.4},
            {"word": " ", "start": 0.4, "end": 0.4},
            {"word": "", "start": -1, "end": -2},  # dropped before its times are checked
            {"word": "world. ", "start": 0.5, "end": 0.9},
        ]
        assert from_records(records) == [
            WordAlignment("Hello", 0, 400),
            WordAlignment("world.", 500, 900),
        ]

    def test_word_must_be_a_string(self) -> None:
        with pytest.raises(ConversionError, match="word 0 must be a string"):
            from_records([{"word": 5, "start": 0, "end": 1}])


class TestTimeValidation:
    @pytest.mark.parametrize(
        ("start", "end", "message"),
        [
            (-0.1, 0.2, "negative"),
            (0.5, 0.2, "before it starts"),
            (float("nan"), 0.2, "finite"),
            (0.1, float("inf"), "finite"),
            (True, 0.2, "must be a number"),
            ("soon", 0.2, "must be a number"),
        ],
    )
    def test_invalid_times(self, start: Any, end: Any, message: str) -> None:
        with pytest.raises(ConversionError, match=message) as info:
            from_records([{"word": "late", "start": start, "end": end}])
        assert "'late'" in str(info.value)  # the error names the word

    def test_numpy_scalar_times(self) -> None:
        np = pytest.importorskip("numpy")
        records = [
            {"word": "a", "start": np.int64(100), "end": np.int64(300)},
            {"word": "b", "start": np.float32(350.4), "end": np.float64(610.6)},
        ]
        assert from_records(records, unit="ms") == [
            WordAlignment("a", 100, 300),
            WordAlignment("b", 350, 611),
        ]
        with pytest.raises(ConversionError, match="must be a number"):
            from_records([{"word": "a", "start": np.bool_(True), "end": 1}])

    def test_out_of_order_words(self) -> None:
        records = [
            {"word": "one", "start": 1.0, "end": 1.3},
            {"word": "two", "start": 0.4, "end": 0.7},
        ]
        with pytest.raises(ConversionError, match="word 1 .'two'. .*time order"):
            from_records(records)

    def test_overlapping_words_in_order_are_kept(self) -> None:
        # Recognisers' word boundaries may overlap slightly; order is what matters.
        records = [
            {"word": "one", "start": 1.0, "end": 1.35},
            {"word": "two", "start": 1.3, "end": 1.7},
        ]
        assert [a.start_ms for a in from_records(records)] == [1000, 1300]

    def test_milliseconds_read_as_seconds_are_detected(self) -> None:
        records = [{"word": w, "start": s * 1000, "end": e * 1000} for w, s, e in JFK_WORDS]
        with pytest.raises(ConversionError, match="probably in milliseconds"):
            from_records(records)

    def test_seconds_read_as_milliseconds_are_detected(self) -> None:
        payload = assemblyai_transcript()
        for word, (_, s, e) in zip(payload["words"], JFK_WORDS, strict=True):
            word["start"], word["end"] = s, e
        with pytest.raises(ConversionError, match="probably in seconds"):
            from_assemblyai(payload)

    def test_errors_are_sdk_errors(self) -> None:
        assert issubclass(ConversionError, ProsodyProtocolError)


# ---------------------------------------------------------------------------
# load_word_timings
# ---------------------------------------------------------------------------


class TestLoadWordTimings:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (openai_whisper_result(), JFK_EXPECTED),
            (openai_api_verbose_json(), JFK_EXPECTED),
            (deepgram_response(diarize=True), _spoken_by("0", JFK_EXPECTED)),
            (assemblyai_transcript(), JFK_EXPECTED),
            (google_response(), GOOGLE_EXPECTED),
            (openai_whisper_result()["segments"], JFK_EXPECTED),
            (assemblyai_transcript()["words"], JFK_EXPECTED),
            (deepgram_response()["results"]["channels"][0]["alternatives"][0]["words"],
             JFK_EXPECTED),
            ([{"word": w, "start": s, "end": e} for w, s, e in JFK_WORDS], JFK_EXPECTED),
            ([dataclasses.asdict(a) for a in JFK_EXPECTED], JFK_EXPECTED),
            (JFK_EXPECTED, JFK_EXPECTED),
            ([], []),
        ],
        ids=[
            "openai-whisper", "openai-api", "deepgram", "assemblyai", "google",
            "segment-list", "assemblyai-words", "deepgram-words", "records-seconds",
            "records-ms", "word-alignments", "empty",
        ],
    )
    def test_detects_shape(self, payload: Any, expected: list[WordAlignment]) -> None:
        assert load_word_timings(payload) == expected

    def test_json_file_by_path_and_by_string(self, tmp_path: Path) -> None:
        path = tmp_path / "deepgram.json"
        path.write_text(json.dumps(deepgram_response()), encoding="utf-8")
        assert load_word_timings(path) == JFK_EXPECTED
        assert load_word_timings(str(path)) == JFK_EXPECTED

    @pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
    def test_json_file_encodings(self, tmp_path: Path, encoding: str) -> None:
        # Windows tools write a BOM; PowerShell's ">" writes UTF-16.
        path = tmp_path / "timings.json"
        path.write_text(json.dumps(assemblyai_transcript()), encoding=encoding)
        assert load_word_timings(path) == JFK_EXPECTED

    def test_undecodable_file(self, tmp_path: Path) -> None:
        path = tmp_path / "timings.json"
        path.write_bytes(b'{"words": ["\xff\xfe\x80"]}')
        with pytest.raises(ConversionError, match="not valid JSON"):
            load_word_timings(path)

    def test_repo_ground_truth_file(self) -> None:
        # tests/fixtures/audio/*.json: {"words": [{"word", "start_ms", "end_ms"}], ...}
        path = AUDIO_DIR / "speech_pauses.json"
        truth = json.loads(path.read_text(encoding="utf-8"))
        assert load_word_timings(path) == [
            WordAlignment(w["word"], w["start_ms"], w["end_ms"]) for w in truth["words"]
        ]

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text('{"words": [', encoding="utf-8")
        with pytest.raises(ConversionError, match="not valid JSON"):
            load_word_timings(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            load_word_timings(tmp_path / "missing.json")

    @pytest.mark.parametrize(
        "payload",
        [
            {"transcript": "hello"},
            [{"token": "a", "t": 1}],
            ["a", "b"],
            42,
            b"[]",
            # Google v2 BatchRecognize with inline results
            {"results": {"gs://bucket/call.wav": {"transcript": {"results": []}}}},
            # AWS Transcribe: a results object and a status, but neither
            # Deepgram nor AssemblyAI
            {
                "jobName": "call",
                "status": "COMPLETED",
                "results": {
                    "transcripts": [{"transcript": "Hello."}],
                    "items": [{"start_time": "0.1", "end_time": "0.5", "type": "pronunciation",
                               "alternatives": [{"content": "Hello"}]}],
                },
            },
        ],
        ids=["object", "records", "strings", "number", "bytes", "google-batch", "aws"],
    )
    def test_unrecognised_shapes(self, payload: Any) -> None:
        with pytest.raises(ConversionError, match="unrecognised|must be"):
            load_word_timings(payload)

    def test_google_response_without_speech(self) -> None:
        assert load_word_timings({"totalBilledTime": "3s", "requestId": "729462"}) == []


class TestParseWordTimings:
    def test_json_text(self) -> None:
        assert parse_word_timings(json.dumps(assemblyai_transcript())) == JFK_EXPECTED

    @pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-32"])
    def test_json_bytes(self, encoding: str) -> None:
        body = json.dumps(deepgram_response()).encode(encoding)
        assert parse_word_timings(body) == JFK_EXPECTED

    def test_text_with_a_byte_order_mark(self) -> None:
        assert parse_word_timings("\ufeff" + json.dumps(openai_whisper_result())) == JFK_EXPECTED

    def test_parsed_data(self) -> None:
        assert parse_word_timings(google_response()) == GOOGLE_EXPECTED

    @pytest.mark.parametrize(
        "text",
        [
            str(AUDIO_DIR / "speech_pauses.json"),  # a readable word timing file
            str(AUDIO_DIR),  # a directory
            "speech_pauses.json",
        ],
        ids=["file", "directory", "relative"],
    )
    def test_never_reads_a_file(self, text: str) -> None:
        # load_word_timings() reads the first path; parse_word_timings() must
        # treat it as (invalid) JSON text, so untrusted input cannot make it
        # read, probe or block on files.
        assert load_word_timings(AUDIO_DIR / "speech_pauses.json")
        with pytest.raises(ConversionError, match="not valid JSON"):
            parse_word_timings(text)

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("", "not valid JSON"),
            ("   ", "not valid JSON"),
            ('{"words": [', "not valid JSON"),
            ("[" * 100_000, "not valid JSON"),  # too deeply nested to parse
            ("42", "must be a JSON object or array"),
            ("null", "must be a JSON object or array"),
            ('"words"', "must be a JSON object or array"),
        ],
        ids=["empty", "blank", "truncated", "deep", "number", "null", "string"],
    )
    def test_invalid_text(self, text: str, message: str) -> None:
        with pytest.raises(ConversionError, match=message):
            parse_word_timings(text)

    def test_punctuation_restoration_is_linear(self) -> None:
        """Words that differ from the text used to be matched with difflib in
        time growing with the product of their numbers: 28,000 repeated words
        with one extra token at each end (1.1 MB of JSON) took about a minute."""
        count = 30_000
        body = json.dumps({
            "text": "b " + "a " * count + "c",
            "words": [{"word": "a", "start": i * 0.1, "end": i * 0.1 + 0.05}
                      for i in range(count)],
        })
        started = time.perf_counter()
        words = parse_word_timings(body)
        assert time.perf_counter() - started < 2.0
        assert len(words) == count

    def test_large_input_still_restores_punctuation(self) -> None:
        """Beyond difflib's budget the linear matching still restores the
        punctuation around words missing from, or misheard in, the word list."""
        repeats = 60  # 1,320 words: past the budget, so the linear matching is used
        text = " ".join([JFK_TEXT] * repeats)
        expected = [w for _ in range(repeats) for w, _, _ in JFK_WORDS]
        bare = [_bare(w) for w in expected]
        bare[3] = "yellow"  # misheard: keeps its own form
        del bare[500]  # a word missing from the word list
        bare.insert(1000, "um")  # a word missing from the text
        payload = {
            "text": text,
            "words": [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.3}
                      for i, w in enumerate(bare)],
        }
        restored = [w.word for w in from_whisper(payload)]
        want = list(expected)
        want[3] = "yellow"
        del want[500]
        want.insert(1000, "um")
        assert restored == want


# ---------------------------------------------------------------------------
# Into the pipeline
# ---------------------------------------------------------------------------


def test_restored_punctuation_lets_the_assembler_split_sentences() -> None:
    """OpenAI API word timings become IML with the real, punctuated text."""
    words = from_whisper(openai_api_verbose_json())
    features = [SpanFeatures(start_ms=w.start_ms, end_ms=w.end_ms, text=w.word) for w in words]
    doc = IMLAssembler().assemble(words, features, [])
    assert IMLParser().to_plain_text(doc) == JFK_TEXT
    assert len(doc.utterances) == 2  # split at "you."
