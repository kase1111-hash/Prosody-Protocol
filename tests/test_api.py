"""Tests for the Prosody Protocol REST API (Phase 9).

Covers:
- Acceptance criteria: correct status codes, structured errors, OpenAPI spec
- Health endpoint: capabilities and limits
- Validation endpoint: valid/invalid IML
- Text-to-IML endpoint
- IML-to-SSML endpoint
- Synthesize endpoint: WAV audio response, duration cap, engine/voice
- Audio-to-IML endpoint: file upload, language field, unreadable audio,
  word timings, transcript and prosody profile fields
- IML-to-prompt endpoint: annotated transcript and chat messages for LLMs
- Error handling: malformed input returns 400 not 500
- Request limits: body size (including chunked bodies), text length, rate limit
- Worker processes: the event loop stays free during long conversions
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import operator
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
pytest.importorskip("numpy")
pytest.importorskip("parselmouth")

from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from prosody_protocol import IMLParser, Prosody
from prosody_protocol.exceptions import IMLValidationError
from prosody_protocol.server import _worker
from prosody_protocol.server.app import app, create_app
from prosody_protocol.server.config import Settings
from prosody_protocol.server.errors import APIError
from prosody_protocol.server.jobs import JobRunner
from prosody_protocol.server.middleware import RateLimitMiddleware
from prosody_protocol.server.routes.convert import _save_upload

AUDIO_FIXTURES = Path(__file__).parent / "fixtures" / "audio"
SPEECH_WAV = AUDIO_FIXTURES / "speech_pauses.wav"
PROFILE_FIXTURES = Path(__file__).parent / "fixtures" / "profiles"
EXAMPLES = Path(__file__).parent.parent / "examples"


def speech_words_json(shape: str = "records") -> str:
    """The exact word timings of SPEECH_WAV as JSON text, in the given *shape*."""
    words = json.loads((AUDIO_FIXTURES / "speech_pauses.json").read_text())["words"]
    if shape == "records":
        return json.dumps(words)
    assert shape == "whisper"
    return json.dumps({"segments": [{"words": [
        {"word": " " + w["word"], "start": w["start_ms"] / 1000, "end": w["end_ms"] / 1000}
        for w in words
    ]}]})


def pause_profile(boost: float) -> str:
    """A profile whose only mapping matches SPEECH_WAV (2 pauses in 6 word boundaries)."""
    return json.dumps({
        "profile_version": "0.1.0",
        "user_id": "caller_7",
        "prosody_mappings": [
            {
                "pattern": {"pause_frequency": "high"},
                "interpretation": {"emotion": "uncertain", "confidence_boost": boost},
            }
        ],
    })


def post_speech(http: TestClient, **data: Any) -> httpx.Response:
    """POST SPEECH_WAV to audio-to-iml with the form fields *data*."""
    files: dict[str, Any] = {"audio": ("speech.wav", SPEECH_WAV.read_bytes(), "audio/wav")}
    for name in [k for k, v in data.items() if isinstance(v, tuple)]:
        files[name] = data.pop(name)
    return http.post("/v1/convert/audio-to-iml", files=files, data=data)

needs_espeak = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="needs espeak-ng")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """A client for an app with the default settings except the rate limit.

    Shared by the module, so its worker processes start once; its lifespan
    stops them at the end. Without a rate limit, the number of tests using
    it cannot push a later one over PP_RATE_LIMIT.
    """
    application = create_app(Settings(rate_limit_per_minute=0))
    with TestClient(application, raise_server_exceptions=False) as http:
        yield http


def make_client(**settings: Any) -> TestClient:
    """A client for an app with explicit settings (rate limit off unless given)."""
    settings.setdefault("rate_limit_per_minute", 0)
    return TestClient(create_app(Settings(**settings)), raise_server_exceptions=False)


def long_speech_wav(path: Path, seconds: float) -> Path:
    """Write *seconds* of real speech (the speech fixture repeated) to *path*."""
    with wave.open(str(SPEECH_WAV)) as src:
        params = src.getparams()
        frames = src.readframes(src.getnframes())
    repeats = int(seconds * params.framerate / params.nframes) + 1
    with wave.open(str(path), "wb") as out:
        out.setparams(params)
        out.writeframes(frames * repeats)
    return path


def multipart_body(field: str, filename: str, payload: bytes) -> tuple[bytes, str]:
    boundary = "prosodyboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def chunks(data: bytes, size: int = 65536) -> Iterator[bytes]:
    """A generator body: sent with Transfer-Encoding: chunked, no Content-Length."""
    for i in range(0, len(data), size):
        yield data[i : i + size]


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self) -> None:
        resp = TestClient(app).get("/v1/health")  # The app ASGI servers load.
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_has_version_string(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        data = resp.json()
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_health_reports_optional_backends(self, client: TestClient) -> None:
        capabilities = client.get("/v1/health").json()["capabilities"]
        assert capabilities == {
            "whisper": importlib.util.find_spec("whisper") is not None,
            "espeak_ng": shutil.which("espeak-ng") is not None,
            "ffmpeg": shutil.which("ffmpeg") is not None,
        }

    def test_health_reports_limits(self) -> None:
        limits = make_client(
            max_upload_size_mb=3, max_text_chars=1234, max_synth_seconds=45.0,
            max_words_chars=5678,
        ).get("/v1/health").json()["limits"]
        assert limits == {
            "max_upload_bytes": 3 * 1024 * 1024,
            "max_json_bytes": 24 * 1234 + 65536,
            "max_text_chars": 1234,
            "max_words_chars": 5678,
            "max_synth_seconds": 45.0,
            "max_audio_seconds": 600.0,
            "job_timeout_s": 900.0,
            "rate_limit_per_minute": 0,
        }


# ---------------------------------------------------------------------------
# Validation endpoint
# ---------------------------------------------------------------------------


class TestValidateEndpoint:
    def test_valid_iml(self, client: TestClient) -> None:
        resp = client.post("/v1/validate", json={"iml": "<utterance>Hello</utterance>"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["issues"] == []

    def test_invalid_iml_missing_confidence(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/validate",
            json={"iml": '<utterance emotion="happy">Hello</utterance>'},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["issues"]) > 0
        assert data["issues"][0]["severity"] == "error"
        assert data["issues"][0]["rule"]
        assert data["issues"][0]["message"]

    def test_valid_iml_with_emotion_and_confidence(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/validate",
            json={"iml": '<utterance emotion="happy" confidence="0.8">Hello</utterance>'},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True

    def test_malformed_xml_returns_structured_error(self, client: TestClient) -> None:
        """Malformed XML should return validation issues, not a 500."""
        resp = client.post(
            "/v1/validate",
            json={"iml": "<utterance>unclosed"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["issues"]) > 0

    def test_syntax_error_reports_line_and_column(self, client: TestClient) -> None:
        resp = client.post("/v1/validate", json={"iml": "<utterance>\n <prosody>x</utterance>"})
        issue = resp.json()["issues"][0]
        assert issue["rule"] == "V1"
        assert issue["line"] == 2
        assert isinstance(issue["column"], int)

    def test_missing_body_returns_422(self, client: TestClient) -> None:
        """Missing request body should return 422."""
        resp = client.post("/v1/validate")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Text-to-IML endpoint
# ---------------------------------------------------------------------------


class TestTextToIMLEndpoint:
    def test_simple_text(self, client: TestClient) -> None:
        resp = client.post("/v1/convert/text-to-iml", json={"text": "Hello world."})
        assert resp.status_code == 200
        data = resp.json()
        assert "<utterance" in data["iml"]
        assert data["plain_text"] == "Hello world."

    def test_text_with_context(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/convert/text-to-iml",
            json={"text": "That's GREAT!", "context": "Previous conversation."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "GREAT" in data["iml"]

    def test_context_is_passed_to_the_model(self, client: TestClient) -> None:
        text = "I cannot believe this happened again!"
        without = client.post("/v1/convert/text-to-iml", json={"text": text}).json()
        with_context = client.post(
            "/v1/convert/text-to-iml", json={"text": text, "context": "the user is frustrated"}
        ).json()
        assert with_context["iml"] != without["iml"]
        assert with_context["plain_text"] == without["plain_text"] == text

    def test_emphasis_in_output(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/convert/text-to-iml",
            json={"text": "Oh, that's GREAT."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert '<emphasis level="strong">' in data["iml"]

    def test_empty_text(self, client: TestClient) -> None:
        resp = client.post("/v1/convert/text-to-iml", json={"text": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert "iml" in data

    def test_missing_text_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/convert/text-to-iml", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# IML-to-SSML endpoint
# ---------------------------------------------------------------------------


class TestIMLToSSMLEndpoint:
    def test_basic_conversion(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/convert/iml-to-ssml",
            json={"iml": "<utterance>Hello world.</utterance>"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "<speak" in data["ssml"]
        assert "<s>" in data["ssml"]
        assert "Hello world." in data["ssml"]

    def test_prosody_mapped(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/convert/iml-to-ssml",
            json={"iml": '<utterance><prosody pitch="+10%">loud</prosody></utterance>'},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert 'pitch="+10%"' in data["ssml"]

    def test_malformed_iml_returns_400(self, client: TestClient) -> None:
        """Malformed IML should return 400, not 500."""
        resp = client.post(
            "/v1/convert/iml-to-ssml",
            json={"iml": "<utterance>unclosed"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_strict_rejects_invalid_iml_with_issues(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/convert/iml-to-ssml",
            json={
                "iml": '<utterance emotion="angry">Stop <pause/> now</utterance>',
                "strict": True,
            },
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"] == "validation_error"
        assert {"V3", "V5"} <= {issue["rule"] for issue in data["issues"]}

    def test_invalid_iml_converted_leniently_by_default(self, client: TestClient) -> None:
        """Spec 6.2: consumers degrade gracefully; clients of the old API keep working."""
        resp = client.post(
            "/v1/convert/iml-to-ssml",
            json={"iml": '<utterance emotion="angry">Stop now</utterance>'},
        )
        assert resp.status_code == 200
        assert "Stop now" in resp.json()["ssml"]


# ---------------------------------------------------------------------------
# Synthesize endpoint
# ---------------------------------------------------------------------------


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        frames: int = w.getnframes()
        rate: int = w.getframerate()
    return frames / rate


class TestSynthesizeEndpoint:
    def test_returns_wav_audio(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/synthesize",
            json={"iml": "<utterance>Hello.</utterance>"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        # WAV files start with "RIFF".
        assert resp.content[:4] == b"RIFF"

    def test_wav_has_content_disposition(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/synthesize",
            json={"iml": "<utterance>Hello.</utterance>"},
        )
        assert "attachment" in resp.headers.get("content-disposition", "")

    def test_malformed_iml_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/synthesize",
            json={"iml": "<utterance>unclosed"},
        )
        assert resp.status_code == 400

    def test_missing_body_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/synthesize")
        assert resp.status_code == 422

    def test_long_pause_rejected_by_duration_cap(self, client: TestClient) -> None:
        """A 70-byte request must not produce a 30-minute WAV."""
        resp = client.post(
            "/v1/synthesize",
            json={"iml": '<utterance>Hi<pause duration="1800000"/>there</utterance>'},
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "conversion_error"
        assert "max_duration_s=120" in resp.json()["detail"]

    def test_duration_cap_follows_setting(self) -> None:
        iml = '<utterance>Hi<pause duration="4000"/>there</utterance>'
        body = {"iml": iml, "engine": "tones"}
        capped = make_client(max_synth_seconds=3.0).post("/v1/synthesize", json=body)
        assert capped.status_code == 400
        assert "max_duration_s=3" in capped.json()["detail"]
        allowed = make_client(max_synth_seconds=10.0).post("/v1/synthesize", json=body)
        assert allowed.status_code == 200
        assert 4.0 < wav_seconds(allowed.content) < 10.0

    def test_negative_pause_is_never_a_500(self, client: TestClient) -> None:
        iml = '<utterance>a<pause duration="-500"/>b</utterance>'
        strict = client.post("/v1/synthesize", json={"iml": iml, "strict": True})
        assert strict.status_code == 400
        data = strict.json()
        assert data["error"] == "validation_error"
        assert [issue["rule"] for issue in data["issues"]] == ["V6"]
        assert data["issues"][0]["line"] == 1
        # Leniently, the invalid pause is ignored.
        lenient = client.post("/v1/synthesize", json={"iml": iml, "engine": "tones"})
        assert lenient.status_code == 200
        assert lenient.content[:4] == b"RIFF"

    def test_invalid_iml_synthesized_unless_strict(self, client: TestClient) -> None:
        iml = '<utterance emotion="angry">Stop now</utterance>'
        resp = client.post("/v1/synthesize", json={"iml": iml})
        assert resp.status_code == 200
        assert resp.content[:4] == b"RIFF"
        strict = client.post("/v1/synthesize", json={"iml": iml, "strict": True})
        assert strict.status_code == 400
        assert [issue["rule"] for issue in strict.json()["issues"]] == ["V3"]

    def test_tones_engine_reported_in_header(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/synthesize", json={"iml": "<utterance>Hello.</utterance>", "engine": "tones"}
        )
        assert resp.status_code == 200
        assert resp.headers["x-prosody-engine"] == "tones"

    @needs_espeak
    def test_auto_engine_uses_espeak_when_installed(self, client: TestClient) -> None:
        resp = client.post("/v1/synthesize", json={"iml": "<utterance>Hello.</utterance>"})
        assert resp.headers["x-prosody-engine"] == "espeak"

    @needs_espeak
    def test_default_voice_follows_document_language(self, client: TestClient) -> None:
        iml = '<iml version="0.1.0" language="fr-FR"><utterance>Bonjour à tous.</utterance></iml>'
        default = client.post("/v1/synthesize", json={"iml": iml})
        english = client.post(
            "/v1/synthesize", json={"iml": iml, "voice": "en_US-female-medium"}
        )
        assert default.status_code == english.status_code == 200
        assert default.content != english.content

    def test_unknown_engine_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/synthesize", json={"iml": "<utterance>Hi.</utterance>", "engine": "coqui"}
        )
        assert resp.status_code == 422

    def test_unknown_voice_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/synthesize", json={"iml": "<utterance>Hi.</utterance>", "voice": "klingon"}
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "conversion_error"
        assert "klingon" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Audio-to-IML endpoint (file upload)
# ---------------------------------------------------------------------------


def upload_dirs() -> set[str]:
    return {p for p in os.listdir(tempfile.gettempdir()) if p.startswith("prosody-protocol-upload")}


class TestAudioToIMLEndpoint:
    def test_upload_wav_returns_iml(self, client: TestClient) -> None:
        audio_file = AUDIO_FIXTURES / "tone_440hz.wav"
        with open(audio_file, "rb") as f:
            resp = client.post(
                "/v1/convert/audio-to-iml",
                files={"audio": ("test.wav", f, "audio/wav")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "<utterance" in data["iml"] or "utterance" in data["iml"]
        assert "plain_text" in data

    def test_upload_with_language(self, client: TestClient) -> None:
        """The README sends language as a multipart form field."""
        with open(SPEECH_WAV, "rb") as f:
            resp = client.post(
                "/v1/convert/audio-to-iml",
                files={"audio": ("test.wav", f, "audio/wav")},
                data={"language": "fr-FR"},
            )
        assert resp.status_code == 200
        assert 'language="fr-FR"' in resp.json()["iml"]

    def test_language_query_parameter_still_works(self, client: TestClient) -> None:
        with open(SPEECH_WAV, "rb") as f:
            resp = client.post(
                "/v1/convert/audio-to-iml",
                files={"audio": ("test.wav", f, "audio/wav")},
                params={"language": "de-DE"},
            )
        assert resp.status_code == 200
        assert 'language="de-DE"' in resp.json()["iml"]

    def test_conflicting_languages_return_422(self, client: TestClient) -> None:
        with open(SPEECH_WAV, "rb") as f:
            resp = client.post(
                "/v1/convert/audio-to-iml",
                files={"audio": ("test.wav", f, "audio/wav")},
                data={"language": "fr-FR"},
                params={"language": "de-DE"},
            )
        assert resp.status_code == 422

    def test_languages_differing_only_in_case_agree(self, client: TestClient) -> None:
        with open(SPEECH_WAV, "rb") as f:
            resp = client.post(
                "/v1/convert/audio-to-iml",
                files={"audio": ("test.wav", f, "audio/wav")},
                data={"language": "fr-FR"},
                params={"language": "FR-fr"},
            )
        assert resp.status_code == 200, resp.text
        assert 'language="fr-FR"' in resp.json()["iml"]

    @pytest.mark.parametrize("where", ["data", "params"])
    def test_empty_language_means_none(self, client: TestClient, where: str) -> None:
        with open(SPEECH_WAV, "rb") as f:
            resp = client.post(
                "/v1/convert/audio-to-iml",
                files={"audio": ("test.wav", f, "audio/wav")},
                **{where: {"language": ""}},
            )
        assert resp.status_code == 200, resp.text
        assert "language=" not in resp.json()["iml"]

    @pytest.mark.parametrize("where", ["data", "params"])
    def test_posix_locale_language_is_read_as_bcp47(self, client: TestClient, where: str) -> None:
        """en_US is en-US, as in the SDK and the CLI."""
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files={"audio": ("speech.wav", SPEECH_WAV.read_bytes(), "audio/wav")},
            **{where: {"language": "en_US"}},
        )
        assert resp.status_code == 200, resp.text
        assert 'language="en-US"' in resp.json()["iml"]

    def test_same_language_in_both_forms_does_not_conflict(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files={"audio": ("speech.wav", SPEECH_WAV.read_bytes(), "audio/wav")},
            data={"language": "en_US"},
            params={"language": "en-us"},
        )
        assert resp.status_code == 200, resp.text

    def test_invalid_language_returns_422(self, client: TestClient) -> None:
        with open(SPEECH_WAV, "rb") as f:
            resp = client.post(
                "/v1/convert/audio-to-iml",
                files={"audio": ("test.wav", f, "audio/wav")},
                data={"language": "not a language"},
            )
        assert resp.status_code == 422

    def test_reports_transcript_source_and_warnings(self, client: TestClient) -> None:
        with open(SPEECH_WAV, "rb") as f:
            resp = client.post(
                "/v1/convert/audio-to-iml", files={"audio": ("speech.wav", f, "audio/wav")}
            )
        data = resp.json()
        assert data["transcript_source"] in ("whisper", "none")
        if data["transcript_source"] == "none":
            assert "[speech]" in data["plain_text"]
            assert data["warnings"], "placeholder text must be flagged"

    def test_missing_file_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/convert/audio-to-iml")
        assert resp.status_code == 422

    @pytest.mark.parametrize(
        ("payload", "expected"), [(b"not audio", "x.wav"), (b"", "empty")]
    )
    def test_unreadable_upload_returns_400(
        self, client: TestClient, payload: bytes, expected: str
    ) -> None:
        resp = client.post(
            "/v1/convert/audio-to-iml", files={"audio": ("x.wav", payload, "audio/wav")}
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"] == "audio_processing_error"
        assert expected in data["detail"]
        # The client's file name, not the server's temporary path.
        assert tempfile.gettempdir() not in data["detail"]

    def test_audio_longer_than_limit_returns_400(self, tmp_path: Path) -> None:
        audio = long_speech_wav(tmp_path / "long.wav", seconds=8).read_bytes()
        capped = make_client(max_audio_seconds=5.0).post(
            "/v1/convert/audio-to-iml", files={"audio": ("long.wav", audio, "audio/wav")}
        )
        assert capped.status_code == 400
        assert capped.json()["error"] == "audio_processing_error"
        assert "max_duration_s=5" in capped.json()["detail"]
        assert "'long.wav'" in capped.json()["detail"]
        allowed = make_client(max_audio_seconds=20.0).post(
            "/v1/convert/audio-to-iml", files={"audio": ("long.wav", audio, "audio/wav")}
        )
        assert allowed.status_code == 200

    def test_non_multipart_request_returns_415(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/convert/audio-to-iml",
            content=SPEECH_WAV.read_bytes(),
            headers={"content-type": "audio/wav"},
        )
        assert resp.status_code == 415
        assert resp.json()["error"] == "unsupported_media_type"

    @needs_ffmpeg
    def test_ogg_upload_is_converted(self, client: TestClient, tmp_path: Path) -> None:
        ogg = tmp_path / "speech.ogg"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(SPEECH_WAV), str(ogg)],
            check=True,
        )
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files={"audio": ("speech.ogg", ogg.read_bytes(), "audio/ogg")},
        )
        assert resp.status_code == 200, resp.text
        assert "<pause" in resp.json()["iml"]

    def test_temporary_files_are_removed(self, client: TestClient) -> None:
        before = upload_dirs()
        with open(SPEECH_WAV, "rb") as f:
            ok = client.post("/v1/convert/audio-to-iml", files={"audio": ("a.wav", f)})
        bad = client.post("/v1/convert/audio-to-iml", files={"audio": ("b.wav", b"junk")})
        assert (ok.status_code, bad.status_code) == (200, 400)
        assert upload_dirs() == before

    def test_save_upload_refuses_oversized_stream(self, tmp_path: Path) -> None:
        with pytest.raises(APIError) as excinfo:
            _save_upload(io.BytesIO(b"x" * 101), tmp_path / "upload", max_bytes=100)
        assert excinfo.value.status_code == 413
        _save_upload(io.BytesIO(b"x" * 100), tmp_path / "upload", max_bytes=100)
        assert (tmp_path / "upload").stat().st_size == 100


class TestAudioToIMLFields:
    """The words, transcript and profile form fields."""

    @pytest.mark.parametrize("shape", ["records", "whisper"])
    def test_words_become_the_transcript(self, client: TestClient, shape: str) -> None:
        resp = post_speech(client, words=speech_words_json(shape), language="en-US")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["transcript_source"] == "words"
        assert data["plain_text"] == "I told you to call me yesterday."
        # A single utterance without calibration: only the missing baseline is noted.
        assert [w for w in data["warnings"] if "speaker baseline" not in w] == []
        assert data["profile_matches"] == []
        # The emphasised word and the 600 ms pause are marked.
        assert "<emphasis" in data["iml"] and "told" in data["iml"]
        # The 600 ms gap between "you" and "to", as measured.
        assert 550 <= int(re.findall(r'<pause duration="(\d+)"', data["iml"])[0]) <= 650

    def test_words_as_a_file(self, client: TestClient) -> None:
        """curl -F words=@speech.whisper.json sends the words as a file part."""
        whisper = speech_words_json("whisper").encode("utf-16")
        resp = post_speech(client, words=("speech.whisper.json", whisper, "application/json"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["plain_text"] == "I told you to call me yesterday."

    @pytest.mark.parametrize(
        "words",
        [
            str(AUDIO_FIXTURES / "speech_pauses.json"),  # a path is not JSON text
            json.dumps(str(AUDIO_FIXTURES / "speech_pauses.json")),
            "not json",
            '[{"word": "hi", "start": 2.0, "end": 1.0}]',
            '{"unknown": []}',
            '[{"word": "a\\u0001b", "start_ms": 0, "end_ms": 100}]',
            "[" * 50_000,
            # Words that overlap: each would have the audio analysed again.
            '[{"word": "a", "start_ms": 0, "end_ms": 4000}, '
            '{"word": "b", "start_ms": 1, "end_ms": 4000}]',
        ],
    )
    def test_invalid_words_return_422(self, client: TestClient, words: str) -> None:
        resp = post_speech(client, words=words)
        assert resp.status_code == 422, resp.text
        [error] = resp.json()["detail"]
        assert error["loc"] == ["body", "words"]
        assert "Invalid word timings" in error["msg"]

    def test_transcript(self, client: TestClient) -> None:
        resp = post_speech(client, transcript="I told you to call me yesterday.")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["transcript_source"] == "transcript"
        assert data["plain_text"] == "I told you to call me yesterday."

    def test_transcript_as_a_file(self, client: TestClient) -> None:
        text = "Café au lait.\n".encode()
        resp = post_speech(client, transcript=("speech.txt", text, "text/plain"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["plain_text"] == "Café au lait."

    def test_words_and_transcript_together_return_422(self, client: TestClient) -> None:
        resp = post_speech(client, words=speech_words_json(), transcript="I told you.")
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["loc"] == ["body", "transcript"]

    @pytest.mark.parametrize(
        "transcript", ["bad \x01 control", ("t.txt", b"\xff\xfe not utf-8", "text/plain")]
    )
    def test_unusable_transcript_returns_422(
        self, client: TestClient, transcript: str | tuple[str, bytes, str]
    ) -> None:
        resp = post_speech(client, transcript=transcript)
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["loc"] == ["body", "transcript"]

    @pytest.mark.parametrize(("boost", "applied"), [(0.3, False), (0.6, True)])
    def test_profile_is_applied_and_reported(
        self, client: TestClient, boost: float, applied: bool
    ) -> None:
        resp = post_speech(client, words=speech_words_json(), profile=pause_profile(boost))
        assert resp.status_code == 200, resp.text
        data = resp.json()
        [match] = data["profile_matches"]
        assert match["utterance"] == 0
        assert match["pattern"] == {"pause_frequency": "high"}
        assert match["observed"]["pause_frequency"] == "high"
        assert match["emotion"] == "uncertain"
        # One utterance and no calibration: the classifier has no baseline
        # (confidence 0), so the boost alone is the confidence.
        assert match["confidence"] == pytest.approx(boost)
        assert match["applied"] is applied
        # Profile use is shown by the matched pattern; the user_id is never written.
        assert ('x-profile="pause_frequency=high"' in data["iml"]) is applied
        assert "caller_7" not in data["iml"]
        assert ('emotion="uncertain"' in data["iml"]) is applied

    def test_profile_as_a_file(self, client: TestClient) -> None:
        profile = (PROFILE_FIXTURES / "autism_spectrum.json").read_bytes()
        resp = post_speech(client, profile=("profile.json", profile, "application/json"))
        assert resp.status_code == 200, resp.text

    @pytest.mark.parametrize(
        ("profile", "message"),
        [
            ("{not json", "Invalid JSON"),
            ('{"profile_version": "0.1.0", "user_id": "u", "prosody_mappings": [], '
             '"extra": 1}', "unknown key"),
            (pause_profile(0.3).replace('"high"', '"often"'), "P6"),
            (pause_profile(0.3).replace("0.3", "NaN"), "not a JSON number"),
            ((PROFILE_FIXTURES / "invalid_empty_mappings.json").read_text(), "P3"),
            ("[" * 50_000, "Invalid JSON"),  # nested too deeply for the JSON parser
        ],
    )
    def test_invalid_profile_returns_400(
        self, client: TestClient, profile: str, message: str
    ) -> None:
        resp = post_speech(client, profile=profile)
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"] == "profile_error"
        assert message in resp.json()["detail"]

    @pytest.mark.parametrize(
        ("field", "value", "setting"),
        [
            ("transcript", "a" * 101, "PP_MAX_TEXT_CHARS"),
            ("transcript", ("t.txt", b"a" * 101, "text/plain"), "PP_MAX_TEXT_CHARS"),
            ("profile", " " * 101, "PP_MAX_TEXT_CHARS"),
            ("words", "[" + " " * 200 + "]", "PP_MAX_WORDS_CHARS"),
            ("words", ("w.json", b"[" + b" " * 200 + b"]", "application/json"),
             "PP_MAX_WORDS_CHARS"),
        ],
    )
    def test_fields_over_their_limit_return_413(
        self, field: str, value: str | tuple[str, bytes, str], setting: str
    ) -> None:
        http = make_client(max_text_chars=100, max_words_chars=200)
        resp = post_speech(http, **{field: value})
        assert resp.status_code == 413, resp.text
        assert resp.json()["error"] == "text_too_large"
        assert setting in resp.json()["detail"]

    def test_fields_within_their_limits_are_accepted(self) -> None:
        http = make_client(max_text_chars=100, max_words_chars=len(speech_words_json()))
        resp = post_speech(http, words=speech_words_json())
        assert resp.status_code == 200, resp.text
        resp = post_speech(http, transcript="a" * 100)
        assert resp.status_code == 200, resp.text

    def test_invalid_fields_do_not_take_a_worker(self) -> None:
        """Fields are checked before the job is admitted, so no worker starts."""
        http = make_client()
        resp = post_speech(http, words="not json")
        assert resp.status_code == 422
        assert http.app.state.jobs.worker_pids() == []  # type: ignore[attr-defined]

    def test_overlapping_words_are_rejected_before_analysis(self) -> None:
        """A few KB of words that each span the whole recording used to take a
        worker for minutes and gigabytes; now they are a 422 up front."""
        http = make_client()
        words = json.dumps([
            {"word": "x", "start_ms": i, "end_ms": 596_000 if i % 5 < 2 else i + 300}
            for i in range(2000)
        ])
        resp = post_speech(http, words=words)
        assert resp.status_code == 422, resp.text
        [error] = resp.json()["detail"]
        assert error["loc"] == ["body", "words"]
        assert "may overlap by at most 500 ms" in error["msg"]
        assert http.app.state.jobs.worker_pids() == []  # type: ignore[attr-defined]

    def test_profile_emotion_xml_cannot_hold_is_rejected_up_front(self) -> None:
        """It used to fail only when the mapping applied, after the analysis,
        as a 400 conversion_error."""
        http = make_client()
        profile = pause_profile(0.6).replace('"uncertain"', '"calm\\u0001"')
        resp = post_speech(http, words=speech_words_json(), profile=profile)
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"] == "profile_error"
        assert "XML does not allow" in resp.json()["detail"]
        assert http.app.state.jobs.worker_pids() == []  # type: ignore[attr-defined]

    def test_calibration_sets_the_baseline(self, client: TestClient) -> None:
        """REST had no way to send calibration audio, so a single utterance
        never got an overall pitch or loudness."""
        raised = AUDIO_FIXTURES / "speech_raised.wav"
        words = json.dumps(json.loads(
            (AUDIO_FIXTURES / "speech_raised.json").read_text()
        )["words"])
        files: list[tuple[str, Any]] = [("audio", ("raised.wav", raised.read_bytes()))]
        before = upload_dirs()
        plain = client.post("/v1/convert/audio-to-iml", files=files, data={"words": words})
        calibration = AUDIO_FIXTURES / "speech_calibration.wav"
        calibrated = client.post(
            "/v1/convert/audio-to-iml",
            files=[*files, ("calibration", ("calm.wav", calibration.read_bytes()))],
            data={"words": words},
        )
        assert upload_dirs() == before
        assert plain.status_code == calibrated.status_code == 200, calibrated.text
        assert 'pitch="+' not in plain.json()["iml"]
        assert any("speaker baseline" in w for w in plain.json()["warnings"])
        assert not any("speaker baseline" in w for w in calibrated.json()["warnings"])
        # The whole utterance is higher than the speaker's calm voice.
        utterance = IMLParser().parse(calibrated.json()["iml"]).utterances[0]
        [child] = utterance.children
        assert isinstance(child, Prosody) and child.pitch and child.pitch.startswith("+")

    def test_several_calibration_recordings(self, client: TestClient) -> None:
        """Earlier turns of a conversation can be the baseline together."""
        raised = AUDIO_FIXTURES / "speech_raised.wav"
        words = json.dumps(json.loads(
            (AUDIO_FIXTURES / "speech_raised.json").read_text()
        )["words"])
        before = upload_dirs()
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files=[
                ("audio", ("raised.wav", raised.read_bytes())),
                ("calibration", ("turn1.wav", (AUDIO_FIXTURES / "speech_calibration.wav")
                                 .read_bytes())),
                ("calibration", ("turn2.wav", SPEECH_WAV.read_bytes())),
            ],
            data={"words": words},
        )
        assert resp.status_code == 200, resp.text
        assert upload_dirs() == before
        utterance = IMLParser().parse(resp.json()["iml"]).utterances[0]
        [child] = utterance.children
        assert isinstance(child, Prosody) and child.pitch and child.pitch.startswith("+")

    def test_unusable_calibration_is_named_in_the_error(self, client: TestClient) -> None:
        silence = (AUDIO_FIXTURES / "silence_1s.wav").read_bytes()
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files=[("audio", ("a.wav", SPEECH_WAV.read_bytes())),
                   ("calibration", ("quiet.wav", silence))],
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"] == "audio_processing_error"
        assert "'quiet.wav'" in resp.json()["detail"]
        assert tempfile.gettempdir() not in resp.json()["detail"]

    def test_skipped_calibration_is_named_in_the_warning(self, client: TestClient) -> None:
        """Warnings, like errors, name the client's files, not server paths."""
        silence = (AUDIO_FIXTURES / "silence_1s.wav").read_bytes()
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files=[("audio", ("a.wav", SPEECH_WAV.read_bytes())),
                   ("calibration", ("quiet.wav", silence)),
                   ("calibration", ("turn1.wav", SPEECH_WAV.read_bytes()))],
            data={"words": speech_words_json()},
        )
        assert resp.status_code == 200, resp.text
        warnings = resp.json()["warnings"]
        skipped = [w for w in warnings if "were not used" in w]
        assert skipped and "calibration 'quiet.wav'" in skipped[0], warnings
        assert not any(tempfile.gettempdir() in w or "upload" in w for w in warnings), warnings

    def test_empty_calibration_counts_as_absent(self, client: TestClient) -> None:
        resp = post_speech(client, words=speech_words_json(), calibration="")
        assert resp.status_code == 200, resp.text

    def test_calibration_must_be_a_file(self) -> None:
        http = make_client()
        resp = post_speech(http, calibration="calm.wav")
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["loc"] == ["body", "calibration"]
        assert http.app.state.jobs.worker_pids() == []  # type: ignore[attr-defined]

    def test_calibration_recordings_are_limited(self) -> None:
        http = make_client()
        calibration = ("calm.wav", SPEECH_WAV.read_bytes())
        resp = http.post(
            "/v1/convert/audio-to-iml",
            files=[("audio", ("a.wav", SPEECH_WAV.read_bytes())),
                   *[("calibration", calibration)] * 6],
        )
        assert resp.status_code == 422
        assert "At most 5 calibration recordings" in resp.json()["detail"][0]["msg"]
        assert http.app.state.jobs.worker_pids() == []  # type: ignore[attr-defined]

    def test_example_files(self, client: TestClient) -> None:
        """The examples/README.md curl command."""
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files={
                "audio": ("speech.wav", (EXAMPLES / "speech.wav").read_bytes(), "audio/wav"),
                "words": ("speech.whisper.json", (EXAMPLES / "speech.whisper.json").read_bytes()),
            },
            data={"language": "en-US"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["transcript_source"] == "words"
        assert data["plain_text"] == (EXAMPLES / "speech.txt").read_text().strip()

    def test_example_profile(self, client: TestClient) -> None:
        """The examples/README.md curl command with a profile."""
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files={
                "audio": ("m.wav", (EXAMPLES / "monotone.wav").read_bytes(), "audio/wav"),
                "words": ("m.json", (EXAMPLES / "monotone.deepgram.json").read_bytes()),
                "profile": ("p.json", (EXAMPLES / "profile.json").read_bytes()),
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert [(m["emotion"], m["applied"]) for m in data["profile_matches"]] == [
            ("calm", True), ("calm", True), ("calm", True), ("joyful", True)
        ]
        assert 'x-profile="pitch_contour=flat rate=fast"' in data["iml"]
        assert "example_user" not in data["iml"]


# ---------------------------------------------------------------------------
# IML to LLM prompt
# ---------------------------------------------------------------------------


SARCASM = (
    '<utterance emotion="sarcastic" confidence="0.87">Oh, that\'s '
    '<prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">GREAT</prosody>.'
    '<pause duration="800"/> Really.</utterance>'
)


class TestIMLToPromptEndpoint:
    def test_context_system_prompt_and_messages(self, client: TestClient) -> None:
        from prosody_protocol.llm import SYSTEM_PROMPT, build_messages, to_llm_context

        resp = client.post("/v1/convert/iml-to-prompt", json={"iml": SARCASM})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["context"] == to_llm_context(SARCASM)
        assert "GREAT (higher pitch, louder, sharply falling)" in data["context"]
        assert "[pause 0.8s]" in data["context"]
        assert "sarcastic" in data["context"]
        assert data["system_prompt"] == SYSTEM_PROMPT
        assert data["messages"] == build_messages(SARCASM)

    def test_options(self, client: TestClient) -> None:
        from prosody_protocol.llm import build_messages, to_llm_context

        body = {
            "iml": SARCASM,
            "min_confidence": 0.9,
            "include_numbers": True,
            "instruction": "Summarize.",
        }
        data = client.post("/v1/convert/iml-to-prompt", json=body).json()
        options = {"min_confidence": 0.9, "include_numbers": True}
        assert data["context"] == to_llm_context(SARCASM, **options)
        assert "not reliably detected" in data["context"]
        assert "+15%" in data["context"]
        assert data["messages"] == build_messages(SARCASM, "Summarize.", **options)
        assert data["messages"][1]["content"].endswith("Summarize.")

    def test_malformed_iml_returns_400(self, client: TestClient) -> None:
        resp = client.post("/v1/convert/iml-to-prompt", json={"iml": "<utterance>oops"})
        assert resp.status_code == 400
        assert resp.json()["error"] == "iml_parse_error"

    def test_strict_rejects_invalid_iml(self, client: TestClient) -> None:
        invalid = '<utterance emotion="angry">No confidence.</utterance>'
        lenient = client.post("/v1/convert/iml-to-prompt", json={"iml": invalid})
        assert lenient.status_code == 200
        strict = client.post("/v1/convert/iml-to-prompt", json={"iml": invalid, "strict": True})
        assert strict.status_code == 400
        assert strict.json()["error"] == "validation_error"
        assert [i["rule"] for i in strict.json()["issues"]] == ["V3"]

    @pytest.mark.parametrize("min_confidence", [-0.1, 1.5])
    def test_invalid_min_confidence_returns_422(
        self, client: TestClient, min_confidence: float
    ) -> None:
        resp = client.post(
            "/v1/convert/iml-to-prompt", json={"iml": SARCASM, "min_confidence": min_confidence}
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize(
        ("path", "body", "echoed"),
        [
            ("/v1/convert/iml-to-prompt", '{"iml": "<utterance>hi</utterance>", '
             '"min_confidence": NaN}', "nan"),
            ("/v1/convert/iml-to-prompt", '{"iml": "<utterance>hi</utterance>", '
             '"min_confidence": -Infinity}', "-inf"),
            ("/v1/validate", '{"iml": NaN}', "nan"),
            ("/v1/convert/iml-to-ssml", '{"iml": Infinity}', "inf"),
            ("/v1/convert/text-to-iml", '{"text": NaN}', "nan"),
        ],
    )
    def test_non_finite_json_numbers_return_422(
        self, client: TestClient, path: str, body: str, echoed: str
    ) -> None:
        """Python's JSON parser accepts NaN and Infinity; echoing them back in
        the 422 used to fail with a 500 internal_error."""
        resp = client.post(path, content=body, headers={"Content-Type": "application/json"})
        assert resp.status_code == 422, resp.text
        [error] = resp.json()["detail"]
        assert error["input"] == echoed


# ---------------------------------------------------------------------------
# OpenAPI spec
# ---------------------------------------------------------------------------


class TestOpenAPISpec:
    def test_openapi_json_available(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_openapi_has_all_paths(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        expected = [
            "/v1/health",
            "/v1/validate",
            "/v1/synthesize",
            "/v1/convert/audio-to-iml",
            "/v1/convert/text-to-iml",
            "/v1/convert/iml-to-ssml",
            "/v1/convert/iml-to-prompt",
        ]
        for path in expected:
            assert path in paths, f"Missing path: {path}"

    def test_openapi_info_has_version(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        info = resp.json()["info"]
        assert "version" in info
        assert info["title"] == "Prosody Protocol API"

    def test_synthesize_documents_wav_response(self, client: TestClient) -> None:
        responses = client.get("/openapi.json").json()["paths"]["/v1/synthesize"]["post"][
            "responses"
        ]
        assert list(responses["200"]["content"]) == ["audio/wav"]
        assert list(responses["400"]["content"]) == ["application/json"]

    def test_post_endpoints_document_error_responses(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        error_ref = "#/components/schemas/ErrorResponse"
        for path, item in spec["paths"].items():
            if "post" not in item:
                continue
            responses = item["post"]["responses"]
            for status in ("400", "413", "429"):
                schema = responses[status]["content"]["application/json"]["schema"]
                assert schema == {"$ref": error_ref}, (path, status)
            assert "422" in responses
        audio = spec["paths"]["/v1/convert/audio-to-iml"]["post"]
        assert "415" in audio["responses"]
        # Only the endpoints that run in a worker process can be busy.
        busy = {path for path, item in spec["paths"].items()
                if "503" in item.get("post", {}).get("responses", {})}
        assert busy == {"/v1/convert/audio-to-iml", "/v1/synthesize"}

    def test_audio_language_is_a_form_field(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        body_ref = spec["paths"]["/v1/convert/audio-to-iml"]["post"]["requestBody"]["content"][
            "multipart/form-data"
        ]["schema"]["$ref"]
        body = spec["components"]["schemas"][body_ref.rsplit("/", 1)[1]]
        assert set(body["properties"]) == {
            "audio", "language", "words", "transcript", "profile", "calibration"
        }

    def test_audio_response_declares_profile_matches(self, client: TestClient) -> None:
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        assert {"transcript_source", "warnings", "profile_matches"} <= set(
            schemas["AudioToIMLResponse"]["properties"]
        )
        assert set(schemas["PromptResponse"]["properties"]) == {
            "context", "system_prompt", "messages"
        }


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_iml_parse_error_returns_400(self, client: TestClient) -> None:
        """ConversionError from SSML conversion returns 400."""
        resp = client.post(
            "/v1/convert/iml-to-ssml",
            json={"iml": "not xml at all <<<"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"] in ("conversion_error", "iml_parse_error")

    def test_validation_endpoint_never_500_on_bad_input(self, client: TestClient) -> None:
        """Any string input to validate should return 200 with issues, not crash."""
        bad_inputs = [
            "",
            "   ",
            "not xml",
            "<foo>bar</foo>",
            '<utterance emotion="x">no conf</utterance>',
        ]
        for iml in bad_inputs:
            resp = client.post("/v1/validate", json={"iml": iml})
            assert resp.status_code == 200, f"Got {resp.status_code} for {iml!r}"
            data = resp.json()
            assert "valid" in data
            assert "issues" in data

    def test_form_field_over_starlettes_part_limit_is_text_too_large(
        self, client: TestClient
    ) -> None:
        """A words text field over 1 MiB used to get Starlette's 400
        {"detail": "Part exceeded maximum size of 1024KB."}, without an error code."""
        words = json.dumps(
            [{"word": "\u65e5" * 400_000, "start_ms": 0, "end_ms": 500}], ensure_ascii=False
        )
        # Under PP_MAX_WORDS_CHARS (1,000,000 characters), over 1 MiB of UTF-8.
        assert len(words) < 1_000_000 and len(words.encode()) > 1024 * 1024
        resp = post_speech(client, words=words)
        assert resp.status_code == 413, resp.text
        assert resp.json()["error"] == "text_too_large"
        assert "send it as a file" in resp.json()["detail"]

    def test_unparsable_body_has_an_error_code(self, client: TestClient) -> None:
        """JSON nested too deeply used to get {"detail": "There was an error parsing the body"}."""
        body = '{"iml": "<utterance/>", "x": ' + "[" * 100_000 + "]" * 100_000 + "}"
        resp = client.post(
            "/v1/validate", content=body, headers={"content-type": "application/json"}
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_body"
        assert "cannot be parsed" in resp.json()["detail"]

    def test_unknown_paths_and_methods_have_error_codes(self, client: TestClient) -> None:
        resp = client.get("/v1/nothing-here")
        assert (resp.status_code, resp.json()) == (
            404, {"error": "not_found", "detail": "Not Found"}
        )
        resp = client.get("/v1/validate")
        assert resp.status_code == 405
        assert resp.json() == {"error": "method_not_allowed", "detail": "Method Not Allowed"}
        assert resp.headers["allow"] == "POST"

    def test_schema_errors_have_an_error_code(self, client: TestClient) -> None:
        resp = client.post("/v1/validate", json={})
        assert resp.status_code == 422
        assert resp.json()["error"] == "invalid_request"
        assert resp.json()["detail"][0]["loc"] == ["body", "iml"]

    def test_unexpected_error_returns_json_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(self: object, text: str) -> None:
            raise RuntimeError("secret internal state")

        monkeypatch.setattr("prosody_protocol.IMLValidator.validate", broken)
        resp = client.post("/v1/validate", json={"iml": "<utterance>Hi</utterance>"})
        assert resp.status_code == 500
        assert resp.json()["error"] == "internal_error"
        assert "secret" not in resp.text


# ---------------------------------------------------------------------------
# Request size limits
# ---------------------------------------------------------------------------


def big_validate_body(size: int) -> bytes:
    """A JSON body for /v1/validate of at least *size* bytes."""
    return json.dumps({"iml": "<utterance>" + "a " * (size // 2) + "</utterance>"}).encode()


class TestUploadSizeLimit:
    def test_oversized_upload_rejected(self, client: TestClient) -> None:
        """Uploads exceeding max_upload_size_mb should return 413."""
        from prosody_protocol.server.app import settings

        # Create a payload just over the limit
        over_limit = b"x" * (settings.max_upload_bytes + 1)
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files={"audio": ("big.wav", over_limit, "audio/wav")},
            headers={"content-length": str(len(over_limit))},
        )
        assert resp.status_code == 413
        assert "payload_too_large" in resp.json()["error"]

    def test_small_upload_passes_size_check(self, client: TestClient) -> None:
        """A small upload is not rejected by the size middleware."""
        small_payload = b"RIFF" + b"\x00" * 100
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files={"audio": ("small.wav", small_payload, "audio/wav")},
        )
        # Not audio, so the analysis rejects it -- with a 400, not a 413 or 500.
        assert resp.status_code == 400
        assert resp.json()["error"] == "audio_processing_error"

    def test_chunked_json_body_over_limit_rejected(self) -> None:
        client = make_client(max_upload_size_mb=1)
        body = big_validate_body(2 * 1024 * 1024)
        resp = client.post(
            "/v1/validate", content=chunks(body), headers={"content-type": "application/json"}
        )
        assert resp.status_code == 413
        assert resp.json()["error"] == "payload_too_large"

    def test_chunked_upload_over_limit_rejected(self) -> None:
        client = make_client(max_upload_size_mb=1)
        body, content_type = multipart_body("audio", "big.wav", b"\x00" * (2 * 1024 * 1024))
        resp = client.post(
            "/v1/convert/audio-to-iml",
            content=chunks(body),
            headers={"content-type": content_type},
        )
        assert resp.status_code == 413
        assert resp.json()["error"] == "payload_too_large"

    def test_understated_content_length_rejected(self) -> None:
        client = make_client(max_upload_size_mb=1)
        body = big_validate_body(2 * 1024 * 1024)
        resp = client.post(
            "/v1/validate",
            content=body,
            headers={"content-type": "application/json", "content-length": "100"},
        )
        assert resp.status_code == 413

    def test_chunked_body_under_limit_accepted(self) -> None:
        client = make_client(max_upload_size_mb=1, max_text_chars=10**7)
        body = big_validate_body(1024 * 1024 - 100)
        assert len(body) <= 1024 * 1024
        resp = client.post(
            "/v1/validate", content=chunks(body), headers={"content-type": "application/json"}
        )
        assert resp.status_code == 200
        assert resp.json()["valid"] is True


def junk_json_body(size: int) -> bytes:
    """A valid /v1/validate body of about *size* bytes, mostly empty objects in an extra key.

    The kind of body that used to cost about 25 times its size in memory
    (and seconds of event loop time) before any field was checked.
    """
    count = max(0, (size - 64) // 3)
    return b'{"iml":"<utterance>x</utterance>","x":[' + b",".join([b"{}"] * count) + b"]}"


class TestJSONSizeLimit:
    """JSON requests get PP_MAX_JSON_BYTES, far below the upload limit."""

    def test_default_is_derived_from_the_text_limit(self) -> None:
        assert Settings().max_json_bytes == 24 * 100_000 + 65536
        assert Settings(max_text_chars=10).max_json_bytes == 24 * 10 + 65536
        assert Settings(max_json_bytes=5000).json_body_limit == 5000
        # The upload limit applies to every request.
        assert Settings(max_upload_size_mb=1, max_text_chars=10**6).json_body_limit == 2**20

    @pytest.mark.parametrize("chunked", [False, True])
    @pytest.mark.parametrize(
        "path", ["/v1/validate", "/v1/convert/iml-to-ssml", "/v1/convert/iml-to-prompt",
                 "/v1/convert/text-to-iml", "/v1/synthesize"],
    )
    def test_json_body_over_the_limit_is_413(self, path: str, chunked: bool) -> None:
        """A 3 MB body used to be parsed in full (the limit was PP_MAX_UPLOAD_MB, 50 MB)."""
        body = junk_json_body(3 * 1024 * 1024)
        headers = {"content-type": "application/json"}
        with make_client() as http:
            resp = http.post(path, content=chunks(body) if chunked else body, headers=headers)
        assert resp.status_code == 413, resp.text
        assert resp.json()["error"] == "payload_too_large"
        assert "PP_MAX_JSON_BYTES" in resp.json()["detail"]
        assert str(24 * 100_000 + 65536) in resp.json()["detail"]

    def test_limit_follows_the_setting(self) -> None:
        body = junk_json_body(5000)
        assert make_client(max_json_bytes=len(body)).post(
            "/v1/validate", content=body, headers={"content-type": "application/json"}
        ).status_code == 200
        resp = make_client(max_json_bytes=len(body) - 1).post(
            "/v1/validate", content=body, headers={"content-type": "application/json"}
        )
        assert resp.status_code == 413

    @pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded"])
    def test_every_non_multipart_body_is_limited(self, content_type: str) -> None:
        resp = make_client(max_json_bytes=1000).post(
            "/v1/convert/audio-to-iml", content=b"x" * 2000, headers={"content-type": content_type}
        )
        assert resp.status_code == 413
        assert "PP_MAX_JSON_BYTES" in resp.json()["detail"]

    def test_uploads_keep_the_upload_limit(self) -> None:
        body, content_type = multipart_body("audio", "big.wav", b"\x00" * (3 * 1024 * 1024))
        resp = make_client().post(
            "/v1/convert/audio-to-iml", content=body, headers={"content-type": content_type}
        )
        # Read and analysed (it is not audio), not refused for its size.
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"] == "audio_processing_error"

    def test_two_fields_of_escaped_text_at_the_text_limit_fit(self) -> None:
        """json.dumps escapes non-ASCII: an emoji is 12 bytes. The default fits both fields."""
        text = "\U0001F600" * 1000
        body = json.dumps({"text": text, "context": text}).encode()
        assert len(body) > 24_000
        resp = make_client(max_text_chars=1000).post(
            "/v1/convert/text-to-iml", content=body, headers={"content-type": "application/json"}
        )
        assert resp.status_code == 200, resp.text

    def test_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PP_MAX_JSON_BYTES", "4096")
        assert Settings().max_json_bytes == 4096
        monkeypatch.setenv("PP_MAX_JSON_BYTES", "")
        assert Settings().max_json_bytes == 24 * 100_000 + 65536
        for bad in ("0", "lots", "-5"):
            monkeypatch.setenv("PP_MAX_JSON_BYTES", bad)
            with pytest.raises(ValueError, match="PP_MAX_JSON_BYTES"):
                Settings()


class TestTextLimit:
    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/v1/validate", {"iml": "<utterance>" + "a" * 100 + "</utterance>"}),
            ("/v1/convert/iml-to-ssml", {"iml": "<utterance>" + "a" * 100 + "</utterance>"}),
            ("/v1/synthesize", {"iml": "<utterance>" + "a" * 100 + "</utterance>"}),
            ("/v1/convert/text-to-iml", {"text": "a" * 101}),
            ("/v1/convert/text-to-iml", {"text": "Hi.", "context": "a" * 101}),
            ("/v1/convert/iml-to-prompt", {"iml": "<utterance>" + "a" * 100 + "</utterance>"}),
            ("/v1/convert/iml-to-prompt", {"iml": "<utterance/>", "instruction": "a" * 101}),
        ],
    )
    def test_text_over_limit_returns_413(self, path: str, body: dict[str, str]) -> None:
        resp = make_client(max_text_chars=100).post(path, json=body)
        assert resp.status_code == 413
        assert resp.json()["error"] == "text_too_large"
        assert "PP_MAX_TEXT_CHARS" in resp.json()["detail"]

    def test_text_at_limit_accepted(self) -> None:
        resp = make_client(max_text_chars=100).post("/v1/convert/text-to-iml", json={
            "text": "a" * 100
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def limited(rpm: int, **kwargs: Any) -> tuple[RateLimitMiddleware, FakeClock]:
    """A rate limiter in front of a trivial app, on a clock the test controls."""
    clock = FakeClock()
    middleware = RateLimitMiddleware(
        PlainTextResponse("ok"), requests_per_minute=rpm, clock=clock, **kwargs
    )
    return middleware, clock


def hit(
    middleware: RateLimitMiddleware, peer: str = "203.0.113.9", forwarded: str | None = None
) -> tuple[int, dict[str, str]]:
    """Send one request through *middleware*; return its status and headers."""
    headers = [] if forwarded is None else [(b"x-forwarded-for", forwarded.encode())]
    scope = {"type": "http", "method": "GET", "path": "/", "query_string": b"",
             "headers": headers, "client": (peer, 5000)}
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> MutableMapping[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    start = sent[0]
    status: int = start["status"]
    return status, {k.decode(): v.decode() for k, v in start["headers"]}


class TestRateLimit:
    def test_limit_returns_429_with_retry_after(self) -> None:
        client = make_client(rate_limit_per_minute=3)
        body = {"iml": "<utterance/>"}
        statuses = [client.post("/v1/validate", json=body).status_code for _ in range(3)]
        assert statuses == [200, 200, 200]
        resp = client.post("/v1/validate", json=body)
        assert resp.status_code == 429
        assert resp.json()["error"] == "rate_limited"
        assert 1 <= int(resp.headers["retry-after"]) <= 60

    def test_health_is_not_rate_limited(self) -> None:
        client = make_client(rate_limit_per_minute=2)
        assert [client.get("/v1/health").status_code for _ in range(5)] == [200] * 5
        # Health checks did not use up the client's budget either.
        assert client.post("/v1/validate", json={"iml": "<utterance/>"}).status_code == 200

    def test_forwarded_for_ignored_from_untrusted_peer(self) -> None:
        middleware, _ = limited(2)
        statuses = [hit(middleware, forwarded=f"198.51.100.{i}")[0] for i in range(3)]
        assert statuses == [200, 200, 429]

    def test_forwarded_for_used_from_trusted_proxy(self) -> None:
        application = create_app(
            Settings(rate_limit_per_minute=2, trusted_proxies=["10.0.0.0/8"])
        )
        proxied = TestClient(application, client=("10.1.2.3", 5000))

        def validate(forwarded: str) -> int:
            resp = proxied.post(
                "/v1/validate",
                json={"iml": "<utterance/>"},
                headers={"x-forwarded-for": forwarded},
            )
            return int(resp.status_code)

        # Four different clients behind the proxy each get their own budget ...
        assert [validate(f"198.51.100.{i}") for i in range(4)] == [200] * 4
        # ... and one client is still limited.
        assert [validate("198.51.100.77") for _ in range(3)] == [200, 200, 429]

    def test_spoofed_forwarded_entries_are_skipped(self) -> None:
        proxies = Settings(trusted_proxies=["10.0.0.1"]).trusted_proxy_networks()
        middleware, _ = limited(1, trusted_proxies=proxies)
        # The proxy appends the address it saw; entries to its left came from the client.
        first = hit(middleware, "10.0.0.1", "1.1.1.1, 198.51.100.7")
        second = hit(middleware, "10.0.0.1", "2.2.2.2, 198.51.100.7")
        assert (first[0], second[0]) == (200, 429)
        # Trusted hops on the right are skipped too.
        third = hit(middleware, "10.0.0.1", "198.51.100.7, 10.0.0.1")
        assert third[0] == 429

    def test_window_slides_and_retry_after_counts_down(self) -> None:
        middleware, clock = limited(2)
        assert hit(middleware)[0] == 200
        clock.now += 10
        assert hit(middleware)[0] == 200
        clock.now += 20
        status, headers = hit(middleware)
        assert status == 429
        assert headers["retry-after"] == "30"  # The first request expires at +60 s.
        clock.now += 30.5
        assert hit(middleware)[0] == 200

    def test_tracked_clients_are_bounded(self) -> None:
        middleware, clock = limited(5, max_clients=50)
        for i in range(500):
            hit(middleware, f"198.51.{i // 256}.{i % 256}")
        assert len(middleware._hits) <= 50
        clock.now += 61
        hit(middleware, "203.0.113.1")
        assert len(middleware._hits) == 1  # Idle clients are forgotten.

    def test_invalid_trusted_proxy_setting_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PP_TRUSTED_PROXIES", "10.0.0.0/8, proxy.local")
        with pytest.raises(ValueError, match="proxy.local"):
            Settings()


class TestSettings:
    def test_environment_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PP_MAX_TEXT_CHARS", "500")
        monkeypatch.setenv("PP_MAX_WORDS_CHARS", "2000")
        monkeypatch.setenv("PP_MAX_SYNTH_SECONDS", "30.5")
        monkeypatch.setenv("PP_MAX_AUDIO_SECONDS", "90")
        monkeypatch.setenv("PP_MAX_CONCURRENT_JOBS", "3")
        monkeypatch.setenv("PP_MAX_QUEUED_JOBS", "0")
        monkeypatch.setenv("PP_TRUSTED_PROXIES", "10.0.0.1, 172.16.0.0/12")
        settings = Settings()
        assert settings.max_text_chars == 500
        assert settings.max_words_chars == 2000
        assert settings.max_synth_seconds == 30.5
        assert settings.max_audio_seconds == 90.0
        assert settings.max_concurrent_jobs == 3
        assert settings.max_queued_jobs == 0
        assert settings.trusted_proxies == ["10.0.0.1", "172.16.0.0/12"]

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("PP_MAX_UPLOAD_MB", "lots"),
            ("PP_MAX_UPLOAD_MB", "0"),
            ("PP_RATE_LIMIT", "-1"),
            ("PP_MAX_WORDS_CHARS", "0"),
            ("PP_PORT", "70000"),
            ("PP_MAX_SYNTH_SECONDS", "0"),
            ("PP_MAX_SYNTH_SECONDS", "inf"),
            ("PP_MAX_SYNTH_SECONDS", "nan"),
            ("PP_MAX_AUDIO_SECONDS", "-5"),
            ("PP_MAX_CONCURRENT_JOBS", "0"),
            ("PP_MAX_QUEUED_JOBS", "-1"),
        ],
    )
    def test_invalid_values_name_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, name: str, value: str
    ) -> None:
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            Settings()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("port", 0),
            ("port", 65_536),
            ("max_upload_size_mb", -1),
            ("max_upload_size_mb", 1.5),
            ("rate_limit_per_minute", -1),
            ("max_text_chars", 0),
            ("max_words_chars", -1),
            ("max_synth_seconds", 0.0),
            ("max_synth_seconds", float("nan")),
            ("max_audio_seconds", float("inf")),
            ("max_concurrent_jobs", 0),
            ("max_queued_jobs", -1),
        ],
    )
    def test_invalid_explicit_values_are_rejected(self, field: str, value: object) -> None:
        """create_app(Settings(...)) must fail at startup, not 500 on every request."""
        with pytest.raises(ValueError, match=field):
            Settings(**{field: value})  # type: ignore[arg-type]


    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_host_means_loopback(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """PP_HOST= (e.g. PP_HOST: ${PP_HOST} in a compose file, unset) used to bind
        every interface: uvicorn takes an empty host as all of them."""
        monkeypatch.setenv("PP_HOST", value)
        assert Settings().host == "127.0.0.1"
        assert Settings(host=value).host == "127.0.0.1"
        monkeypatch.setenv("PP_HOST", " 0.0.0.0 ")
        assert Settings().host == "0.0.0.0"

    def test_run_with_empty_host_binds_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        uvicorn = pytest.importorskip("uvicorn")
        hosts: list[str] = []
        monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: hosts.append(kwargs["host"]))
        monkeypatch.setenv("PP_HOST", "")
        from prosody_protocol.server import run

        run()
        run(host="")
        run(host=" ")
        assert hosts == ["127.0.0.1"] * 3

    def test_stt_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert Settings().stt_model == "base"
        monkeypatch.setenv("PP_STT_MODEL", " large-v3 ")
        assert Settings().stt_model == "large-v3"
        monkeypatch.setenv("PP_STT_MODEL", "")
        assert Settings().stt_model == "base"
        with pytest.raises(ValueError, match="PP_STT_MODEL"):
            Settings(stt_model=3)  # type: ignore[arg-type]

    def test_run_uses_host_port_and_debug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        uvicorn = pytest.importorskip("uvicorn")
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(
            {"app": app, **kwargs}
        ))
        monkeypatch.setenv("PP_HOST", "0.0.0.0")
        monkeypatch.setenv("PP_PORT", "9123")
        monkeypatch.setenv("PP_DEBUG", "1")
        from prosody_protocol.server import run

        run()
        run(host="127.0.0.2", port=9000)
        run(port=0)  # any free port; used to become PP_PORT
        assert calls == [
            {"app": "prosody_protocol.server.app:app", "host": "0.0.0.0", "port": 9123,
             "log_level": "debug"},
            {"app": "prosody_protocol.server.app:app", "host": "127.0.0.2", "port": 9000,
             "log_level": "debug"},
            {"app": "prosody_protocol.server.app:app", "host": "0.0.0.0", "port": 0,
             "log_level": "debug"},
        ]
        for bad in (-5, 99_999):
            with pytest.raises(ValueError, match="port must be between 0 and 65535"):
                run(port=bad)
        assert len(calls) == 3


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------


class TestJobRunner:
    def test_crashed_worker_is_replaced(self) -> None:
        runner = JobRunner(max_workers=1)

        async def scenario() -> int:
            with pytest.raises(APIError) as excinfo:
                await runner.run(os._exit, 3)
            assert excinfo.value.status_code == 500
            return await runner.run(operator.add, 2, 3)

        try:
            assert asyncio.run(scenario()) == 5
        finally:
            runner.shutdown()

    @staticmethod
    async def until(condition: Any, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition():
            assert time.monotonic() < deadline, "timed out"
            await asyncio.sleep(0.05)

    def test_a_dead_worker_fails_only_its_own_job(self) -> None:
        """Killing the worker used to fail every running and queued job with a 500."""
        runner = JobRunner(max_workers=1, max_queued=1)

        async def scenario() -> tuple[BaseException | None, object]:
            running = asyncio.ensure_future(runner.run(time.sleep, 60))
            await self.until(lambda: runner.running == 1)
            queued = asyncio.ensure_future(runner.run(operator.add, 2, 3))
            await asyncio.sleep(0.2)
            [pid] = runner.worker_pids()
            os.kill(pid, 9)
            failed, result = await asyncio.gather(running, queued, return_exceptions=True)
            return failed, result

        try:
            failed, result = asyncio.run(scenario())
        finally:
            runner.shutdown()
        assert result == 5, result
        assert isinstance(failed, APIError)
        assert failed.status_code == 500
        assert "exited unexpectedly (killed by SIGKILL" in failed.detail

    def test_other_workers_keep_running(self) -> None:
        runner = JobRunner(max_workers=2)

        async def scenario() -> list[object]:
            slow = asyncio.ensure_future(runner.run(time.sleep, 2))
            await self.until(lambda: runner.running == 1)
            return await asyncio.gather(slow, runner.run(os._exit, 3), return_exceptions=True)

        try:
            slow, crashed = asyncio.run(scenario())
        finally:
            runner.shutdown()
        assert slow is None  # time.sleep's result: the job finished.
        assert isinstance(crashed, APIError) and "exit status 3" in crashed.detail

    def test_worker_killed_while_idle_is_replaced_quietly(self) -> None:
        runner = JobRunner(max_workers=1)
        try:
            assert asyncio.run(runner.run(operator.add, 1, 1)) == 2
            [pid] = runner.worker_pids()
            os.kill(pid, 9)
            deadline = time.monotonic() + 30
            while pid in runner.worker_pids():
                assert time.monotonic() < deadline
                time.sleep(0.05)
            assert asyncio.run(runner.run(operator.add, 2, 3)) == 5
            assert runner.worker_pids() != [pid]
        finally:
            runner.shutdown()

    def test_cancelled_request_stops_its_job(self) -> None:
        """A cancelled request used to leave its job running to the end."""
        runner = JobRunner(max_workers=1, max_queued=0)

        async def scenario() -> None:
            async def request() -> None:
                with runner.admit():
                    await runner.run(time.sleep, 30)

            task = asyncio.ensure_future(request())
            await self.until(lambda: runner.running == 1)
            [pid] = runner.worker_pids()
            started = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # The worker was killed, so its place frees long before 30 s.
            await self.until(lambda: runner.admitted == 0 and runner.running == 0, 10.0)
            assert time.monotonic() - started < 10.0
            assert pid not in runner.worker_pids()
            # The next job gets a fresh worker.
            with runner.admit():
                assert await runner.run(operator.add, 2, 3) == 5

        try:
            asyncio.run(scenario())
        finally:
            runner.shutdown()

    def test_client_that_goes_away_stops_its_job(self) -> None:
        runner = JobRunner(max_workers=1, max_queued=0)
        checks: list[bool] = []

        async def gone() -> bool:
            checks.append(runner.running == 1)
            return runner.running == 1  # the client leaves once the job has started

        async def scenario() -> None:
            started = time.monotonic()
            with pytest.raises(APIError) as excinfo, runner.admit():
                await runner.run(time.sleep, 30, is_disconnected=gone)
            assert excinfo.value.status_code == 499
            assert excinfo.value.error == "client_closed_request"
            await self.until(lambda: runner.admitted == 0 and runner.running == 0, 10.0)
            assert time.monotonic() - started < 10.0
            assert checks and checks[-1]

        try:
            asyncio.run(scenario())
        finally:
            runner.shutdown()

    def test_client_that_stays_gets_the_result(self) -> None:
        runner = JobRunner(max_workers=1)

        async def connected() -> bool:
            return False

        async def scenario() -> None:
            with runner.admit():
                assert await runner.run(
                    operator.mul, 6, 7, is_disconnected=connected
                ) == 42

        try:
            asyncio.run(scenario())
        finally:
            runner.shutdown()

    def test_job_over_the_time_limit_is_stopped(self) -> None:
        runner = JobRunner(max_workers=1, job_timeout_s=0.5)

        async def scenario() -> None:
            started = time.monotonic()
            with pytest.raises(APIError) as excinfo, runner.admit():
                await runner.run(time.sleep, 30)
            assert excinfo.value.status_code == 504
            assert excinfo.value.error == "job_timeout"
            assert "PP_JOB_TIMEOUT_S" in excinfo.value.detail
            assert time.monotonic() - started < 10.0
            # Its worker was replaced; a quick job still runs.
            with runner.admit():
                assert await runner.run(operator.add, 1, 1) == 2

        try:
            asyncio.run(scenario())
        finally:
            runner.shutdown()

    def test_exceptions_keep_the_worker_traceback(self) -> None:
        runner = JobRunner(max_workers=1)
        try:
            with pytest.raises(ZeroDivisionError) as excinfo:
                asyncio.run(runner.run(operator.truediv, 1, 0))
            # Logged with the request's traceback by the internal_error handler.
            remote = str(excinfo.value.__cause__)
            assert "Traceback" in remote and "ZeroDivisionError: division by zero" in remote
            assert asyncio.run(runner.run(operator.add, 2, 3)) == 5
        finally:
            runner.shutdown()

    def test_validation_issues_survive_the_worker(self) -> None:
        runner = JobRunner(max_workers=1)
        iml = '<utterance emotion="sad">Oh.</utterance>'
        try:
            with pytest.raises(IMLValidationError) as excinfo:
                asyncio.run(runner.run(_worker.synthesize, iml, None, "tones", 10.0, True))
        finally:
            runner.shutdown()
        assert [issue.rule for issue in excinfo.value.issues] == ["V3"]

    def test_admission_is_bounded(self) -> None:
        runner = JobRunner(max_workers=1, max_queued=1)
        with runner.admit(), runner.admit(), pytest.raises(APIError) as excinfo, runner.admit():
            pass
        assert excinfo.value.status_code == 503
        assert excinfo.value.error == "server_busy"
        assert excinfo.value.headers == {"Retry-After": "10"}
        # Places are given back, also when the job fails.
        with pytest.raises(RuntimeError), runner.admit():
            raise RuntimeError("job failed")
        assert runner.admitted == 0

    def test_busy_server_refuses_jobs_with_503(self) -> None:
        application = create_app(
            Settings(rate_limit_per_minute=0, max_concurrent_jobs=1, max_queued_jobs=0)
        )
        http = TestClient(application, raise_server_exceptions=False)
        before = upload_dirs()
        with application.state.jobs.admit():  # A job is running and nothing may wait.
            synth = http.post("/v1/synthesize", json={"iml": "<utterance>Hi.</utterance>"})
            with open(SPEECH_WAV, "rb") as f:
                audio = http.post("/v1/convert/audio-to-iml", files={"audio": ("a.wav", f)})
            assert upload_dirs() == before, "a refused upload must not be copied"
        for resp in (synth, audio):
            assert resp.status_code == 503
            assert resp.json()["error"] == "server_busy"
            assert resp.headers["retry-after"] == "10"
        ok = http.post("/v1/synthesize", json={"iml": "<utterance>Hi.</utterance>"})
        assert ok.status_code == 200


# A stand-in for openai-whisper: logs each model load next to itself and
# "transcribes" any audio as "hello" plus the language it was asked for.
FAKE_WHISPER = """
import os
from pathlib import Path


class _Model:
    def transcribe(self, audio, **options):
        language = options.get("language", "en")
        words = [
            {"word": " hello", "start": 0.1, "end": 0.4},
            {"word": " " + language, "start": 0.5, "end": 0.9},
        ]
        return {"language": language, "segments": [{"words": words}]}


def load_model(name):
    with Path(__file__).with_name("loads.log").open("a") as log:
        log.write(f"{os.getpid()} {name}\\n")
    return _Model()
"""


class TestWhisperInWorkers:
    def test_whisper_model_is_loaded_once_per_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "whisper.py").write_text(FAKE_WHISPER)
        monkeypatch.syspath_prepend(str(tmp_path))  # Spawned workers inherit sys.path.
        application = create_app(Settings(rate_limit_per_minute=0, max_concurrent_jobs=1))
        with TestClient(application, raise_server_exceptions=False) as http:
            for language in ["en-US", "fr-FR", "de-DE", "en-US", "es-ES"]:
                with open(SPEECH_WAV, "rb") as f:
                    resp = http.post(
                        "/v1/convert/audio-to-iml",
                        files={"audio": ("speech.wav", f)},
                        data={"language": language},
                    )
                assert resp.status_code == 200, resp.text
                data = resp.json()
                assert data["transcript_source"] == "whisper"
                # Each request still transcribes in its own language.
                assert data["plain_text"] == f"hello {language[:2]}"
                assert f'language="{language}"' in data["iml"]
        loads = (tmp_path / "loads.log").read_text().splitlines()
        assert len(loads) == 1, loads

    def test_model_follows_the_setting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The server always loaded "base"; PP_STT_MODEL now chooses the model."""
        (tmp_path / "whisper.py").write_text(FAKE_WHISPER)
        monkeypatch.syspath_prepend(str(tmp_path))
        application = create_app(Settings(rate_limit_per_minute=0, stt_model="large-v3"))
        with TestClient(application, raise_server_exceptions=False) as http:
            resp = post_speech(http)
        assert resp.status_code == 200, resp.text
        assert resp.json()["transcript_source"] == "whisper"
        [load] = (tmp_path / "loads.log").read_text().splitlines()
        assert load.endswith(" large-v3")

    @pytest.mark.parametrize(
        ("broken", "status", "error"),
        [
            ("load_model", 503, "speech_recognition_unavailable"),
            ("transcribe", 500, "speech_recognition_failed"),
        ],
    )
    def test_whisper_failures_are_server_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        broken: str,
        status: int,
        error: str,
    ) -> None:
        """A model that cannot be downloaded used to be a 400 audio_processing_error,
        which tells the client its upload was bad."""
        fake = FAKE_WHISPER + (
            "\n\ndef load_model(name):\n    raise OSError('no network')\n"
            if broken == "load_model"
            else "\n\ndef _fail(self, audio, **options):\n    raise OSError('no network')\n"
            "\n\n_Model.transcribe = _fail\n"
        )
        (tmp_path / "whisper.py").write_text(fake)
        monkeypatch.syspath_prepend(str(tmp_path))
        application = create_app(Settings(rate_limit_per_minute=0, max_concurrent_jobs=1))
        with TestClient(application, raise_server_exceptions=False) as http:
            resp = post_speech(http)
            # Words need no speech recognition, so they still work.
            ok = post_speech(http, words=speech_words_json())
        assert resp.status_code == status, resp.text
        assert resp.json()["error"] == error
        assert "no network" in resp.json()["detail"]
        assert "tmp" not in resp.json()["detail"].replace(str(tmp_path), "")
        if status == 503:
            assert resp.headers["retry-after"] == "60"
            assert "Send the words" in resp.json()["detail"]
        assert ok.status_code == 200, ok.text


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A real uvicorn server on a free port, stopped after the module."""
    pytest.importorskip("uvicorn")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp_path_factory.mktemp("uvicorn") / "server.log"
    env = {**os.environ, "PP_MAX_UPLOAD_MB": "10", "PP_RATE_LIMIT": "0"}
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "prosody_protocol.server.app:app",
             "--port", str(port)],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                if httpx.get(f"{base_url}/v1/health", trust_env=False).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                pytest.fail("uvicorn did not start:\n" + log_path.read_text())
            time.sleep(0.2)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


class TestServerEntryPoint:
    def test_empty_pp_host_binds_loopback(self, tmp_path: Path) -> None:
        """PP_HOST= used to make the server listen on every interface."""
        pytest.importorskip("uvicorn")
        port = free_port()
        log_path = tmp_path / "server.log"
        env = {**os.environ, "PP_HOST": "", "PP_PORT": str(port)}
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "prosody_protocol.server"],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
        try:
            deadline = time.monotonic() + 60
            while "Uvicorn running on" not in log_path.read_text():
                if proc.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("the server did not start:\n" + log_path.read_text())
                time.sleep(0.2)
            assert f"Uvicorn running on http://127.0.0.1:{port}" in log_path.read_text()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


class TestLiveServer:
    """Behaviour that only a real HTTP server shows."""

    def test_chunked_body_over_limit_rejected(self, live_server: str) -> None:
        body = big_validate_body(12 * 1024 * 1024)
        with httpx.Client(trust_env=False, timeout=60) as http:
            resp = http.post(
                f"{live_server}/v1/validate",
                content=chunks(body),
                headers={"content-type": "application/json"},
            )
        assert resp.status_code == 413
        assert resp.json()["error"] == "payload_too_large"

    def test_json_floods_are_cut_off_and_health_stays_responsive(
        self, live_server: str
    ) -> None:
        """Three 9 MB JSON bodies of empty objects used to be parsed in full on
        the event loop (a GB of memory each at 48 MB), stalling /v1/health."""
        body = junk_json_body(9 * 1024 * 1024)
        statuses: list[int] = []

        def flood(chunked: bool) -> None:
            with httpx.Client(trust_env=False, timeout=120) as http:
                resp = http.post(
                    f"{live_server}/v1/convert/iml-to-ssml",
                    content=chunks(body) if chunked else body,
                    headers={"content-type": "application/json"},
                )
            statuses.append(resp.status_code)

        senders = [threading.Thread(target=flood, args=(i % 2 == 0,)) for i in range(4)]
        for sender in senders:
            sender.start()
        latencies = []
        with httpx.Client(trust_env=False, timeout=60) as http:
            while any(sender.is_alive() for sender in senders):
                start = time.monotonic()
                assert http.get(f"{live_server}/v1/health").status_code == 200
                latencies.append(time.monotonic() - start)
                time.sleep(0.05)
        for sender in senders:
            sender.join()
        assert statuses == [413] * 4
        assert max(latencies, default=0.0) < 1.0, latencies

    def test_health_responsive_during_long_conversion(
        self, live_server: str, tmp_path: Path
    ) -> None:
        audio = long_speech_wav(tmp_path / "long.wav", seconds=240).read_bytes()
        outcome: dict[str, object] = {}

        def convert() -> None:
            with httpx.Client(trust_env=False, timeout=300) as http:
                resp = http.post(
                    f"{live_server}/v1/convert/audio-to-iml",
                    files={"audio": ("long.wav", audio, "audio/wav")},
                )
            outcome["status"] = resp.status_code

        worker = threading.Thread(target=convert)
        worker.start()
        latencies = []  # Of health checks sent while the conversion was in flight.
        with httpx.Client(trust_env=False, timeout=60) as http:
            while worker.is_alive():
                start = time.monotonic()
                assert http.get(f"{live_server}/v1/health").status_code == 200
                latencies.append(time.monotonic() - start)
                time.sleep(0.1)
        worker.join()
        assert outcome["status"] == 200
        # Blocking the event loop would hold a health check for the whole conversion.
        assert max(latencies) < 1.0, latencies
        assert len(latencies) >= 3, "the conversion finished too quickly to measure"
