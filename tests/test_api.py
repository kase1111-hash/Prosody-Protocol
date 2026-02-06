"""Tests for the Prosody Protocol REST API (Phase 9).

Covers:
- Acceptance criteria: correct status codes, structured errors, OpenAPI spec
- Health endpoint
- Validation endpoint: valid/invalid IML
- Text-to-IML endpoint
- IML-to-SSML endpoint
- Synthesize endpoint: WAV audio response
- Audio-to-IML endpoint: file upload
- Error handling: malformed input returns 400 not 500
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app

AUDIO_FIXTURES = Path(__file__).parent / "fixtures" / "audio"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_has_version_string(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        data = resp.json()
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0


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
        assert data["plain_text"] is not None

    def test_text_with_context(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/convert/text-to-iml",
            json={"text": "That's GREAT!", "context": "Previous conversation."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "GREAT" in data["iml"]

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


# ---------------------------------------------------------------------------
# Synthesize endpoint
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Audio-to-IML endpoint (file upload)
# ---------------------------------------------------------------------------


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
        audio_file = AUDIO_FIXTURES / "tone_440hz.wav"
        with open(audio_file, "rb") as f:
            resp = client.post(
                "/v1/convert/audio-to-iml",
                files={"audio": ("test.wav", f, "audio/wav")},
                data={"language": "en-US"},
            )
        assert resp.status_code == 200

    def test_missing_file_returns_422(self, client: TestClient) -> None:
        resp = client.post("/v1/convert/audio-to-iml")
        assert resp.status_code == 422


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
        ]
        for path in expected:
            assert path in paths, f"Missing path: {path}"

    def test_openapi_info_has_version(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        info = resp.json()["info"]
        assert "version" in info
        assert info["title"] == "Prosody Protocol API"


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


# ---------------------------------------------------------------------------
# Upload size enforcement
# ---------------------------------------------------------------------------


class TestUploadSizeLimit:
    def test_oversized_upload_rejected(self, client: TestClient) -> None:
        """Uploads exceeding max_upload_size_mb should return 413."""
        from api.app import settings

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
        """A small upload should not be rejected by the size middleware."""
        small_payload = b"RIFF" + b"\x00" * 100
        resp = client.post(
            "/v1/convert/audio-to-iml",
            files={"audio": ("small.wav", small_payload, "audio/wav")},
        )
        # May fail on audio processing, but should NOT be 413
        assert resp.status_code != 413
