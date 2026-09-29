"""Every module agrees on what a language tag is.

IML documents (validator V29, iml-1.0.xsd), dataset entries (D6,
dataset-entry.schema.json) and the SDK's constructors all use
``prosody_protocol._types.LANGUAGE_TAG_RE``. Documents and datasets are
checked as written; tags passed to the SDK's APIs may use the POSIX locale
form (``en_US``), which is read as ``en-US``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prosody_protocol import DatasetLoader, IMLAssembler, IMLToSSML, IMLValidator, WordAlignment
from prosody_protocol._types import is_language_tag, normalize_language_tag

SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"

VALID = ["en", "en-US", "fr-FR", "zh-Hant-TW", "ja", "x-klingon", "sgn-BE-FR", "english"]
INVALID = ["", "en_US", "en US", "en-", "-en", "123", "en-US-toolongsubtag", "én"]


def _entry(language: str) -> dict[str, object]:
    return {
        "id": "e1",
        "timestamp": "2025-01-15T10:30:00Z",
        "source": "recorded",
        "language": language,
        "audio_file": "audio/e1.wav",
        "transcript": "Hello world.",
        "iml": "<utterance>Hello world.</utterance>",
        "speaker_id": "speaker_01",
        "emotion_label": "neutral",
        "annotator": "human",
        "consent": True,
    }


@pytest.mark.parametrize("tag", VALID + INVALID)
def test_documents_and_datasets_agree(tag: str) -> None:
    expected = tag in VALID
    assert is_language_tag(tag) is expected
    v29 = IMLValidator().validate(f'<iml language="{tag}"><utterance>Hi.</utterance></iml>')
    assert (not any(i.rule == "V29" for i in v29.issues)) is expected
    d6 = DatasetLoader().validate_entry(_entry(tag))
    # An empty language is D-rule "missing field", not D6; either way it is invalid.
    assert d6.valid is expected


@pytest.mark.parametrize("tag", VALID + INVALID)
def test_dataset_schema_agrees(tag: str) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / "dataset-entry.schema.json").read_text(encoding="utf-8"))
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(_entry(tag)))
    assert (not errors) is (tag in VALID)


def test_api_inputs_accept_the_posix_locale_form() -> None:
    assert normalize_language_tag("en_US") == "en-US"
    assert IMLToSSML(default_language="en_US").default_language == "en-US"
    doc = IMLAssembler().assemble([WordAlignment("Hi.", 0, 300)], [], [], language="en_US")
    assert doc.language == "en-US"


@pytest.mark.parametrize("tag", ["", "en US", "en-", "123", 5])
def test_api_inputs_reject_other_values(tag: object) -> None:
    with pytest.raises(ValueError, match="BCP 47"):
        normalize_language_tag(tag)
    with pytest.raises(ValueError, match="BCP 47"):
        IMLToSSML(default_language=tag)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="BCP 47"):
        IMLAssembler().assemble(
            [WordAlignment("Hi.", 0, 300)], [], [], language=tag  # type: ignore[arg-type]
        )


def test_mavis_bridge_checks_its_language() -> None:
    pytest.importorskip("numpy")
    from prosody_protocol import MavisBridge

    assert MavisBridge(language="en_GB").language == "en-GB"
    with pytest.raises(ValueError, match="BCP 47"):
        MavisBridge(language="en GB")
