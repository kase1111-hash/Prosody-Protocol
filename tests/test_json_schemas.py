"""The JSON schemas in schemas/ agree with the SDK loaders.

Covers:
- Both schemas are valid Draft 2020-12 schemas
- Every profile and dataset-entry fixture gets the same verdict from the
  schema and from ProfileLoader / DatasetLoader
- A set of edge cases (types, enums, extra keys, path traversal, ...) gets
  the same verdict from both
- Entries exported by MavisBridge and categorize_features output conform

The loaders check more than JSON Schema can express (IML validity, audio
existence, duplicate ids, NaN); those rules are tested in test_datasets.py
and test_profiles.py.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

jsonschema = pytest.importorskip("jsonschema")

from prosody_protocol._types import PauseInterval, SpanFeatures
from prosody_protocol.datasets import DatasetLoader
from prosody_protocol.exceptions import ProfileError
from prosody_protocol.profiles import ProfileLoader, categorize_features

ROOT = Path(__file__).parent.parent
SCHEMAS = ROOT / "schemas"
FIXTURES = Path(__file__).parent / "fixtures"
PROFILE_FIXTURES = sorted((FIXTURES / "profiles").glob("*.json"))
ENTRY_FIXTURES = sorted((FIXTURES / "datasets").glob("*/entries/*.json"))


def _validator(name: str) -> Any:
    schema = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


PROFILE_SCHEMA = _validator("prosody-profile.schema.json")
ENTRY_SCHEMA = _validator("dataset-entry.schema.json")


def _schema_accepts(validator: Any, instance: object) -> bool:
    return not list(validator.iter_errors(instance))


def _loader_accepts_profile(data: object) -> bool:
    loader = ProfileLoader()
    try:
        profile = loader.load_json(data)  # type: ignore[arg-type]
    except ProfileError:
        return False
    return loader.validate(profile).valid


def _loader_accepts_entry(entry: object) -> bool:
    return DatasetLoader().validate_entry(entry).valid  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fixtures conform
# ---------------------------------------------------------------------------


class TestFixtures:
    def test_fixture_sets_are_not_empty(self) -> None:
        assert len(PROFILE_FIXTURES) >= 5
        assert len(ENTRY_FIXTURES) >= 13

    @pytest.mark.parametrize("path", PROFILE_FIXTURES, ids=lambda p: p.name)
    def test_profile_fixture_verdicts_agree(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = not path.name.startswith("invalid_")
        assert _schema_accepts(PROFILE_SCHEMA, data) is expected
        assert _loader_accepts_profile(data) is expected

    @pytest.mark.parametrize(
        "path", ENTRY_FIXTURES, ids=lambda p: f"{p.parent.parent.name}/{p.name}"
    )
    def test_dataset_entry_fixtures_conform(self, path: Path) -> None:
        entry = json.loads(path.read_text(encoding="utf-8"))
        assert _schema_accepts(ENTRY_SCHEMA, entry), list(ENTRY_SCHEMA.iter_errors(entry))
        result = DatasetLoader().validate_entry(entry, dataset_dir=path.parent.parent)
        assert result.valid, result.issues


# ---------------------------------------------------------------------------
# Profiles: loader and schema agree
# ---------------------------------------------------------------------------

_SPEC_PROFILE: dict[str, Any] = {
    "profile_version": "0.1.0",
    "user_id": "user_789",
    "description": "Autism spectrum - monotone speech with rate-based expression",
    "prosody_mappings": [
        {
            "pattern": {"pitch_contour": "flat", "rate": "fast"},
            "interpretation": {"emotion": "excitement", "confidence_boost": 0.15},
        },
        {
            "pattern": {"volume": "spike"},
            "interpretation": {"emotion": "emphasis_not_anger", "confidence_boost": 0.20},
        },
    ],
}


def _profile(**changes: Any) -> dict[str, Any]:
    """The spec profile with top-level keys replaced (_DELETE removes one)."""
    data = copy.deepcopy(_SPEC_PROFILE)
    for key, value in changes.items():
        if value is _DELETE:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def _mapping(pattern: Any = None, **interpretation: Any) -> dict[str, Any]:
    data = _profile()
    mapping = data["prosody_mappings"][0]
    if pattern is not None:
        mapping["pattern"] = pattern
    mapping["interpretation"].update(interpretation)
    return data


_DELETE = object()

_PROFILE_CASES: list[tuple[str, dict[str, Any], bool]] = [
    ("spec example", _profile(), True),
    ("description null", _profile(description=None), True),
    ("description missing", _profile(description=_DELETE), True),
    ("description not a string", _profile(description=5), False),
    ("prerelease version", _profile(profile_version="1.0.0-alpha.1"), True),
    ("two-part version", _profile(profile_version="1.0"), False),
    ("version not a string", _profile(profile_version=1), False),
    ("empty user_id", _profile(user_id=""), False),
    ("missing user_id", _profile(user_id=_DELETE), False),
    ("empty mappings", _profile(prosody_mappings=[]), False),
    ("mappings not a list", _profile(prosody_mappings={"a": 1}), False),
    ("extra top-level key", _profile(evil=1), False),
    ("metadata object", _profile(metadata={"author": "clinic", "extra": 1}), True),
    ("metadata not an object", _profile(metadata="x"), False),
    ("metadata null", _profile(metadata=None), False),
    ("metadata author not a string", _profile(metadata={"author": 5}), False),
    ("metadata created_at not a string", _profile(metadata={"created_at": 20250101}), False),
    ("metadata clinical_context null", _profile(metadata={"clinical_context": None}), False),
    ("metadata with every known key", _profile(metadata={
        "created_at": "2025-01-15T10:30:00Z", "updated_at": "2025-02-01T09:00:00Z",
        "author": "clinic", "clinical_context": "ASD, adult",
    }), True),
    ("integer boost", _mapping(confidence_boost=1), True),
    ("zero boost", _mapping(confidence_boost=0), True),
    ("boolean boost", _mapping(confidence_boost=True), False),
    ("string boost", _mapping(confidence_boost="0.1"), False),
    ("boost above 1", _mapping(confidence_boost=1.5), False),
    ("negative boost", _mapping(confidence_boost=-0.1), False),
    ("empty emotion", _mapping(emotion=""), False),
    ("custom emotion", _mapping(emotion="thinking_carefully"), True),
    ("extra interpretation key", _mapping(intensity=2), False),
    ("empty pattern", _mapping(pattern={}), False),
    ("unknown pattern key", _mapping(pattern={"pitch_countour": "flat"}), False),
    ("unknown pattern value", _mapping(pattern={"pitch": "very_high"}), False),
    ("non-string pattern value", _mapping(pattern={"pitch": 5}), False),
    ("every pattern key", _mapping(pattern={
        "pitch": "high", "pitch_contour": "rise-sharp", "volume": "spike", "rate": "slow",
        "quality": "creaky", "pause_frequency": "low", "emphasis_frequency": "normal",
    }), True),
    # The format the README once showed; it is not the spec 7.1 format.
    ("README user_profile format", {
        "user_profile": "user_789",
        "prosody_mappings": {"monotone_with_fast_rate": "excitement"},
    }, False),
]


def _extra_mapping_key() -> dict[str, Any]:
    data = _profile()
    data["prosody_mappings"][0]["weight"] = 2
    return data


_PROFILE_CASES.append(("extra mapping key", _extra_mapping_key(), False))


class TestProfileSchemaAgreement:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [(data, expected) for _, data, expected in _PROFILE_CASES],
        ids=[name for name, _, _ in _PROFILE_CASES],
    )
    def test_loader_and_schema_agree(self, data: dict[str, Any], expected: bool) -> None:
        assert _schema_accepts(PROFILE_SCHEMA, data) is expected
        assert _loader_accepts_profile(data) is expected

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_non_json_numbers_rejected(self, constant: str, tmp_path: Path) -> None:
        """JSON has no NaN or Infinity (RFC 8259), so the schema cannot see
        them; Python's json module accepts them, so the loader must refuse."""
        text = json.dumps(_SPEC_PROFILE).replace("0.15", constant)
        path = tmp_path / "profile.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ProfileError, match=constant.lstrip("-")):
            ProfileLoader().load(path)

    def test_categorize_features_output_is_schema_vocabulary(self) -> None:
        """Every value categorize_features produces can be used in a pattern."""
        rising = [150.0 * 2 ** (i / 40) for i in range(40)]  # +12 semitones
        spans = [
            SpanFeatures(start_ms=i * 300, end_ms=i * 300 + 250, text=f"w{i}",
                         f0_mean=rising[i * 4], f0_contour=rising[i * 4: i * 4 + 4],
                         intensity_mean=62.0 + (15.0 if i == 3 else 0.0),
                         speech_rate=6.5, quality="breathy")
            for i in range(10)
        ]
        baseline = [
            SpanFeatures(start_ms=0, end_ms=250, text="x", f0_mean=150.0,
                         f0_contour=[150.0] * 4, intensity_mean=60.0, speech_rate=4.5)
        ]
        pauses = [PauseInterval(start_ms=i * 300 + 250, end_ms=i * 300 + 300 + 250)
                  for i in range(0, 9, 2)]
        observed = categorize_features(spans, pauses, baseline=baseline)
        assert set(observed) == {
            "pitch", "pitch_contour", "volume", "rate", "quality",
            "pause_frequency", "emphasis_frequency",
        }
        profile = _mapping(pattern=observed)
        assert _schema_accepts(PROFILE_SCHEMA, profile), list(PROFILE_SCHEMA.iter_errors(profile))


# ---------------------------------------------------------------------------
# Dataset entries: loader and schema agree
# ---------------------------------------------------------------------------

_ENTRY: dict[str, Any] = {
    "id": "e1",
    "timestamp": "2025-01-15T10:30:00Z",
    "source": "recorded",
    "language": "en-US",
    "audio_file": "audio/e1.wav",
    "transcript": "Hello world.",
    "iml": "<utterance>Hello world.</utterance>",
    "speaker_id": "speaker_01",
    "emotion_label": "neutral",
    "annotator": "human",
    "consent": True,
}


def _entry(**changes: Any) -> dict[str, Any]:
    data = dict(_ENTRY)
    for key, value in changes.items():
        if value is _DELETE:
            data.pop(key, None)
        else:
            data[key] = value
    return data


_ENTRY_CASES: list[tuple[str, dict[str, Any], bool]] = [
    ("valid", _entry(), True),
    ("null speaker", _entry(speaker_id=None), True),
    ("no speaker", _entry(speaker_id=_DELETE), True),
    ("metadata object", _entry(metadata={"session": 1}), True),
    ("non-ISO timestamp (advisory)", _entry(timestamp="yesterday"), True),
    ("consent false", _entry(consent=False), False),
    ("consent missing", _entry(consent=_DELETE), False),
    ("consent as string", _entry(consent="true"), False),
    ("consent as 1", _entry(consent=1), False),
    ("unknown source", _entry(source="scraped"), False),
    ("unknown annotator", _entry(annotator="nobody"), False),
    ("not a BCP-47 tag", _entry(language="not a language"), False),
    ("empty timestamp", _entry(timestamp=""), False),
    ("empty transcript", _entry(transcript=""), False),
    ("blank transcript", _entry(transcript="   "), False),
    ("empty emotion", _entry(emotion_label=""), False),
    ("missing iml", _entry(iml=_DELETE), False),
    ("id not a string", _entry(id=5), False),
    ("extra field", _entry(notes="x"), False),
    ("speaker_id not a string", _entry(speaker_id=42), False),
    ("metadata not an object", _entry(metadata="x"), False),
    ("metadata null", _entry(metadata=None), False),
    ("parent traversal", _entry(audio_file="../../../../etc/passwd"), False),
    ("inner traversal", _entry(audio_file="audio/../../x.wav"), False),
    ("trailing ..", _entry(audio_file="audio/.."), False),
    ("backslash traversal", _entry(audio_file="audio\\..\\x.wav"), False),
    ("absolute POSIX path", _entry(audio_file="/etc/hostname"), False),
    ("Windows drive path", _entry(audio_file="C:\\audio\\x.wav"), False),
    ("UNC path", _entry(audio_file="\\\\host\\share\\x.wav"), False),
    ("dots inside a name", _entry(audio_file="audio/take..2.wav"), True),
    ("dot segment", _entry(audio_file="./audio/e1.wav"), True),
    ("blank audio path", _entry(audio_file="  "), False),
    # '.' in a regex does not match a newline, so a pattern written with '.*'
    # let these through while the loader rejected them.
    ("newline before traversal", _entry(audio_file="\n/../../etc/passwd"), False),
    ("newline inside traversal", _entry(audio_file="a\n/../../../x"), False),
    ("line separator before traversal", _entry(audio_file="a\u2028/../x"), False),
    ("NUL byte", _entry(audio_file="audio/a\x00b.wav"), False),
    ("tab in a name", _entry(audio_file="audio/take\t1.wav"), False),
    ("trailing newline", _entry(audio_file="audio/e1.wav\n"), False),
    ("non-ASCII name", _entry(audio_file="audio/grüße.wav"), True),
]


class TestDatasetEntrySchemaAgreement:
    @pytest.mark.parametrize(
        ("entry", "expected"),
        [(entry, expected) for _, entry, expected in _ENTRY_CASES],
        ids=[name for name, _, _ in _ENTRY_CASES],
    )
    def test_loader_and_schema_agree(self, entry: dict[str, Any], expected: bool) -> None:
        assert _schema_accepts(ENTRY_SCHEMA, entry) is expected
        assert _loader_accepts_entry(entry) is expected

    def test_invalid_iml_is_beyond_the_schema(self) -> None:
        """The schema cannot check IML; the loader does (rule D7)."""
        entry = _entry(iml='<utterance emotion="sad">no confidence</utterance>')
        assert _schema_accepts(ENTRY_SCHEMA, entry)
        assert not _loader_accepts_entry(entry)

    def test_mavis_export_conforms(self, tmp_path: Path) -> None:
        pytest.importorskip("numpy")
        from prosody_protocol.mavis_bridge import MavisBridge, PhonemeEvent

        events = [
            PhonemeEvent("t", start_ms=0, duration_ms=90, volume=0.5, pitch_hz=180.0),
            PhonemeEvent("o", start_ms=90, duration_ms=120, volume=0.5, pitch_hz=185.0),
            PhonemeEvent("m", start_ms=300, duration_ms=100, volume=0.9, pitch_hz=280.0),
        ]
        MavisBridge().export_dataset(
            [{"events": events, "transcript": "Tom & <Jerry>", "session_id": "s1",
              "emotion_label": "joyful", "speaker_id": "u1"}],
            tmp_path / "ds",
            consent=True,
        )
        stored = json.loads((tmp_path / "ds" / "entries" / "mavis_s1.json").read_text())
        assert _schema_accepts(ENTRY_SCHEMA, stored), list(ENTRY_SCHEMA.iter_errors(stored))
        assert _loader_accepts_entry(stored)
