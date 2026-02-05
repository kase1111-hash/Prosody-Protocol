# Prosody Protocol -- Execution Guide

A step-by-step coding plan for bringing the Prosody Protocol from specification to production. Each phase is broken into discrete milestones with concrete deliverables, file layouts, dependencies, and acceptance criteria.

---

## Table of Contents

1. [Phase 1: Specification Finalization](#phase-1-specification-finalization)
2. [Phase 2: Project Scaffolding](#phase-2-project-scaffolding)
3. [Phase 3: Core SDK -- IML Parser & Validator](#phase-3-core-sdk----iml-parser--validator)
4. [Phase 4: Audio-to-IML Pipeline](#phase-4-audio-to-iml-pipeline)
5. [Phase 5: IML-to-Audio Synthesis](#phase-5-iml-to-audio-synthesis)
6. [Phase 6: Text-to-IML Prediction](#phase-6-text-to-iml-prediction)
7. [Phase 7: IML-to-SSML Conversion](#phase-7-iml-to-ssml-conversion)
8. [Phase 8: Prosody Profiles (Accessibility)](#phase-8-prosody-profiles-accessibility)
9. [Phase 9: REST API](#phase-9-rest-api)
10. [Phase 10: Dataset Infrastructure](#phase-10-dataset-infrastructure)
11. [Phase 11: Model Training Pipelines](#phase-11-model-training-pipelines)
12. [Phase 12: Evaluation & Benchmarks](#phase-12-evaluation--benchmarks)
13. [Phase 13: Documentation & Adoption](#phase-13-documentation--adoption)
14. [Dependency Graph](#dependency-graph)

---

## Phase 1: Specification Finalization

**Goal:** Lock the v1.0 spec so all downstream code targets a stable contract.

**Status:** Mostly complete -- `spec.md` exists with core tags defined.

### 1.1 Remaining Spec Tasks

| Task | Deliverable | Acceptance Criteria |
|------|-------------|---------------------|
| Add XML Schema Definition (XSD) | `schemas/iml-1.0.xsd` | Validates all Appendix C examples in `spec.md` |
| Add JSON Schema for prosody profiles | `schemas/prosody-profile.schema.json` | Validates the profile example in spec Section 7.1 |
| Write RFC-2119 conformance summary | Appendix D in `spec.md` | Lists every MUST/SHOULD/MAY requirement with section references |
| Resolve `<segment>` stability | Update Section 3.5 and 9.3 | Promote to stable or document migration path |

### 1.2 XSD Implementation

Create `schemas/iml-1.0.xsd` that enforces:

```
- <iml> root element with version and language attributes
- <utterance> requires confidence when emotion is present
- <pause> is empty, duration is xs:positiveInteger
- <prosody> attribute patterns: pitch matches (+|-)\d+(%|st) or \d+Hz
- <segment> only valid as direct child of <utterance>
- Nesting depth constraint (max 2 levels for prosody/emphasis)
```

### 1.3 Prosody Profile JSON Schema

Create `schemas/prosody-profile.schema.json` covering:

```json
{
  "required": ["profile_version", "user_id", "prosody_mappings"],
  "properties": {
    "profile_version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" },
    "user_id": { "type": "string" },
    "description": { "type": "string" },
    "prosody_mappings": {
      "type": "array",
      "items": {
        "required": ["pattern", "interpretation"],
        "properties": {
          "pattern": { "type": "object" },
          "interpretation": {
            "required": ["emotion"],
            "properties": {
              "emotion": { "type": "string" },
              "confidence_boost": { "type": "number", "minimum": 0, "maximum": 1 }
            }
          }
        }
      }
    }
  }
}
```

---

## Phase 2: Project Scaffolding

**Goal:** Set up the Python package, tooling, CI, and directory structure so all subsequent phases have a home.

### 2.1 Target Directory Layout

```
Prosody-Protocol/
├── README.md
├── spec.md
├── CLAUDE.md
├── EXECUTION_GUIDE.md
├── LICENSE                         # MIT for code
├── CONTRIBUTING.md
├── pyproject.toml                  # PEP 621 project metadata
├── setup.cfg                       # Optional legacy compat
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Lint + test on push/PR
│       └── publish.yml             # PyPI publish on release tag
├── schemas/
│   ├── iml-1.0.xsd
│   └── prosody-profile.schema.json
├── src/
│   └── prosody_protocol/
│       ├── __init__.py             # Public API re-exports
│       ├── py.typed                # PEP 561 marker
│       ├── parser.py               # IMLParser
│       ├── validator.py            # IMLValidator
│       ├── models.py               # Dataclasses: Utterance, Prosody, Pause, etc.
│       ├── audio_to_iml.py         # AudioToIML
│       ├── iml_to_audio.py         # IMLToAudio
│       ├── text_to_iml.py          # TextToIML
│       ├── iml_to_ssml.py          # IMLToSSML
│       ├── prosody_analyzer.py     # ProsodyAnalyzer
│       ├── profiles.py             # Prosody profile loader & applier
│       ├── exceptions.py           # Custom exception hierarchy
│       └── _version.py             # Single-source version
├── tests/
│   ├── conftest.py                 # Shared fixtures (sample IML strings, audio files)
│   ├── test_parser.py
│   ├── test_validator.py
│   ├── test_models.py
│   ├── test_audio_to_iml.py
│   ├── test_iml_to_audio.py
│   ├── test_text_to_iml.py
│   ├── test_iml_to_ssml.py
│   ├── test_prosody_analyzer.py
│   ├── test_profiles.py
│   └── fixtures/
│       ├── valid/                  # Valid IML documents
│       ├── invalid/                # Invalid IML documents (for validator tests)
│       └── audio/                  # Short audio clips for integration tests
├── datasets/
│   └── README.md                   # Dataset directory placeholder
├── docs/
│   ├── API.md
│   └── quickstart.md
└── api/                            # REST API server
    ├── __init__.py
    ├── app.py                      # FastAPI application
    ├── routes/
    │   ├── convert.py              # /v1/convert/audio-to-iml
    │   ├── synthesize.py           # /v1/synthesize
    │   └── validate.py             # /v1/validate
    └── config.py
```

### 2.2 `pyproject.toml` Configuration

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "prosody-protocol"
dynamic = ["version"]
description = "SDK for the Intent Markup Language (IML) -- preserving prosodic intent in speech-to-text."
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Kase Branham" }]

dependencies = [
    "lxml>=5.0",           # XML parsing
]

[project.optional-dependencies]
audio = [
    "librosa>=0.10",       # Audio feature extraction
    "soundfile>=0.12",     # Audio I/O
    "openai-whisper>=2023", # STT backbone
    "praat-parselmouth>=0.4", # Acoustic analysis (F0, jitter, shimmer)
]
ml = [
    "torch>=2.0",
    "transformers>=4.30",  # Text-to-IML prediction models
]
api = [
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "python-multipart>=0.0.6",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.5",
    "lxml-stubs",
]
all = ["prosody-protocol[audio,ml,api,dev]"]

[tool.hatch.version]
path = "src/prosody_protocol/_version.py"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers -v"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.mypy]
python_version = "3.10"
strict = true
```

### 2.3 CI Pipeline (`.github/workflows/ci.yml`)

Steps:

1. Checkout code
2. Set up Python 3.10, 3.11, 3.12 matrix
3. `pip install -e ".[dev]"`
4. `ruff check src/ tests/`
5. `mypy src/`
6. `pytest --cov=prosody_protocol --cov-report=xml`
7. Upload coverage to Codecov (optional)

### 2.4 Acceptance Criteria

- `pip install -e .` succeeds with only core dependencies
- `pytest` discovers the test directory (0 tests is fine at this stage)
- `ruff check` and `mypy` pass on empty source modules
- CI runs green on GitHub Actions

---

## Phase 3: Core SDK -- IML Parser & Validator

**Goal:** Parse IML XML into Python dataclasses and validate documents against the spec.

**Depends on:** Phase 1 (schemas), Phase 2 (scaffolding)

### 3.1 Data Models (`src/prosody_protocol/models.py`)

Define immutable dataclasses mirroring the IML tag set:

```python
@dataclass(frozen=True)
class Pause:
    duration: int  # milliseconds

@dataclass(frozen=True)
class Prosody:
    text: str
    pitch: str | None          # "+15%", "-2st", "185Hz"
    pitch_contour: str | None  # "rise", "fall", "fall-rise", etc.
    volume: str | None         # "+6dB", "-3dB"
    rate: str | None           # "fast", "150%"
    quality: str | None        # "breathy", "tense", etc.
    children: tuple[...]       # Nested Emphasis, Pause, text spans
    # Extended attributes (optional)
    f0_mean: float | None = None
    f0_range: str | None = None
    intensity_mean: float | None = None
    jitter: float | None = None
    shimmer: float | None = None
    hnr: float | None = None

@dataclass(frozen=True)
class Emphasis:
    level: str  # "strong", "moderate", "reduced"
    children: tuple[...]

@dataclass(frozen=True)
class Segment:
    tempo: str | None    # "rushed", "steady", "drawn-out"
    rhythm: str | None   # "staccato", "legato", "syncopated"
    children: tuple[...]

@dataclass(frozen=True)
class Utterance:
    emotion: str | None
    confidence: float | None
    speaker_id: str | None
    children: tuple[...]  # Text spans, Prosody, Pause, Emphasis, Segment

@dataclass(frozen=True)
class IMLDocument:
    version: str | None
    language: str | None
    utterances: tuple[Utterance, ...]
```

### 3.2 Parser (`src/prosody_protocol/parser.py`)

**Class:** `IMLParser`

**Implementation approach:**

1. Use `lxml.etree` for XML parsing (handles malformed XML with clear errors).
2. Recursive descent through element tree, constructing `models.*` objects.
3. Handle mixed content (text + child elements) by walking `element.text`, `child.tail` carefully.
4. Return `IMLDocument`.

**Public API:**

```python
class IMLParser:
    def parse(self, iml_string: str) -> IMLDocument: ...
    def parse_file(self, path: str | Path) -> IMLDocument: ...
    def to_plain_text(self, doc: IMLDocument) -> str: ...
    def to_iml_string(self, doc: IMLDocument) -> str: ...
```

### 3.3 Validator (`src/prosody_protocol/validator.py`)

**Class:** `IMLValidator`

**Validation rules to implement (from spec Sections 5-6):**

| # | Rule | Severity |
|---|------|----------|
| V1 | Document is well-formed XML | ERROR |
| V2 | At least one `<utterance>` exists | ERROR |
| V3 | `confidence` present when `emotion` is set | ERROR |
| V4 | `confidence` is float between 0.0 and 1.0 | ERROR |
| V5 | `<pause>` has `duration` attribute | ERROR |
| V6 | `<pause>` `duration` is a positive integer | ERROR |
| V7 | `<pause>` has no child content | ERROR |
| V8 | `<emphasis>` has `level` attribute | ERROR |
| V9 | `<emphasis>` `level` is one of: strong, moderate, reduced | WARNING |
| V10 | `<segment>` is direct child of `<utterance>` | ERROR |
| V11 | `<segment>` not nested in another `<segment>` | ERROR |
| V12 | Nesting depth of `<prosody>`/`<emphasis>` does not exceed 2 | WARNING |
| V13 | `pitch` value matches valid format | WARNING |
| V14 | `volume` value matches valid format | WARNING |
| V15 | `emotion` is from core vocabulary | INFO |
| V16 | No unknown elements present | INFO |

**Public API:**

```python
@dataclass
class ValidationIssue:
    severity: Literal["error", "warning", "info"]
    rule: str          # e.g. "V3"
    message: str
    line: int | None
    column: int | None

@dataclass
class ValidationResult:
    valid: bool                        # True if no errors
    issues: list[ValidationIssue]

class IMLValidator:
    def validate(self, iml_string: str) -> ValidationResult: ...
    def validate_file(self, path: str | Path) -> ValidationResult: ...
```

### 3.4 Test Plan

| Test File | Coverage |
|-----------|----------|
| `test_models.py` | Construct each model, verify immutability, repr |
| `test_parser.py` | Parse every Appendix C example; round-trip (parse -> serialize -> parse); malformed XML errors; mixed content ordering |
| `test_validator.py` | One test per validation rule (V1-V16); valid documents pass; documents with multiple errors return all of them |

**Test fixtures:** Place IML files in `tests/fixtures/valid/` and `tests/fixtures/invalid/` with names like `valid_sarcasm.xml`, `invalid_missing_confidence.xml`, etc.

### 3.5 Acceptance Criteria

- `IMLParser().parse(s)` round-trips all Appendix C examples
- `IMLParser().to_plain_text(doc)` strips all markup, preserving text only
- `IMLValidator().validate(s).valid` returns `True` for all valid fixtures
- `IMLValidator().validate(s)` catches every spec violation in invalid fixtures
- 100% branch coverage on parser and validator
- All types pass `mypy --strict`

---

## Phase 4: Audio-to-IML Pipeline

**Goal:** Convert an audio file into an IML-annotated transcript.

**Depends on:** Phase 3 (parser/models for output construction)

### 4.1 Architecture

```
audio.wav
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  STT Engine  │────>│  Prosody Analyzer │────>│  IML Assembler   │
│  (Whisper)   │     │  (Parselmouth +   │     │  (models.py)     │
│              │     │   librosa)        │     │                  │
│  text +      │     │  F0, intensity,   │     │  Utterance(      │
│  timestamps  │     │  rate, quality    │     │    emotion=...,  │
└──────────────┘     └──────────────────┘     │    children=...  │
                                               │  )               │
                                               └──────────────────┘
                                                       │
                                                       ▼
                                                   IML string
```

### 4.2 Sub-Components

#### 4.2.1 `ProsodyAnalyzer` (`src/prosody_protocol/prosody_analyzer.py`)

Extracts acoustic features from audio given word-level timestamps.

**Inputs:** Audio file path + word-level alignments (from Whisper).

**Outputs per word/span:**

| Feature | Source Library | Method |
|---------|---------------|--------|
| F0 mean, range, contour | `parselmouth` | `Sound.to_pitch()` -> `Pitch.get_value_at_time()` |
| Intensity mean, range | `parselmouth` | `Sound.to_intensity()` |
| Speech rate | `librosa` | Syllable nuclei detection via amplitude envelope |
| Jitter, shimmer | `parselmouth` | `call(point_process, "Get jitter (local)")` |
| HNR | `parselmouth` | `call(harmonicity, "Get mean")` |
| Pause detection | `librosa` | Silent interval detection via RMS threshold |
| Voice quality | Derived | Classify from jitter/shimmer/HNR thresholds |

**Public API:**

```python
@dataclass
class SpanFeatures:
    start_ms: int
    end_ms: int
    text: str
    f0_mean: float | None
    f0_range: tuple[float, float] | None
    f0_contour: list[float] | None
    intensity_mean: float | None
    intensity_range: float | None
    speech_rate: float | None
    jitter: float | None
    shimmer: float | None
    hnr: float | None
    quality: str | None  # "modal", "breathy", "tense", "creaky"

class ProsodyAnalyzer:
    def analyze(self, audio_path: str, alignments: list[WordAlignment]) -> list[SpanFeatures]: ...
```

#### 4.2.2 Emotion Classifier

**Approach options (implement in order of complexity):**

1. **Rule-based baseline:** Map prosodic feature combinations to emotions using heuristic thresholds (e.g., high F0 + high intensity + fast rate -> "angry"). Include confidence based on how strongly features match.
2. **Pre-trained model:** Use a speech emotion recognition (SER) model (e.g., `wav2vec2` fine-tuned on emotion datasets). Wrap behind a common interface so the backend is swappable.

```python
class EmotionClassifier(Protocol):
    def classify(self, audio_path: str, start_ms: int, end_ms: int) -> tuple[str, float]:
        """Returns (emotion_label, confidence)."""
        ...

class RuleBasedEmotionClassifier:
    """Heuristic classifier using prosodic feature thresholds."""
    ...

class ModelEmotionClassifier:
    """Neural SER model wrapper."""
    ...
```

#### 4.2.3 IML Assembler

Takes STT text, prosody features, and emotion classification, and constructs an `IMLDocument`:

1. Group words into utterances (by sentence boundaries or pause thresholds > 1000ms).
2. For each utterance, classify emotion.
3. For each word span, determine if prosody deviates significantly from speaker baseline.
4. Wrap deviating spans in `<prosody>` tags with relative values.
5. Detect emphasis from intensity/F0 spikes.
6. Insert `<pause>` elements for significant gaps (> 200ms).

#### 4.2.4 `AudioToIML` (`src/prosody_protocol/audio_to_iml.py`)

Top-level orchestrator:

```python
class AudioToIML:
    def __init__(
        self,
        stt_model: str = "base",
        emotion_classifier: EmotionClassifier | None = None,
        include_extended: bool = False,
    ): ...

    def convert(self, audio_path: str | Path) -> str:
        """Returns IML XML string."""
        ...

    def convert_to_doc(self, audio_path: str | Path) -> IMLDocument:
        """Returns parsed IMLDocument."""
        ...
```

### 4.3 Test Plan

| Test | Strategy |
|------|----------|
| `ProsodyAnalyzer` unit tests | Use short synthetic audio (sine waves, silence) with known F0/intensity |
| Emotion classifier unit tests | Feed known feature vectors, assert expected labels |
| `AudioToIML` integration test | Use 3-5 short recorded samples with hand-annotated expected IML; assert structural correctness (right tags present), not exact attribute values |
| Round-trip test | `AudioToIML` -> `IMLParser().parse()` succeeds and `IMLValidator().validate()` is valid |

### 4.4 Acceptance Criteria

- Given a WAV file with clear sarcasm, the output IML contains `emotion="sarcastic"` with confidence > 0.6
- Pauses > 500ms appear as `<pause>` elements
- Extended attributes (F0, intensity, etc.) appear only when `include_extended=True`
- Output always passes `IMLValidator`
- Runs on CPU (GPU optional for neural emotion classifier)

---

## Phase 5: IML-to-Audio Synthesis

**Goal:** Generate speech audio from IML markup, preserving prosodic intent.

**Depends on:** Phase 3 (parser to read IML input)

### 5.1 Architecture

```
IML string
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  IML Parser  │────>│  SSML Converter  │────>│  TTS Engine      │
│              │     │  (iml_to_ssml)   │     │  (Coqui / Piper  │
│  IMLDocument │     │                  │     │   / ElevenLabs)  │
└──────────────┘     │  SSML string     │     │                  │
                     └──────────────────┘     │  audio bytes     │
                                               └──────────────────┘
```

### 5.2 Implementation

```python
class IMLToAudio:
    def __init__(
        self,
        voice: str = "en_US-female-medium",
        engine: Literal["coqui", "piper", "elevenlabs"] = "coqui",
    ): ...

    def synthesize(self, iml_string: str) -> AudioSegment: ...
    def synthesize_to_file(self, iml_string: str, output_path: str | Path) -> None: ...
```

**Strategy:**

1. Parse IML to `IMLDocument`.
2. Convert to SSML using `IMLToSSML` (Phase 7 -- implement that first or in parallel).
3. Pass SSML to TTS engine.
4. If TTS engine doesn't support full SSML, apply post-processing: pitch shifting, volume adjustment, pause insertion using `librosa`/`pydub`.

### 5.3 Acceptance Criteria

- Given IML with `pitch="+15%"`, output audio has measurably higher F0 than baseline
- `<pause duration="800"/>` produces ~800ms of silence in the output
- `<emphasis level="strong">` words are louder/higher-pitched than surrounding text
- Output is valid WAV/MP3

---

## Phase 6: Text-to-IML Prediction

**Goal:** Predict likely prosody for plain text (no audio input).

**Depends on:** Phase 3 (models/parser for output)

### 6.1 Architecture

```
plain text + optional context
    │
    ▼
┌──────────────────────────┐     ┌──────────────────┐
│  Prosody Prediction Model │────>│  IML Assembler   │
│  (fine-tuned transformer) │     │                  │
│                           │     │  IMLDocument     │
│  Per-token predictions:   │     └──────────────────┘
│  - emotion                │
│  - emphasis               │
│  - pitch direction        │
│  - pause locations        │
└──────────────────────────┘
```

### 6.2 Implementation Phases

**Phase 6a -- Rule-based baseline:**

Use punctuation, capitalization, and lexical cues:

| Cue | Prediction |
|-----|------------|
| `!` at end | `emotion="frustrated"` or `"joyful"`, `pitch="+5%"` |
| `?` at end | `pitch_contour="rise"` |
| ALL CAPS word | `<emphasis level="strong">` |
| `...` | `<pause duration="500"/>` |
| `"` quoted speech | Separate `<utterance>` |
| Sentiment lexicon hit | Set `emotion` with lower confidence |

**Phase 6b -- ML model:**

Fine-tune a sequence labeling model (e.g., `bert-base-uncased`) to predict per-token prosodic labels. Training data comes from the dataset infrastructure (Phase 10).

```python
class TextToIML:
    def __init__(
        self,
        model: str = "rule-based",  # or "prosody-bert-large"
        default_confidence: float = 0.6,
    ): ...

    def predict(self, text: str, context: str | None = None) -> str:
        """Returns IML XML string."""
        ...
```

### 6.3 Acceptance Criteria

- "Oh, that's GREAT." produces `<emphasis level="strong">GREAT</emphasis>`
- "Really?..." produces `pitch_contour="rise"` and a `<pause>`
- All output passes `IMLValidator`
- Rule-based model requires zero external dependencies beyond `lxml`

---

## Phase 7: IML-to-SSML Conversion

**Goal:** Convert IML markup to SSML for consumption by standard TTS engines.

**Depends on:** Phase 3 (parser)

### 7.1 Tag Mapping

| IML | SSML |
|-----|------|
| `<utterance>` | `<s>` (sentence) |
| `<prosody pitch="+15%" volume="+6dB" rate="fast">` | `<prosody pitch="+15%" volume="+6dB" rate="fast">` (direct mapping) |
| `<pause duration="800"/>` | `<break time="800ms"/>` |
| `<emphasis level="strong">` | `<emphasis level="strong">` (direct mapping) |
| `<segment>` | Flatten -- apply attributes to children |
| `emotion`, `confidence` | Stripped (SSML has no equivalent) |
| `quality="breathy"` | No standard SSML mapping; drop or use vendor extensions |
| Extended attributes | Stripped |

### 7.2 Implementation

```python
class IMLToSSML:
    def __init__(self, vendor: str | None = None): ...

    def convert(self, iml_string: str) -> str:
        """Returns SSML XML string."""
        ...

    def convert_doc(self, doc: IMLDocument) -> str: ...
```

### 7.3 Acceptance Criteria

- Output is valid SSML 1.1
- Round-trip: IML -> SSML -> TTS engine produces audio without errors
- Attributes that have no SSML equivalent are silently dropped
- `<pause duration="800"/>` becomes `<break time="800ms"/>`

---

## Phase 8: Prosody Profiles (Accessibility)

**Goal:** Load, validate, and apply user prosody profiles to modify emotion classification.

**Depends on:** Phase 3 (validator), Phase 4 (emotion classifier)

### 8.1 Implementation

```python
@dataclass
class ProsodyMapping:
    pattern: dict[str, str]           # e.g. {"pitch_contour": "flat", "rate": "fast"}
    interpretation_emotion: str
    confidence_boost: float

@dataclass
class ProsodyProfile:
    profile_version: str
    user_id: str
    description: str | None
    mappings: list[ProsodyMapping]

class ProfileLoader:
    def load(self, path: str | Path) -> ProsodyProfile: ...
    def load_json(self, data: dict) -> ProsodyProfile: ...
    def validate(self, profile: ProsodyProfile) -> ValidationResult: ...

class ProfileApplier:
    def apply(
        self,
        profile: ProsodyProfile,
        features: SpanFeatures,
        base_emotion: str,
        base_confidence: float,
    ) -> tuple[str, float]:
        """Returns (adjusted_emotion, adjusted_confidence)."""
        ...
```

### 8.2 Profile Matching Logic

1. For each `ProsodyMapping` in the profile, check if **all** pattern keys match the observed features.
2. Matching is fuzzy for continuous values (e.g., `"rate": "fast"` matches `speech_rate > 5.0`).
3. If multiple mappings match, use the one with the most specific pattern (most keys).
4. Apply `confidence_boost` (capped at 1.0).

### 8.3 Acceptance Criteria

- Profile JSON from spec Section 7.1 loads and validates successfully
- Applying the autism-spectrum profile to flat-pitch + fast-rate features produces `emotion="excitement"`
- Invalid profiles (missing fields, bad version) fail validation with clear errors
- `confidence_boost` never produces confidence > 1.0

---

## Phase 9: REST API

**Goal:** Expose SDK functionality over HTTP.

**Depends on:** Phase 3-7 (all SDK components it wraps)

### 9.1 Endpoints

| Method | Path | Request | Response | SDK Class |
|--------|------|---------|----------|-----------|
| `POST` | `/v1/convert/audio-to-iml` | `multipart/form-data` with `audio` file + optional `language` | `{ "iml": "...", "plain_text": "..." }` | `AudioToIML` |
| `POST` | `/v1/synthesize` | `{ "iml": "...", "voice": "..." }` | Audio file (WAV) | `IMLToAudio` |
| `POST` | `/v1/validate` | `{ "iml": "..." }` | `{ "valid": bool, "issues": [...] }` | `IMLValidator` |
| `POST` | `/v1/convert/text-to-iml` | `{ "text": "...", "context": "..." }` | `{ "iml": "..." }` | `TextToIML` |
| `POST` | `/v1/convert/iml-to-ssml` | `{ "iml": "..." }` | `{ "ssml": "..." }` | `IMLToSSML` |
| `GET`  | `/v1/health` | -- | `{ "status": "ok", "version": "..." }` | -- |

### 9.2 Implementation (`api/app.py`)

Use FastAPI with:

- Pydantic models for request/response validation
- Dependency injection for SDK class instances
- File upload handling via `python-multipart`
- Error handling middleware that returns structured error responses
- CORS middleware for browser clients
- OpenAPI docs auto-generated at `/docs`

### 9.3 Deployment Considerations

- Containerize with a `Dockerfile` (Python slim base, multi-stage build)
- GPU support optional via `nvidia/cuda` base image
- Health check endpoint for orchestrators
- Rate limiting via middleware or reverse proxy

### 9.4 Test Plan

| Test | Strategy |
|------|----------|
| Endpoint unit tests | Use `httpx.AsyncClient` with FastAPI `TestClient` |
| File upload test | POST a WAV fixture to `/v1/convert/audio-to-iml` |
| Validation error test | POST invalid IML to `/v1/validate`, assert issues returned |
| OpenAPI schema test | Fetch `/openapi.json`, validate it's well-formed |

### 9.5 Acceptance Criteria

- All endpoints return correct status codes (200, 400, 422, 500)
- Audio upload and download work end-to-end
- Invalid IML returns structured validation errors, not 500
- OpenAPI spec is auto-generated and accurate
- Docker image builds and runs successfully

---

## Phase 10: Dataset Infrastructure

**Goal:** Create the directory structure, schemas, loaders, and contribution pipeline for training datasets.

**Depends on:** Phase 1 (schemas), Phase 3 (parser/validator for IML annotations)

### 10.1 Dataset Schema

Each dataset entry is a JSON object:

```json
{
  "id": "string",
  "timestamp": "ISO-8601",
  "source": "mavis | recorded | synthetic",
  "language": "BCP-47",
  "audio_file": "relative/path.wav",
  "transcript": "plain text",
  "iml": "<utterance>...</utterance>",
  "speaker_id": "string | null",
  "emotion_label": "string",
  "annotator": "human | model | hybrid",
  "consent": true,
  "metadata": {}
}
```

Create `schemas/dataset-entry.schema.json` to validate this.

### 10.2 Dataset Loader

```python
class DatasetLoader:
    def load(self, dataset_dir: str | Path) -> Dataset: ...
    def validate_entry(self, entry: dict) -> ValidationResult: ...
    def iter_entries(self, dataset_dir: str | Path) -> Iterator[DatasetEntry]: ...
    def split(self, dataset: Dataset, train: float, val: float, test: float) -> tuple: ...
```

### 10.3 Dataset Directories

```
datasets/
├── README.md                    # Overview, licensing, contribution guide
├── mavis-corpus/
│   ├── metadata.json            # { "size": 0, "version": "0.1.0", ... }
│   ├── entries/                 # One JSON file per entry
│   ├── audio/                   # WAV files
│   └── README.md
├── emotional-speech/
│   ├── metadata.json
│   ├── entries/
│   ├── audio/
│   └── README.md
├── conversational/
│   ├── metadata.json
│   ├── entries/
│   ├── audio/
│   └── README.md
└── accessibility/
    ├── metadata.json
    ├── entries/
    ├── audio/
    └── README.md
```

### 10.4 Acceptance Criteria

- `DatasetLoader` can load a directory of entries and validate all of them
- Invalid entries (missing fields, bad IML) produce clear validation errors
- Train/val/test split is deterministic given a seed
- Audio files referenced in entries exist on disk (checked during validation)

---

## Phase 11: Model Training Pipelines

**Goal:** Train and evaluate models for prosody detection and emotion classification.

**Depends on:** Phase 4 (feature extraction), Phase 10 (datasets)

### 11.1 Training Tasks

| Task | Input | Output | Architecture |
|------|-------|--------|-------------|
| Speech Emotion Recognition (SER) | Audio segment | Emotion label + confidence | wav2vec2 + classification head |
| Prosody Feature Prediction | Audio + text alignment | Per-word prosodic features | Multi-task regression on F0, intensity, rate |
| Text-to-Prosody | Plain text | Per-token prosodic labels | BERT + sequence labeling |
| Pitch Contour Classification | F0 sequence | Contour label (rise, fall, etc.) | 1D CNN or LSTM |

### 11.2 Pipeline Structure

```
training/
├── configs/
│   ├── ser_wav2vec2.yaml
│   ├── text_to_prosody_bert.yaml
│   └── pitch_contour_cnn.yaml
├── scripts/
│   ├── train.py               # Unified training entry point
│   ├── evaluate.py            # Evaluation + metric reporting
│   ├── export.py              # Export to ONNX / TorchScript
│   └── data_prep.py           # Convert dataset entries to model input format
└── checkpoints/               # Saved model weights (gitignored)
```

### 11.3 Training Script Interface

```bash
# Train SER model
python training/scripts/train.py --config training/configs/ser_wav2vec2.yaml \
    --dataset datasets/emotional-speech \
    --output training/checkpoints/ser_v1

# Evaluate
python training/scripts/evaluate.py --checkpoint training/checkpoints/ser_v1 \
    --dataset datasets/emotional-speech --split test
```

### 11.4 Acceptance Criteria

- Training script runs end-to-end on a small synthetic dataset (10 samples)
- Evaluation script produces precision/recall/F1 per emotion class
- Exported model loads and runs inference via the SDK classes
- Configs are YAML; no hardcoded hyperparameters in scripts

---

## Phase 12: Evaluation & Benchmarks

**Goal:** Define metrics, build evaluation harness, and publish baseline benchmarks.

**Depends on:** Phase 4, 6, 11

### 12.1 Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Emotion Accuracy** | % of utterances with correct emotion label | > 75% (basic), > 85% (with context) |
| **Emotion F1 (macro)** | Per-class balance | > 0.70 |
| **Confidence Calibration** | ECE (Expected Calibration Error) | < 0.10 |
| **Pitch Direction Accuracy** | Correct contour label (rise/fall/etc.) | > 80% |
| **Pause Detection F1** | Precision/recall of detected pauses | > 0.85 |
| **IML Validity Rate** | % of generated IML that passes validation | 100% |
| **Round-trip Fidelity** | Audio -> IML -> Audio prosodic similarity | Pearson r > 0.7 for F0 contour |

### 12.2 Benchmark Harness

```python
class Benchmark:
    def __init__(self, dataset: Dataset, converter: AudioToIML): ...

    def run(self) -> BenchmarkReport: ...

@dataclass
class BenchmarkReport:
    emotion_accuracy: float
    emotion_f1: dict[str, float]       # per-class
    confidence_ece: float
    pitch_accuracy: float
    pause_f1: float
    validity_rate: float
    num_samples: int
    duration_seconds: float
```

### 12.3 Acceptance Criteria

- Benchmark runs against the emotional-speech dataset
- Report is saved as JSON for tracking over time
- CI can run benchmarks on a subset and fail if metrics regress

---

## Phase 13: Documentation & Adoption

**Goal:** Make the project usable by external developers and researchers.

**Depends on:** All prior phases

### 13.1 Documentation Deliverables

| Document | Location | Content |
|----------|----------|---------|
| Quick Start Guide | `docs/quickstart.md` | Install, parse first IML, validate, convert audio |
| API Reference | `docs/API.md` | All public classes and methods with examples |
| REST API Docs | Auto-generated | FastAPI `/docs` endpoint (Swagger UI) |
| Integration Guides | `docs/integrations/` | Whisper, Claude, ElevenLabs, Coqui TTS |
| Contributing Guide | `CONTRIBUTING.md` | Code style, PR process, dataset contribution |
| Changelog | `CHANGELOG.md` | Per-version changes |

### 13.2 Package Publishing

1. Set up PyPI publishing via GitHub Actions (on tag push).
2. Package name: `prosody-protocol`.
3. Include `py.typed` marker for PEP 561.
4. Publish to TestPyPI first, then PyPI.

### 13.3 Adoption Checklist

- [ ] `pip install prosody-protocol` works
- [ ] `from prosody_protocol import IMLParser, IMLValidator` works
- [ ] README quick-start code runs without modification
- [ ] At least 3 integration examples are tested end-to-end
- [ ] API server is deployable via Docker
- [ ] Benchmark results published in README or docs

---

## Dependency Graph

This shows which phases must complete before others can start. Phases at the same level can run in parallel.

```
Phase 1: Spec Finalization
    │
    ▼
Phase 2: Project Scaffolding
    │
    ├───────────────────────────────────┐
    ▼                                   ▼
Phase 3: Parser & Validator        Phase 10: Dataset Infra
    │                                   │
    ├──────────┬──────────┬─────────┐   │
    ▼          ▼          ▼         ▼   │
Phase 4:   Phase 6:   Phase 7:  Phase 8 │
Audio→IML  Text→IML   IML→SSML Profiles │
    │          │          │         │   │
    │          │          ▼         │   │
    │          │      Phase 5:     │   │
    │          │      IML→Audio    │   │
    │          │          │         │   │
    ├──────────┴──────────┴─────────┤   │
    ▼                               ▼   ▼
Phase 9: REST API              Phase 11: Training
    │                               │
    ▼                               ▼
Phase 12: Evaluation & Benchmarks
    │
    ▼
Phase 13: Documentation & Adoption
```

### Recommended Execution Order

| Sprint | Phases | Rationale |
|--------|--------|-----------|
| 1 | 1, 2 | Foundation -- schemas and scaffolding must exist first |
| 2 | 3 | Core SDK -- everything depends on parser and validator |
| 3 | 7, 8 | IML-to-SSML and profiles are self-contained, moderate scope |
| 4 | 4 | Audio-to-IML is the flagship feature, largest scope |
| 5 | 5, 6 | Synthesis and text prediction can proceed in parallel |
| 6 | 9 | REST API wraps everything built so far |
| 7 | 10 | Dataset infrastructure for training |
| 8 | 11, 12 | Training and evaluation together |
| 9 | 13 | Documentation and release preparation |

---

## Appendix: Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| XML parser | `lxml` | Fast, full XPath/XSLT support, good error messages |
| Audio analysis | `parselmouth` (Praat) + `librosa` | Parselmouth is the gold standard for acoustic phonetics; librosa for general audio |
| STT engine | OpenAI Whisper | Best open-source STT with word-level timestamps |
| TTS engine | Coqui TTS (default) | Open-source, SSML support, local inference |
| ML framework | PyTorch + HuggingFace | Ecosystem support, pre-trained models available |
| API framework | FastAPI | Async, auto-docs, Pydantic integration |
| Build system | Hatch | Modern, PEP 621, single-file config |
| Serialization | Immutable dataclasses | Thread-safe, hashable, good for caching |
