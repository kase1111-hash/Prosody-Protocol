# Changelog

All notable changes to the Prosody Protocol SDK are documented here.

This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0a2] - 2026-02-06

### Added

#### Mavis Data Bridge (Phase 16)
- `MavisBridge` for converting Mavis PhonemeEvent streams to prosody_protocol datasets
- `PhonemeEvent` dataclass mirroring Mavis data types
- Feature extraction (7-dimensional) for sklearn training pipelines
- Batch feature extraction and dataset export
- Mavis integration guide (`docs/integrations/mavis.md`)

### Changed

#### Security Hardening (Phase 14)
- XML parser: explicit XXE prevention with `resolve_entities=False, no_network=True`
- Training model serialization: replaced `pickle` with `joblib` (safer deserialization)
- API upload size enforcement via `UploadSizeLimitMiddleware` (configurable, default 50MB)
- Validator V17/V18: consent and processing attribute validation
- Validator handles non-element nodes (entities, PIs) gracefully

#### Dependency & Spec Cleanup (Phase 15)
- Moved `torch` and `transformers` to new `ml-neural` extra (not required for sklearn baselines)
- `ml` extra now only requires `scikit-learn` and `pyyaml`
- Promoted `<segment>` tag to stable status in spec.md and CLAUDE.md

#### API Production Readiness (Phase 17)
- Rate limiting middleware (configurable via `PP_RATE_LIMIT` env var, default 60 req/min)
- CORS restricted to configured origins only (no wildcard default)
- Dockerfile runs as non-root user, includes `api/` directory, env var configuration
- `api/config.py` reads all settings from environment variables

#### Testing & CI (Phase 18)
- Added `test-ml-extras` CI job for sklearn training pipeline
- Added `test-api-extras` CI job for FastAPI endpoints
- Total test count: 526 (from 491)

### Infrastructure
- `pyproject.toml`: new `ml-neural` and `all-neural` extras groups

## [0.1.0a1] - 2026-02-05

### Added

#### Core SDK (Phases 1-5)
- `IMLParser` for parsing IML XML into structured `IMLDocument` objects
- `IMLValidator` with 16 validation rules (V1-V16) covering well-formedness, semantics, and spec compliance
- `IMLAssembler` for programmatic IML document construction
- `ProsodyAnalyzer` for extracting acoustic features (F0, intensity, jitter, shimmer, HNR) via Praat
- `AudioToIML` converter with Whisper STT integration and prosodic feature annotation
- `IMLToAudio` synthesizer producing WAV audio from IML markup
- `IMLToSSML` converter for mapping IML to SSML for TTS engines
- Data models: `IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment`
- `EmotionClassifier` protocol and `RuleBasedEmotionClassifier` baseline

#### Text-to-IML Prediction (Phase 6)
- `TextToIML` rule-based prosody predictor from plain text
- Sentiment lexicon with emotion detection from punctuation, capitalization, and word lists

#### Prosody Profiles (Phase 8)
- `ProfileLoader` for loading and validating JSON prosody profiles
- `ProfileApplier` for applying atypical prosody mappings (accessibility feature)
- Profile validation rules P1-P8

#### REST API (Phase 9)
- FastAPI server with 6 endpoints: audio-to-iml, text-to-iml, iml-to-ssml, synthesize, validate, health
- Error handling middleware for all SDK exception types
- Auto-generated Swagger/OpenAPI docs at `/docs`

#### Dataset Infrastructure (Phase 10)
- `DatasetLoader` with directory loading, entry validation (D1-D8), and lazy iteration
- `DatasetEntry` and `Dataset` dataclasses
- Deterministic train/val/test splitting with seeded randomization
- JSON Schema for dataset entries (`schemas/dataset-entry.schema.json`)

#### Model Training Pipelines (Phase 11)
- Config-driven training with YAML configuration files
- `ModelRegistry` with pluggable model architecture system
- Baseline models: logistic regression (SER), decision tree (text-to-prosody), random forest (pitch contour)
- Unified `train.py`, `evaluate.py`, `export.py`, and `data_prep.py` scripts
- Per-class precision/recall/F1 evaluation metrics

#### Evaluation & Benchmarks (Phase 12)
- `Benchmark` harness for evaluating AudioToIML converters against labelled datasets
- `BenchmarkReport` with 7 metrics: emotion accuracy, per-class F1, confidence ECE, pitch accuracy, pause F1, validity rate
- JSON report persistence for tracking metrics over time
- Regression detection for CI integration with baseline comparison and threshold checks

#### Documentation & Adoption (Phase 13)
- Quick Start Guide (`docs/quickstart.md`)
- API Reference (`docs/API.md`)
- Integration guides for Whisper, Claude, ElevenLabs, and Coqui TTS
- Contributing guide (`CONTRIBUTING.md`)
- Dockerfile for API server deployment
- GitHub Actions CI and PyPI publish workflows
- PEP 561 `py.typed` marker

### Infrastructure
- `pyproject.toml` with optional dependency groups: `audio`, `ml`, `api`, `dev`
- Comprehensive test suite (491+ tests)
- IML test fixtures in `tests/fixtures/`

[0.1.0a2]: https://github.com/kase1111-hash/Prosody-Protocol/releases/tag/v0.1.0a2
[0.1.0a1]: https://github.com/kase1111-hash/Prosody-Protocol/releases/tag/v0.1.0a1
