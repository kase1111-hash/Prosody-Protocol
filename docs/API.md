# API Reference

> **Status:** Pre-alpha. Method signatures are stable; implementations are in progress.

## Core Classes

### `IMLParser`

Parse IML XML into structured Python objects.

| Method | Description |
|--------|-------------|
| `parse(iml_string) -> IMLDocument` | Parse an IML XML string |
| `parse_file(path) -> IMLDocument` | Parse an IML file from disk |
| `to_plain_text(doc) -> str` | Extract plain text, stripping markup |
| `to_iml_string(doc) -> str` | Serialize back to XML |

### `IMLValidator`

Validate IML documents against the specification.

| Method | Description |
|--------|-------------|
| `validate(iml_string) -> ValidationResult` | Validate an XML string |
| `validate_file(path) -> ValidationResult` | Validate a file |

### `AudioToIML`

Convert audio files to IML-annotated transcripts.

| Method | Description |
|--------|-------------|
| `convert(audio_path) -> str` | Returns IML XML string |
| `convert_to_doc(audio_path) -> IMLDocument` | Returns parsed document |

### `IMLToAudio`

Synthesize speech from IML markup.

| Method | Description |
|--------|-------------|
| `synthesize(iml_string) -> bytes` | Returns WAV audio bytes |
| `synthesize_to_file(iml_string, path) -> None` | Write audio to file |

### `TextToIML`

Predict prosody for plain text.

| Method | Description |
|--------|-------------|
| `predict(text, context=None) -> str` | Returns IML XML string |

### `IMLToSSML`

Convert IML to SSML for TTS engines.

| Method | Description |
|--------|-------------|
| `convert(iml_string) -> str` | Returns SSML XML string |
| `convert_doc(doc) -> str` | Convert from IMLDocument |

### `ProsodyAnalyzer`

Extract acoustic features from audio.

| Method | Description |
|--------|-------------|
| `analyze(audio_path, alignments) -> list[SpanFeatures]` | Analyze prosody |

### `ProfileLoader` / `ProfileApplier`

Load and apply user prosody profiles.

| Method | Description |
|--------|-------------|
| `ProfileLoader.load(path) -> ProsodyProfile` | Load from JSON file |
| `ProfileApplier.apply(profile, features, emotion, confidence) -> (str, float)` | Apply profile |

## Data Models

See `prosody_protocol.models` for: `IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment`.

## Exceptions

All exceptions inherit from `ProsodyProtocolError`:

- `IMLParseError` -- malformed XML
- `IMLValidationError` -- spec rule violations
- `ProfileError` -- profile loading/application failures
- `AudioProcessingError` -- audio pipeline failures
- `ConversionError` -- format conversion failures

## REST API

When running the API server (`pip install prosody-protocol[api]`):

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/convert/audio-to-iml` | Audio file to IML |
| POST | `/v1/convert/text-to-iml` | Plain text to IML |
| POST | `/v1/convert/iml-to-ssml` | IML to SSML |
| POST | `/v1/synthesize` | IML to audio |
| POST | `/v1/validate` | Validate IML |
| GET | `/v1/health` | Health check |

Interactive docs available at `/docs` when the server is running.
