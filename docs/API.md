# API Reference

## Core Classes

### `IMLParser`

Parse IML XML into structured Python objects.

| Method | Description |
|--------|-------------|
| `parse(iml_string) -> IMLDocument` | Parse an IML XML string |
| `parse_file(path) -> IMLDocument` | Parse an IML file from disk |
| `to_plain_text(doc) -> str` | Extract plain text, stripping markup |
| `to_iml_string(doc) -> str` | Serialize back to XML |

```python
from prosody_protocol import IMLParser

parser = IMLParser()
doc = parser.parse('<utterance emotion="joyful" confidence="0.9">Hello!</utterance>')
print(doc.utterances[0].emotion)  # "joyful"
```

### `IMLValidator`

Validate IML documents against the specification (16 rules: V1-V16).

| Method | Description |
|--------|-------------|
| `validate(iml_string) -> ValidationResult` | Validate an XML string |
| `validate_file(path) -> ValidationResult` | Validate a file |

```python
from prosody_protocol import IMLValidator

result = IMLValidator().validate(iml_string)
if not result.valid:
    for issue in result.issues:
        print(f"[{issue.severity}] {issue.rule}: {issue.message}")
```

### `AudioToIML`

Convert audio files to IML-annotated transcripts using Whisper STT and prosodic analysis.

| Method | Description |
|--------|-------------|
| `convert(audio_path) -> str` | Returns IML XML string |
| `convert_to_doc(audio_path) -> IMLDocument` | Returns parsed document |

```python
from prosody_protocol import AudioToIML

converter = AudioToIML(stt_model="base")
iml = converter.convert("speech.wav")
```

### `IMLToAudio`

Synthesize speech from IML markup (WAV, 22050 Hz, 16-bit PCM).

| Method | Description |
|--------|-------------|
| `synthesize(iml_string) -> bytes` | Returns WAV audio bytes |
| `synthesize_to_file(iml_string, path) -> None` | Write audio to file |
| `synthesize_doc(doc) -> bytes` | Synthesize from IMLDocument |

### `TextToIML`

Predict prosody for plain text using rule-based heuristics.

| Method | Description |
|--------|-------------|
| `predict(text, context=None) -> str` | Returns IML XML string |
| `predict_document(text, context=None) -> IMLDocument` | Returns parsed document |

```python
from prosody_protocol import TextToIML

predictor = TextToIML()
iml = predictor.predict("I can't believe this!", context="frustrated")
```

### `IMLToSSML`

Convert IML to SSML for TTS engines.

| Method | Description |
|--------|-------------|
| `convert(iml_string) -> str` | Returns SSML XML string |
| `convert_doc(doc) -> str` | Convert from IMLDocument |

### `IMLAssembler`

Programmatically build IML documents.

| Method | Description |
|--------|-------------|
| `begin_utterance(**attrs)` | Start a new utterance |
| `end_utterance()` | Close the current utterance |
| `add_word(text)` | Add a text span |
| `add_pause(duration)` | Add a pause element |
| `add_prosody(**attrs)` / `end_prosody()` | Wrap text in prosody |
| `add_emphasis(level)` / `end_emphasis()` | Wrap text in emphasis |
| `build() -> IMLDocument` | Build the final document |

## Analysis

### `ProsodyAnalyzer`

Extract acoustic features from audio using Praat.

| Method | Description |
|--------|-------------|
| `analyze(audio_path, alignments) -> list[SpanFeatures]` | Extract per-word features |
| `detect_pauses(audio_path, min_pause_ms=200) -> list[PauseInterval]` | Find silence gaps |

### `EmotionClassifier` (Protocol)

Interface for emotion classifiers. Implement `classify(features) -> (emotion, confidence)`.

### `RuleBasedEmotionClassifier`

Heuristic classifier mapping prosodic feature ranges to emotions.

| Method | Description |
|--------|-------------|
| `classify(features) -> tuple[str, float]` | Returns (emotion, confidence) |

Supported emotions: `neutral`, `angry`, `frustrated`, `joyful`, `sad`, `fearful`, `sarcastic`, `calm`.

## Profiles

### `ProfileLoader`

Load and validate JSON prosody profiles for accessibility.

| Method | Description |
|--------|-------------|
| `load(path) -> ProsodyProfile` | Load profile from JSON file |
| `load_json(data) -> ProsodyProfile` | Load from a dict |
| `validate(profile) -> ValidationResult` | Validate a profile (rules P1-P8) |

### `ProfileApplier`

Apply prosody profiles to modify emotion classification.

| Method | Description |
|--------|-------------|
| `apply(profile, features, emotion, confidence) -> tuple[str, float]` | Apply profile mappings |

## Datasets

### `DatasetLoader`

Load, validate, and split labelled datasets.

| Method | Description |
|--------|-------------|
| `load(dataset_dir) -> Dataset` | Load all entries from a directory |
| `iter_entries(dataset_dir) -> Iterator[DatasetEntry]` | Lazy iteration |
| `validate_entry(entry, dataset_dir=None) -> ValidationResult` | Validate against rules D1-D8 |
| `split(dataset, train=0.8, val=0.1, test=0.1, seed=42) -> tuple` | Deterministic split |

```python
from prosody_protocol import DatasetLoader

loader = DatasetLoader()
dataset = loader.load("datasets/emotional-speech")
train, val, test = loader.split(dataset)
```

### `DatasetEntry`

Frozen dataclass for a single dataset entry with fields: `id`, `timestamp`, `source`, `language`, `audio_file`, `transcript`, `iml`, `emotion_label`, `annotator`, `consent`, `speaker_id`, `metadata`.

### `Dataset`

Mutable dataclass with `name`, `entries`, `metadata`, and a `size` property.

## Benchmarks

### `Benchmark`

Evaluation harness for benchmarking AudioToIML converters.

| Method | Description |
|--------|-------------|
| `run(max_samples=None) -> BenchmarkReport` | Run the full benchmark |

```python
from prosody_protocol import Benchmark, DatasetLoader, AudioToIML

dataset = DatasetLoader().load("datasets/emotional-speech")
benchmark = Benchmark(dataset, AudioToIML(), dataset_dir="datasets/emotional-speech")
report = benchmark.run(max_samples=100)
```

### `BenchmarkReport`

Aggregated metrics from a benchmark run.

| Field | Description |
|-------|-------------|
| `emotion_accuracy` | Fraction of correct emotion predictions |
| `emotion_f1` | Per-class F1 scores (dict) |
| `confidence_ece` | Expected Calibration Error |
| `pitch_accuracy` | Pitch contour prediction accuracy |
| `pause_f1` | Pause detection F1 score |
| `validity_rate` | Fraction of valid generated IML |
| `num_samples` | Total entries processed |
| `duration_seconds` | Wall-clock time |

| Method | Description |
|--------|-------------|
| `save(path)` | Save as JSON |
| `load(path) -> BenchmarkReport` | Load from JSON (classmethod) |
| `check_regression(baseline, thresholds) -> list[str]` | CI regression check |
| `to_dict() -> dict` | JSON-serializable dictionary |

## Data Models

| Class | Description |
|-------|-------------|
| `IMLDocument` | Root document with `utterances`, `version`, `language` |
| `Utterance` | Spoken phrase with `children`, `emotion`, `confidence`, `speaker_id` |
| `Prosody` | Prosodic span with `pitch`, `pitch_contour`, `volume`, `rate`, `quality`, and extended attributes |
| `Pause` | Silence gap with `duration` (ms) |
| `Emphasis` | Stressed span with `level` (strong/moderate/reduced) |
| `Segment` | Clause-level grouping with `tempo` and `rhythm` |

## Exceptions

All inherit from `ProsodyProtocolError`:

| Exception | Description |
|-----------|-------------|
| `IMLParseError` | Malformed XML (includes line/column) |
| `IMLValidationError` | Spec rule violations |
| `ProfileError` | Profile loading/application failures |
| `AudioProcessingError` | Audio pipeline failures |
| `ConversionError` | Format conversion failures |
| `DatasetError` | Dataset loading/validation failures |
| `TrainingError` | Model training/evaluation failures |

## REST API

Install with `pip install prosody-protocol[api]` and run:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/convert/audio-to-iml` | Audio file to IML |
| POST | `/v1/convert/text-to-iml` | Plain text to IML |
| POST | `/v1/convert/iml-to-ssml` | IML to SSML |
| POST | `/v1/synthesize` | IML to audio (WAV) |
| POST | `/v1/validate` | Validate IML |
| GET | `/v1/health` | Health check |

Interactive docs available at `/docs` (Swagger UI) when the server is running.
