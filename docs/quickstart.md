# Quick Start

## Installation

```bash
pip install prosody-protocol
```

For audio processing capabilities:

```bash
pip install prosody-protocol[audio]
```

For the REST API server:

```bash
pip install prosody-protocol[api]
```

## Parse IML

```python
from prosody_protocol import IMLParser

parser = IMLParser()
doc = parser.parse('''
<utterance emotion="sarcastic" confidence="0.87">
  Oh, that's
  <prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">
    GREAT
  </prosody>.
</utterance>
''')

print(doc.utterances[0].emotion)  # "sarcastic"
print(parser.to_plain_text(doc))  # "Oh, that's GREAT."
```

## Validate IML

```python
from prosody_protocol import IMLValidator

validator = IMLValidator()
result = validator.validate('<utterance emotion="angry">Missing confidence</utterance>')

print(result.valid)  # False
for issue in result.issues:
    print(f"[{issue.severity}] {issue.rule}: {issue.message}")
```

## Convert Audio to IML

```python
from prosody_protocol import AudioToIML

converter = AudioToIML(stt_model="base")
iml_string = converter.convert("recording.wav")
print(iml_string)
```

## Predict Prosody from Text

```python
from prosody_protocol import TextToIML

predictor = TextToIML()
iml = predictor.predict("I can't believe this happened!", context="frustrated")
print(iml)
```

## Convert IML to SSML

```python
from prosody_protocol import IMLToSSML

converter = IMLToSSML()
ssml = converter.convert(iml_string)
print(ssml)
```

## Synthesize Audio from IML

```python
from prosody_protocol import IMLToAudio

synthesizer = IMLToAudio()
wav_bytes = synthesizer.synthesize('''
<utterance emotion="calm" confidence="0.9">
  Please <prosody rate="slow">listen carefully</prosody>.
</utterance>
''')

with open("output.wav", "wb") as f:
    f.write(wav_bytes)
```

## Load Prosody Profiles

```python
from prosody_protocol import ProfileLoader, ProfileApplier

loader = ProfileLoader()
profile = loader.load("profiles/autism_spectrum.json")

applier = ProfileApplier()
emotion, confidence = applier.apply(profile, features, "neutral", 0.5)
```

## Work with Datasets

```python
from prosody_protocol import DatasetLoader

loader = DatasetLoader()
dataset = loader.load("datasets/emotional-speech")

print(f"Dataset: {dataset.name}, {dataset.size} entries")

train, val, test = loader.split(dataset)
for entry in train:
    print(f"{entry.id}: {entry.emotion_label}")
```

## Run Benchmarks

```python
from prosody_protocol import Benchmark, BenchmarkReport, AudioToIML, DatasetLoader

loader = DatasetLoader()
dataset = loader.load("datasets/emotional-speech")
converter = AudioToIML()

benchmark = Benchmark(dataset, converter, dataset_dir="datasets/emotional-speech")
report = benchmark.run()

print(f"Emotion accuracy: {report.emotion_accuracy:.2%}")
print(f"Validity rate: {report.validity_rate:.2%}")

# Save for tracking
report.save("benchmarks/latest.json")
```

## Run the REST API

```bash
pip install prosody-protocol[api]
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000/docs` for interactive API documentation.

## Next Steps

- [API Reference](API.md) -- full class and method documentation
- [Integration Guides](integrations/) -- Whisper, Claude, ElevenLabs, Coqui TTS
- [Specification](../spec.md) -- formal IML spec
