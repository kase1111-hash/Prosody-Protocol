# Quick Start

## Installation

```bash
pip install prosody-protocol
```

For audio processing capabilities:

```bash
pip install prosody-protocol[audio]
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

## Convert IML to SSML

```python
from prosody_protocol import IMLToSSML

converter = IMLToSSML()
ssml = converter.convert(iml_string)
print(ssml)
```

> **Note:** The SDK is in pre-alpha. Many classes currently raise
> `NotImplementedError` and will be implemented in upcoming phases.
> See the [Execution Guide](../EXECUTION_GUIDE.md) for the roadmap.
