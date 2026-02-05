# Coqui TTS Integration

Use Prosody Protocol with Coqui TTS for open-source speech synthesis.

## Concept

Coqui TTS is an open-source text-to-speech engine that supports SSML input.
The `IMLToSSML` converter translates IML prosody markup into SSML that Coqui
can use to generate expressive speech.

## Setup

```bash
pip install prosody-protocol TTS
```

## Basic Usage

```python
from prosody_protocol import IMLToSSML

converter = IMLToSSML()

iml = '''<utterance emotion="calm">
  Please <prosody rate="slow" pitch="-3%">take your time</prosody>.
</utterance>'''

ssml = converter.convert(iml)
print(ssml)
```

## With Coqui TTS

```python
from TTS.api import TTS
from prosody_protocol import IMLToSSML, IMLParser

# Initialize Coqui TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Convert IML to plain text (Coqui basic mode)
parser = IMLParser()
doc = parser.parse(iml)
text = parser.to_plain_text(doc)

# Synthesize
tts.tts_to_file(text=text, file_path="output.wav")
```

## Using Prosodic Cues for Voice Selection

Use the emotion from IML to select appropriate Coqui voices or styles:

```python
from prosody_protocol import IMLParser

parser = IMLParser()
doc = parser.parse(iml)
emotion = doc.utterances[0].emotion

# Map emotions to voice styles
voice_map = {
    "calm": "tts_models/en/ljspeech/tacotron2-DDC",
    "joyful": "tts_models/en/ljspeech/glow-tts",
    "sad": "tts_models/en/ljspeech/tacotron2-DDC",
}

model_name = voice_map.get(emotion, voice_map["calm"])
tts = TTS(model_name=model_name)
```

## IML-to-Audio via Built-in Synthesizer

For quick prototyping, use the SDK's built-in synthesizer (no external TTS needed):

```python
from prosody_protocol import IMLToAudio

synthesizer = IMLToAudio()
wav_bytes = synthesizer.synthesize(iml)

with open("output.wav", "wb") as f:
    f.write(wav_bytes)
```

The built-in synthesizer produces basic sine-wave audio with pitch and volume
modulation from IML attributes. For production use, pair with Coqui TTS or
ElevenLabs for natural-sounding output.

## Text-to-IML-to-Speech Pipeline

Generate expressive speech from plain text:

```python
from prosody_protocol import TextToIML, IMLToSSML

# 1. Predict prosody from text
predictor = TextToIML()
iml = predictor.predict(
    "I can't believe this happened!",
    context="frustrated",
)

# 2. Convert to SSML
ssml = IMLToSSML().convert(iml)

# 3. Synthesize with Coqui
tts.tts_to_file(text=IMLParser().to_plain_text(IMLParser().parse(iml)),
                file_path="frustrated_output.wav")
```
