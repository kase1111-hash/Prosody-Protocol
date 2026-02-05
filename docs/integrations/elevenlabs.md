# ElevenLabs Integration

Convert IML to SSML for speech synthesis with ElevenLabs.

## Concept

IML captures prosodic intent. ElevenLabs can synthesize speech from SSML.
The `IMLToSSML` converter bridges these formats, translating IML prosody
attributes into SSML that ElevenLabs' TTS engine understands.

## Setup

```bash
pip install prosody-protocol elevenlabs
```

## Basic Usage

```python
from prosody_protocol import IMLToSSML

converter = IMLToSSML()

iml = '''<utterance emotion="empathetic">
  I understand <prosody pitch="-5%" rate="slow" quality="breathy">
    how you feel
  </prosody>.
</utterance>'''

ssml = converter.convert(iml)
print(ssml)
```

Output SSML:
```xml
<speak>
  <s>I understand <prosody pitch="-5%" rate="slow">how you feel</prosody>.</s>
</speak>
```

## With ElevenLabs TTS

```python
from elevenlabs import generate, set_api_key
from prosody_protocol import IMLToSSML

set_api_key("your-api-key")

converter = IMLToSSML()
ssml = converter.convert(iml)

audio = generate(
    text=ssml,
    voice="Bella",
    model="eleven_multilingual_v2",
)

with open("output.mp3", "wb") as f:
    f.write(audio)
```

## Round-Trip: Audio to IML to Speech

```python
from prosody_protocol import AudioToIML, IMLToSSML

# 1. Analyze original speech
audio_converter = AudioToIML()
iml = audio_converter.convert("original_speech.wav")

# 2. Convert to SSML
ssml_converter = IMLToSSML()
ssml = ssml_converter.convert(iml)

# 3. Synthesize with ElevenLabs (preserving prosodic intent)
audio = generate(text=ssml, voice="Josh")
```

## Handling Emphasis and Pauses

IML emphasis and pauses map directly to SSML:

```python
iml = '''<utterance>
  I <emphasis level="strong">really</emphasis> need this
  <pause duration="500"/> done today.
</utterance>'''

ssml = IMLToSSML().convert(iml)
# <speak><s>I <emphasis level="strong">really</emphasis> need this
# <break time="500ms"/> done today.</s></speak>
```

## Batch Synthesis

```python
from pathlib import Path
from prosody_protocol import IMLParser, IMLToSSML

converter = IMLToSSML()

for iml_file in Path("iml_files/").glob("*.xml"):
    iml = iml_file.read_text()
    ssml = converter.convert(iml)
    output = iml_file.with_suffix(".ssml")
    output.write_text(ssml)
```
