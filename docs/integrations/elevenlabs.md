# ElevenLabs

[ElevenLabs](https://elevenlabs.io) turns text into natural-sounding
speech. It is not an SSML engine, though: it reads plain text and accepts
only a small part of SSML, so most of what an IML document says about
delivery cannot be passed to it. This page shows what does survive and how
to send it, and where to go when you need the rest.

The ElevenLabs calls on this page are not run by the docs tests (they need
an API key); the `prosody_protocol` parts are.

## What ElevenLabs accepts

ElevenLabs' help center (as of 2026) lists two SSML tags, written inline
in the text; check
[Do pauses and SSML phoneme tags work with the API?](https://elevenlabs.io/docs/help-center/technical/do-pauses-and-ssml-phoneme-tags-work-with-the-api)
for the current list:

- `<break time="1.5s" />` pauses, up to 3 seconds, on all models except
  Eleven v3;
- `<phoneme>` tags on some English models (Flash v2 and Turbo v2).

It lists no `<prosody>` (pitch, rate, volume) and no `<emphasis>`. So from
IML only the words and the pauses carry over. Emphasis, pitch, pitch
contours, loudness and emotion are lost; ElevenLabs' voices choose their
own delivery from the text. Eleven v3 takes no SSML breaks; it takes
bracketed audio tags in the text instead (such as `[pause]` or
`[excited]`, see ElevenLabs' prompting guide), and mapping IML emotions
to them is up to you.

Compare the full SSML that `IMLToSSML` writes for `examples/sarcasm.iml`:

```python
from pathlib import Path

from prosody_protocol import IMLToSSML

iml = Path("examples/sarcasm.iml").read_text(encoding="utf-8")
print(IMLToSSML().convert(iml))
```

```text
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><s>
    I've been on hold for <prosody pitch="+8%" volume="+4dB">forty minutes</prosody>.
  </s><s>
    Oh, that's just
    <prosody pitch="+15%" rate="80%"><prosody contour="(0%,+0%) (50%,-15%) (100%,+15%)">wonderful</prosody></prosody>.
    <break time="600ms"/>
    Really <emphasis level="strong">great</emphasis> service.
  </s></speak>
```

Sending this to ElevenLabs gains nothing: it does not interpret the tags
outside its short list. Send the words with the pauses as `<break>` tags
instead:

```python
from prosody_protocol import IMLParser, Pause


def to_elevenlabs_text(iml: str) -> str:
    """The words of an IML document, with its pauses as ElevenLabs <break> tags."""

    def render(children) -> str:
        parts = []
        for child in children:
            if isinstance(child, str):
                parts.append(child)
            elif isinstance(child, Pause):
                if child.duration:  # 0 means a missing or invalid duration
                    seconds = min(child.duration, 3000) / 1000  # ElevenLabs' maximum
                    parts.append(f' <break time="{seconds:g}s" /> ')
            else:  # <prosody>, <emphasis>, <segment>: keep the words
                parts.append(render(child.children))
        return "".join(parts)

    doc = IMLParser().parse(iml)
    return " ".join(" ".join(render(u.children).split()) for u in doc.utterances)


text = to_elevenlabs_text(iml)
print(text)
```

```text
I've been on hold for forty minutes. Oh, that's just wonderful. <break time="0.6s" /> Really great service.
```

## Send it to ElevenLabs

With the current SDK (`pip install elevenlabs`):

<!-- docs-test: skip -->
```python
import os

from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
audio = client.text_to_speech.convert(
    voice_id="JBFqnCBsd6RMkjVDRZzb",       # any voice ID from your account
    model_id="eleven_multilingual_v2",
    text=text,
    output_format="mp3_44100_128",
)
with open("customer.mp3", "wb") as f:
    for chunk in audio:                    # convert() returns an iterator of bytes
        f.write(chunk)
```

`VoiceSettings(stability=..., style=..., speed=...)`, passed as
`voice_settings=`, changes the delivery of a whole request; nothing in the
text controls a single word.

## From a recording

The same works for IML measured from audio: a recognizer gives the words,
`AudioToIML` finds the pauses, and ElevenLabs speaks them in another voice:

```python
from prosody_protocol import AudioToIML, load_word_timings

words = load_word_timings("examples/speech.whisper.json")
measured = AudioToIML().convert("examples/speech.wav", words=words)
print(to_elevenlabs_text(measured))
```

```text
I never said <break time="0.66s" /> she stole my money.
```

The emphasis on "stole", the sentence's point, is lost on the way.

## When you need the prosody

- **SSML engines.** Azure AI Speech, Google Cloud Text-to-Speech and
  Amazon Polly accept much more SSML than ElevenLabs: breaks, `<prosody>`
  rate, pitch and volume, and `<emphasis>` on some voices. Support differs
  by voice type (neural, standard, ...) and pitch contours are rarely
  honored, so check the vendor's SSML reference for the voice you use.
  `IMLToSSML(speaker_voices={"customer": "en-US-JennyNeural"})` wraps each
  speaker's utterances in a `<voice>` element for such engines.
- **espeak-ng, locally.** `IMLToAudio` speaks IML with espeak-ng, including
  pitch, volume, rate, pauses, emphasis and contours. The voice is robotic,
  but the prosody is there, which makes it useful for checking what a
  document says: `prosody-protocol synthesize examples/sarcasm.iml -o
  sarcasm.wav`.
