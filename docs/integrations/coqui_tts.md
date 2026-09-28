# Coqui TTS

[Coqui TTS](https://github.com/idiap/coqui-ai-TTS) is an open-source
text-to-speech toolkit that runs locally; its XTTS v2 model clones a voice
from a short reference recording. (Coqui, the company, closed in 2024; the
toolkit is maintained by the community as the `coqui-tts` package, which
still imports as `TTS`.)

Coqui TTS takes plain text. It has no SSML input, and XTTS has no controls
for pitch, loudness or emphasis within a sentence, so `IMLToSSML` output is
of no use to it. What you can carry over from IML:

| IML | With Coqui TTS |
|-----|----------------|
| the words | the `text` |
| `<pause>` | synthesize the text between pauses separately and insert silence of the pause's length |
| a whole-utterance `rate` | XTTS's `speed` argument (1.0 is normal) |
| `pitch`, `volume`, `<emphasis>`, `pitch_contour`, `emotion` | lost; XTTS imitates the reference recording's voice and, to a degree, its manner, so a reference spoken in the wanted mood is the closest you get |

If you need the prosody itself, use an engine that renders it (see
[below](#when-you-need-the-prosody)).

The Coqui calls on this page are not run by the docs tests (they need
PyTorch and a model download); the `prosody_protocol` parts are.

## Split the document at its pauses

```python
from pathlib import Path

from prosody_protocol import IMLParser, Pause


def split_at_pauses(iml: str) -> list[tuple[str, int]]:
    """The document as (text, silence in ms after it) chunks, split at its pauses."""
    chunks: list[tuple[str, int]] = []
    words: list[str] = []

    def walk(children) -> None:
        for child in children:
            if isinstance(child, str):
                words.append(child)
            elif isinstance(child, Pause):
                chunks.append((" ".join("".join(words).split()), child.duration))
                words.clear()
            else:  # <prosody>, <emphasis>, <segment>: keep the words
                walk(child.children)

    for utterance in IMLParser().parse(iml).utterances:
        walk(utterance.children)
        words.append(" ")
    chunks.append((" ".join("".join(words).split()), 0))
    return [(text, ms) for text, ms in chunks if text or ms]


iml = Path("examples/sarcasm.iml").read_text(encoding="utf-8")
for chunk in split_at_pauses(iml):
    print(chunk)
```

```text
("I've been on hold for forty minutes. Oh, that's just wonderful.", 600)
('Really great service.', 0)
```

The same for IML measured from a recording, whose pauses `AudioToIML`
found in the audio:

```python
from prosody_protocol import AudioToIML, load_word_timings

words = load_word_timings("examples/monotone.deepgram.json")
for chunk in split_at_pauses(AudioToIML().convert("examples/monotone.wav", words=words)):
    print(chunk)
```

```text
('I read the list.', 310)
('The room is booked.', 300)
('I have the slides.', 310)
('And we got the grant!', 0)
```

## Speak the chunks with XTTS

```bash
pip install coqui-tts
```

<!-- docs-test: skip -->
```python
import wave

import numpy as np
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
rate = tts.synthesizer.output_sample_rate

pieces = []
for text, pause_ms in split_at_pauses(iml):
    if text:
        samples = tts.tts(text=text, speaker_wav="reference.wav", language="en")
        pieces.append(np.asarray(samples, dtype=np.float32))
    pieces.append(np.zeros(int(rate * pause_ms / 1000), dtype=np.float32))

audio = np.clip(np.concatenate(pieces), -1.0, 1.0)
with wave.open("customer.wav", "wb") as out:
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(rate)
    out.writeframes((audio * 32767).astype("<i2").tobytes())
```

`reference.wav` is a few seconds of the voice to clone (use a recording you
have the right to use). `tts.tts` returns the samples; `tts.tts_to_file(text=...,
speaker_wav=..., language=..., file_path=...)` writes one chunk straight to a
file. For an utterance that IML marks as faster or slower as a whole (such
as `<prosody rate="170%">` around all of it), pass `speed=1.7` to that
chunk's `tts.tts` call.

Check the model's license before you build on it. XTTS v2 is released
under the Coqui Public Model License (CPML), which allows non-commercial
use only. The first load downloads the model and stops at an interactive
prompt until you accept the CPML (or state that you hold a commercial
license); in a script or a container, set `COQUI_TOS_AGREED=1` to accept
it without the prompt, which blocks otherwise.

## When you need the prosody

- **espeak-ng, locally.** `IMLToAudio` speaks IML with espeak-ng, rendering
  pitch, volume, rate, pauses, emphasis and pitch contours. The voice is
  robotic, but the delivery follows the markup:

  ```bash
  prosody-protocol synthesize examples/sarcasm.iml -o sarcasm.wav
  ```

- **SSML engines.** Azure AI Speech, Google Cloud Text-to-Speech and Amazon
  Polly accept `IMLToSSML`'s output to a much larger degree (breaks,
  `<prosody>` rate, pitch and volume, and on some voices `<emphasis>`);
  check the vendor's SSML reference for the voice you use.

`IMLToAudio(engine="coqui")` is not implemented and raises
`ConversionError`.
