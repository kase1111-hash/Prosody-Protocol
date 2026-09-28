# Whisper

[Whisper](https://github.com/openai/whisper) transcribes speech with word
timestamps; Prosody Protocol measures how those words were said. There are
two ways to combine them:

1. **Let `AudioToIML` run Whisper** (the `whisper` extra). Simplest, but it
   installs PyTorch and downloads a model.
2. **Run Whisper yourself** (locally, with faster-whisper or WhisperX, or
   through the OpenAI API) and pass its words to `AudioToIML`. The SDK then
   needs only the `audio` extra.

The Whisper calls on this page are not run by the docs tests (they need the
model); the `prosody_protocol` parts are.

## Built-in: `AudioToIML(stt="whisper")`

Install the `whisper` extra, from a clone or from GitHub:

```bash
pip install -e ".[whisper]"                     # in a clone
pip install "prosody-protocol[whisper] @ git+https://github.com/kase1111-hash/Prosody-Protocol"   # or without one
```

`prosody-protocol doctor` then reports `[ok     ] speech recognition`.
Without words or a transcript, `AudioToIML` now transcribes the audio:

<!-- docs-test: skip -->
```python
from prosody_protocol import AudioToIML

converter = AudioToIML(stt="whisper", stt_model="small", language="en-US")
result = converter.convert_detailed("examples/speech.wav")
print(result.transcript_source)   # "whisper"
print(result.iml)
```

- `stt="whisper"` requires Whisper; `stt="auto"` (the default) uses it when
  it is installed and emits `[speech]` placeholders otherwise, with a
  warning. Without the extra, `stt="whisper"` raises `AudioProcessingError`
  (`prosody-protocol from-audio --stt whisper` exits with status 2 and an
  install hint).
- `stt_model` is the Whisper model size (`tiny`, `base` (default), `small`,
  `medium`, `large-v3`, ...). The model is loaded on the first conversion
  and reused by the same converter, so create one converter for a batch.
- `language="fr-FR"` is passed to Whisper as `fr` and labels the document.
  With `language=None`, Whisper detects the language, and its code (`en`)
  labels the document.
- Whisper gets the audio already decoded by the SDK (16 kHz mono), so its
  timestamps refer to exactly the samples that are measured. Load and
  transcription failures raise `AudioProcessingError`.

The same from the command line: `prosody-protocol from-audio speech.wav
--stt whisper --whisper-model small`. The REST server uses Whisper when it
is installed (`/v1/health` reports `"whisper": true`).

## Bring your own Whisper output

### openai-whisper

<!-- docs-test: skip -->
```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("examples/speech.wav", word_timestamps=True)
```

`result` is a dict with `segments[].words[]`, each word with `word`, `start`
and `end` in seconds. `examples/speech.whisper.json` is such a result (saved
with `json.dump`), for `examples/speech.wav`:

```python
import json

from prosody_protocol import AudioToIML
from prosody_protocol.alignment import from_whisper

with open("examples/speech.whisper.json", encoding="utf-8") as f:
    result = json.load(f)

words = from_whisper(result)
print(words[:3])
print(AudioToIML(language="en-US").convert("examples/speech.wav", words=words))
```

```text
[WordAlignment(word='I', start_ms=250, end_ms=564), WordAlignment(word='never', start_ms=614, end_ms=1081), WordAlignment(word='said', start_ms=1131, end_ms=1536)]
<iml version="0.1.0" language="en-US"><utterance>I never said<pause duration="660"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
```

`load_word_timings("words.json")` detects the format from a file, and
`prosody-protocol from-audio speech.wav --words words.json` does the same on
the command line.

### faster-whisper and WhisperX

<!-- docs-test: skip -->
```python
from faster_whisper import WhisperModel

model = WhisperModel("small")
segments, info = model.transcribe("examples/speech.wav", word_timestamps=True)
words = from_whisper(list(segments))
```

`from_whisper` reads faster-whisper's segment objects (or a generator of
them), and WhisperX's aligned result (`result["segments"]` with `words`).
WhisperX leaves words it could not align without times; each joins its
neighboring timed word (the previous one, or the next one at the start of a
segment or sentence).

### The OpenAI transcription API

`whisper-1` returns word timestamps with
`response_format="verbose_json", timestamp_granularities=["word"]`, and
`from_whisper` reads the result (restoring punctuation from its `text`).
See the [speech-to-text guide](speech-to-text.md#openai-transcription-api).

## Your own pipeline

`AudioToIML` runs three steps, which you can also run yourself, for
example to reuse the measurements:

```python
from prosody_protocol import IMLAssembler, IMLParser, ProsodyAnalyzer

analyzer = ProsodyAnalyzer(max_duration_s=600)
features = analyzer.analyze("examples/speech.wav", words)   # one SpanFeatures per word
pauses = analyzer.detect_pauses("examples/speech.wav")
doc = IMLAssembler().assemble(words, features, pauses, language="en-US")
print(IMLParser().to_iml_string(doc) == AudioToIML(language="en-US").convert("examples/speech.wav", words=words))
print(features[4].text, round(features[4].f0_mean), "Hz")
```

```text
True
stole 134 Hz
```

Pass `reference_features=analyzer.analyze("calibration.wav", ...)` to
`assemble` to measure against a neutral recording of the same speaker
(what `AudioToIML(calibration_audio=...)` does). See the
[API reference](../API.md#imlassembler) for what the assembler writes.

## Many files

Reuse one converter, so the Whisper model is loaded once:

```python
from pathlib import Path

from prosody_protocol import load_word_timings

converter = AudioToIML(language="en-US")
recordings = {"speech": "speech.whisper.json", "monotone": "monotone.deepgram.json"}
for name, timings in recordings.items():
    words = load_word_timings(Path("examples") / timings)
    iml = converter.convert(Path("examples") / f"{name}.wav", words=words)
    Path(f"{name}.iml").write_text(iml + "\n", encoding="utf-8")
    print(f"wrote {name}.iml")
```

```text
wrote speech.iml
wrote monotone.iml
```

With the built-in Whisper, drop `words=`. Set `max_duration_s` when the
recordings come from users.

## What to expect

- Whisper's word timestamps are estimates and can be off by tens of
  milliseconds. Pauses are measured in the audio itself, so they do not
  depend on them.
- The transcript is only as good as Whisper's; misrecognized words are
  measured all the same.
- Emotion labels come from a rule-based heuristic that needs a baseline of
  the same speaker (`calibration_audio`, or at least three utterances) and
  abstains otherwise. It was checked on synthetic speech, not a real-speech
  benchmark: treat its labels as estimates.
