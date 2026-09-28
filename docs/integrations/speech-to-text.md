# Word timings from a speech-to-text service

`AudioToIML` measures *how* words were said. It needs to know *which* words
were said, and when: a list of words with start and end times. Any speech
recognizer that returns word timestamps can supply them, so you can keep
the one you already use and skip the `whisper` extra (and PyTorch).

`prosody_protocol.alignment` turns the output of common services into
`WordAlignment` lists; `load_word_timings` and `parse_word_timings` detect
which service a JSON document came from. The vendor calls on this page are
examples of each service's current Python SDK and are not run by the docs
tests; the `prosody_protocol` parts are.

| Service | Ask for | Adapter |
|---------|---------|---------|
| [OpenAI transcription API](#openai-transcription-api) | `model="whisper-1"`, `response_format="verbose_json"`, `timestamp_granularities=["word"]` | `from_whisper` |
| [Deepgram](#deepgram) | `smart_format=True` (or `punctuate=True`) | `from_deepgram` |
| [AssemblyAI](#assemblyai) | a completed transcript (words are always included) | `from_assemblyai` |
| [Google Cloud Speech-to-Text](#google-cloud-speech-to-text) | `enable_word_time_offsets=True`, `enable_automatic_punctuation=True` | `from_google` |
| local Whisper, faster-whisper, WhisperX | `word_timestamps=True` | `from_whisper`; see the [Whisper guide](whisper.md) |
| [anything else](#anything-else) | word, start and end per word | `from_records` |

Ask for punctuation: the assembler splits utterances at sentence ends, and
without punctuation it can only split at long pauses.

## The pattern

```python
from prosody_protocol import AudioToIML, load_word_timings

words = load_word_timings("examples/speech.whisper.json")   # a file, or parsed JSON
print(AudioToIML().convert("examples/speech.wav", words=words))
```

```text
<iml version="0.1.0"><utterance>I never said<pause duration="660"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
```

`load_word_timings` takes a file path or data you already parsed (a dict,
a list, or an SDK response object); a `str` is always read as a file path.
For JSON *text*, such as a request body from a user, use
`parse_word_timings`, which never touches the file system. The command line
takes the same files: `prosody-protocol from-audio audio.wav --words
words.json`, and so does the REST API's `words` field.

Every adapter strips whitespace from words, drops empty tokens and rounds
times to whole milliseconds. It rejects (with `ConversionError`) times that
are not finite, negative, end before they start or are out of order, and,
with three or more words, a median word length over 10 s or under 5 ms (a
sign of milliseconds read as seconds, or the reverse). `AudioToIML` also
rejects words that overlap an earlier word by more than 500 ms (`ValueError`):
give it the words of one speaker, one after another. For a conversation,
convert each speaker's channel or turns separately.

## OpenAI transcription API

Only `whisper-1` returns word timestamps (`verbose_json` with
`timestamp_granularities=["word"]`); the `gpt-4o-*-transcribe` models return
text only, which you can still pass as `transcript=`.

<!-- docs-test: skip -->
```python
from openai import OpenAI

client = OpenAI()   # OPENAI_API_KEY
with open("examples/speech.wav", "rb") as audio:
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
```

The response's `words` have no punctuation; `from_whisper` restores it from
the response's `text`. It accepts the SDK object or its JSON:

```python
from prosody_protocol.alignment import from_whisper

response = {   # the shape of response.model_dump(), shortened
    "text": "I never said she stole my money.",
    "language": "english",
    "duration": 4.2,
    "words": [
        {"word": "I", "start": 0.25, "end": 0.56},
        {"word": "never", "start": 0.61, "end": 1.08},
        {"word": "said", "start": 1.13, "end": 1.54},
        {"word": "she", "start": 2.14, "end": 2.49},
        {"word": "stole", "start": 2.54, "end": 3.01},
        {"word": "my", "start": 3.06, "end": 3.48},
        {"word": "money", "start": 3.53, "end": 3.9},
    ],
}
words = from_whisper(response)
print(words[-1])
```

```text
WordAlignment(word='money.', start_ms=3530, end_ms=3900)
```

## Deepgram

<!-- docs-test: skip -->
```python
from deepgram import DeepgramClient

client = DeepgramClient()   # DEEPGRAM_API_KEY
with open("examples/monotone.wav", "rb") as audio:
    response = client.listen.v1.media.transcribe_file(
        request=audio.read(), model="nova-3", smart_format=True
    )
```

`from_deepgram` reads `results.channels[channel].alternatives[alternative].words`
and prefers `punctuated_word`. `examples/monotone.deepgram.json` has the
shape of a Deepgram response:

```python
import json

from prosody_protocol.alignment import from_deepgram

with open("examples/monotone.deepgram.json", encoding="utf-8") as f:
    response = json.load(f)
words = from_deepgram(response, channel=0)
print(words[:4])
print(AudioToIML().convert("examples/monotone.wav", words=words))
```

```text
[WordAlignment(word='I', start_ms=150, end_ms=431), WordAlignment(word='read', start_ms=461, end_ms=839), WordAlignment(word='the', start_ms=869, end_ms=1124), WordAlignment(word='list.', start_ms=1154, end_ms=1518)]
<iml version="0.1.0"><utterance>I read the list.</utterance><utterance><pause duration="310"/>The room is booked.</utterance><utterance><pause duration="300"/>I have the slides.</utterance><utterance><pause duration="310"/><prosody rate="170%">And we got the grant!</prosody></utterance></iml>
```

With `multichannel=True`, pick the channel with `channel=`.

## AssemblyAI

<!-- docs-test: skip -->
```python
import assemblyai as aai

aai.settings.api_key = "..."   # or ASSEMBLYAI_API_KEY
transcript = aai.Transcriber().transcribe("examples/speech.wav")
```

Times are in milliseconds, and the transcript must be completed:
`from_assemblyai` raises `ConversionError` for a queued, processing or
failed one. It accepts the SDK's `Transcript` or the JSON of
`GET /v2/transcript/{id}`:

```python
from prosody_protocol.alignment import from_assemblyai

transcript = {
    "id": "...",
    "status": "completed",
    "text": "I never said she stole my money.",
    "words": [
        {"text": "I", "start": 250, "end": 564, "confidence": 0.98, "speaker": None},
        {"text": "never", "start": 614, "end": 1081, "confidence": 0.99, "speaker": None},
        {"text": "said", "start": 1131, "end": 1536, "confidence": 0.99, "speaker": None},
    ],
}
print(from_assemblyai(transcript))
```

```text
[WordAlignment(word='I', start_ms=250, end_ms=564), WordAlignment(word='never', start_ms=614, end_ms=1081), WordAlignment(word='said', start_ms=1131, end_ms=1536)]
```

## Google Cloud Speech-to-Text

Word time offsets are off by default; turn them on.

<!-- docs-test: skip -->
```python
from google.cloud import speech

client = speech.SpeechClient()
config = speech.RecognitionConfig(
    language_code="en-US",
    enable_word_time_offsets=True,
    enable_automatic_punctuation=True,
)
with open("examples/speech.wav", "rb") as f:
    audio = speech.RecognitionAudio(content=f.read())
response = client.recognize(config=config, audio=audio)
```

`from_google` takes the response object (times are `timedelta`s) or its
JSON (times such as `"1.300s"`, v1 `startTime` or v2 `startOffset`). A
response without speech gives `[]`; results with transcripts but no word
offsets raise `ConversionError`. With `enable_separate_recognition_per_channel`,
pass `channel_tag=`; with speaker diarization, the final result that repeats
every word with its speaker is used.

```python
from prosody_protocol.alignment import from_google

response = {
    "results": [
        {
            "alternatives": [
                {
                    "transcript": "I never said",
                    "confidence": 0.93,
                    "words": [
                        {"startTime": "0.250s", "endTime": "0.564s", "word": "I"},
                        {"startTime": "0.614s", "endTime": "1.081s", "word": "never"},
                        {"startTime": "1.131s", "endTime": "1.536s", "word": "said"},
                    ],
                }
            ],
            "languageCode": "en-us",
        }
    ]
}
print(from_google(response))
```

```text
[WordAlignment(word='I', start_ms=250, end_ms=564), WordAlignment(word='never', start_ms=614, end_ms=1081), WordAlignment(word='said', start_ms=1131, end_ms=1536)]
```

## Anything else

`from_records` reads any list of records (dicts or objects) with
configurable keys, in seconds or milliseconds; numeric strings work, so CSV
rows do too:

```python
import csv
import io

from prosody_protocol.alignment import from_records

rows = csv.DictReader(io.StringIO("token,begin,finish\nHello,120,480\nthere.,510,900\n"))
print(from_records(rows, word_key="token", start_key="begin", end_key="finish", unit="ms"))
```

```text
[WordAlignment(word='Hello', start_ms=120, end_ms=480), WordAlignment(word='there.', start_ms=510, end_ms=900)]
```

A list of `{"word", "start_ms", "end_ms"}` records (what
`dataclasses.asdict(WordAlignment(...))` gives) is detected by
`load_word_timings` without any options. You can also build the list
yourself: `WordAlignment("Hello", 120, 480)`.

## Untrusted input

A web service that accepts word timings from its users should parse them
with `parse_word_timings`, after limiting their size, and set
`AudioToIML(max_duration_s=...)` for the audio:

```python
from prosody_protocol import ConversionError, parse_word_timings

body = b'[{"word": "hi", "start": 0.1, "end": "soon"}]'
try:
    parse_word_timings(body)
except ConversionError as error:
    print(error)
```

```text
records: word 0 ('hi') end time must be a number, got 'soon'
```

The REST API does this for its `words` field, with `PP_MAX_WORDS_CHARS` and
`PP_MAX_AUDIO_SECONDS` as the limits.
