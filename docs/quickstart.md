# Quick Start

This page walks through the SDK with the files in [`examples/`](../examples/).
The Python blocks run in order, from the root of a clone, and the outputs
shown are from real runs.

- [Install](#install)
- [Parse and validate IML](#parse-and-validate-iml)
- [Annotate a recording](#annotate-a-recording)
- [Give the result to an LLM](#give-the-result-to-an-llm)
- [Predict prosody for text](#predict-prosody-for-text)
- [Convert IML to SSML](#convert-iml-to-ssml)
- [Speak IML](#speak-iml)
- [Apply a prosody profile](#apply-a-prosody-profile)
- [Datasets and benchmarks](#datasets-and-benchmarks)
- [Run the REST API](#run-the-rest-api)
- [Next steps](#next-steps)

## Install

The package is not on PyPI yet. Install it from GitHub:

```bash
pip install "prosody-protocol[audio] @ git+https://github.com/kase1111-hash/Prosody-Protocol"
```

or from a clone, which also gives you `examples/`, the training scripts and
the tests (this page assumes a clone):

```bash
git clone https://github.com/kase1111-hash/Prosody-Protocol
cd Prosody-Protocol
pip install -e ".[audio]"
```

Python 3.10 or newer. The core (parsing, validation, SSML, LLM context,
text-to-IML, profiles, datasets) needs only `lxml`. The extras add the rest:

| Extra | Adds | For |
|-------|------|-----|
| `audio` | numpy, praat-parselmouth | measuring prosody in recordings, speaking IML, benchmarks, the Mavis bridge |
| `whisper` | `audio` + openai-whisper (pulls in PyTorch) | built-in speech recognition, when you have no word timings or transcript |
| `api` | `audio` + fastapi, uvicorn, python-multipart | the REST API |
| `ml` | `audio` + scikit-learn, joblib, pyyaml | the `training/` scripts (source checkout only) |
| `dev` | pytest, ruff, mypy, httpx, jsonschema, ... | development |
| `all` | `audio`, `whisper`, `ml`, `api` | everything except `dev` |

Combine them as usual: `pip install -e ".[audio,api]"`, or
`pip install "prosody-protocol[audio,api] @ git+https://github.com/kase1111-hash/Prosody-Protocol"`.

Two optional system programs: **espeak-ng** makes `IMLToAudio` speak (without
it you get a tone preview, not speech), and **ffmpeg** lets the SDK read
OGG/Opus, WebM, M4A and other formats besides WAV, AIFF, FLAC and MP3.
Install them with `sudo apt install espeak-ng ffmpeg` or
`brew install espeak-ng ffmpeg`.

`prosody-protocol doctor` lists what your installation can do:

```bash
prosody-protocol doctor
```

```text
prosody-protocol 0.1.0a3 (Python 3.11.15)
[ok     ] core (lxml 6.1.3)
          enables: validate, to-text, to-ssml, to-prompt, from-text
[ok     ] audio analysis (numpy 2.4.6, praat-parselmouth 0.4.7)
          enables: from-audio, benchmark, synthesize
[missing] speech recognition (openai-whisper)
          enables: from-audio transcribes audio given without --words or --transcript (otherwise each stretch of speech is a [speech] placeholder)
          install: pip install "prosody-protocol[whisper] @ git+https://github.com/kase1111-hash/Prosody-Protocol" (pulls in PyTorch)
[ok     ] speech synthesis (espeak-ng at /usr/bin/espeak-ng)
          enables: synthesize speaks IML (otherwise it renders a tone preview, not speech)
[ok     ] audio decoding (ffmpeg at /usr/bin/ffmpeg)
          enables: from-audio reads OGG/Opus, WebM, M4A and other formats besides WAV, AIFF, FLAC and MP3
[ok     ] REST API (fastapi 0.141.1, uvicorn 0.54.0, python-multipart 0.0.32)
          enables: serve
[ok     ] training baselines (scikit-learn 1.9.1)
          enables: the training/ scripts of a source checkout
```

The `install:` hints install from GitHub; in a clone,
`pip install -e ".[whisper]"` does the same.

## Parse and validate IML

```python
from prosody_protocol import IMLParser, IMLValidator

iml = """<utterance emotion="sarcastic" confidence="0.87">
  Oh, that's <prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">GREAT</prosody>.
</utterance>"""

parser = IMLParser()
doc = parser.parse(iml)
utterance = doc.utterances[0]
print(utterance.emotion, utterance.confidence)
print(parser.to_plain_text(doc))
```

```text
sarcastic 0.87
Oh, that's GREAT.
```

`IMLValidator` checks a document against the spec's rules (V1-V33, listed
in the [API reference](API.md#validation-rules)). Errors make a document
invalid; warnings do not:

```python
validator = IMLValidator()

result = validator.validate('<utterance emotion="angry">Give it back!</utterance>')
print(result.valid)
for issue in result.issues:
    print(issue.severity, issue.rule, issue.message)

result = validator.validate('<utterance><prosody volume="+80dB">Hey!</prosody></utterance>')
print(result.valid, [(issue.rule, issue.severity) for issue in result.warnings])
```

```text
False
error V3 <utterance> has emotion="angry" but no confidence attribute
True [('V33', 'warning')]
```

`result.raise_for_errors()` raises `IMLValidationError` (with the issues in
`.issues`) when the document is invalid.

## Annotate a recording

`AudioToIML` measures how the words of a recording were said: pauses,
stressed words, pitch and loudness relative to the speaker, pitch movement,
tempo and voice quality. It needs the words with their timings. Bring them
from any speech recognizer: `load_word_timings` reads the JSON of Whisper,
the OpenAI transcription API, Deepgram, AssemblyAI and Google Cloud
Speech-to-Text, and detects which one it is.

`examples/speech.wav` is a synthetic voice (espeak-ng) saying "I never said
-- she STOLE my money.", with "stole" higher and louder than the other
words; `examples/speech.whisper.json` holds its word timings as
openai-whisper returns them.

```python
from prosody_protocol import AudioToIML, load_word_timings

words = load_word_timings("examples/speech.whisper.json")
print(words[:2])

converter = AudioToIML(language="en-US")
annotated = converter.convert_detailed("examples/speech.wav", words=words)
print(annotated.iml)
print(annotated.transcript_source, annotated.warnings)
```

```text
[WordAlignment(word='I', start_ms=250, end_ms=564), WordAlignment(word='never', start_ms=614, end_ms=1081)]
<iml version="0.1.0" language="en-US"><utterance>I never said<pause duration="610"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
words ("No speaker baseline: a single utterance without calibration_audio has nothing to compare its pitch, loudness and rate with, so they were not marked, and no emotion was estimated from them. Pass calibration_audio (recordings of the speaker's usual speech, such as their earlier turns) to assess them.",)
```

The pause after "said" (610 ms; the gap in the recording is 600 ms) and the
stressed "stole" are marked. `convert()` returns just the IML string and
`convert_to_doc()` the parsed document, and both issue the warnings as
`UserWarning`s; `convert_detailed()` also says where the words came from
(`transcript_source`) and returns the warnings: anything that degraded the
output or could not be assessed.

There is no emotion on this utterance, and no pitch or loudness for it as a
whole, and the warning says why. Pitch, loudness and emotion are judged
against the speaker's usual voice, and a single sentence gives nothing to
compare with. The SDK gets that voice from recordings of the same speaker
talking as usual (`calibration_audio`, see below) or, without them, from a
recording of at least three utterances, most of them at the speaker's usual
level. Even then, an utterance whose estimate is below
`min_emotion_confidence` (0.5) carries no `emotion` attribute. Emotion
labels come from a rule-based heuristic that was checked on synthetic
espeak-ng speech, not on a benchmark of real speech; treat them as rough
estimates, and never as the only basis for a decision that matters (spec
Section 8 covers emotional data and the uses IML must not be put to).

### Compare with the speaker's earlier turns

In a conversation, the user's earlier turns are the natural baseline: pass
them as `calibration_audio`, one recording or several. Each file is
analyzed once and cached, so a converter reused across turns can be given
the growing list. To have two turns, this cuts `examples/monotone.wav`
before its last sentence. The files this page writes go in a temporary
directory, `workdir`, so the clone stays clean:

```python
import tempfile
import wave
from pathlib import Path

from prosody_protocol import WordAlignment

workdir = Path(tempfile.mkdtemp(prefix="prosody-"))

# Split examples/monotone.wav before its last sentence: an earlier turn of
# three sentences, then the turn to convert, "And we got the grant!".
monotone_words = load_word_timings("examples/monotone.deepgram.json")
cut_ms = monotone_words[12].start_ms - 150    # 150 ms before "And"
with wave.open("examples/monotone.wav") as recording:
    params = recording.getparams()
    frames = recording.readframes(params.nframes)
cut = cut_ms * params.framerate // 1000 * params.sampwidth * params.nchannels
for name, data in (("earlier.wav", frames[:cut]), ("turn.wav", frames[cut:])):
    with wave.open(str(workdir / name), "wb") as out:
        out.setparams(params)
        out.writeframes(data)
turn_words = [
    WordAlignment(w.word, w.start_ms - cut_ms, w.end_ms - cut_ms) for w in monotone_words[12:]
]

assistant = AudioToIML(stt="none")
print(assistant.convert_detailed(workdir / "turn.wav", words=turn_words).iml)
assistant.calibration_audio = [workdir / "earlier.wav"]    # the turns so far
result = assistant.convert_detailed(workdir / "turn.wav", words=turn_words)
print(result.iml)
print(result.warnings)
```

```text
<iml version="0.1.0"><utterance>And we got the grant!</utterance></iml>
<iml version="0.1.0"><utterance><prosody rate="170%">And we got the grant!</prosody></utterance></iml>
()
```

Alone, the turn is its own baseline and nothing is marked (with the "No
speaker baseline" warning). Against the earlier turn, it is much faster than
this speaker's usual pace. The rule-based estimator still reports no
emotion here (its best guess is "neutral" at 0.3); the
[profile section](#apply-a-prosody-profile) shows how this speaker's fast,
flat speech is read. The calibration should be the same speaker, recorded
with the same setup; with word timings that carry several speaker labels it
is not used, and a warning says so. On the command line, repeat `from-audio
--calibration EARLIER.wav` for several recordings; over REST, repeat the
`calibration` field (up to 5 recordings).

### Without word timings

With only the text (`transcript=`), nothing says where each word is, so no
pauses, stresses or word-level pitch can be placed. The whole text is one
utterance, and its overall delivery is measured only against
`calibration_audio`; without it the result is the plain text, with a
warning. No timings are made up:

```python
text = open("examples/speech.txt", encoding="utf-8").read().strip()
result = converter.convert_detailed("examples/speech.wav", transcript=text)
print(result.iml)
print(result.transcript_source)
```

```text
<iml version="0.1.0" language="en-US"><utterance>I never said she stole my money.</utterance></iml>
transcript
```

With neither, `AudioToIML` transcribes the audio with Whisper if the
`whisper` extra is installed (`transcript_source == "whisper"`). Otherwise,
or with `stt="none"`, each stretch of speech becomes a `[speech]`
placeholder, and a warning says so:

```python
result = AudioToIML(stt="none").convert_detailed("examples/speech.wav")
print(result.iml)
print(result.transcript_source)
print(result.warnings[0])
```

```text
<iml version="0.1.0"><utterance><prosody pitch_contour="rise-fall">[speech]</prosody></utterance> <utterance><pause duration="610"/><prosody pitch_contour="rise-fall">[speech]</prosody></utterance></iml>
none
No transcript (stt='none'): each stretch of speech is a '[speech]' placeholder. For real words, give word timings (words) or a transcript, or install 'prosody-protocol[whisper]'.
```

`convert()` and `convert_to_doc()` issue the same warnings as
`UserWarning`s. The [speech-to-text guide](integrations/speech-to-text.md)
shows how to get word timings from each service, and the
[Whisper guide](integrations/whisper.md) covers local transcription.

## Give the result to an LLM

An LLM reads words well but not attribute values. `to_llm_context` turns
IML into an annotated transcript:

```python
from prosody_protocol import to_llm_context

print(to_llm_context(annotated.document))
```

```text
I never said [pause 0.6s] she **stole** (much higher pitch, falling) my money.
```

The ordinary fall at the end of a statement (and rise at the end of a
question) is not noted; other contours are.

`examples/sarcasm.iml` is a hand-annotated customer call with emotion labels:

```python
from pathlib import Path

sarcasm = Path("examples/sarcasm.iml").read_text(encoding="utf-8")
print(to_llm_context(sarcasm))
```

```text
customer: I've been on hold for {forty minutes} (slightly higher pitch, louder).
Delivery: sounds frustrated (estimated, 82%).
customer: Oh, that's just wonderful (higher pitch, slower, falling then rising). [pause 0.6s] Really **great** service.
Delivery: sounds sarcastic (estimated, 74%).
```

Emotions are named only at a confidence of at least `min_confidence`
(default 0.5) and always as estimates. `build_messages` adds a system prompt
that explains the notation and cautions the model against over-trusting
prosodic cues, and returns chat messages for any chat API:

```python
from prosody_protocol import build_messages

messages = build_messages(sarcasm, "Reply to the customer.")
print([m["role"] for m in messages])
print(messages[1]["content"])
```

```text
['system', 'user']
<transcript>
customer: I've been on hold for {forty minutes} (slightly higher pitch, louder).
Delivery: sounds frustrated (estimated, 82%).
customer: Oh, that's just wonderful (higher pitch, slower, falling then rising). [pause 0.6s] Really **great** service.
Delivery: sounds sarcastic (estimated, 74%).
</transcript>

Reply to the customer.
```

The [Claude guide](integrations/claude.md) sends these messages to Claude.

## Predict prosody for text

`TextToIML` guesses markup for plain text from punctuation, capitals and
cue words (a rule-based baseline, not a trained model). Sentences without
clear cues get no emotion:

```python
from prosody_protocol import TextToIML

predictor = TextToIML()
print(predictor.predict("Oh, that's GREAT."))
print(predictor.predict("Oh, that's great news!"))
print(predictor.predict("The app crashed again.", context="the user is frustrated"))
```

```text
<iml version="0.1.0"><utterance emotion="sarcastic" confidence="0.6"><prosody pitch_contour="fall-rise">Oh, that's <emphasis level="strong">GREAT</emphasis>.</prosody></utterance></iml>
<iml version="0.1.0"><utterance emotion="joyful" confidence="0.6"><prosody pitch="+5%" volume="+3dB">Oh, that's great news!</prosody></utterance></iml>
<iml version="0.1.0"><utterance emotion="frustrated" confidence="0.5">The app crashed again.</utterance></iml>
```

## Convert IML to SSML

`IMLToSSML` writes SSML 1.1 for speech synthesizers. Pitch, volume, rate,
pitch contours, emphasis and pauses are mapped; emotion, voice quality and
the extended measurements have no SSML equivalent and are left out:

```python
from prosody_protocol import IMLToSSML

print(IMLToSSML().convert(
    '<utterance>I <emphasis level="strong">really</emphasis> need this '
    '<pause duration="500"/> done today.</utterance>'
))
```

```text
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><s>I <emphasis level="strong">really</emphasis> need this <break time="500ms"/> done today.</s></speak>
```

How much of this a synthesizer honors differs by vendor; see the
[ElevenLabs](integrations/elevenlabs.md) and [Coqui TTS](integrations/coqui_tts.md)
guides.

## Speak IML

`IMLToAudio` returns a WAV file (mono, 16-bit, 22050 Hz). With espeak-ng
installed it speaks the document, with its pitch, volume, rate, pauses,
emphasis and pitch contours; the voice is recognizably synthetic. Without
espeak-ng it renders a tone preview (one tone per word, following the
prosody), not speech, and warns. The file goes in `workdir`, from above:

```python
from prosody_protocol import IMLToAudio

synthesizer = IMLToAudio()                  # engine="auto"
wav = synthesizer.synthesize(sarcasm)
(workdir / "sarcasm.wav").write_bytes(wav)
print(synthesizer.backend)                  # "espeak", or "tones" without espeak-ng
```

`IMLToAudio(engine="tones")` always renders the preview, and
`IMLToAudio(voice="en-us+m3")` picks an espeak-ng voice. Like `IMLToSSML`,
it rejects invalid IML (`IMLValidationError`) unless you pass `strict=False`.

## Apply a prosody profile

Some speakers express intent in ways the default reading gets wrong: for
example, a flat voice that is calm rather than bored. A prosody profile
(spec Section 7), written with the speaker, maps their patterns to what
they mean. `examples/monotone.wav` is such a speaker, and
`examples/profile.json` their profile. Without it, no emotion is reported:

```python
from prosody_protocol import ProfileLoader

words = load_word_timings("examples/monotone.deepgram.json")   # Deepgram's format
print(AudioToIML().convert("examples/monotone.wav", words=words))
```

```text
<iml version="0.1.0"><utterance>I read the list.</utterance> <utterance><pause duration="310"/>The room is booked.</utterance> <utterance><pause duration="300"/>I have the slides.</utterance> <utterance><pause duration="290"/><prosody rate="165%">And we got the grant!</prosody></utterance></iml>
```

With it, a mapping that matches an utterance sets its emotion, with the
default reading's confidence plus the mapping's `confidence_boost`:

```python
profile = ProfileLoader().load("examples/profile.json")
result = AudioToIML(profile=profile).convert_detailed("examples/monotone.wav", words=words)
print(result.iml)
for match in result.profile_matches:
    print(match.utterance, match.pattern, match.emotion, match.confidence, match.applied)
```

```text
<iml version="0.1.0"><utterance emotion="calm" confidence="0.61" x-profile="pitch_contour=flat">I read the list.</utterance> <utterance emotion="calm" confidence="0.62" x-profile="pitch_contour=flat"><pause duration="310"/>The room is booked.</utterance> <utterance emotion="calm" confidence="0.61" x-profile="pitch_contour=flat"><pause duration="300"/>I have the slides.</utterance> <utterance emotion="joyful" confidence="0.61" x-profile="pitch_contour=flat rate=fast"><pause duration="290"/><prosody rate="165%">And we got the grant!</prosody></utterance></iml>
0 {'pitch_contour': 'flat'} calm 0.61 True
1 {'pitch_contour': 'flat'} calm 0.62 True
2 {'pitch_contour': 'flat'} calm 0.61 True
3 {'pitch_contour': 'flat', 'rate': 'fast'} joyful 0.61 True
```

`x-profile` records which pattern matched, so downstream consumers can see
that a profile, not the default reading, set the emotion; the profile's
`user_id` is never written into the IML. Profiles are matched against the
speaker's usual voice: here, the recording's three ordinary sentences. For
a single sentence, also pass `calibration_audio=`, such as the earlier turn
from [above](#compare-with-the-speakers-earlier-turns):

```python
turn = AudioToIML(calibration_audio=[workdir / "earlier.wav"], profile=profile)
result = turn.convert_detailed(workdir / "turn.wav", words=turn_words)
print(to_llm_context(result.document))
```

```text
And we got the grant!
Delivery: overall much faster; sounds joyful (estimated, 60%; interpreted with the speaker's prosody profile).
```

`categorize_features` and `ProfileApplier` expose the matching step on its
own (see the [API reference](API.md#profiles)).

## Datasets and benchmarks

A dataset is a directory with `metadata.json`, one JSON file per entry in
`entries/` (see `schemas/dataset-entry.schema.json`) and the recordings in
`audio/`. An entry's `metadata.word_timings` may hold its word timings, in
any format `load_word_timings` reads. This builds a two-entry dataset from
the example recordings, in `workdir`:

```python
import json
import shutil

root = workdir / "my-dataset"
(root / "entries").mkdir(parents=True)
(root / "audio").mkdir()
(root / "metadata.json").write_text(json.dumps({"name": "my-dataset", "version": "0.1.0"}))

entries = {
    "speech": ("neutral", "speech.whisper.json",
               '<utterance>I never said <pause duration="600"/> she '
               '<emphasis level="strong">stole</emphasis> my money.</utterance>'),
    "monotone": ("calm", "monotone.deepgram.json",
                 '<utterance emotion="calm" confidence="0.8">I read the list. '
                 'The room is booked. I have the slides. And we got the grant!</utterance>'),
}
for i, (name, (label, timings, iml)) in enumerate(entries.items()):
    shutil.copy(f"examples/{name}.wav", root / "audio" / f"{name}.wav")
    entry = {
        "id": name, "timestamp": "2026-09-01T12:00:00Z", "source": "synthetic",
        "language": "en-US", "audio_file": f"audio/{name}.wav",
        "transcript": IMLParser().to_plain_text(IMLParser().parse(iml)),
        "iml": iml, "emotion_label": label, "annotator": "human",
        "consent": True, "speaker_id": f"speaker_{i}",
        "metadata": {"word_timings": json.loads(Path("examples", timings).read_text(encoding="utf-8"))},
    }
    (root / "entries" / f"{name}.json").write_text(json.dumps(entry, indent=2))
```

`DatasetLoader` validates every entry (rules D1-D12) and raises one
`DatasetError` listing every problem; entries without `"consent": true` are
never loaded. `Benchmark` runs a converter on each recording and scores the
output against the labels. A converter whose `convert` takes `words=` gets
each entry's word timings, or else (`transcript=`) its transcript:

```python
from prosody_protocol import Benchmark, DatasetLoader

dataset = DatasetLoader().load(root, check_audio=True)
print(dataset.name, dataset.size)

report = Benchmark(dataset, AudioToIML(stt="none")).run()
for metric, value in report.to_dict().items():
    if metric != "duration_seconds":
        print(f"{metric}: {value}")
print(report.check_regression(thresholds={"validity_rate": 1.0}))
print(report.check_regression(thresholds={"emotion_accuracy": 0.5}))
```

```text
my-dataset 2
emotion_accuracy: None
emotion_coverage: 0.0
emotion_f1: {'calm': 0.0, 'neutral': 0.0}
emotion_f1_macro: 0.0
confidence_ece: None
pitch_accuracy: None
pitch_coverage: None
pause_f1: 0.4
validity_rate: 1.0
failure_rate: 0.0
num_samples: 2
num_failures: 0
num_unaligned: 0
word_sources: {'timings': 2, 'transcript': 0, 'stt': 0}
abstention_label: None
[]
['emotion_accuracy = 0.0000 < threshold 0.5000 (over all entries: those without an emotion count as wrong; add an emotion_coverage threshold to score only the entries with one)']
```

The converter got each entry's word timings (`word_sources`). Neither
output carries an emotion: "speech" is a single sentence without
calibration, and the estimates for "monotone" stay below 0.5. An output
without an emotion is an abstention, so `emotion_accuracy`, measured over
the outputs that have one, is `None`, and `emotion_coverage` is 0; the
per-class F1 counts abstentions as misses. A threshold on `emotion_accuracy`
without an `emotion_coverage` threshold counts abstentions as wrong, so
abstaining cannot pass it. (`Benchmark(..., abstention_label="neutral")`
scores an abstention as "neutral" instead.) The pause
F1 is 0.4: the "speech" pause is found, and the three pauses between the
"monotone" sentences are not in its ground truth. Metrics with nothing to
compare are `None`. `check_regression` also fails on any failed conversion
unless a `failure_rate` threshold allows it. Two entries measure nothing,
of course: benchmark on dozens of entries per class at least. The same from
the command line is `prosody-protocol benchmark` followed by the dataset's
directory (`root`).

The `training/` directory of a clone trains small scikit-learn baselines on
such datasets; see [training/README.md](../training/README.md).

## Run the REST API

With the `api` extra:

```bash
prosody-protocol serve               # http://127.0.0.1:8000, docs at /docs
```

In another terminal:

```bash
curl -F audio=@examples/speech.wav -F words=@examples/speech.whisper.json -F language=en-US http://127.0.0.1:8000/v1/convert/audio-to-iml
```

```json
{"iml":"<iml version=\"0.1.0\" language=\"en-US\"><utterance>I never said<pause duration=\"610\"/> she <emphasis level=\"strong\"><prosody pitch=\"+47%\" pitch_contour=\"fall\">stole</prosody></emphasis> my <prosody pitch_contour=\"fall\">money.</prosody></utterance></iml>","plain_text":"I never said she stole my money.","transcript_source":"words","warnings":["No speaker baseline: a single utterance without calibration_audio has nothing to compare its pitch, loudness and rate with, so they were not marked, and no emotion was estimated from them. Pass calibration_audio (recordings of the speaker's usual speech, such as their earlier turns) to assess them."],"profile_matches":[]}
```

The server binds to 127.0.0.1 by default; set `PP_HOST`, `PP_PORT` and the
other `PP_*` settings (upload size, rate limit, worker processes) as
described in the [API reference](API.md#rest-api). The same app runs under
any ASGI server as `prosody_protocol.server.app:app`, or with
`python -m prosody_protocol.server`.

## Next steps

- [CLI reference](cli.md): every `prosody-protocol` command and flag
- [API reference](API.md): the Python classes and the REST endpoints
- Integration guides: [speech-to-text services](integrations/speech-to-text.md),
  [Whisper](integrations/whisper.md), [Claude](integrations/claude.md),
  [ElevenLabs](integrations/elevenlabs.md), [Coqui TTS](integrations/coqui_tts.md),
  [Mavis](integrations/mavis.md)
- [Specification](../spec.md): the formal IML spec
