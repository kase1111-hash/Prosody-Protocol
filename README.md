# Prosody Protocol

**A protocol for preserving human intent across the speech-to-text boundary.**

![Status](https://img.shields.io/badge/status-alpha-orange)
![Version](https://img.shields.io/badge/version-0.1.0a3-orange)
![Spec license](https://img.shields.io/badge/spec-CC--BY--4.0-green)
![Code license](https://img.shields.io/badge/code-MIT-green)

Prosody Protocol defines **Intent Markup Language (IML)**, an XML format that
keeps *how* something was said (pitch, loudness, tempo, pauses, emphasis,
voice quality) next to *what* was said. This repository holds the
[specification](spec.md) and `prosody_protocol`, a Python SDK with a
command line and a REST API that measure prosody in recordings, write and
validate IML, and hand it to language models and speech synthesizers.

**Status: alpha (0.1.0a3).** The specification is a draft (0.1.0-alpha).
The SDK works end to end on the shipped examples and has about 2,500 tests,
but it is not on PyPI yet, its emotion labels come from a rule-based
heuristic, and its audio analysis has been validated on synthetic
(espeak-ng) speech only. See [What is real and what is heuristic](#what-is-real-and-what-is-heuristic).

---

## The problem

Speech-to-text keeps the words and drops the delivery:

```text
Speaker:        "Oh, that's GREAT."   (pitch jumps on "great", then falls sharply)
Speech-to-text: "oh that's great"
LLM reads:      genuine enthusiasm
LLM replies:    "I'm glad you're happy!"
```

Pitch contours, loudness, tempo, pauses and voice quality are often what
separates sarcasm from sincerity, urgency from calm, and hesitation from
confidence. A plain transcript cannot carry them, so every system
downstream of it has to guess.

IML writes them down:

```xml
<utterance emotion="sarcastic" confidence="0.87">
  Oh, that's
  <prosody pitch="+15%" volume="+6dB" pitch_contour="fall-sharp">GREAT</prosody>.
</utterance>
```

and the SDK turns IML into a transcript a language model can read:

```text
Oh, that's GREAT (higher pitch, louder, sharply falling).
Delivery: sounds sarcastic (estimated, 87%).
```

This example is annotated by hand. What the SDK can measure from a
recording today is narrower: pauses, stressed words, pitch and loudness
relative to the speaker, pitch movement, tempo and voice quality, plus a
rough estimate of a few emotions when it has enough of the speaker's
speech to compare with. It does not detect sarcasm from audio.

---

## Install

The package is not on PyPI yet. Install it from GitHub:

```bash
pip install "prosody-protocol[audio] @ git+https://github.com/kase1111-hash/Prosody-Protocol"
```

or from a clone, which also gives you the examples, the training scripts and
the tests:

```bash
git clone https://github.com/kase1111-hash/Prosody-Protocol
cd Prosody-Protocol
pip install -e ".[audio]"
```

Python 3.10 or newer. The core needs only `lxml`; extras add the rest:

| Extra | Adds | Enables |
|-------|------|---------|
| *(none)* | `lxml` | Parsing, validation, IML to SSML, IML to LLM context, text to IML, prosody profiles, datasets, word-timing adapters |
| `audio` | `numpy`, `praat-parselmouth` | Measuring prosody in recordings (`AudioToIML`, `ProsodyAnalyzer`), speaking IML (`IMLToAudio`), `Benchmark`, `MavisBridge` |
| `whisper` | `audio` + `openai-whisper` (pulls in PyTorch) | Built-in speech recognition, for when you have no word timings or transcript |
| `api` | `audio` + `fastapi`, `uvicorn`, `python-multipart` | The REST API (`prosody-protocol serve`) |
| `ml` | `audio` + `scikit-learn`, `joblib`, `pyyaml` | The `training/` scripts (source checkout only) |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `lxml-stubs`, `httpx`, `jsonschema` | Development |
| `all` | `audio`, `whisper`, `ml`, `api` | Everything except `dev` |

Two optional system tools:

- **espeak-ng** lets `IMLToAudio` and `prosody-protocol synthesize` speak
  IML. Without it they render a tone preview, not speech.
- **ffmpeg** lets the SDK read OGG/Opus, WebM, M4A and other formats
  besides WAV, AIFF, FLAC and MP3, and gives exact MP3 timing.

Install them with `sudo apt install espeak-ng ffmpeg` (Debian/Ubuntu) or
`brew install espeak-ng ffmpeg` (macOS); on Windows use the installers from
the espeak-ng and ffmpeg projects. `prosody-protocol doctor` shows what your
installation can do:

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
          install: pip install 'prosody-protocol[whisper]' (pulls in PyTorch)
[ok     ] speech synthesis (espeak-ng at /usr/bin/espeak-ng)
          enables: synthesize speaks IML (otherwise it renders a tone preview, not speech)
...
```

The install hints that `doctor` and error messages print
(`pip install 'prosody-protocol[whisper]'`) assume a PyPI release. Until
there is one, add an extra with `pip install -e ".[whisper]"` in a clone,
or with the `git+https` command above. Where the `prosody-protocol` script
is not on your `PATH`, `python -m prosody_protocol` runs the same command.

---

## Quickstart

From the root of a clone with the `audio` extra installed. `examples/speech.wav`
is a synthetic voice (espeak-ng) saying "I never said -- she STOLE my money.",
with "stole" higher and louder and a pause after "said".
`examples/speech.whisper.json` holds its word timings in the shape Whisper
returns them.

```bash
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json
```

```xml
<iml version="0.1.0"><utterance>I never said<pause duration="660"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
```

The SDK measured the recording with Praat, found the stressed word and the
pause, and wrote them as IML. It wrote no `emotion`: one sentence without a
recording of the speaker's normal voice gives it nothing to compare with,
and it leaves the label out rather than guess. For a language model, ask
for the annotated transcript instead:

```bash
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json --prompt
```

```text
I never said [pause 0.7s] she **stole** (much higher pitch, falling) my money (falling).
```

The same in Python:

```python
from prosody_protocol import AudioToIML, load_word_timings, to_llm_context

words = load_word_timings("examples/speech.whisper.json")
result = AudioToIML(language="en-US").convert_detailed("examples/speech.wav", words=words)
print(result.iml)
print(to_llm_context(result.document))
```

```text
<iml version="0.1.0" language="en-US"><utterance>I never said<pause duration="660"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
I never said [pause 0.7s] she **stole** (much higher pitch, falling) my money (falling).
```

**Where the words come from.** The SDK measures *how* words were said; it
needs to know *which* words and when. You have three options:

- **Word timings from any speech recognizer.** `load_word_timings()` (and
  `--words`) reads the output of openai-whisper, faster-whisper, WhisperX,
  the OpenAI transcription API (`verbose_json`), Deepgram, AssemblyAI,
  Google Cloud Speech-to-Text, or a list of `{word, start_ms, end_ms}`
  records, and detects which it is. This is the recommended path: no model
  download, and you keep the recognizer you already use.
- **A transcript without timings** (`transcript=`, `--transcript`). Prosody
  is then measured for the utterance as a whole.
- **Built-in Whisper**, with the `whisper` extra. Without it and without
  words or a transcript, each stretch of speech becomes a `[speech]`
  placeholder and the result says so in its warnings.

[examples/README.md](examples/README.md) has more to try, including the
accessibility profile below. Further documentation:
[docs/quickstart.md](docs/quickstart.md) (a longer tour),
[docs/cli.md](docs/cli.md) (every command and option),
[docs/API.md](docs/API.md) (Python and REST reference) and
[docs/integrations/](docs/integrations/) (speech recognizers, LLMs, speech
synthesizers, Mavis).

---

## IML at a glance

An IML document is one `<utterance>`, or several wrapped in `<iml>`:

```xml
<iml version="0.1.0" language="en-US">
  <utterance emotion="frustrated" confidence="0.82" speaker_id="customer">
    I've been on hold for <prosody pitch="+8%" volume="+4dB">forty minutes</prosody>.
  </utterance>
  <utterance speaker_id="customer">
    Well<pause duration="800"/> I <emphasis level="strong">said</emphasis> I'm fine.
  </utterance>
</iml>
```

| Element | Attributes | Notes |
|---------|------------|-------|
| `<iml>` | `version`, `language` (BCP 47), `consent` (`explicit`, `implicit`, `none`), `processing` (`local`, `remote`, `hybrid`) | Optional wrapper; contains only utterances |
| `<utterance>` | `emotion`, `confidence` (0.0-1.0, required with `emotion`), `speaker_id` | One spoken phrase or sentence |
| `<prosody>` | `pitch` (`+15%`, `-2st`, `185Hz`), `pitch_contour` (`rise`, `fall`, `rise-fall`, `fall-rise`, `rise-sharp`, `fall-sharp`, `flat`), `volume` (`+6dB`), `rate` (`fast`, `medium`, `slow` or `150%`), `quality` (`modal`, `breathy`, `tense`, `creaky`, `whispery`, `harsh`) | Relative values are against the speaker's baseline |
| `<pause>` | `duration` (whole milliseconds, required) | Always empty: `<pause duration="800"/>` |
| `<emphasis>` | `level` (`strong`, `moderate`, `reduced`; required) | Not directly inside another `<emphasis>` |
| `<segment>` | `tempo` (`rushed`, `steady`, `drawn-out`), `rhythm` (`staccato`, `legato`, `syncopated`) | Only as a direct child of `<utterance>` |

The core emotion vocabulary is `neutral`, `sincere`, `sarcastic`,
`frustrated`, `joyful`, `uncertain`, `angry`, `sad`, `fearful`, `surprised`,
`disgusted`, `calm` and `empathetic`. `<prosody>` may also carry measured
values for research (`f0_mean`, `f0_range`, `intensity_mean`, `speech_rate`,
`jitter`, `shimmer`, `hnr`, ...), and custom attributes use an `x-` prefix.
`<prosody>` and `<emphasis>` should not nest more than two levels deep.
[spec.md](spec.md) is the full specification, with an RFC 2119 conformance
summary in Appendix D; [schemas/](schemas/) has an XML Schema and JSON
schemas for profiles and dataset entries.

Check a document from the command line or from Python:

```bash
prosody-protocol validate examples/sarcasm.iml
```

```python
from prosody_protocol import IMLValidator

result = IMLValidator().validate(
    '<utterance emotion="sarcastic">Oh, <emphasis>great</emphasis>.</utterance>'
)
print(result.valid)
for issue in result.errors:
    print(issue.rule, issue.message)
```

```text
False
V3 <utterance> has emotion="sarcastic" but no confidence attribute
V8 <emphasis> is missing required attribute 'level'
```

`result.warnings` holds SHOULD-level issues (such as nesting deeper than
two levels, or values implausible for a human voice), which do not make a
document invalid, and `result.raise_for_errors()` raises
`IMLValidationError` when there are errors.

---

## Use cases

### Hand a voice turn to a language model

`to_llm_context()` renders IML as an annotated transcript, and
`build_messages()` wraps it in chat messages with a system prompt that
explains the notation and warns the model not to over-trust it. The
messages are plain dicts that work with any chat API.

```python
from prosody_protocol import build_messages, to_llm_context

iml = open("examples/sarcasm.iml", encoding="utf-8").read()
print(to_llm_context(iml))
messages = build_messages(iml, "Reply to the customer.")
```

```text
customer: I've been on hold for {forty minutes} (slightly higher pitch, louder).
Delivery: sounds frustrated (estimated, 82%).
customer: Oh, that's just wonderful (higher pitch, slower, falling then rising). [pause 0.6s] Really **great** service.
Delivery: sounds sarcastic (estimated, 74%).
```

Emotions below a confidence of 0.5 are shown as "emotion not reliably
detected" (`min_confidence=` changes this), and measured values appear only
with `include_numbers=True`. With the Anthropic SDK (`pip install anthropic`):

<!-- docs-test: skip -->
```python
import anthropic

client = anthropic.Anthropic()
response = client.beta.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    system=messages[0]["content"],
    messages=messages[1:],
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",  # if a safety classifier declines, retry on another model
)
if response.stop_reason == "refusal":
    print("The model declined to answer.")
else:
    print(next((block.text for block in response.content if block.type == "text"), ""))
```

`fallbacks` needs a recent `anthropic` package (1.8.0 has it); with
an older one, call `client.messages.create` without the `betas` and
`fallbacks` lines. `prosody-protocol to-prompt FILE --messages` prints the
same messages as JSON, and the REST API has
`POST /v1/convert/iml-to-prompt`.

**Confirming risky actions.** Prosody can tell an agent that a spoken
command might not be meant literally ("Oh sure, just archive EVERYTHING,
that'll help."). It cannot tell the agent what was asked. Prosodic cues are
probabilistic, so they must never be the only basis for a consequential
decision. Decide from the words which action was requested and whether it
is destructive, and always confirm destructive actions. Prosody can then
add caution, but it never removes a safeguard. A check that works this
way and fails closed:

```python
from prosody_protocol import IMLParser

SETTLED = {"calm", "neutral", "sincere"}


def needs_confirmation(iml: str, destructive: bool, min_confidence: float = 0.7) -> bool:
    """Whether to ask the user before acting on a spoken request.

    *destructive* comes from the words and the requested action, never from
    prosody. Anything else is confirmed too, unless there are words and every
    utterance was labeled settled with enough confidence. Missing labels mean "ask".
    """
    if destructive:
        return True
    parser = IMLParser()
    doc = parser.parse(iml)
    if not parser.to_plain_text(doc):
        return True
    return not all(
        u.emotion in SETTLED and u.confidence is not None and u.confidence >= min_confidence
        for u in doc.utterances
    )


archive = "Archive last week's reports."
print(needs_confirmation(
    '<utterance emotion="sarcastic" confidence="0.87">Oh sure, just archive everything.</utterance>',
    destructive=False,
))
print(needs_confirmation(f"<utterance>{archive}</utterance>", destructive=False))
print(needs_confirmation("<utterance></utterance>", destructive=False))
print(needs_confirmation(
    f'<utterance emotion="calm" confidence="0.9">{archive}</utterance>', destructive=False
))
print(needs_confirmation(
    '<utterance emotion="calm" confidence="0.9">Delete the temporary files.</utterance>',
    destructive=True,
))
```

```text
True
True
True
False
True
```

Only the calm, labeled request for a reversible action goes ahead without
confirmation. With the built-in heuristic even that one would ask: the
heuristic keeps `neutral` and `calm` below its reporting threshold of 0.5,
so its output never carries them. Only a prosody profile or a trained
classifier can supply a settled label. Treat a recording converted without
real words (`result.transcript_source == "none"`, `[speech]` placeholders)
as no request at all.

### Accessibility: prosody profiles

Some people's prosody is easily misread: a flat, monotone voice sounds bored
or annoyed to listeners (and to classifiers) when the speaker is calm or
excited. A prosody profile ([spec Section 7](spec.md#7-user-prosody-profiles)),
written with the speaker, maps their patterns to what they mean.
[examples/profile.json](examples/profile.json) belongs to the speaker of
`examples/monotone.wav`; two of its mappings:

```json
{
  "profile_version": "0.1.0",
  "user_id": "example_user",
  "prosody_mappings": [
    {"pattern": {"pitch_contour": "flat", "rate": "fast"},
     "interpretation": {"emotion": "joyful", "confidence_boost": 0.3}},
    {"pattern": {"pitch_contour": "flat"},
     "interpretation": {"emotion": "calm", "confidence_boost": 0.15}}
  ]
}
```

```python
from prosody_protocol import AudioToIML, ProfileLoader, load_word_timings, to_llm_context

profile = ProfileLoader().load("examples/profile.json")
words = load_word_timings("examples/monotone.deepgram.json")  # Deepgram's response shape
result = AudioToIML(profile=profile).convert_detailed("examples/monotone.wav", words=words)
print(to_llm_context(result.document))
```

```text
I read the list.
Delivery: sounds calm (estimated, 61%; interpreted with the speaker's prosody profile).
[pause 0.3s] The room is booked.
Delivery: sounds calm (estimated, 62%; interpreted with the speaker's prosody profile).
[pause 0.3s] I have the slides.
Delivery: sounds calm (estimated, 61%; interpreted with the speaker's prosody profile).
[pause 0.3s] And we got the grant!
Delivery: overall much faster; sounds joyful (estimated, 60%; interpreted with the speaker's prosody profile).
```

Without the profile the same recording gets no emotion at all: the
heuristic's best guess for every sentence is "neutral" below 0.5
confidence. Spec Section 7.2 asks that profile use be reported
downstream, and the SDK reports it in three places:

- Each utterance whose emotion a mapping set carries `x-profile` with the
  matched pattern (for example `x-profile="pitch_contour=flat rate=fast"`).
- `result.profile_matches` lists every match as a `ProfileMatch`.
- `to_llm_context()` marks those emotions as "interpreted with the
  speaker's prosody profile", and the system prompt of `build_messages()`
  (`prosody_protocol.llm.SYSTEM_PROMPT`) tells the model to prefer that
  reading.

The profile's `user_id` is never written into IML.

### Speech synthesis: SSML and espeak-ng

`TextToIML` predicts markup for plain text with rules (punctuation,
capitals, cue words); `IMLToSSML` converts IML to SSML 1.1 for speech
engines; `IMLToAudio` speaks it with espeak-ng.

```python
from prosody_protocol import IMLToAudio, IMLToSSML, TextToIML

iml = TextToIML().predict("Oh, that's GREAT.")
print(iml)
print(IMLToSSML().convert(iml))
IMLToAudio().synthesize_to_file(iml, "great.wav")
```

```text
<iml version="0.1.0"><utterance emotion="sarcastic" confidence="0.6"><prosody pitch_contour="fall-rise">Oh, that's <emphasis level="strong">GREAT</emphasis>.</prosody></utterance></iml>
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><s><prosody contour="(0%,+0%) (50%,-15%) (100%,+15%)">Oh, that's <emphasis level="strong">GREAT</emphasis>.</prosody></s></speak>
```

espeak-ng speech is robotic, but it is speech, and IML's pitch, volume,
rate, pauses, emphasis and contours are adapted to what espeak-ng renders.
Without espeak-ng, `IMLToAudio` warns and renders a tone preview (one tone
per word), which is not speech. `IMLToSSML(vendor="espeak-ng")` writes SSML
adapted to espeak-ng; standard SSML suits engines that accept it. From the
command line: `prosody-protocol from-text`, `to-ssml` and `synthesize`.

### Datasets, benchmarks and training

A dataset is a directory with `metadata.json`, `entries/*.json` (see
[schemas/dataset-entry.schema.json](schemas/dataset-entry.schema.json)) and
`audio/`. `DatasetLoader` validates every entry when it loads it and never
loads one without `"consent": true`. `Benchmark` scores any converter with a
`convert(audio_path) -> str` method against a dataset:

```python
import json
import shutil
import tempfile
from pathlib import Path

from prosody_protocol import AudioToIML, Benchmark, DatasetLoader, load_word_timings

root = Path(tempfile.mkdtemp()) / "demo"
(root / "entries").mkdir(parents=True)
(root / "audio").mkdir()
shutil.copy("examples/speech.wav", root / "audio" / "speech.wav")
entry = {
    "id": "speech_001",
    "timestamp": "2026-09-01T12:00:00Z",
    "source": "synthetic",
    "language": "en-US",
    "audio_file": "audio/speech.wav",
    "transcript": "I never said she stole my money.",
    "iml": '<utterance>I never said<pause duration="600"/> she '
           '<emphasis level="strong">stole</emphasis> my money.</utterance>',
    "emotion_label": "neutral",
    "annotator": "human",
    "consent": True,
}
(root / "entries" / "speech_001.json").write_text(json.dumps(entry), encoding="utf-8")
dataset = DatasetLoader().load(root, check_audio=True)


class WithWords:
    """A converter that uses known word timings instead of speech recognition."""

    def convert(self, audio_path):
        words = load_word_timings("examples/speech.whisper.json")
        return AudioToIML().convert(audio_path, words=words)


report = Benchmark(dataset, WithWords()).run()
print(report.pause_f1, report.validity_rate, report.failure_rate)
```

```text
1.0 1.0 0.0
```

`prosody-protocol benchmark DATASET_DIR` runs the default `AudioToIML`
(which needs the `whisper` extra to get real words from the audio) and
exits 1 on failed conversions or on regressions against a saved report. [datasets/README.md](datasets/README.md) describes the
format; `MavisBridge` exports sessions of the
[Mavis](https://github.com/kase1111-hash/Mavis) vocal typing instrument as
datasets. [training/README.md](training/README.md) trains three
scikit-learn baselines (speech emotion recognition, text-to-prosody,
pitch contour) from a source checkout. The repository ships no training
corpus, and the baselines are small models meant for inspection and
comparison, not production use.

---

## REST API

With the `api` extra:

```bash
prosody-protocol serve
```

The server binds to 127.0.0.1:8000 by default (`--host`, `--port`, or
`PP_HOST`/`PP_PORT`). `python -m prosody_protocol.server` does the same,
and the [Dockerfile](Dockerfile) runs it in a container with espeak-ng and
ffmpeg. Interactive OpenAPI docs are at `/docs`.

| Endpoint | Does |
|----------|------|
| `GET /v1/health` | Version, optional backends found (whisper, espeak-ng, ffmpeg) and limits |
| `POST /v1/validate` | Validate IML: `{"iml": ...}` returns `{valid, issues}` |
| `POST /v1/convert/audio-to-iml` | Multipart: `audio`, and optionally `words`, `transcript`, `profile`, `language` |
| `POST /v1/convert/text-to-iml` | `{"text": ..., "context": ...}` to predicted IML |
| `POST /v1/convert/iml-to-ssml` | IML to SSML 1.1 |
| `POST /v1/convert/iml-to-prompt` | IML to an annotated transcript, system prompt and chat messages |
| `POST /v1/synthesize` | IML to `audio/wav` (espeak-ng, or a tone preview without it) |

```bash
curl -F audio=@examples/speech.wav -F words=@examples/speech.whisper.json \
     -F language=en-US http://127.0.0.1:8000/v1/convert/audio-to-iml
```

```json
{"iml":"<iml version=\"0.1.0\" language=\"en-US\"><utterance>I never said<pause duration=\"660\"/> she <emphasis level=\"strong\"><prosody pitch=\"+47%\" pitch_contour=\"fall\">stole</prosody></emphasis> my <prosody pitch_contour=\"fall\">money.</prosody></utterance></iml>","plain_text":"I never said she stole my money.","transcript_source":"words","warnings":[],"profile_matches":[]}
```

Uploads, text fields, audio length, synthesis length, rate limits and the
number of worker processes are limited by `PP_*` environment variables.
[docs/API.md](docs/API.md) documents the endpoints, errors and settings.

---

## What is real and what is heuristic

| Part | What it is | How far it has been checked |
|------|------------|-----------------------------|
| Spec, parser, validator | IML 0.1.0-alpha; the validator implements the spec's rules (V1-V33) | Tested against every spec example, the XML Schema and hand-written edge cases |
| Prosody measurement (`ProsodyAnalyzer`) | Praat, through parselmouth: F0, intensity, pauses, speech rate, jitter, shimmer, HNR | Synthetic espeak-ng speech with known timings and pitch; **no benchmark on real recordings yet** |
| Markup (`AudioToIML`, `IMLAssembler`) | Rules on measured values relative to the speaker's own voice: pauses of 200 ms or more, emphasis, pitch contour, utterance-level shifts | Same as above |
| Emotion labels | `RuleBasedEmotionClassifier`: deviations of pitch, loudness, rate and pitch movement from the speaker's baseline; labels only `neutral`, `calm`, `sad`, `angry`, `joyful`, `fearful` | A heuristic, not a trained model. It abstains (no `emotion`) below 0.5 confidence, needs calibration audio or at least three utterances of the same speaker, and cannot hear sarcasm or frustration |
| Speech recognition | None built in unless you install the `whisper` extra | Bring word timings from any recognizer |
| `TextToIML` | Rules: punctuation, capitals, cue words, simple sarcasm idioms | A text heuristic; confidences between 0.5 and 0.8 |
| `IMLToAudio` | espeak-ng (robotic but real speech); a tone preview without it | Pitch and rate mappings measured against espeak-ng |
| Training baselines | scikit-learn logistic regression, decision tree, random forest | Pipeline tests on small synthetic data only |

Limits that follow from this:

- **Prosodic cues are probabilistic.** They vary between people, languages,
  cultures and recording conditions. Never use them as the only basis for
  a consequential decision. IML must not be used for deception detection,
  covert emotional surveillance or profiling people in hiring, lending or
  law enforcement ([spec Section 8.2](spec.md#82-prohibited-use-cases)).
- **Emotion data is personal data** ([spec Section 8.1](spec.md#81-emotional-data-as-pii)):
  get explicit consent before collecting it, and process locally where you
  can. The SDK analyzes audio on your machine and sends nothing anywhere
  (the `whisper` extra downloads its model on first use).
- For emotion labels on short recordings, pass a recording of the same
  speaker talking normally (`calibration_audio=`, `--calibration`).
  For better labels, plug in your own classifier
  (`AudioToIML(emotion_classifier=...)`).
- The analysis is tuned on English synthetic speech. IML itself is
  language-agnostic; interpreting it is not.

**Why not SSML?** SSML tells a synthesizer how to speak (text to speech).
IML describes how someone did speak (speech to text), with confidence on
every inferred label. The vocabularies overlap, and `IMLToSSML` converts
one to the other.

---

## Roadmap

- A benchmark on real, consented speech recordings, and results published
  from it.
- A trained speech-emotion backend, evaluated on that benchmark, behind the
  existing `emotion_classifier` interface.
- A PyPI release.
- Spec work listed in [spec Section 10](spec.md#10-future-work): turn-taking
  and overlap, backchannels, laughter and other non-verbal sounds,
  cross-lingual interpretation, streaming IML.

---

## Project layout

```text
spec.md                   IML specification (draft 0.1.0-alpha)
schemas/                  XML Schema for IML, JSON schemas for profiles and dataset entries
src/prosody_protocol/     the Python SDK
    parser.py, validator.py, models.py      IML documents
    audio_to_iml.py, prosody_analyzer.py    recordings to IML (audio extra)
    assembler.py, emotion_classifier.py     markup and emotion rules
    alignment.py                            word timings from speech recognizers
    llm.py                                  IML for language models
    iml_to_ssml.py, iml_to_audio.py         IML to SSML and speech
    text_to_iml.py, profiles.py             text to IML, prosody profiles
    datasets.py, benchmarks.py, mavis_bridge.py
    cli.py                                  the prosody-protocol command
    server/                                 the REST API (api extra)
examples/                 recordings, word timings, a profile and IML to try
training/                 scikit-learn baselines (source checkout only)
datasets/                 dataset format documentation
docs/                     API and CLI reference, quickstart, integration guides
tests/                    pytest suite, including tests that run the docs' examples
```

---

## Contributing

Contributions to the specification, the SDK, datasets and research are
welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers the development setup,
tests and conventions, and [CHANGELOG.md](CHANGELOG.md) lists changes by
version. Report problems and propose spec changes in
[GitHub Issues](https://github.com/kase1111-hash/Prosody-Protocol/issues).

Related projects by the same author:

- [Mavis](https://github.com/kase1111-hash/Mavis): a vocal typing
  instrument whose sessions can become IML datasets.
- [Intent-Engine](https://github.com/kase1111-hash/Intent-Engine): a
  prosody-aware assistant that consumes IML.
- [Agent-OS](https://github.com/kase1111-hash/Agent-OS): a natural-language
  operating system for AI agents.

## License

- Specification (`spec.md`, `schemas/`): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
  Credit "Prosody Protocol / True North Construction LLC".
- Code: MIT, see [LICENSE](LICENSE).
- Datasets: each dataset states its own license.

## Citation

```bibtex
@misc{prosody_protocol_2026,
  title  = {Intent Markup Language: Preserving Prosodic Information for AI Systems},
  author = {{True North Construction LLC}},
  year   = {2026},
  url    = {https://github.com/kase1111-hash/Prosody-Protocol},
  note   = {Specification 0.1.0-alpha, SDK 0.1.0a3}
}
```

Maintainer: Kase Branham (kase1111@gmail.com).
