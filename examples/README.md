# Examples

Small files to try the SDK, the `prosody-protocol` command and the REST API
with. Run the commands from the root of a clone. The commands that read
audio need the `audio` extra (`pip install -e ".[audio]"`); `synthesize`
speaks with espeak-ng when it is installed. `prosody-protocol doctor` shows
what your installation can do. The [quick start](../docs/quickstart.md)
walks through the same files in Python, and the
[CLI reference](../docs/cli.md) lists every option.

| File | What it is |
|------|------------|
| `speech.wav` | "I never said -- she STOLE my money." in a synthetic voice (espeak-ng), 4.2 s, 16 kHz mono. "stole" is higher and louder than the other words, and there is a 600 ms pause after "said". |
| `speech.whisper.json` | The word timings of `speech.wav` as an openai-whisper `transcribe(..., word_timestamps=True)` result: `segments[].words[]`, with Whisper's leading spaces and times in seconds (recognizer statistics such as tokens and log-probabilities are left out). The timings are exact: the audio was built word by word. |
| `speech.txt` | The plain transcript of `speech.wav`. |
| `monotone.wav` | Four short sentences in a flat, monotone synthetic voice, 6.4 s, 16 kHz mono: three at an ordinary pace, then "And we got the grant!" much faster. The speaker of `profile.json`. |
| `monotone.deepgram.json` | The word timings of `monotone.wav` as a Deepgram transcription response (`results.channels[0].alternatives[0].words[]`). `--words` recognizes the shape of each supported recognizer's output. |
| `sarcasm.iml` | A two-utterance IML document: a frustrated customer, then a sarcastic one. |
| `profile.json` | A prosody profile (spec Section 7) for a speaker whose flat, monotone delivery is easily misread: a flat voice is their calm, and fast speech their excitement. |
| `make_examples.py` | Regenerates `speech.wav`, `speech.whisper.json`, `speech.txt`, `monotone.wav` and `monotone.deepgram.json` (needs espeak-ng and the `audio` extra). |

## Annotate a recording

Bring the words from any speech recognizer (here, Whisper's output) and let
the SDK measure how they were said:

```sh
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json --language en-US
```

```xml
<iml version="0.1.0" language="en-US"><utterance>I never said<pause duration="660"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
```

The stressed word and the pause are marked. (The pause is 660 ms rather
than 600 because the silence starts where the "d" of "said" fades out.) The
same, as a transcript an LLM can read:

```sh
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json --prompt
```

```text
I never said [pause 0.7s] she **stole** (much higher pitch, falling) my money (falling).
```

`--json` reports where the words came from (`"transcript_source": "words"`)
and any warnings. No emotion is reported: a single sentence without
calibration audio (`--calibration`) does not show the speaker's usual voice
to compare with.

```sh
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json --json
```

With only a transcript (no timings), prosody is measured for the utterance
as a whole; with neither, Whisper transcribes the audio if it is installed
(`pip install -e ".[whisper]"`), and otherwise each stretch of
speech is a `[speech]` placeholder:

```sh
prosody-protocol from-audio examples/speech.wav --transcript-file examples/speech.txt
```

## Apply a prosody profile

A prosody profile, written with the speaker, says what their atypical
prosody means. `monotone.wav` is the speaker of `profile.json`. On its own,
the recording gets no emotion: the flat voice gives the default reading too
little to go on (its best guess for every sentence is "neutral", at a
confidence of 0.30 to 0.47, below the 0.5 needed to report it).

```sh
prosody-protocol from-audio examples/monotone.wav --words examples/monotone.deepgram.json
```

```xml
<iml version="0.1.0"><utterance>I read the list.</utterance><utterance><pause duration="310"/>The room is booked.</utterance><utterance><pause duration="300"/>I have the slides.</utterance><utterance><pause duration="310"/><prosody rate="170%">And we got the grant!</prosody></utterance></iml>
```

With the profile, a mapping that matches an utterance takes precedence
(spec 7.2): its emotion is used, with the default reading's confidence
plus the mapping's `confidence_boost`:

```sh
prosody-protocol from-audio examples/monotone.wav --words examples/monotone.deepgram.json --profile examples/profile.json
```

```xml
<iml version="0.1.0"><utterance emotion="calm" confidence="0.61" x-profile="pitch_contour=flat">I read the list.</utterance><utterance emotion="calm" confidence="0.62" x-profile="pitch_contour=flat"><pause duration="310"/>The room is booked.</utterance><utterance emotion="calm" confidence="0.61" x-profile="pitch_contour=flat"><pause duration="300"/>I have the slides.</utterance><utterance emotion="joyful" confidence="0.6" x-profile="pitch_contour=flat rate=fast"><pause duration="310"/><prosody rate="170%">And we got the grant!</prosody></utterance></iml>
```

The profile's use is reported (spec 7.2). Each utterance whose emotion a
mapping set carries `x-profile` with the pattern that matched, not the
profile's `user_id`: IML ends up in logs, datasets and prompts, and should
not name the person. Standard error explains each match:

```text
note: prosody profile 'example_user' set utterance 1 to 'calm' (confidence 0.61; matched pitch_contour=flat)
note: prosody profile 'example_user' set utterance 2 to 'calm' (confidence 0.62; matched pitch_contour=flat)
note: prosody profile 'example_user' set utterance 3 to 'calm' (confidence 0.61; matched pitch_contour=flat)
note: prosody profile 'example_user' set utterance 4 to 'joyful' (confidence 0.60; matched pitch_contour=flat, rate=fast)
```

and `--json` lists the matches in `profile_matches`, including those whose
confidence stays below 0.5 (`"applied": false`). Profiles are judged
against the speaker's usual voice: here the recording's three ordinary
sentences. For a single sentence, also pass a recording of the speaker
talking normally with `--calibration`; without one, a mapping can only
supply an emotion when its `confidence_boost` is 0.5 or more. `speech.wav`
is neither flat nor fast, so none of the profile's mappings match it:

```sh
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json --profile examples/profile.json --json
```

## Work with IML

```sh
prosody-protocol validate examples/sarcasm.iml
prosody-protocol to-text examples/sarcasm.iml
prosody-protocol to-ssml examples/sarcasm.iml
prosody-protocol to-prompt examples/sarcasm.iml
prosody-protocol to-prompt examples/sarcasm.iml --messages --instruction "Reply to the customer."
prosody-protocol synthesize examples/sarcasm.iml -o sarcasm.wav
prosody-protocol from-text "Oh great, another meeting."
```

`to-prompt` gives:

```text
customer: I've been on hold for {forty minutes} (slightly higher pitch, louder).
Delivery: sounds frustrated (estimated, 82%).
customer: Oh, that's just wonderful (higher pitch, slower, falling then rising). [pause 0.6s] Really **great** service.
Delivery: sounds sarcastic (estimated, 74%).
```

and `--messages` wraps it, with a system prompt that explains the notation,
as chat messages for any chat-completion API.

## Python

```python
from prosody_protocol import AudioToIML, ProfileLoader, load_word_timings, to_llm_context

words = load_word_timings("examples/speech.whisper.json")
result = AudioToIML(language="en-US").convert_detailed("examples/speech.wav", words=words)
print(result.iml)
print(to_llm_context(result.document))

# With the speaker's prosody profile, result.profile_matches reports its use.
profile = ProfileLoader().load("examples/profile.json")
words = load_word_timings("examples/monotone.deepgram.json")
result = AudioToIML(profile=profile).convert_detailed("examples/monotone.wav", words=words)
for match in result.profile_matches:
    print(match.utterance, match.pattern, match.emotion, match.confidence, match.applied)
```

## REST API

Start the server with `prosody-protocol serve` (needs the `api` extra:
`pip install -e ".[api]"`), then:

```sh
curl -F audio=@examples/speech.wav -F words=@examples/speech.whisper.json -F language=en-US http://127.0.0.1:8000/v1/convert/audio-to-iml
curl -F audio=@examples/monotone.wav -F words=@examples/monotone.deepgram.json -F profile=@examples/profile.json http://127.0.0.1:8000/v1/convert/audio-to-iml
python -c 'import json; print(json.dumps({"iml": open("examples/sarcasm.iml").read()}))' | curl -H 'Content-Type: application/json' -d @- http://127.0.0.1:8000/v1/convert/iml-to-prompt
```

The [API reference](../docs/API.md#rest-api) documents every endpoint, its
fields and errors, and the server's `PP_*` settings.
