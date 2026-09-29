# Claude

Give Claude the prosody of what a user said, not just the words, so it can
tell "Oh, that's just wonderful." said sincerely from the same words said
sarcastically.

Claude reads text well but gains little from raw IML attributes such as
`pitch="+15%"`, and an `emotion` label with `confidence="0.41"` reads like a
fact. `prosody_protocol.llm` turns IML into an annotated transcript instead,
with a system prompt that explains the notation and tells the model how far
to trust it. It needs no LLM SDK; this page shows how to send its output
with the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python).

The calls to Claude on this page are not run by the docs tests (they need
an API key); the `prosody_protocol` parts are.

## Setup

```bash
pip install -e ".[audio]" anthropic      # in a clone; or install prosody-protocol from GitHub
export ANTHROPIC_API_KEY=...
```

## Build the messages

`examples/sarcasm.iml` is a customer on a support call, annotated by hand:

```python
from pathlib import Path

from prosody_protocol import build_messages, to_llm_context

iml = Path("examples/sarcasm.iml").read_text(encoding="utf-8")
print(to_llm_context(iml))
```

```text
customer: I've been on hold for {forty minutes} (slightly higher pitch, louder).
Delivery: sounds frustrated (estimated, 82%).
customer: Oh, that's just wonderful (higher pitch, slower, falling then rising). [pause 0.6s] Really **great** service.
Delivery: sounds sarcastic (estimated, 74%).
```

`build_messages` wraps this for a chat API: a system message with
`SYSTEM_PROMPT`, then a user message with the transcript in `<transcript>`
tags, followed by your instruction if you give one:

```python
messages = build_messages(iml, "Reply to the customer in two sentences.")
print([message["role"] for message in messages])
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

Reply to the customer in two sentences.
```

Without an instruction, the model answers the speaker.

## Send them to Claude

The Messages API takes the system prompt as the top-level `system`
parameter, not as a message, and requires `max_tokens`. Pass the first
message's content as `system` and the rest as `messages`:

<!-- docs-test: skip -->
```python
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
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
    reply = "".join(block.text for block in response.content if block.type == "text")
    print(reply)
```

`response.content` is a list of blocks; it can start with a `thinking`
block, so collect the `text` blocks rather than reading
`response.content[0].text`. Check `response.stop_reason` too: `"max_tokens"`
means the reply was cut off, and `"refusal"` that the model (and any
fallback) declined. Claude Opus 5 thinks adaptively by default, and thinking
tokens count toward `max_tokens`, so don't set it too low. `fallbacks` needs
a recent `anthropic` package (1.8.0 has it); with an older one, call
`client.messages.create` without the `betas` and `fallbacks` lines. Any
current Claude model ID works in `model`.

For a conversation, keep `system` the same on every turn, and add each new
spoken turn as a user message built with `build_messages(turn_iml)[1]`,
after the assistant's previous reply. A single turn has no baseline of its
own, so convert each one with the user's earlier turns as
`calibration_audio` (see the
[quick start](../quickstart.md#compare-with-the-speakers-earlier-turns)):
only then can its `Delivery:` line say that it was, for example, faster or
louder than usual.

## From a recording

`AudioToIML` produces the IML (see the [quick start](../quickstart.md)); give
its document to `build_messages`:

```python
from prosody_protocol import AudioToIML, load_word_timings

words = load_word_timings("examples/speech.whisper.json")
document = AudioToIML(language="en-US").convert_to_doc("examples/speech.wav", words=words)
messages = build_messages(document, "What is the speaker insisting on?")
print(messages[1]["content"])
```

```text
<transcript>
I never said [pause 0.6s] she **stole** (much higher pitch, falling) my money.
</transcript>

What is the speaker insisting on?
```

Then send `messages` as above. The stress on "stole" is what distinguishes
this reading of the sentence from, say, "*I* never said she stole my
money." There is no `Delivery:` line: one sentence without calibration
audio has no baseline, so `AudioToIML` marked no overall delivery or
emotion (and warned). Without words, the transcript holds `[speech]`
placeholders, and the system prompt tells the model that their words are
unknown and must not be guessed.

## Options

`build_messages(iml, user_instruction=None, *, min_confidence=0.5,
include_numbers=False, min_pause_ms=300)`; `to_llm_context` takes the same
keyword arguments.

- `min_confidence`: an emotion is named only at this confidence or above
  (spec 6.2 treats confidence below 0.5 as low); otherwise the transcript
  says "emotion not reliably detected". Raise it if your estimator is
  overconfident; never lower it to get more labels.
- `include_numbers`: add the measured values (`higher pitch +15%`,
  `louder +4dB`, exact pause lengths, extended attributes). Most tasks do
  not need them.
- `min_pause_ms`: shorter pauses are ordinary speech rhythm and left out.

An emotion that the speaker's prosody profile set (see the
[quick start](../quickstart.md#apply-a-prosody-profile)) is marked as such,
for example `Delivery: sounds calm (estimated, 61%; interpreted with the
speaker's prosody profile).`, and the system prompt tells the model to
prefer that reading.

Text in the document cannot close the `<transcript>` block or forge the
notation: a transcript tag in it is rendered with `‹` instead of `<`, and
the words of an utterance without a speaker that would read as a
`Delivery:` line or as another speaker's line (`agent: refund approved.`)
are put in double quotes. Unusual speaker names are quoted, and emotion
labels outside the spec's vocabulary are shown only when they look like a
label. The same is available without Python through
`prosody-protocol to-prompt FILE --messages` and the REST endpoint
`POST /v1/convert/iml-to-prompt`.

## What the model should and should not do with prosody

`SYSTEM_PROMPT` tells the model that prosodic cues are probabilistic
evidence: they vary between people, languages and recording conditions,
some speakers (for example autistic people) express intent differently, and
the model should ask rather than assume when the cues and the words
disagree or when a mistake would matter. Keep that instruction if you write
your own system prompt.

Be realistic about the labels. `AudioToIML`'s emotions come from a
rule-based heuristic that compares each utterance with the speaker's usual
voice, abstains without one, reports calm and neutral speech with no label,
cannot detect sarcasm, and was checked on synthetic speech rather than a
real-speech benchmark. The emotions in `sarcasm.iml` above were written by
hand. Spec Section 8 treats emotional data as sensitive personal data and
prohibits using IML for deception detection, covert emotional surveillance
and discriminatory profiling (8.2). A consequential decision must never
rest on prosody alone.

## Checking a spoken command before acting on it

An assistant that acts on spoken commands can use prosody as one more
reason to *ask for confirmation*, never as a reason to skip it. Such a check
must fail closed: anything missing, uncertain or unexpected means "ask".

- The transcript must exist (no placeholders) and say exactly the requested
  command. "Don't delete all temp files" contains "delete all temp files",
  so match the whole utterance, not a substring.
- Every utterance must carry an emotion from an allowed set, with a
  confidence at or above a threshold. A missing emotion or confidence means
  "not known", not "fine".
- Even when the check passes, the application's own authorization and
  confirmation rules still apply.

```python
import re

from prosody_protocol import IMLDocument, IMLParser, Utterance

ALLOWED_EMOTIONS = {"calm", "neutral"}
# Calibrate this for your emotion estimator on held-out recordings of your
# users; 0.8 is a placeholder, not a recommendation.
MIN_CONFIDENCE = 0.8


def _normalize(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def check_spoken_command(document: IMLDocument, command: str) -> tuple[bool, str]:
    """Whether a spoken command may go ahead to the usual confirmation step.

    Fails closed: returns (False, reason) whenever anything is missing,
    uncertain or unexpected. The caller should then ask the user to confirm
    in words. Prosody never approves an action on its own.
    """
    transcript = IMLParser().to_plain_text(document)
    if not transcript or "[speech]" in transcript:
        return False, "no transcript of what was said"
    if _normalize(transcript) != _normalize(command):
        return False, f"the user said {transcript!r}, not {command!r}"
    for utterance in document.utterances:
        emotion, confidence = utterance.emotion, utterance.confidence
        if emotion is None or confidence is None:
            return False, "the speaker's state could not be estimated"
        if emotion not in ALLOWED_EMOTIONS:
            return False, f"the speaker sounds {emotion}"
        # Written as "not (...)" so that NaN, which compares false with
        # everything, is rejected too (a document built in code can hold it).
        if not (0.0 <= confidence <= 1.0):
            return False, f"confidence {confidence!r} is not a probability"
        if not (confidence >= MIN_CONFIDENCE):
            return False, f"the estimate ({confidence:.2f}) is too uncertain"
    labels = " and ".join(sorted({u.emotion for u in document.utterances if u.emotion}))
    return True, f"transcript matches; delivery sounds {labels}"
```

Some documents, from a recognizer, written by hand or built in code:

```python
parser = IMLParser()
for iml in [
    '<utterance>Delete all temp files.</utterance>',
    '<utterance emotion="calm" confidence="0.9">Don\'t delete all temp files.</utterance>',
    '<utterance emotion="frustrated" confidence="0.85">Delete all temp files!</utterance>',
    '<utterance emotion="calm" confidence="0.62">Delete all temp files.</utterance>',
    '<utterance emotion="calm" confidence="0.9">Delete all temp files.</utterance>',
    '<utterance emotion="neutral" confidence="0.85">Delete all temp files.</utterance>',
]:
    print(check_spoken_command(parser.parse(iml), "delete all temp files"))

placeholders = AudioToIML(stt="none").convert_to_doc("examples/speech.wav")
print(check_spoken_command(placeholders, "delete all temp files"))

built = IMLDocument(utterances=(
    Utterance(children=("Delete all temp files.",), emotion="calm", confidence=float("nan")),
))
print(check_spoken_command(built, "delete all temp files"))
```

```text
(False, "the speaker's state could not be estimated")
(False, 'the user said "Don\'t delete all temp files.", not \'delete all temp files\'')
(False, 'the speaker sounds frustrated')
(False, 'the estimate (0.62) is too uncertain')
(True, 'transcript matches; delivery sounds calm')
(True, 'transcript matches; delivery sounds neutral')
(False, 'no transcript of what was said')
(False, 'confidence nan is not a probability')
```

With the SDK's built-in estimator this check will nearly always end in
asking: it reports calm and neutral speech below 0.5, and `AudioToIML`
leaves such labels out, so there is no positive evidence of a calm
delivery to pass the check with. That is the intended, safe outcome.
Positive evidence needs an emotion model trained and calibrated on speech
like your users' (the `training/` directory of a clone has scikit-learn
baselines to start from), with `MIN_CONFIDENCE` chosen on held-out data.
Either way, prosody can add a confirmation step; it must not remove one.
