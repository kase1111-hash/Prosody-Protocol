# Command-line reference

Installing the package (see the [quick start](quickstart.md#install))
installs one command, `prosody-protocol`, with a subcommand per task:

| Command | Does | Needs |
|---------|------|-------|
| [`validate`](#validate) | check IML documents against the spec | core |
| [`to-text`](#to-text) | print the plain text of an IML document | core |
| [`to-ssml`](#to-ssml) | convert IML to SSML 1.1 | core |
| [`to-prompt`](#to-prompt) | describe IML as an annotated transcript for an LLM | core |
| [`from-text`](#from-text) | predict IML markup for plain text | core |
| [`from-audio`](#from-audio) | annotate a recording | `audio` extra |
| [`synthesize`](#synthesize) | speak IML to a WAV file | `audio` extra; espeak-ng for speech |
| [`benchmark`](#benchmark) | score `AudioToIML` against a labeled dataset | `audio` extra |
| [`serve`](#serve) | run the REST API | `api` extra |
| [`doctor`](#doctor) | show which optional capabilities are installed | core |

`prosody-protocol COMMAND --help` prints each command's options, and
`prosody-protocol --version` the version. Where the script is not on
`PATH`, `python -m prosody_protocol` runs the same command line. The
examples on this page run from the root of a clone, with the files in
[`examples/`](../examples/).

## Common behavior

**Input.** An argument of `-` reads standard input: any FILE, TEXT or AUDIO
argument, and `--words`, `--transcript-file`, `--profile` and
`--calibration`. Only one input of a command can come from stdin. Text
input must be UTF-8 (a byte order mark is allowed).

```bash
echo '<utterance>Hi.</utterance>' | prosody-protocol validate -
cat examples/speech.whisper.json | prosody-protocol from-audio examples/speech.wav --words - --prompt
```

**Output.** Results go to standard output, or to the file named by
`-o/--output` (UTF-8, with a trailing newline). Warnings go to standard
error as `warning: ...` as they happen, each distinct message once; Python's
warning filters (`-W`, `PYTHONWARNINGS`) still apply.

**Exit status.**

| Status | Meaning |
|--------|---------|
| 0 | success |
| 1 | the input was read but rejected: invalid IML, a failed conversion or analysis, a benchmark regression or failed conversions, a server that could not start |
| 2 | usage error; a file or directory that cannot be read or written (including a missing AUDIO or DATASET_DIR); a missing optional dependency; invalid `PP_*` settings for `serve` |
| 130 | interrupted (Ctrl-C) |

Expected errors print one `error: ...` line, never a traceback. A missing
extra names what to install:

```text
$ prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json
error: from-audio needs the 'audio' extra (numpy is not installed): pip install "prosody-protocol[audio] @ git+https://github.com/kase1111-hash/Prosody-Protocol"
```

In a clone, `pip install -e ".[audio]"` does the same.

## validate

```text
prosody-protocol validate [--json] [--strict] FILE [FILE ...]
```

Checks each file against the IML rules (V1-V33, listed in the
[API reference](API.md#validation-rules)). Each issue is printed as
`FILE:LINE:COL: SEVERITY RULE MESSAGE` (the column is known only for XML
syntax and encoding errors), then one summary line per file.

| Option | Effect |
|--------|--------|
| `--json` | print a JSON list of `{file, valid, errors, warnings, issues: [{severity, rule, message, line, column}]}` instead |
| `--strict` | also fail on warnings (the spec's SHOULD rules) |

Exit status: 0 when every file passes, 1 when any is invalid (or, with
`--strict`, has warnings), 2 when a file cannot be read (the others are
still checked).

```bash
prosody-protocol validate examples/sarcasm.iml
```

```text
examples/sarcasm.iml: valid
```

```bash
echo '<utterance emotion="angry">Give it back!</utterance>' | prosody-protocol validate -
echo '<utterance><prosody volume="+80dB">Hey!</prosody></utterance>' | prosody-protocol validate --strict -
```

```text
<stdin>:1: error V3 <utterance> has emotion="angry" but no confidence attribute
<stdin>: invalid (1 error)

<stdin>:1: warning V33 volume="+80dB" is more than 40 dB from the baseline; such a value is almost always a measurement or conversion error (spec 6.4)
<stdin>: fails --strict (1 warning)
```

Both exit with status 1. Without `--strict`, the second document is
`valid (1 warning)` and exits 0. Info notes, such as V15 for an emotion
outside the core vocabulary, never affect the result. With `--json`:

```bash
echo '<utterance emotion="angry">Give it back!</utterance>' | prosody-protocol validate --json -
```

```json
[
  {
    "file": "<stdin>",
    "valid": false,
    "errors": 1,
    "warnings": 0,
    "issues": [
      {
        "severity": "error",
        "rule": "V3",
        "message": "<utterance> has emotion=\"angry\" but no confidence attribute",
        "line": 1,
        "column": null
      }
    ]
  }
]
```

## to-text

```text
prosody-protocol to-text FILE [-o OUT]
```

Prints the words of the document without markup, with whitespace
normalized and utterances joined by a space.

```bash
prosody-protocol to-text examples/sarcasm.iml
```

```text
I've been on hold for forty minutes. Oh, that's just wonderful. Really great service.
```

## to-ssml

```text
prosody-protocol to-ssml FILE [--vendor espeak-ng] [--lenient] [-o OUT]
```

Converts IML to SSML 1.1 (`IMLToSSML`). Pitch, volume, rate, pitch contours,
emphasis and pauses are mapped; emotion, voice quality and the extended
measurements are not.

| Option | Effect |
|--------|--------|
| `--vendor espeak-ng` | SSML adapted to what espeak-ng renders (rescaled pitch, volume as percentages, contours as pitch steps); not portable |
| `--lenient` | convert a document with validation errors, ignoring invalid values (spec 6.2); without it such a document is an error (exit 1) |

```bash
prosody-protocol to-ssml examples/sarcasm.iml
```

```xml
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><s>
    I've been on hold for <prosody pitch="+8%" volume="+4dB">forty minutes</prosody>.
  </s><s>
    Oh, that's just
    <prosody pitch="+15%" rate="80%"><prosody contour="(0%,+0%) (50%,-15%) (100%,+15%)">wonderful</prosody></prosody>.
    <break time="600ms"/>
    Really <emphasis level="strong">great</emphasis> service.
  </s></speak>
```

```bash
prosody-protocol to-ssml examples/sarcasm.iml --vendor espeak-ng -o sarcasm.ssml
```

## to-prompt

```text
prosody-protocol to-prompt FILE [--min-confidence F] [--numbers] [--messages [--instruction TEXT]] [-o OUT]
```

Describes the document as an annotated transcript for a large language
model (`to_llm_context`): `*stressed*` and `**strongly stressed**` words,
prosody in words in parentheses, `[pause 0.6s]`, and a `Delivery:` line per
utterance.

| Option | Effect |
|--------|--------|
| `--min-confidence F` | name an emotion only at this confidence or above (default 0.5); lower ones read "emotion not reliably detected" |
| `--numbers` | add the measured values (pitch %, dB, rate %, extended attributes) |
| `--messages` | print chat messages as JSON: the system prompt that explains the notation, and the transcript in `<transcript>` tags (`build_messages`) |
| `--instruction TEXT` | with `--messages`: a task for the model, after the transcript (without `--messages` it is a usage error) |

```bash
prosody-protocol to-prompt examples/sarcasm.iml
prosody-protocol to-prompt examples/sarcasm.iml --numbers --min-confidence 0.8
```

```text
customer: I've been on hold for {forty minutes} (slightly higher pitch, louder).
Delivery: sounds frustrated (estimated, 82%).
customer: Oh, that's just wonderful (higher pitch, slower, falling then rising). [pause 0.6s] Really **great** service.
Delivery: sounds sarcastic (estimated, 74%).

customer: I've been on hold for {forty minutes} (slightly higher pitch +8%, louder +4dB).
Delivery: sounds frustrated (estimated, 82%).
customer: Oh, that's just wonderful (higher pitch +15%, slower at 80% pace, falling then rising). [pause 0.6s] Really **great** service.
Delivery: emotion not reliably detected (confidence 0.74).
```

```bash
prosody-protocol to-prompt examples/sarcasm.iml --messages --instruction "Reply to the customer."
```

prints a list of two messages, `{"role": "system", "content": ...}` (the
system prompt that explains the notation) and:

```json
{
  "role": "user",
  "content": "<transcript>\ncustomer: I've been on hold for {forty minutes} (slightly higher pitch, louder).\nDelivery: sounds frustrated (estimated, 82%).\ncustomer: Oh, that's just wonderful (higher pitch, slower, falling then rising). [pause 0.6s] Really **great** service.\nDelivery: sounds sarcastic (estimated, 74%).\n</transcript>\n\nReply to the customer."
}
```

## from-text

```text
prosody-protocol from-text TEXT [--context TEXT] [-o OUT]
```

Predicts IML for plain text (`TextToIML`, rule-based: punctuation, capitals
and cue words). `--context` is surrounding text, such as the previous turn;
the emotion its cue words point to counts as one more cue in every
sentence. `TEXT` of `-` reads stdin.

```bash
prosody-protocol from-text "Oh great, another meeting."
prosody-protocol from-text "The app crashed again." --context "the user is frustrated"
echo "Wait, you did WHAT?!" | prosody-protocol from-text -
```

```text
<iml version="0.1.0"><utterance emotion="sarcastic" confidence="0.6"><prosody pitch_contour="fall-rise">Oh great, another meeting.</prosody></utterance></iml>
<iml version="0.1.0"><utterance emotion="frustrated" confidence="0.5">The app crashed again.</utterance></iml>
<iml version="0.1.0"><utterance emotion="surprised" confidence="0.6"><prosody pitch_contour="rise-sharp">Wait, you did <emphasis level="strong">WHAT</emphasis>?!</prosody></utterance></iml>
```

## from-audio

```text
prosody-protocol from-audio AUDIO [--words FILE | --transcript TEXT | --transcript-file FILE]
    [--language TAG] [--profile FILE] [--calibration AUDIO]
    [--stt {auto,whisper,none}] [--whisper-model NAME]
    [--extended] [--min-confidence F] [--json | --prompt] [-o OUT]
```

Annotates a recording (`AudioToIML`). WAV, AIFF, FLAC and MP3 are always
read; OGG/Opus, WebM, M4A and other formats need ffmpeg.

Where the words come from, in order of preference:

| Option | Effect |
|--------|--------|
| `--words FILE` | word timings JSON from any recognizer: openai-whisper (and faster-whisper, WhisperX), the OpenAI transcription API's `verbose_json`, Deepgram, AssemblyAI, Google Cloud Speech-to-Text, or a list of `{word, start_ms, end_ms}` records (optionally with `speaker`). The format is detected. Speaker labels from diarization are kept: a new utterance starts where the speaker changes, with its `speaker_id`, and each speaker is measured against their own voice. One speaker's words may overlap by at most 500 ms. |
| `--transcript TEXT`, `--transcript-file FILE` | the text without timings: one utterance, no pauses or word-level markup, and overall delivery only with `--calibration`; without it, little beyond the text |
| `--stt auto` (default) | with neither: Whisper transcribes the audio if the `whisper` extra is installed; otherwise each stretch of speech is a `[speech]` placeholder, with a warning |
| `--stt whisper` | require Whisper when neither words nor a transcript are given (without it: exit 2 and an install hint) |
| `--stt none` | never transcribe: placeholders |
| `--whisper-model NAME` | Whisper model size (default `base`) |

Other options:

| Option | Effect |
|--------|--------|
| `--language TAG` | BCP 47 language tag, e.g. `en-US` (`en_US` is accepted as `en-US`). Labels the output and is passed to Whisper. |
| `--profile FILE` | the speaker's prosody profile (JSON, spec Section 7). Notes on standard error say which utterances it set. |
| `--calibration AUDIO` | a recording of the same speaker talking as usual (for example an earlier turn), used as the baseline for pitch, loudness, rate, emotion and the profile. Repeat it for several recordings (such as several earlier turns); each is analyzed once. Not used when the words carry several speaker labels. |
| `--extended` | add the measurements (`f0_mean`, `jitter`, ...) to the words; the markup and the text are otherwise the same |
| `--min-confidence F` | leave out emotions below this confidence (default 0.5) |
| `--json` | print `{iml, plain_text, transcript_source, warnings, profile_matches}` |
| `--prompt` | print the annotated transcript for an LLM (as `to-prompt`) |

```bash
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json --language en-US
```

```xml
<iml version="0.1.0" language="en-US"><utterance>I never said<pause duration="610"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
```

and, on standard error:

```text
warning: No speaker baseline: a single utterance without calibration_audio has nothing to compare its pitch, loudness and rate with, so they were not marked, and no emotion was estimated from them. Pass calibration_audio (recordings of the speaker's usual speech, such as their earlier turns) to assess them.
```

Warnings say what degraded the output or could not be assessed: no speaker
baseline, `[speech]` placeholders, a transcript without timings, words
without sentence punctuation, word timings that do not match the audio, two
voices without speaker labels, and more. The
[API reference](API.md#conversion-warnings) lists them.

```bash
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json --prompt
```

```text
I never said [pause 0.6s] she **stole** (much higher pitch, falling) my money.
```

```bash
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json --json
```

```json
{
  "iml": "<iml version=\"0.1.0\"><utterance>I never said<pause duration=\"610\"/> she <emphasis level=\"strong\"><prosody pitch=\"+47%\" pitch_contour=\"fall\">stole</prosody></emphasis> my <prosody pitch_contour=\"fall\">money.</prosody></utterance></iml>",
  "plain_text": "I never said she stole my money.",
  "transcript_source": "words",
  "warnings": [
    "No speaker baseline: a single utterance without calibration_audio has nothing to compare its pitch, loudness and rate with, so they were not marked, and no emotion was estimated from them. Pass calibration_audio (recordings of the speaker's usual speech, such as their earlier turns) to assess them."
  ],
  "profile_matches": []
}
```

The utterance has no emotion and no overall pitch or loudness: one sentence
without `--calibration` gives the rule-based estimator no usual voice to
compare with, so it abstains, as the warning says. `--min-confidence 0`
shows what it would otherwise say:

```bash
prosody-protocol from-audio examples/speech.wav --words examples/speech.whisper.json --min-confidence 0
```

```xml
<iml version="0.1.0"><utterance emotion="neutral" confidence="0.0">I never said<pause duration="610"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
```

Without timings, or without any transcript:

```bash
prosody-protocol from-audio examples/speech.wav --transcript-file examples/speech.txt
prosody-protocol from-audio examples/speech.wav --stt none
```

```text
warning: The transcript has no word timings, so only the utterance as a whole was measured: pauses, emphasis and word-level pitch or loudness cannot be placed, and a transcript of several sentences is one utterance. Its overall pitch, loudness, rate and emotion are assessed only against a speaker baseline, which a single utterance gets from calibration_audio. For word-level markup, pass word timings (words=), such as a speech recognizer's word timestamps.
warning: No speaker baseline: a single utterance without calibration_audio has nothing to compare its pitch, loudness and rate with, so they were not marked, and no emotion was estimated from them. Pass calibration_audio (recordings of the speaker's usual speech, such as their earlier turns) to assess them.
<iml version="0.1.0"><utterance>I never said she stole my money.</utterance></iml>
warning: No transcript (stt='none'): each stretch of speech is a '[speech]' placeholder. For real words, give word timings (words) or a transcript, or install 'prosody-protocol[whisper]'.
warning: No speaker baseline: without calibration_audio, the speaker's usual pitch and loudness come from the recording, which needs at least 3 utterances, most of them at a similar level; this one has 2. Pitch, loudness and rate were marked only relative to one another, and no emotion was estimated from them. Pass calibration_audio (recordings of the speaker's usual speech, such as their earlier turns) to assess them.
<iml version="0.1.0"><utterance><prosody pitch_contour="rise-fall">[speech]</prosody></utterance> <utterance><pause duration="610"/><prosody pitch_contour="rise-fall">[speech]</prosody></utterance></iml>
```

A transcript without timings adds little without `--calibration`: nothing
can be placed on the words, and the sentence as a whole has nothing to be
compared with.

With a prosody profile (see the
[quick start](quickstart.md#apply-a-prosody-profile)):

```bash
prosody-protocol from-audio examples/monotone.wav --words examples/monotone.deepgram.json --profile examples/profile.json
```

```text
note: prosody profile 'example_user' set utterance 1 to 'calm' (confidence 0.61; matched pitch_contour=flat)
note: prosody profile 'example_user' set utterance 2 to 'calm' (confidence 0.62; matched pitch_contour=flat)
note: prosody profile 'example_user' set utterance 3 to 'calm' (confidence 0.61; matched pitch_contour=flat)
note: prosody profile 'example_user' set utterance 4 to 'joyful' (confidence 0.61; matched pitch_contour=flat, rate=fast)
<iml version="0.1.0"><utterance emotion="calm" confidence="0.61" x-profile="pitch_contour=flat">I read the list.</utterance> <utterance emotion="calm" confidence="0.62" x-profile="pitch_contour=flat"><pause duration="310"/>The room is booked.</utterance> <utterance emotion="calm" confidence="0.61" x-profile="pitch_contour=flat"><pause duration="300"/>I have the slides.</utterance> <utterance emotion="joyful" confidence="0.61" x-profile="pitch_contour=flat rate=fast"><pause duration="290"/><prosody rate="165%">And we got the grant!</prosody></utterance></iml>
```

The `note:` lines go to standard error (they are left out with `--json`,
whose `profile_matches` holds the same information). A match whose
confidence stays below `--min-confidence` is reported as matched but not
applied, and a profile that matches nothing says so.

Errors: an unreadable audio file or bad input is reported on one line:

```text
$ prosody-protocol from-audio nope.wav --transcript hi
error: cannot read nope.wav: No such file or directory
$ echo '[{"word":"hi","start_ms":0,"end_ms":4000},{"word":"there","start_ms":100,"end_ms":4100}]' | prosody-protocol from-audio examples/speech.wav --words -
error: words[1] ('there', 100-4100 ms) starts 3900 ms before words[0] ('hi', 0-4000 ms) ends; word timings may overlap by at most 500 ms (the words of one speaker, one after another)
$ prosody-protocol from-audio examples/speech.wav --stt whisper
error: from-audio --stt whisper needs the 'whisper' extra (openai-whisper is not installed): pip install "prosody-protocol[whisper] @ git+https://github.com/kase1111-hash/Prosody-Protocol", or give the words with --words or --transcript
```

with exit status 2, 1 and 2.

## synthesize

```text
prosody-protocol synthesize FILE -o OUT.wav [--engine {auto,espeak,tones}] [--voice V] [--lenient]
```

Speaks an IML document to a WAV file (mono, 16-bit, 22050 Hz) with
`IMLToAudio`, and says on standard error what it wrote.

| Option | Effect |
|--------|--------|
| `-o OUT.wav` | required; `-o -` writes the WAV to standard output |
| `--engine auto` (default) | espeak-ng if it is on PATH, otherwise the tone preview with a warning |
| `--engine espeak` | speech from espeak-ng (an error if it is missing) |
| `--engine tones` | a prosody preview, one tone per word; not speech |
| `--voice V` | a language tag (`en-US`), optionally with gender and pitch level (`en_US-female-medium`, `fr-male-low`), or an espeak-ng voice with a variant (`en-us+f3`). Default: a voice for the document's language. |
| `--lenient` | speak a document with validation errors, ignoring invalid values (spec 6.2) |

```bash
prosody-protocol synthesize examples/sarcasm.iml -o sarcasm.wav
prosody-protocol synthesize examples/sarcasm.iml -o preview.wav --engine tones
prosody-protocol synthesize examples/sarcasm.iml -o male.wav --voice en-us+m3
```

```text
wrote sarcasm.wav: speech (espeak-ng)
wrote preview.wav: tone preview, not speech
wrote male.wav: speech (espeak-ng)
```

espeak-ng speech is intelligible but robotic. Without espeak-ng:

```text
warning: espeak-ng was not found on PATH, so IMLToAudio renders a tone preview of the prosody (one tone per word), not speech. Install espeak-ng for speech, or pass engine='tones' to choose the preview explicitly.
wrote sarcasm.wav: tone preview, not speech
```

To play it at once, write to standard output:
`prosody-protocol synthesize examples/sarcasm.iml -o - | aplay`.

## benchmark

```text
prosody-protocol benchmark DATASET_DIR [--save REPORT.json] [--baseline REPORT.json]
    [--tolerance F] [--threshold METRIC=VALUE]... [--max-samples N]
    [--calibration AUDIO] [--words-from {auto,timings,transcript,stt}]
    [--stt {auto,whisper,none}]
    [--abstention-label LABEL] [--language TAG]
```

Loads a dataset with `DatasetLoader` (see the
[quick start](quickstart.md#datasets-and-benchmarks) for the format), runs
`AudioToIML` on every recording and scores the output against the labels
(`Benchmark`). What the converter gets besides the audio is set by
`--words-from`:

| `--words-from` | The converter gets |
|----------------|--------------------|
| `auto` (default) | the entry's word timings (`metadata.word_timings`, any format `--words` reads) when it has them; otherwise nothing when Whisper can find the words (`--stt` is not `none` and the `whisper` extra is installed), since recognized words carry timings; otherwise the entry's transcript |
| `timings` | the word timings when the entry has them, otherwise nothing (speech recognition, see `--stt`) |
| `transcript` | always the transcript |
| `stt` | nothing: Whisper, or `[speech]` placeholders without it |

Pauses and pitch contours can only be scored where the converter has word
timings, given or from Whisper; a bare transcript places no pauses, and
outputs that are only placeholders are left out of those metrics.

| Option | Effect |
|--------|--------|
| `--save REPORT.json` | save the report (its directory must exist) |
| `--baseline REPORT.json` | fail when a metric, or a per-class F1, is worse than this saved report by more than the tolerance, or was measured there but not in this run |
| `--tolerance F` | allowed drop against the baseline (default 0.01) |
| `--threshold METRIC=VALUE` | a limit, repeatable: minimums for `emotion_accuracy`, `emotion_coverage`, `emotion_f1_macro`, `pitch_accuracy`, `pitch_coverage`, `pause_f1`, `validity_rate`; maximums for `confidence_ece`, `failure_rate` (default 0, so any failed conversion fails). An `emotion_accuracy` limit without an `emotion_coverage` limit counts entries without an emotion as wrong (likewise `pitch_accuracy` and `pitch_coverage`) |
| `--max-samples N` | evaluate only the first N entries |
| `--calibration AUDIO` | a recording of the speakers talking as usual, used as the baseline for every entry (repeat for several). Without one, an entry that is a single utterance gets no emotion; tests/fixtures/benchmarks/training_synthetic.json was made with `--stt none --calibration tests/fixtures/datasets/training_synthetic/audio/synth_001.wav` |
| `--abstention-label LABEL` | score an output without an emotion as LABEL (such as `neutral`); by default it is an abstention, which lowers `emotion_coverage` and counts as a miss in the per-class F1, but is left out of `emotion_accuracy` |
| `--stt`, `--language` | as for `from-audio` |

It prints a summary (where the words came from, then the metrics), then
`Passed` or `FAILED:` with the reasons, and exits 1 on any failure. A
baseline that scored abstentions differently (reports saved before
`--abstention-label` existed counted them as `neutral`) is itself a
failure, and its emotion metrics are not compared. Unknown metrics, a
missing dataset directory and an unwritable `--save` path are rejected
(exit 2) before the run. With the small test dataset of a clone, whose
entries have transcripts but no word timings:

```text
$ prosody-protocol benchmark tests/fixtures/datasets/sample --stt none --save report.json
warning: The transcript has no word timings, so only the utterance as a whole was measured: pauses, emphasis and word-level pitch or loudness cannot be placed, and a transcript of several sentences is one utterance. Its overall pitch, loudness, rate and emotion are assessed only against a speaker baseline, which a single utterance gets from calibration_audio. For word-level markup, pass word timings (words=), such as a speech recognizer's word timestamps.
warning: No speaker baseline: a single utterance without calibration_audio has nothing to compare its pitch, loudness and rate with, so they were not marked, and no emotion was estimated from them. Pass calibration_audio (recordings of the speaker's usual speech, such as their earlier turns) to assess them.
Benchmark of sample: 3 entries, 0 failed conversions, 0.1 s
  words from        transcripts 3
  emotion_accuracy  n/a
  emotion_coverage  0.0000
  emotion_f1_macro  0.0000
  confidence_ece    n/a
  pitch_accuracy    n/a
  pitch_coverage    n/a
  pause_f1          n/a
  validity_rate     1.0000
  failure_rate      0.0000
  emotion_f1        joyful 0.00, neutral 0.00, uncertain 0.00
Saved the report to report.json
Passed
$ prosody-protocol benchmark tests/fixtures/datasets/sample --stt none --baseline report.json --threshold emotion_accuracy=0.7
...
FAILED:
  emotion_accuracy = 0.0000 < threshold 0.7000 (over all entries: those without an emotion count as wrong; add an emotion_coverage threshold to score only the entries with one)
$ prosody-protocol benchmark tests/fixtures/datasets/sample --stt none --abstention-label neutral
...
  words from        transcripts 3
  emotion_accuracy  0.3333
  emotion_coverage  1.0000
  emotion_f1_macro  0.1667
...
  emotion_f1        joyful 0.00, neutral 0.50, uncertain 0.00
  (an output without an emotion counts as 'neutral')
Passed
```

(`...` marks lines left out.) `n/a` marks a metric with nothing to compare
(here the dataset has no pitch contours or pauses, and no output carries
an emotion). Every output abstains: each entry is a single sentence
without calibration audio. Three entries measure nothing; the fixture only
shows that the pipeline runs. The repository's own regression check,
`tests/fixtures/benchmarks/training_synthetic.json`, is a saved report of
this kind for the 10-clip `training_synthetic` fixture, made with a
calibration recording (`tests/fixtures/benchmarks/make_baselines.py`).

## serve

```text
prosody-protocol serve [--host HOST] [--port PORT]
```

Runs the REST API with uvicorn (the `api` extra). The host defaults to
`$PP_HOST`, or 127.0.0.1 when `PP_HOST` is unset or empty, and the port to
`$PP_PORT` or 8000; `--port 0` picks a free port. The other settings (upload and JSON body size, rate limit, worker
processes, Whisper model, ...) are `PP_*` environment variables, listed in the
[API reference](API.md#configuration).

```bash
prosody-protocol serve
PP_RATE_LIMIT=0 PP_MAX_CONCURRENT_JOBS=4 prosody-protocol serve --host 0.0.0.0 --port 8080
```

```text
INFO:     Started server process [20190]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Binding to 0.0.0.0 exposes the server to your network; it has no
authentication, so put it behind a reverse proxy that has, and set
`PP_TRUSTED_PROXIES` so rate limiting sees the real clients. Invalid
settings exit with status 2 before the server starts:

```text
$ PP_MAX_QUEUED_JOBS=-1 prosody-protocol serve
error: max_queued_jobs (PP_MAX_QUEUED_JOBS) must be an integer >= 0, got -1
```

A server that cannot start (for example, because the port is in use) prints
uvicorn's log line and `error: the server could not start (see the log
above)`, exit status 1. `python -m prosody_protocol.server` takes the same
options and reports errors the same way.

## doctor

```text
prosody-protocol doctor
```

Lists the optional capabilities: which are installed (with versions), what
each enables, and how to install the missing ones. Always exits 0.

```bash
prosody-protocol doctor
```

On a core-only install:

```text
prosody-protocol 0.1.0a3 (Python 3.11.15)
[ok     ] core (lxml 6.1.3)
          enables: validate, to-text, to-ssml, to-prompt, from-text
[missing] audio analysis (numpy, praat-parselmouth)
          enables: from-audio, benchmark, synthesize
          install: pip install "prosody-protocol[audio] @ git+https://github.com/kase1111-hash/Prosody-Protocol"
[missing] speech recognition (openai-whisper)
          enables: from-audio transcribes audio given without --words or --transcript (otherwise each stretch of speech is a [speech] placeholder)
          install: pip install "prosody-protocol[whisper] @ git+https://github.com/kase1111-hash/Prosody-Protocol" (pulls in PyTorch)
[ok     ] speech synthesis (espeak-ng at /usr/bin/espeak-ng)
          enables: synthesize speaks IML (otherwise it renders a tone preview, not speech)
[ok     ] audio decoding (ffmpeg at /usr/bin/ffmpeg)
          enables: from-audio reads OGG/Opus, WebM, M4A and other formats besides WAV, AIFF, FLAC and MP3
[missing] REST API (fastapi, uvicorn, python-multipart)
          enables: serve
          install: pip install "prosody-protocol[api] @ git+https://github.com/kase1111-hash/Prosody-Protocol"
[missing] training baselines (scikit-learn)
          enables: the training/ scripts of a source checkout
          install: pip install "prosody-protocol[ml] @ git+https://github.com/kase1111-hash/Prosody-Protocol"
```

The `install:` lines install from GitHub, since the package is not on PyPI
yet; in a clone, `pip install -e ".[audio]"` and so on do the same (see
[Install](quickstart.md#install)).
