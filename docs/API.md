# API Reference

The Python SDK (`prosody_protocol`) and the REST API. For a guided tour, see
the [quick start](quickstart.md); for the command line, the
[CLI reference](cli.md).

- [Imports and extras](#imports-and-extras)
- Core: [IMLParser](#imlparser), [IMLValidator](#imlvalidator) and the
  [validation rules](#validation-rules), [data models](#data-models)
- Audio: [AudioToIML](#audiotoiml), [word timings](#word-timings),
  [ProsodyAnalyzer](#prosodyanalyzer), [IMLAssembler](#imlassembler),
  [emotion classifiers](#emotion-classifiers)
- Output: [LLM context](#llm-context), [TextToIML](#texttoiml),
  [IMLToSSML](#imltossml), [IMLToAudio](#imltoaudio)
- Data: [profiles](#profiles), [datasets](#datasets),
  [benchmarks](#benchmarks), [MavisBridge](#mavisbridge)
- [Exceptions](#exceptions)
- [REST API](#rest-api): [running it](#running-the-server),
  [configuration](#configuration), [errors](#errors),
  [endpoints](#endpoints)

The Python examples on this page run in order, from the root of a clone.

## Imports and extras

The classes and functions below are importable from the package root,
except where a module is named (`prosody_protocol.alignment`,
`prosody_protocol.llm`, the module constants and the server):

```python
from prosody_protocol import (
    AudioToIML, IMLAssembler, IMLDocument, IMLParser, IMLToAudio, IMLToSSML,
    IMLValidator, ProsodyAnalyzer, RuleBasedEmotionClassifier, SpeakerBaseline,
    TextToIML, WordAlignment, build_messages, load_word_timings, to_llm_context,
)
```

The core needs only `lxml`. The names that need numpy and praat-parselmouth
(the `audio` extra) are loaded when first used, so `import prosody_protocol`
always works; without the extra, using one of them raises `ImportError`
naming the extra:

| Needs the `audio` extra | Everything else |
|-------------------------|-----------------|
| `AudioToIML`, `ConversionResult`, `ProsodyAnalyzer`, `IMLToAudio`, `Benchmark`, `BenchmarkReport`, `MavisBridge`, `PhonemeEvent` | lxml only |

`from prosody_protocol import *` works on any install: the names that need
an extra are in `__all__` only when it is installed. Built-in speech
recognition in `AudioToIML` also needs the `whisper` extra (openai-whisper,
which pulls in PyTorch). `prosody_protocol.__version__` holds the version.

## IMLParser

```text
IMLParser()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `parse(iml_string)` | `IMLDocument` | Parse an `<iml>` or bare `<utterance>` document |
| `parse_file(path)` | `IMLDocument` | Parse a UTF-8 file (a byte order mark is allowed) |
| `to_plain_text(doc)` | `str` | The words without markup: whitespace runs collapsed, no stray space before closing punctuation, utterances joined by one space |
| `to_iml_string(doc)` | `str` | Serialize a document; a single utterance without document attributes is written as a bare `<utterance>`, and utterances inside `<iml>` are separated by a space |

The parser is lenient about attribute values, which it keeps, and strict
about structure. It raises `IMLParseError` (with `.line` and `.column` when
known) for malformed XML, a DOCTYPE (spec 2.5), a file that is not UTF-8,
text or markup outside any `<utterance>`, a nested `<utterance>` or `<iml>`,
a `<pause>` with content, and nesting too deep to parse; `parse_file`
raises `OSError` for unreadable files. Comments and processing instructions
are dropped. Unknown elements are transparent: their tags are dropped and
their content kept (spec 6.2). Attributes without a typed field (`x-`
extensions, unknown or namespaced attributes, invalid numeric values) are
kept in `extra_attributes` and written back, so a round trip never turns an
invalid document into a valid one.

`to_iml_string` always writes well-formed XML without an XML declaration.
Numbers the parser read are written as they were (`confidence="0.80"` stays
`0.80`). A document built in code with a character XML forbids (such as
`"\x1b"`) in text or an attribute, or an invalid extra attribute name,
raises `ConversionError`; one with a typed number IML cannot hold (a
`confidence` of `nan` or `1.5`, `Pause(duration=-5)`) raises
`IMLValidationError` naming the rule, rather than writing invalid IML.

```python
parser = IMLParser()
doc = parser.parse('<utterance x-source="call-42" emotion="calm" confidence="0.8">Fine.</utterance>')
print(doc.utterances[0].extra_attributes)
print(parser.to_iml_string(doc))
```

```text
(('x-source', 'call-42'),)
<utterance emotion="calm" confidence="0.8" x-source="call-42">Fine.</utterance>
```

Module constants in `prosody_protocol.parser`: `IML_NAMESPACE`
(`"http://prosody-protocol.org/iml/0.1"`, for IML embedded in other XML,
spec 2.3) and `MAX_INTEGER` (2147483647, the largest IML integer, spec 2.6).

## IMLValidator

```text
IMLValidator()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `validate(iml_string)` | `ValidationResult` | Check a document against the rules below |
| `validate_file(path)` | `ValidationResult` | The same for a file; a file that is not UTF-8 is a V30 error, not an exception (`OSError` if unreadable) |

`ValidationResult` (a dataclass): `valid: bool` (no errors), `issues:
list[ValidationIssue]`, the properties `errors` and `warnings` (the issues
of that severity), and `raise_for_errors()`, which raises
`IMLValidationError` (its `.issues` holds the errors) when the document is
invalid.

`ValidationIssue` (a frozen dataclass): `severity` (`"error"`, `"warning"`
or `"info"`), `rule` (`"V3"`), `message`, `line` and `column` (`None` when
unknown; the column is known only for XML syntax and encoding errors).

```python
result = IMLValidator().validate('<utterance emotion="angry">Give it back!</utterance>')
print(result.valid, [(i.rule, i.severity, i.line) for i in result.errors])
```

```text
False [('V3', 'error', 1)]
```

### Validation rules

Errors make a document invalid, warnings (SHOULD rules) do not, and info
notes never matter for validity.

| Rule | Check | Severity |
|------|-------|----------|
| V1 | the document is well-formed XML | error |
| V2 | the root is `<iml>` or `<utterance>`, with at least one `<utterance>` | error |
| V3 | `confidence` is present when `emotion` is set | error |
| V4 | `confidence` is a float from 0.0 to 1.0 | error |
| V5 | `<pause>` has a `duration` | error |
| V6 | `<pause>` duration is a positive integer (at most 2147483647) | error |
| V7 | `<pause>` has no elements (unknown ones included) and no text other than whitespace | error |
| V8 | `<emphasis>` has a `level` | error |
| V9 | `<emphasis>` level is `strong`, `moderate` or `reduced` | error |
| V10 | `<segment>` is a direct child of `<utterance>` | error |
| V11 | `<segment>` is not nested in another `<segment>` | error |
| V12 | `<prosody>`/`<emphasis>` nesting is at most 2 deep | warning |
| V13 | `pitch` is `+15%`, `-3st` or `185Hz` style | error |
| V14 | `volume` is `+6dB` style | error |
| V15 | `emotion` is from the core vocabulary (spec 3.1) | info |
| V16 | no unknown elements | info |
| V17 | `consent` is `explicit`, `implicit` or `none` | error |
| V18 | `processing` is `local`, `remote` or `hybrid` | error |
| V19 | `<iml>` contains only `<utterance>` elements | error |
| V20 | no `<utterance>` or `<iml>` inside another element | error |
| V21 | no `<emphasis>` directly inside `<emphasis>` | error |
| V22 | `rate` is `fast`, `slow`, `medium` or a percentage | error |
| V23 | `pitch_contour` is from the spec vocabulary | error |
| V24 | `quality` is from the spec vocabulary | error |
| V25 | `tempo` is from the spec vocabulary | error |
| V26 | `rhythm` is from the spec vocabulary | error |
| V27 | extended attributes have their Section 4 type (finite floats; integers at most 2147483647) | error |
| V28 | `version` is a Semantic Version | error |
| V29 | `language` has the form of a BCP 47 tag (1-8 letters, then `-` subtags; the form only, so `english` passes) | error |
| V30 | the document is UTF-8 | error |
| V31 | no DOCTYPE declaration | error |
| V32 | attributes are IML-defined, `x-` prefixed or in a foreign namespace (not unprefixed unknown ones, not in the IML namespace) | warning |
| V33 | values are plausible for human speech (spec 6.4): pitch within 24 st (-75% to +300%) or 40-1200 Hz; volume within 40 dB; rate 25-400%; pauses at most 60000 ms; f0 values 40-1200 Hz; speech_rate at most 20 syllables/s; duration_ms at most 3600000 | warning |

The V33 limits are constants in `prosody_protocol.validator`
(`MAX_PITCH_SEMITONES`, `MAX_PITCH_PERCENT`, `MIN_PITCH_PERCENT`,
`MIN_F0_HZ`, `MAX_F0_HZ`, `MAX_VOLUME_DB`, `MIN_RATE_PERCENT`,
`MAX_RATE_PERCENT`, `MAX_PAUSE_MS`, `MAX_SPEECH_RATE`, `MAX_DURATION_MS`).

### Language tags

One rule decides what a language tag is, everywhere: V29, dataset rule D6
and `schemas/dataset-entry.schema.json`, `IMLToSSML`, `AudioToIML`,
`IMLAssembler`, `MavisBridge` and the CLI's `--language`. A tag is a primary
subtag of 1-8 letters followed by `-` and subtags of 1-8 letters or digits
(`en`, `en-US`, `zh-Hant-TW`). Only this form is checked, not the IANA
registry, so `english` passes. Language arguments of the SDK and the CLI
read a POSIX-style `en_US` as `en-US`; IML documents and dataset entries
are checked as written, so `language="en_US"` in a document is a V29 error.
The REST API's `language` field takes only the tag form (`en_US` is a 422).

## Data models

Frozen dataclasses in `prosody_protocol.models` (also exported from the
package root). Mixed content is a `children` tuple of strings and elements.
Every model has `extra_attributes: tuple[tuple[str, str], ...]` for the
attributes without a typed field (namespaced ones in Clark notation,
`{uri}name`).

| Class | Fields |
|-------|--------|
| `IMLDocument` | `utterances`, `version`, `language`, `consent`, `processing`, `extra_attributes` |
| `Utterance` | `children`, `emotion`, `confidence` (float), `speaker_id`, `extra_attributes` |
| `Prosody` | `children`, `pitch`, `pitch_contour`, `volume`, `rate`, `quality`; extended: `f0_mean`, `f0_range`, `f0_contour`, `intensity_mean`, `intensity_range`, `speech_rate`, `duration_ms`, `jitter`, `shimmer`, `hnr`; `extra_attributes` |
| `Pause` | `duration` (ms; 0 when missing or invalid), `extra_attributes` |
| `Emphasis` | `level` (`""` when missing), `children`, `extra_attributes` |
| `Segment` | `children`, `tempo`, `rhythm`, `extra_attributes` |

In a parsed document, numeric fields only hold valid values: an invalid one
is left at its default and its raw text kept in `extra_attributes`. Models
built in code are not checked when they are built; `to_iml_string` checks
them. To build IML in code, construct the models and serialize them:

```python
from prosody_protocol import Emphasis, Pause, Utterance

doc = IMLDocument(
    utterances=(
        Utterance(
            children=("I ", Emphasis(level="strong", children=("really",)),
                      " need this", Pause(duration=500), " done today."),
            emotion="frustrated", confidence=0.7,
        ),
    ),
    version="0.1.0", language="en-US",
)
iml = parser.to_iml_string(doc)
print(iml)
print(IMLValidator().validate(iml).valid)
```

```text
<iml version="0.1.0" language="en-US"><utterance emotion="frustrated" confidence="0.7">I <emphasis level="strong">really</emphasis> need this<pause duration="500"/> done today.</utterance></iml>
True
```

## AudioToIML

Annotates a recording: it measures each word's prosody, finds pauses,
assembles IML and (when it has a baseline to compare with) estimates each
utterance's emotion. Needs the `audio` extra.

```text
AudioToIML(
    stt_model="base",
    emotion_classifier=None,
    include_extended=False,
    language=None,
    *,
    min_emotion_confidence=0.5,
    calibration_audio=None,
    stt="auto",
    max_duration_s=None,
    profile=None,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stt_model` | `"base"` | Whisper model (`"tiny"`, `"base"`, `"small"`, ..., or a checkpoint path), loaded on first use and reused |
| `emotion_classifier` | `None` | an `EmotionClassifier`; default `RuleBasedEmotionClassifier` |
| `include_extended` | `False` | add each word's measurements (`f0_mean`, `f0_range`, `f0_contour`, `intensity_mean`, `intensity_range`, `speech_rate`, `duration_ms`, `jitter`, `shimmer`, `hnr`) in a `<prosody>` around it. The markup and text are otherwise unchanged; an emphasized word inside an utterance-level `<prosody>` gets none (nesting stays within two levels). |
| `language` | `None` | BCP 47 tag; labels the output and is passed to Whisper (as `en` for `en-US`). `None`: Whisper's detected language, or none. `en_US` is read as `en-US`; a value not in the form of a tag (`"en US"`, `"123"`) raises `ValueError`. Only the form is checked (see [language tags](#language-tags)), so `"english"` is accepted and written as is. |
| `min_emotion_confidence` | `0.5` | utterances estimated below it carry no `emotion` or `confidence` |
| `calibration_audio` | `None` | a recording, or a sequence of recordings, of the same speaker talking as usual (such as their earlier turns), recorded with the same setup: the baseline for pitch, loudness, rate, emotion and profiles. See [below](#speaker-baselines-and-calibration). |
| `stt` | `"auto"` | speech recognition when a conversion gets neither `words` nor `transcript`: `"whisper"` (required), `"none"` (placeholders), `"auto"` (Whisper if installed) |
| `max_duration_s` | `None` | reject longer audio (and calibration audio) before loading it; set it for audio from untrusted users |
| `profile` | `None` | the speaker's `ProsodyProfile` (see [profiles](#profiles)); an invalid one raises `ProfileError` |

The `profile` property returns the profile; `calibration_audio` is a
settable attribute (a `Path`, a tuple of them, or `None`). Methods, all
taking `(audio_path, *, words=None, transcript=None)`:

| Method | Returns |
|--------|---------|
| `convert(...)` | the IML string; warnings are issued as `UserWarning` |
| `convert_to_doc(...)` | the `IMLDocument`; warnings as `UserWarning` |
| `convert_detailed(...)` | a `ConversionResult` |

Where the words come from:

1. `words=`: word timings from any recognizer, an iterable of
   `WordAlignment` (see [word timings](#word-timings)). No recognizer runs.
   Where a word's `speaker` label (from diarization) changes, a new
   utterance starts; the label becomes its `speaker_id`, and each speaker
   is measured against their own baseline. One speaker's words may overlap
   by at most `MAX_WORD_OVERLAP_MS` (500 ms); different speakers' words may
   overlap further, but no more than `MAX_OVERLAPPING_SPEAKERS` (3)
   speakers at once. Timings past the end of the audio are clamped to it.
2. `transcript=`: the text without timings. One utterance over the voiced
   part of the audio, with no pauses, emphasis or word-level tags: nothing
   says where the words are, and no timings are made up. Its overall pitch,
   loudness, rate and emotion are judged only against `calibration_audio`;
   without it the result is usually the plain transcript, so a transcript
   alone adds little beyond the text.
3. Otherwise, per `stt`: Whisper transcribes the audio, or each stretch of
   speech becomes a `[speech]` placeholder (`PLACEHOLDER_TOKEN`), with the
   pauses between them, and a warning.

`ConversionResult` (frozen dataclass): `document` (`IMLDocument`), `iml`
(str), `transcript_source` (`"words"`, `"transcript"`, `"whisper"` or
`"none"`), `warnings` (tuple of str, see
[conversion warnings](#conversion-warnings)) and `profile_matches` (tuple
of `ProfileMatch`, see [IMLAssembler](#imlassembler)).

Audio: WAV, AIFF, FLAC and MP3 are read directly; with ffmpeg on PATH also
OGG/Opus, WebM, M4A/MP4, AAC, CAF, AMR, AU and W64 (playlists are never
followed). All audio is analyzed as 16 kHz mono (see
[ProsodyAnalyzer](#prosodyanalyzer)). Analysis takes roughly 1-2.5 s per
minute of audio (slower for very low voices), plus Whisper if it runs.

Raises: `AudioProcessingError` (unreadable, empty, shorter than 100 ms,
sampled below 4 kHz, NaN or out-of-range samples, longer than
`max_duration_s`, Whisper required but missing, no calibration file with
voiced speech); its subclass `SpeechRecognitionError` when the Whisper
model cannot be loaded or transcription fails; `ValueError` (both `words`
and `transcript`, invalid or overlapping timings, text or a speaker label
with characters XML forbids); `TypeError` (wrong types);
`ConversionError` if the assembled document is not valid IML (a bug:
invalid IML is never returned).

```python
words = load_word_timings("examples/speech.whisper.json")
result = AudioToIML(language="en-US").convert_detailed("examples/speech.wav", words=words)
print(result.iml)
print(result.transcript_source, result.warnings, result.profile_matches)
```

```text
<iml version="0.1.0" language="en-US"><utterance>I never said<pause duration="610"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
words ("No speaker baseline: a single utterance without calibration_audio has nothing to compare its pitch, loudness and rate with, so they were not marked, and no emotion was estimated from them. Pass calibration_audio (recordings of the speaker's usual speech, such as their earlier turns) to assess them.",) ()
```

Module constants in `prosody_protocol.audio_to_iml`: `MAX_WORD_OVERLAP_MS`
(500), `MAX_OVERLAPPING_SPEAKERS` (3) and `PLACEHOLDER_TOKEN`
(`"[speech]"`).

### Speaker baselines and calibration

Utterance-level pitch, loudness and rate, word offsets, the default
classifier's emotion and profile matching are all judged against the
speaker's baseline:

- with `calibration_audio`, those recordings;
- without it, the recording's own utterances, when there are at least three
  (`assembler.MIN_BASELINE_UTTERANCES`) and most of them lie near their
  median. Otherwise there is no baseline, and a warning says so: a single
  sentence gets no utterance-level delivery and no emotion, and two
  utterances are marked only relative to each other, without emotion.
  Silence and noise never get an emotion;
- with speaker labels, each speaker's own utterances. `calibration_audio`
  and the profile describe one speaker, so with several labels they are
  not used (a warning says so);
- without labels, utterances whose pitch falls into two groups more than
  7 semitones apart (`assembler.VOICE_SEPARATION_ST`), as from the two
  sides of a call, get no baseline: no utterance-level offsets, no emotion,
  no profile. Voices closer in pitch are not told apart.

`calibration_audio` takes one path or a sequence of paths, such as a user's
earlier turns. Each file is analyzed once and cached while it is listed and
unchanged on disk (path, modification time and size), so a converter reused
across turns can be given the growing list and analyzes only the new
files. A file without voiced speech is skipped with a warning; if no file
has any, `AudioProcessingError` is raised. An empty sequence means no
calibration. The [quick start](quickstart.md#compare-with-the-speakers-earlier-turns)
has a worked example; the pattern:

```python
def convert_turns(turns):
    """Convert one user's turns, oldest first, each against up to five earlier ones.

    *turns* holds (audio path, word timings) pairs.
    """
    converter = AudioToIML(stt="none")
    earlier = []
    for audio, turn_words in turns:
        converter.calibration_audio = earlier[-5:]
        yield converter.convert_detailed(audio, words=turn_words)
        earlier.append(audio)
```

The labels are estimates from a rule-based heuristic that was checked on
synthetic espeak-ng speech, not on a real-speech benchmark.

### Conversion warnings

`ConversionResult.warnings` (and the `UserWarning`s of `convert()`, the
`warning:` lines of the CLI and the `warnings` of the REST response) say
what degraded the output or could not be assessed. With speaker labels, a
note about one speaker starts with `Speaker 'A': `.

| A warning that starts with | Means |
|----------------------------|-------|
| `No transcript (...)` | no words, transcript or Whisper: each stretch of speech is a `[speech]` placeholder |
| `The transcript has no word timings` | `transcript=`: only the utterance as a whole was measured, and only against a speaker baseline |
| `No speaker baseline:` | no `calibration_audio`, and a single utterance, or fewer than three (or most of them at different levels): overall pitch, loudness and rate were not marked, or only relative to one another, and the emotion classifier had no baseline (the wording names what that meant for it) |
| `The utterances' pitch falls into two groups` | two voices without speaker labels, or under one label: no baseline, emotion or profile |
| `calibration_audio describes one speaker`, `The prosody profile describes one speaker` | the words carry several speaker labels, so it was not used |
| `N of the M calibration_audio files contain no voiced speech` | those files were skipped |
| `The words have no sentence punctuation` | the words have pauses but no sentence punctuation, so utterances were split at pauses of 0.5 s or more instead of at sentence ends; ask the recognizer for punctuation |
| `N word(s) start after the end of the audio`, `N word(s) end after the end of the audio` | words start after the audio, or end more than 250 ms after it: measured up to its end |
| `N of M words lie where the audio has no voiced sound` | at least 10% of the words (of 100 ms or more) lie over unvoiced audio: the timings may belong to another recording, or be offset |
| `N s of the audio's M s of voiced speech lie outside the words` | more than half the voiced speech lies outside the words: expected for one speaker's words of several, otherwise mismatched timings |
| `No voiced speech was detected in the audio` | silence or noise: no emotion |
| `No words were supplied`, `The transcript is empty`, `Speech recognition found no words` | the document has no text |
| `The silence of ... s after ...`, `N silences longer than 60 s` | written as `<pause duration="60000"/>`: a silence over a minute was shortened (spec 6.4) |

The two checks on voiced sound run only on audio with at least 1 s of
voiced speech.

## Word timings

`WordAlignment(word, start_ms, end_ms, speaker=None)` (frozen dataclass): a
token as the recognizer wrote it, with its times in milliseconds, and the
speaker label a recognizer with diarization gave it (a string, such as
`"0"`, `"A"` or `"SPEAKER_00"`), or `None`. Its `repr` leaves out a `None`
speaker.

Two functions at the package root read any supported recognizer's output
and detect its format:

| Function | Takes | Description |
|----------|-------|-------------|
| `load_word_timings(source)` | a file path (`str` or path object), or parsed data (dict, list, SDK response object) | a `str` is always a **file path**. Files may be UTF-8 (with or without a BOM), UTF-16 or UTF-32. `OSError` if unreadable. |
| `parse_word_timings(data)` | JSON text (`str`, `bytes`), or parsed data | never touches the file system: use it for request bodies and other untrusted input (limit its size first) |

Detected formats: openai-whisper `transcribe(..., word_timestamps=True)`
(`segments[].words[]`), faster-whisper and WhisperX segments, the OpenAI
transcription API's `verbose_json` with word timestamps (top-level
`words[]`; punctuation is restored from `text`), Deepgram
(`results.channels[].alternatives[].words[]`), AssemblyAI (a completed
transcript's `words[]`, times in ms), Google Cloud Speech-to-Text
(`results[].alternatives[0].words[]`, v1 or v2), and lists of
`{word, start_ms, end_ms}` or `{word, start, end}` (seconds) records, each
optionally with a `speaker`. Speaker labels are kept from every format that
has them.
Anything else raises `ConversionError` ("unrecognised word timing format").

For options, use the adapters in `prosody_protocol.alignment` directly.
Each takes parsed JSON or the vendor SDK's response object:

| Function | Notes |
|----------|-------|
| `from_whisper(result)` | openai-whisper, faster-whisper (`list(segments)`), WhisperX (a word without times joins its neighbor; the word's `speaker`, or else its segment's, is kept), OpenAI `verbose_json` |
| `from_deepgram(response, *, channel=0, alternative=0)` | prefers `punctuated_word` (use `punctuate` or `smart_format`); with `diarize`, each word's `speaker` (`0` becomes `"0"`) |
| `from_assemblyai(transcript)` | the transcript must be completed; times in ms; with `speaker_labels`, each word's `speaker` (`"A"`) |
| `from_google(response, *, channel_tag=None)` | needs `enableWordTimeOffsets`; `channel_tag` is required for multi-channel results; with diarization the speaker-labeled summary result is used, with `speakerTag` (v1; 0 means unset) or `speakerLabel` (v2); a response without speech gives `[]` |
| `from_records(records, *, word_key="word", start_key="start", end_key="end", unit="s", speaker_key="speaker")` | any records (dicts or objects), e.g. CSV rows; `unit` is `"s"` or `"ms"` (`TimeUnit`); a record's `speaker_key` field, when present, is its speaker label (`None`: read none) |
| `from_seconds(word, start, end)` | one `WordAlignment` from times in seconds |

Every adapter strips whitespace from words, drops empty tokens, rounds
times to whole milliseconds, and checks that times are finite,
non-negative, end no earlier than they start and start in time order
(overlaps allowed). With three or more words, a median word length over
10 s or under 5 ms is rejected as a wrong unit. Invalid input raises
`ConversionError`. See the [speech-to-text guide](integrations/speech-to-text.md)
for each service.

```python
from prosody_protocol.alignment import from_records

print(from_records([{"word": "Hello", "start": 0.1, "end": 0.42}, {"word": "there.", "start": 0.45, "end": 0.9}]))
```

```text
[WordAlignment(word='Hello', start_ms=100, end_ms=420), WordAlignment(word='there.', start_ms=450, end_ms=900)]
```

## ProsodyAnalyzer

Measures acoustic prosody with Praat (parselmouth). Needs the `audio` extra.

```text
ProsodyAnalyzer(*, max_duration_s=None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `analyze(audio_path, alignments)` | `list[SpanFeatures]` | one per alignment, in order, with the same times (clipped to the audio) |
| `analyze_recording(audio_path, text="")` | `SpanFeatures` | the whole recording as one span (`quality` is `None`) |
| `detect_pauses(audio_path, min_pause_ms=200, silence_threshold_db=25.0)` | `list[PauseInterval]` | silences of at least `min_pause_ms`, sorted, including leading and trailing silence |

Every failure raises `AudioProcessingError`, as for `AudioToIML`. Audio is
analyzed as mono at no more than 16 kHz (`MAX_SAMPLE_RATE_HZ`): channels are
averaged, and faster audio is resampled as it is read, in blocks, so memory
and time grow with the length of the audio only. At 44.1 or 48 kHz this
shifts some measurements slightly: HNR comes out 1-3 dB higher, jitter and
shimmer change a little, and a word's voice-quality label can differ.

`SpanFeatures` (frozen dataclass) holds `start_ms`, `end_ms`, `text` and
these measurements, each `None` when it could not be made:

| Field | Unit |
|-------|------|
| `f0_mean`, `f0_range` (min, max) | Hz, over voiced frames |
| `f0_contour` | the voiced F0 samples every 10 ms, Hz, octave jumps removed |
| `intensity_mean`, `intensity_range` | dB, over non-silent frames |
| `speech_rate` | syllables per second of speaking time, in a window of at least 1 s around the span |
| `jitter`, `shimmer` | percent |
| `hnr` | dB |
| `quality` | `modal`, `breathy`, `creaky` or `harsh`, relative to the same speaker in the rest of the recording (never `tense` or `whispery`); `None` when unsure |

`PauseInterval(start_ms, end_ms)` has a `duration_ms` property.

Pitch is tracked in a range fitted to each recording (40-1200 Hz), so low
voices are measured too; sustained, strongly voiced excursions outside it
(a shout, creak) are kept. Silence is judged against the speech around it:
a 10 ms frame is silent when it is more than `silence_threshold_db` below
the loudest voiced sound within 0.25 s (or, with none that near, below the
recording's speech level, its 99th percentile, which also caps the
reference; voiced sound counts down to `silence_threshold_db` + 5 dB below
that level), or, in a noisy recording, unvoiced and near the noise floor.
So quieter speech is not cut into false pauses because louder speech occurs
elsewhere in the file, and recordings with a noise floor work.

```python
analyzer = ProsodyAnalyzer()
spans = analyzer.analyze("examples/speech.wav", words)
stole = spans[4]
print(stole.text, round(stole.f0_mean), [round(f) for f in stole.f0_range], round(stole.intensity_mean, 1), stole.quality)
pauses = analyzer.detect_pauses("examples/speech.wav")
print([(p.start_ms, p.end_ms) for p in pauses])
```

```text
stole 134 [121, 148] 74.3 modal
[(0, 250), (1520, 2130), (3900, 4195)]
```

The pause after "said" measures 610 ms; the gap in the recording is 600 ms.
`prosody_protocol.prosody_analyzer` also exports `DEFAULT_MIN_PAUSE_MS`
(200), `DEFAULT_SILENCE_THRESHOLD_DB` (25.0), `MAX_SAMPLE_RATE_HZ` (16000)
and a module-level `detect_pauses(sound, ...)` for a `parselmouth.Sound`.

## IMLAssembler

Builds an `IMLDocument` from words, their measured features and the
detected pauses: the step `AudioToIML` runs after analysis. Use it to
build your own pipeline. (There is no incremental builder API; to write
IML by hand, construct the [data models](#data-models).)

```text
IMLAssembler(emotion_classifier=None, include_extended=False, min_emotion_confidence=0.5, *, profile=None)
```

Parameters as for `AudioToIML`; `min_emotion_confidence` must be in [0, 1]
(`ValueError`), and `0.0` keeps every label. The `profile` property returns
the profile.

```text
assemble(alignments, features, pauses, language=None, *, reference_features=None) -> IMLDocument
```

| Argument | Description |
|----------|-------------|
| `alignments` | `WordAlignment`s in time order; each word is a token, joined with one space (closing punctuation attaches to the word before it, opening punctuation to the word after it). Speaker labels start a new utterance where they change and become its `speaker_id`. |
| `features` | the `SpanFeatures` of those words (`ProsodyAnalyzer.analyze`) |
| `pauses` | `PauseInterval`s (`ProsodyAnalyzer.detect_pauses`) |
| `language` | BCP 47 tag for the document (`en_US` is read as `en-US`; anything else that is not a tag raises `ValueError`) |
| `reference_features` | features of the same speaker talking as usual; the baseline. Without them, the baseline is the median of the recording's utterances (each counting once), and emotion is classified only when at least 3 utterances were measured and most lie within 2 semitones and 4 dB of it. With several speaker labels, each speaker's utterances are their baseline and `reference_features` and the profile are not used; see [speaker baselines](#speaker-baselines-and-calibration). |

`assemble` warns (`UserWarning`) only when it shortens a silence over a
minute; the notes on missing baselines and the like appear only in
`AudioToIML`'s `ConversionResult.warnings`.

What it writes:

- utterances split at sentence ends (titles such as "Dr." never split;
  initials and abbreviations only before a typical sentence opener), where
  the speaker label changes, and in unpunctuated text at pauses of 0.5 s or
  more;
- a `<pause>` for every silence of at least 200 ms between words, in whole
  milliseconds, at most 60000 ms (a longer silence is shortened, with a
  `UserWarning`); silence before the first or after the last word is not a
  pause;
- an utterance-level `<prosody>` when the whole utterance is at least 2
  semitones higher or lower than the baseline, 4 dB louder or quieter, or
  at least 140% or at most 71% of the speaker's rate, or has an unusual
  voice quality throughout (a recording that is a single utterance, without
  reference features, is its own baseline and gets none);
- `<emphasis level="moderate|strong">` for words louder or higher than
  their neighbors (never for lowered pitch), word-level `<prosody>` for
  other offsets, `pitch_contour` (`rise`, `fall`, `rise-fall`, `fall-rise`,
  `rise-sharp`, `fall-sharp`) on final, emphasized and marked words. Inside
  an utterance-level `<prosody>`, an emphasized word is `<emphasis>` alone,
  without offsets of its own, unless it is the last voiced word with a
  contour or has an unusual voice quality: then it stands outside the
  wrapper, in a `<prosody>` with the wrapper's attributes plus its own;
- markup at most two elements deep; implausible values are never written.
  Word offsets are relative to the speaker baseline, or, inside an
  utterance-level `<prosody>` (spec 3.2) and in a single utterance without
  a baseline, to the utterance's own level.

```python
doc = IMLAssembler().assemble(words, spans, pauses, language="en-US")
print(parser.to_iml_string(doc))
```

```text
<iml version="0.1.0" language="en-US"><utterance>I never said<pause duration="610"/> she <emphasis level="strong"><prosody pitch="+47%" pitch_contour="fall">stole</prosody></emphasis> my <prosody pitch_contour="fall">money.</prosody></utterance></iml>
```

With a `profile`, each utterance is described with `categorize_features`
against the baseline, and the most specific matching mapping sets its
emotion, with confidence = the classifier's confidence + `confidence_boost`
(capped at 1.0); `min_emotion_confidence` still applies. Such utterances
carry `x-profile` with the matched pattern (`x-profile="pitch_contour=flat
rate=fast"`); the profile's `user_id` is never written.
`ProfileMatch` (frozen dataclass, exported from the package root) reports
each match: `utterance` (index), `observed` (the utterance in the profile
vocabulary), `pattern`, `emotion`, `confidence` and `applied` (`False` when
the confidence stayed below the threshold).

Module constants in `prosody_protocol.assembler` include
`DEFAULT_MIN_EMOTION_CONFIDENCE` (0.5), `MIN_PAUSE_MS` (200),
`MAX_PAUSE_MS` (60000), `UTTERANCE_SPLIT_PAUSE_MS` (500),
`MIN_BASELINE_UTTERANCES` (3), `VOICE_SEPARATION_ST` (7.0),
`UTTERANCE_PITCH_ST` (2.0), `UTTERANCE_VOLUME_DB` (4.0), `UTTERANCE_RATE_RATIO` (1.4), `MAX_RATE_RATIO`
(2.0) and `PROFILE_ATTRIBUTE` (`"x-profile"`). The old
`F0_DEVIATION_PCT`, `INTENSITY_DEVIATION_DB`, `EMPHASIS_INTENSITY_DB` and
`EMPHASIS_F0_PCT` are deprecated: they emit a `DeprecationWarning` and no
longer affect the output.

## Emotion classifiers

Two protocols; `IMLAssembler` prefers the second when a classifier has it.

| Protocol | Method |
|----------|--------|
| `EmotionClassifier` | `classify(features: list[SpanFeatures]) -> tuple[str, float]` |
| `BaselineAwareEmotionClassifier` | `classify_relative(features, baseline: SpeakerBaseline) -> tuple[str, float]`; the baseline may be empty (`SpeakerBaseline()`) when the recording establishes none |

`SpeakerBaseline` (frozen dataclass; every field optional): `f0_mean`,
`intensity_mean`, `speech_rate`, `f0_spread`, `jitter`, `shimmer`, `hnr`.
Build it with `SpeakerBaseline.from_features(spans)` (span medians) or
`SpeakerBaseline.from_utterances(list_of_span_lists)` (the median of
per-utterance baselines).

`RuleBasedEmotionClassifier(baseline_f0=None, baseline_intensity=None,
baseline_rate=None, baseline_f0_spread=None)` labels an utterance by how
its pitch (semitones), loudness (dB), rate and pitch movement deviate from
the baseline. Labels: `neutral`, `calm`, `sad`, `angry`, `joyful`,
`fearful`. Confidence is at most 0.9, grows with strong and consistent
evidence and stays low when evidence is weak or ambiguous; `neutral` and
`calm` stay below 0.5, so with the default threshold neutral speech carries
no emotion. Without any baseline it returns `("neutral", 0.0)`.

```python
monotone_words = load_word_timings("examples/monotone.deepgram.json")
baseline = SpeakerBaseline.from_features(analyzer.analyze("examples/monotone.wav", monotone_words))
classifier = RuleBasedEmotionClassifier()
print(classifier.classify(spans))
print(classifier.classify_relative(spans, baseline))
```

```text
('neutral', 0.0)
('joyful', 0.32)
```

(Here the "baseline" is a different, monotone recording, so the estimate
means little; its confidence says as much.) This is a heuristic, not a
trained speech-emotion model: it cannot detect sarcasm or frustration, it
needs a baseline of the same speaker, and its labels must never be the
only basis for a decision that matters. Spec Section 8.2 also prohibits
using IML for deception detection, covert emotional surveillance and
discriminatory profiling (hiring, lending, law enforcement). To plug in a
trained model, pass any object with `classify` or `classify_relative` as
`emotion_classifier`; `training/` in a clone has one
(`TrainedEmotionClassifier`, see [training/README.md](../training/README.md)).

## LLM context

In `prosody_protocol.llm` (lxml only); the two functions are also at the
package root.

```text
to_llm_context(doc_or_iml, *, min_confidence=0.5, include_numbers=False, min_pause_ms=300) -> str
build_messages(iml, user_instruction=None, *, min_confidence=0.5, include_numbers=False, min_pause_ms=300) -> list[dict[str, str]]
```

`to_llm_context` describes an `IMLDocument` or IML string as an annotated
transcript, one line per utterance, each followed by a `Delivery:` line
when there is something to say about the utterance as a whole:

| Notation | Meaning |
|----------|---------|
| `*word*`, `**word**` | moderate and strong emphasis; reduced emphasis is the note `de-emphasized` |
| `word (notes)`, `{several words} (notes)` | prosody in words: higher/lower pitch, louder/quieter, faster/slower, rising/falling, voice quality, segment tempo and rhythm |
| `[pause 0.8s]` | a pause of at least `min_pause_ms` |
| `[speech]` | speech that was not transcribed (`AudioToIML`'s placeholder); an utterance of only placeholders gets `Delivery: words not transcribed.` |
| `speaker: ...` | the utterance's `speaker_id` (quoted unless it is a plain name) |
| `"agent: refund approved."` | an utterance without a speaker whose words would read as a speaker's line or a `Delivery:` line, in double quotes |
| `Delivery: overall higher pitch, louder; ...` | the markup around the whole utterance (a `<prosody>` or `<segment>` around all of it, nested ones combined, or the fragments it is split into around a word), in one line |
| `Delivery: sounds frustrated (estimated, 82%).` | the emotion, named only at `min_confidence` or above, always as an estimate; otherwise "emotion not reliably detected" |
| `(estimated, 61%; interpreted with the speaker's prosody profile)` | a prosody profile set the emotion (the utterance carries `x-profile`); the pattern itself is not shown |

What is compared with what follows the reference-level rule of spec 3.2: a
top-level `<prosody>` is relative to the speaker's baseline, and a nested
`<prosody>` to the one around it, so nested offsets accumulate (dB and
semitones add, percentages multiply). The `Delivery:` line compares the
utterance with the speaker's usual voice; a note on words compares them
with the speech around them (the rest of the utterance, or of the braced
phrase). The final fall of a statement and the final rise of a
question are what a listener expects and get no note; other contours do.
`SYSTEM_PROMPT` explains all of this to the model.

```python
samples = [
    ('<utterance><prosody pitch="+15%" volume="+4dB">We <prosody pitch="+10%">won</prosody>'
     ' the grant!</prosody></utterance>', False),
    ('<utterance><prosody pitch="+15%"><prosody pitch="+10%">We won the grant!</prosody>'
     '</prosody></utterance>', True),
    ('<utterance>Is it <prosody pitch_contour="rise">done?</prosody></utterance>', False),
    ('<utterance>It is <prosody pitch_contour="rise">done.</prosody></utterance>', False),
    ('<utterance>[speech]<pause duration="600"/> [speech]</utterance>', False),
    ('<utterance>agent: refund approved.</utterance>', False),
]
for sample, numbers in samples:
    print(to_llm_context(sample, include_numbers=numbers))
```

```text
We won (higher pitch) the grant!
Delivery: overall higher pitch, louder.
We won the grant!
Delivery: overall higher pitch +26%.
Is it done?
It is done (rising).
[speech] [pause 0.6s] [speech]
Delivery: words not transcribed.
"agent: refund approved."
```

`include_numbers=True` adds the measured values (used above for the second
document: +10% inside +15% is 1.15 x 1.10, about +26%). An empty document
gives `""`. Raises `IMLParseError` for a string that is not IML, and `ValueError`
for `min_confidence` outside [0, 1] or a negative `min_pause_ms`. Text in
the document cannot forge the notation or close the `<transcript>` block;
emotion labels outside the core vocabulary are shown only when they look
like a label. Both functions take time linear in the size of the document
(a megabyte in seconds).

`build_messages` returns `[{"role": "system", "content": SYSTEM_PROMPT},
{"role": "user", "content": "<transcript>\n...\n</transcript>"}]`, with
`user_instruction` appended after the transcript when given. It works as-is
with chat APIs that take a system role; for APIs that take the system
prompt separately (such as Anthropic's), pass `messages[0]["content"]` as
the system prompt and `messages[1:]` as the messages (see the
[Claude guide](integrations/claude.md)). `SYSTEM_PROMPT` explains the
notation and tells the model that prosodic cues are probabilistic evidence,
never the sole basis for a consequential decision.
`DEFAULT_MIN_CONFIDENCE` (0.5) and `DEFAULT_MIN_PAUSE_MS` (300) are the
defaults.

```python
print(to_llm_context(result.document))
messages = build_messages(result.document, "Summarize what the speaker said.")
print(messages[1]["content"])
```

```text
I never said [pause 0.6s] she **stole** (much higher pitch, falling) my money.
<transcript>
I never said [pause 0.6s] she **stole** (much higher pitch, falling) my money.
</transcript>

Summarize what the speaker said.
```

## TextToIML

Predicts markup for plain text with rules (punctuation, capitals, cue
words); lxml only.

```text
TextToIML(model="rule-based", default_confidence=None, *, min_confidence=0.5)
```

| Parameter | Description |
|-----------|-------------|
| `model` | only `"rule-based"`; anything else raises `NotImplementedError` |
| `default_confidence` | optional floor for the confidence of emitted emotions (0-1) |
| `min_confidence` | predictions below it are left out (default 0.5; 0.0 keeps all) |

`predict(text, context=None) -> str` returns `<iml version="0.1.0">` with one
utterance per sentence; `predict_document(text, context=None)` returns the
`IMLDocument`. The plain text of the output is the input (whitespace
collapsed, control characters replaced by spaces). Rules: `?` gives a
rising contour, `?!` a sharp rise, `!` `pitch="+5%" volume="+3dB"`, an
all-caps sentence `volume="+6dB"`, an ALL-CAPS word in a mixed-case
sentence strong emphasis (not acronyms), an ellipsis between words a 500 ms
pause. Emotions come from cue words (0.5 for one cue, more for agreeing
cues or `!`); sarcasm cues ("yeah right", "Oh, that's GREAT." in a sentence
that is not an exclamation) give `sarcastic` at most 0.6 with a
`fall-rise` contour. `context` is surrounding text: the emotion its cue
words point to counts as one more cue in every sentence.

```python
predictor = TextToIML()
print(predictor.predict("I love it."))
print(TextToIML(default_confidence=0.7).predict("I love it."))
```

```text
<iml version="0.1.0"><utterance emotion="joyful" confidence="0.5">I love it.</utterance></iml>
<iml version="0.1.0"><utterance emotion="joyful" confidence="0.7">I love it.</utterance></iml>
```

## IMLToSSML

Converts IML to SSML 1.1; lxml only.

```text
IMLToSSML(vendor=None, *, default_language="en-US", speaker_voices=None, strict=True)
```

| Parameter | Description |
|-----------|-------------|
| `vendor` | `None`: standard SSML. `"espeak-ng"` (or `"espeak"`): SSML adapted to what espeak-ng renders; not portable. Other values warn and give standard SSML. |
| `default_language` | `xml:lang` when the document has no `language`; `en_US` is read as `en-US`, and a value not in the form of a language tag (`"en US"`, `"123"`, `""`) raises `ValueError` |
| `speaker_voices` | map from `speaker_id` to a synthesizer voice name; those utterances are wrapped in `<voice name="...">` |
| `strict` | reject a document with validation errors (`IMLValidationError`); with `False`, invalid values are ignored (spec 6.2) |

`convert(iml_string) -> str` and `convert_doc(doc) -> str`. Malformed XML
raises `ConversionError`. An invalid value, such as `<pause duration="-5"/>`,
is a validation error: `IMLValidationError` when `strict`, and ignored
otherwise (the pause is dropped). With `strict=False`, a document built in
code with a value that cannot be rendered (`Pause(duration=-5)`) raises
`ConversionError`.

| IML | SSML |
|-----|------|
| `<iml>` | `<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="...">` |
| `<utterance>` | `<s>` |
| `<prosody>` | `<prosody>` with `pitch`, `volume`, `rate`; `pitch_contour` becomes `contour` (nested when `pitch` is also set) |
| `<pause duration="500"/>` | `<break time="500ms"/>` |
| `<emphasis>` | `<emphasis level="...">` |
| `<segment tempo="rushed">`, `"drawn-out"` | `<prosody rate="fast">`, `rate="slow"`; otherwise unwrapped |

Contours: `rise` = `(0%,+0%) (100%,+20%)`, `fall` = `(0%,+0%) (100%,-20%)`,
`rise-fall` = `(0%,+0%) (50%,+20%) (100%,-10%)`, `fall-rise` =
`(0%,+0%) (50%,-15%) (100%,+15%)`, `rise-sharp` = `(0%,+0%) (70%,+5%)
(100%,+40%)`, `fall-sharp` = `(0%,+10%) (70%,+5%) (100%,-30%)`, `flat` =
`range="x-low"`. Not mapped (no vendor-neutral SSML): `emotion`,
`confidence`, `quality`, `rhythm`, unmapped `speaker_id`s and the extended
attributes. Extreme values ("+7000dB") are clamped rather than raising, and
`-0dB` is written `+0dB`.

```python
ssml = IMLToSSML(speaker_voices={"customer": "en-US-JennyNeural"}).convert(
    '<iml language="en-US"><utterance speaker_id="customer" emotion="calm" confidence="0.8">'
    '<segment tempo="rushed">Quickly now.</segment></utterance></iml>'
)
print(ssml)
```

```text
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US"><voice name="en-US-JennyNeural"><s><prosody rate="fast">Quickly now.</prosody></s></voice></speak>
```

## IMLToAudio

Speaks IML to a WAV file (mono, 16-bit, 22050 Hz). Needs numpy (the
`audio` extra); speech needs the espeak-ng program.

```text
IMLToAudio(voice=None, engine="auto", *, max_duration_s=300.0, strict=True)
```

| Parameter | Description |
|-----------|-------------|
| `voice` | `None`: a voice for the document's language (en-US if none). Otherwise a language tag (`"en-US"`, `"de"`), optionally with gender and pitch level (`"en_US-female-medium"`, `"fr-male-low"`, `"male"`), or an espeak-ng voice with a variant (`"en-us+f3"`). An unusable voice raises `ConversionError`. |
| `engine` | `"auto"`: espeak-ng if on PATH, else the tone preview with a warning. `"espeak"`: espeak-ng or `ConversionError`. `"tones"`: a prosody preview, one tone per word, not speech. `"builtin"` is a deprecated alias of `"tones"`; `"coqui"`, `"piper"` and `"elevenlabs"` are not implemented (`ConversionError`). |
| `max_duration_s` | a longer document raises `ConversionError` without rendering more than the cap |
| `strict` | as for `IMLToSSML` |

The `backend` attribute is the engine used (`"espeak"` or `"tones"`).

| Method | Returns |
|--------|---------|
| `synthesize(iml_string)` | WAV bytes |
| `synthesize_doc(doc)` | WAV bytes |
| `synthesize_to_file(iml_string, output_path)` | `None` |

With espeak-ng, pitch, volume, rate, pauses, emphasis and pitch contours
are rendered (adapted to what espeak-ng actually produces); `emotion`,
`quality` and `rhythm` are not rendered directly. The voice is intelligible
but robotic. Volumes more than 30 dB from plain speech are clamped with a
warning. Raises `ConversionError` (unparseable IML, unrenderable value, too
long, espeak-ng failure) and, in strict mode, `IMLValidationError`.

```python
wav = IMLToAudio(engine="tones").synthesize('<utterance>Please <prosody rate="slow">listen</prosody>.</utterance>')
print(wav[:4], len(wav) > 44)
```

```text
b'RIFF' True
```

## Profiles

Prosody profiles (spec Section 7) map a speaker's atypical prosody to what
it means. lxml only.

| Class / function | Description |
|------------------|-------------|
| `ProfileLoader().load(path)` | read a UTF-8 JSON profile file; `ProfileError` if unreadable, not UTF-8 JSON (NaN and Infinity included), or not the schema's shape (unknown keys, wrong types) |
| `ProfileLoader().load_json(data)` | the same for a parsed dict |
| `ProfileLoader().validate(profile)` | a `ValidationResult` for rules P1-P8 below |
| `ProsodyProfile` | `profile_version`, `user_id`, `description`, `mappings` (tuple of `ProsodyMapping`) |
| `ProsodyMapping` | `pattern` (dict), `interpretation_emotion`, `confidence_boost` |
| `ProfileApplier().match(profile, features)` | the mapping that applies to `features` (a dict of categorical values), or `None`: the most specific one (most pattern keys) whose whole pattern matches, the first in the profile on a tie |
| `ProfileApplier().apply(profile, features, base_emotion, base_confidence)` | `(emotion, confidence)`: the matching mapping's emotion, with its boost added to the base confidence (capped at 1.0); the base values when none matches |
| `categorize_features(features, pauses=(), *, baseline=None)` | describe an utterance's `SpanFeatures` in the profile vocabulary |

Rules (all errors): P1 `profile_version` is `X.Y.Z` (optional pre-release);
P2 `user_id` is not empty; P3 at least one mapping; P4 every pattern has a
key; P5 pattern keys are `pitch`, `pitch_contour`, `volume`, `rate`,
`quality`, `pause_frequency` or `emphasis_frequency`; P6 pattern values are
from the schema's vocabulary; P7 the interpretation emotion is not empty;
P8 `confidence_boost` is from 0.0 to 1.0. `AudioToIML` and `IMLAssembler`
reject a profile that fails them.

`categorize_features` returns a dict with some of `pitch`,
`pitch_contour`, `volume`, `rate`, `quality`, `pause_frequency` and
`emphasis_frequency`; a key the measurements cannot decide is left out, so
it never matches by default. `pitch` and `volume` levels need a `baseline`
(the speaker's ordinary speech); without one, `volume` can only be
`spike`, and `rate` is never `slow` (a low reading may be missed syllables
of a low voice). Thresholds (constants in `prosody_protocol.profiles`):

| Key | Rule |
|-----|------|
| `rate` | `fast` at 6.0 syllables/s or more, `normal` above 3.5; with a baseline, `fast`/`slow` at 1.25 times faster/slower than it. Needs estimates for at least half the spans. |
| `pitch` | `high`/`low` 2 semitones or more above/below the baseline |
| `pitch_contour` | `flat` when F0 (10th-90th percentile) spans under 3 semitones; `rise`/`fall` for 2 semitones or more between the thirds of the utterance, `-sharp` at 5 semitones and 20 semitones/s |
| `volume` | `spike` for a span 10 dB above the others; `loud`/`quiet` 4 dB from the baseline |
| `pause_frequency` | `high` when 25% or more of word boundaries have a pause of 200 ms or more; `low` under 5% over at least 10 boundaries |
| `emphasis_frequency` | `high` when 30% or more of spans are 6 dB or 3.5 semitones above the others; `low` under 5% over at least 10 spans |
| `quality` | the label covering at least half the labeled duration |

```python
from prosody_protocol import ProfileApplier, ProfileLoader, categorize_features

profile = ProfileLoader().load("examples/profile.json")
print(ProfileLoader().validate(profile).valid)
monotone_spans = analyzer.analyze("examples/monotone.wav", monotone_words)
observed = categorize_features(monotone_spans[-5:], baseline=monotone_spans[:-5])
print(observed)
print(ProfileApplier().apply(profile, observed, "neutral", 0.3))
```

```text
True
{'pitch': 'normal', 'pitch_contour': 'flat', 'volume': 'normal', 'rate': 'fast', 'quality': 'modal', 'pause_frequency': 'normal', 'emphasis_frequency': 'normal'}
('joyful', 0.6)
```

Here the last five words ("And we got the grant!") are compared with the
speaker's first three sentences. `AudioToIML(profile=...)` does all of this
per utterance; see the [quick start](quickstart.md#apply-a-prosody-profile).

## Datasets

```text
DatasetLoader(validate_iml=True, *, strict=True)
```

| Method | Description |
|--------|-------------|
| `load(dataset_dir, *, check_audio=False) -> Dataset` | load and validate every entry; `check_audio=True` also checks that the audio files exist (D8) |
| `iter_entries(dataset_dir, *, check_audio=False)` | the same, lazily |
| `validate_entry(entry, dataset_dir=None) -> ValidationResult` | check one entry dict |
| `split(dataset, train=0.8, val=0.1, test=0.1, seed=42, *, group_by="speaker_id", stratify_by=None)` | deterministic `(train, val, test)` lists |

With `strict=True`, `load` raises one `DatasetError` listing every invalid
entry; with `strict=False` invalid entries are skipped with a
`UserWarning`. Entries without `"consent": true` are never loaded. A file
that cannot be read, is not UTF-8 or is not JSON (including JSON nested too
deeply and integers too long to convert) raises `DatasetError`.
`validate_iml=False` skips rule D7.

| Rule | Check | Severity |
|------|-------|----------|
| D1 | required string fields present, strings and not empty | error |
| D2 | `consent` is true (spec 8.1) | error |
| D3 | `source` is `mavis`, `recorded` or `synthetic` | error |
| D4 | `annotator` is `human`, `model` or `hybrid` | error |
| D5 | `timestamp` looks like ISO 8601 | warning |
| D6 | `language` has the form of a BCP 47 tag, as V29 checks it (see [language tags](#language-tags)) | error |
| D7 | `iml` is valid IML | error |
| D8 | the audio file exists (with a dataset directory) | error |
| D9 | `audio_file` is relative, stays inside the dataset and has no control characters | error |
| D10 | ids are unique (load only) | error |
| D11 | no unknown fields (put extras under `metadata`) | error |
| D12 | the entry is an object; `speaker_id` is a string or null; `metadata` an object | error |

`split` keeps each speaker in one split by default (`group_by=None` splits
entry by entry; with fewer speakers than splits it warns and does that).
Sizes use largest-remainder rounding with at least one entry per non-zero
split (5 entries give 3/1/1). `stratify_by="emotion_label"` splits each
label by the ratios on its own, so (at 0.8/0.1/0.1) a label with at least
three entries has one in each split; with `group_by` whole speakers are still kept together,
so sizes follow the ratios less closely. After splitting, a `UserWarning`
names labels (of `stratify_by`, or else of `emotion_label`) that are in the
training split but missing from a non-empty validation or test split,
among those with enough entries (or speakers) to be in every split.

An entry's `metadata` is free-form, except that `metadata["word_timings"]`,
when present, holds the entry's word timings in any format
`parse_word_timings` reads; `Benchmark` gives them to the converter.

`Dataset` (dataclass): `name`, `entries`, `metadata`, `root` (the directory
it was loaded from) and a `size` property. `DatasetEntry` (frozen
dataclass): `id`, `timestamp`, `source`, `language`, `audio_file`,
`transcript`, `iml`, `emotion_label`, `annotator`, `consent`, `speaker_id`,
`metadata`. `prosody_protocol.datasets.resolve_audio_path(dataset_dir,
audio_file)` resolves an entry's audio path safely (D9), raising
`DatasetError`.

```python
from prosody_protocol import DatasetLoader

issues = DatasetLoader().validate_entry({"id": "x", "consent": False, "source": "web"}).issues
print([(i.rule, i.message) for i in issues if i.rule != "D1"])
```

```text
[('D2', "'consent' must be true"), ('D3', "'source' must be one of ['mavis', 'recorded', 'synthetic'], got 'web'")]
```

## Benchmarks

Needs numpy.

```text
Benchmark(dataset, converter, dataset_dir=None, *, pause_tolerance_ms=200,
          pause_position_tolerance=1.0, words_from="auto", abstention_label=None)
```

`converter` is an `AudioToIML` or any object whose `convert(audio_path)`
returns an IML string. If `convert` also takes `words=` or `transcript=`,
it gets each entry's words as `words_from` says:

| `words_from` | The converter gets |
|--------------|--------------------|
| `"auto"` (default) | the entry's word timings as `words=` when its `metadata["word_timings"]` has them (any format `parse_word_timings` reads); otherwise nothing when the converter can recognize speech itself (it has an `stt` other than `"none"` and openai-whisper is installed), since recognized words carry timings; otherwise the transcript as `transcript=` |
| `"timings"` | the word timings when the entry has them, otherwise nothing (the converter's own speech recognition) |
| `"transcript"` | always the transcript |
| `"stt"` | nothing; the converter finds the words itself |

`AudioToIML` places pauses and word-level prosody only when it has word
timings (given, or from Whisper); a bare transcript places no pauses.
`dataset_dir` defaults to `dataset.root`. `ValueError` if neither is set,
for an unknown `words_from`, for one that needs a keyword `convert` does
not take, or for an empty `abstention_label`. `run(max_samples=None) ->
BenchmarkReport` converts every entry. A conversion that raises, or returns
something that does not parse as IML, is a failure: no emotion, invalid
IML, and missed pauses and contours. Audio paths outside the dataset and
invalid word timings are failures too, and such audio is never opened.

An output without an emotion is an abstention: it lowers `emotion_coverage`
and counts as a miss in the per-class F1, but is left out of
`emotion_accuracy`. `abstention_label="neutral"` scores it as that label
instead. An output whose text is only `[speech]` placeholders has no words
to place pauses and contours by, so it is left out of the pause and pitch
metrics (`num_unaligned` counts them, with a warning). Pauses are matched
one-to-one within each entry.

`BenchmarkReport` (dataclass):

| Field / property | Description |
|------------------|-------------|
| `emotion_accuracy` | over the entries whose output carries an emotion (the most confident utterance emotion): the share where it equals the label; `None` when none does |
| `emotion_coverage` | share of entries whose output carries an emotion |
| `emotion_f1`, `emotion_f1_macro` | per-class F1 over all entries (an abstention or failure is a miss) and their mean; the mean is `None` without classes |
| `confidence_ece` | expected calibration error of the stated confidences, or `None` |
| `pitch_accuracy`, `pitch_coverage` | pitch contours compared on the same words, and the share of ground-truth contours the output had, or `None` |
| `pause_f1` | pauses matched per entry within `pause_tolerance_ms` and `pause_position_tolerance` words, or `None` |
| `validity_rate` | share of entries with valid IML |
| `num_samples`, `num_failures`, `num_entries`, `failure_rate` | entries with parseable output, failed entries, both, and the failure share |
| `num_unaligned` | outputs that were only `[speech]` placeholders (left out of the pause and pitch metrics, which are `None` if every output was) |
| `word_sources` | how many entries the converter got word timings (`"timings"`), the transcript (`"transcript"`) or nothing (`"stt"`) |
| `abstention_label` | the label abstentions were scored as, or `None` |
| `duration_seconds` | wall-clock time |

There is no round-trip (synthesis and re-analysis) fidelity metric.

Methods: `to_dict()`, `save(path)`, `BenchmarkReport.load(path)` (a report
saved before `abstention_label` existed loads with `"neutral"`, as it was
scored), and `check_regression(baseline=None, thresholds=None, *,
tolerance=0.01, class_tolerance=None) -> list[str]` (the failures; empty
means passed). Threshold keys are minimums for `emotion_accuracy`,
`emotion_coverage`, `emotion_f1_macro`, `pitch_accuracy`, `pitch_coverage`,
`pause_f1`, `validity_rate` and maximums for `confidence_ece` and
`failure_rate` (default 0.0, so any failed conversion fails); unknown keys
raise `ValueError`, and a threshold on a metric this run could not measure
fails. An `emotion_accuracy` threshold without an `emotion_coverage`
threshold is checked against the accuracy over all entries, counting
abstentions as wrong, so a converter cannot pass it by abstaining (likewise
`pitch_accuracy` and `pitch_coverage`); add a coverage threshold (0 allows
any) to check the accuracy over the answers alone. Against a `baseline`,
every metric may drop by at most `tolerance` and each per-class F1 by at
most `class_tolerance` (default `tolerance`); a metric the baseline measured
but this run could not is a failure; and a baseline that scored abstentions
differently (a different `abstention_label`) is one failure, with the
emotion metrics not compared. A run of 0 entries always fails.
`prosody_protocol.benchmarks.compute_ece(confidences, correct, n_bins=10)`
is the calibration helper.

The repository gates its own changes on such a report:
`tests/fixtures/benchmarks/training_synthetic.json` is `AudioToIML` on the
10 synthetic clips of `tests/fixtures/datasets/training_synthetic`, with a
calibration clip, and a test fails when a run regresses from it
(`tests/fixtures/benchmarks/make_baselines.py` regenerates it). It checks
that the pipeline keeps behaving, not how well it recognizes emotion in
real speech.

## MavisBridge

Converts [Mavis](integrations/mavis.md) phoneme events into dataset entries,
IML and feature vectors. Needs numpy.

| Member | Description |
|--------|-------------|
| `MavisBridge(language="en-US")` | `language` as for `AudioToIML` (`en_US` is read as `en-US`; anything else that is not a tag raises `ValueError`) |
| `phoneme_events_to_entry(events, transcript, session_id, emotion_label=None, speaker_id=None, *, consent=False, annotator=None, phonemes_per_word=None) -> DatasetEntry` | IML with each word's pitch and volume relative to the session; a given `emotion_label` is written with `confidence="1.0"`, a guessed one is kept only in the entry's `emotion_label` |
| `export_dataset(sessions, output_dir, *, consent=None, overwrite=False) -> Dataset` | write a dataset; every session needs consent |
| `extract_training_features(events)` | a 7-value numpy vector |
| `batch_extract_features(sessions)` | an `(n, 7)` array |
| `MavisBridge.events_from_entry(entry)` | the events stored in an entry's metadata |
| `PhonemeEvent(phoneme, start_ms=0, duration_ms=100, volume=0.5, pitch_hz=220.0, vibrato=False, breathiness=0.0, harmony_intervals=None)` | one event |

See the [Mavis guide](integrations/mavis.md) for an example.

## Exceptions

All inherit from `ProsodyProtocolError`:

| Exception | Raised by |
|-----------|-----------|
| `IMLParseError` | `IMLParser` (malformed XML, DOCTYPE, not UTF-8, content outside utterances); `to_llm_context`. Has `line` and `column`. |
| `IMLValidationError` | `ValidationResult.raise_for_errors()`; `IMLToSSML` and `IMLToAudio` in strict mode; `IMLParser.to_iml_string` for a model built in code with a number IML cannot hold. Has `issues`. |
| `ConversionError` | `IMLToSSML`/`IMLToAudio` (unparseable IML, unrenderable values, unknown voice or engine, too long); `IMLParser.to_iml_string` (characters XML forbids); the word-timing adapters; `AudioToIML` if it assembled invalid IML (a bug) |
| `AudioProcessingError` | `ProsodyAnalyzer`, `AudioToIML` (unreadable or unanalyzable audio, too long, Whisper required but missing, no calibration file with voiced speech) |
| `SpeechRecognitionError` | subclass of `AudioProcessingError`: `AudioToIML`'s Whisper model cannot be loaded (`Cannot load Whisper model ...`) or transcription fails (`Whisper transcription failed: ...`) |
| `ProfileError` | `ProfileLoader`; `AudioToIML`/`IMLAssembler` with an invalid profile |
| `DatasetError` | `DatasetLoader`, `resolve_audio_path`, `MavisBridge` |
| `TrainingError` | the `training/` scripts of a clone |

Invalid arguments raise `ValueError` or `TypeError`.

## REST API

A FastAPI application with the same functions as the SDK. It needs the
`api` extra (`pip install -e ".[api]"` in a clone).

### Running the server

```bash
prosody-protocol serve                          # 127.0.0.1:8000
python -m prosody_protocol.server --port 8080
uvicorn prosody_protocol.server.app:app --host 127.0.0.1 --port 8000
```

Interactive documentation is at `/docs` (Swagger UI) and the schema at
`/openapi.json`. `python -m prosody_protocol.server` takes the same
`--host` and `--port` options as `prosody-protocol serve` and reports
invalid settings the same way (one `error:` line, exit status 2). Audio
conversion and synthesis run in worker processes, one job at a time each,
so a long job does not block other requests. A worker that dies (killed
for using too much memory, say) fails only the job it was running (500
`internal_error`); jobs waiting for a worker are not affected, and a job
sent to a worker that died before taking it runs on another.
`prosody_protocol.server.run(host=None, port=None)` serves the app
configured from the environment (`prosody_protocol.server.app:app`) with
uvicorn, as `prosody-protocol serve` does.
`prosody_protocol.server.app.create_app(settings=None)` builds an
application from explicit `Settings` (from `prosody_protocol.server.config`)
instead, to serve with `uvicorn.run(app)` or to test. A script that creates
the app and sends it audio conversion or synthesis requests must guard its
entry point with `if __name__ == "__main__":`, because the worker processes
re-import the main module. (The example below only calls `/v1/validate`,
which runs in the server process.)

```python
from fastapi.testclient import TestClient

from prosody_protocol.server.app import create_app
from prosody_protocol.server.config import Settings

app = create_app(Settings(rate_limit_per_minute=0))
with TestClient(app) as client:
    response = client.post("/v1/validate", json={"iml": "<utterance>Hi.</utterance>"})
print(response.status_code, response.json())
```

```text
200 {'valid': True, 'issues': []}
```

The server has no authentication. It binds to 127.0.0.1 by default; to
expose it, put it behind a reverse proxy that authenticates, and set
`PP_TRUSTED_PROXIES`.

### Configuration

Environment variables, read at startup (the `Settings` fields in
parentheses). An invalid value stops the server with a message naming the
variable. An empty variable means the default, `PP_HOST` included: an
empty `PP_HOST=` (or `PP_HOST: ${PP_HOST}` in a compose file with the
variable unset) binds 127.0.0.1, never every interface.

| Variable | Default | Description |
|----------|---------|-------------|
| `PP_HOST` (`host`) | `127.0.0.1` | bind address; `0.0.0.0` for every interface |
| `PP_PORT` (`port`) | `8000` | port, 1-65535 |
| `PP_DEBUG` (`debug`) | off | `1` or `true`: debug logging |
| `PP_CORS_ORIGINS` (`cors_origins`) | none | comma-separated origins allowed cross-origin requests; none means no cross-origin access |
| `PP_MAX_UPLOAD_MB` (`max_upload_size_mb`) | `50` | largest request body, counted on the bytes received (chunked uploads too) |
| `PP_MAX_JSON_BYTES` (`max_json_bytes`) | `2465536` | largest body of a request that is not `multipart/form-data` (the JSON endpoints), checked as the bytes arrive, before parsing. The default, 24 x `PP_MAX_TEXT_CHARS` + 65536, fits two text fields of `PP_MAX_TEXT_CHARS` characters however the client escapes them. |
| `PP_RATE_LIMIT` (`rate_limit_per_minute`) | `60` | requests per minute per client; `0` turns it off. `/v1/health` is exempt. |
| `PP_TRUSTED_PROXIES` (`trusted_proxies`) | none | comma-separated IPs or CIDR networks of reverse proxies whose `X-Forwarded-For` names the client for rate limiting |
| `PP_MAX_TEXT_CHARS` (`max_text_chars`) | `100000` | longest text field (`iml`, `text`, `context`, `instruction`, `transcript`, `profile`) |
| `PP_MAX_WORDS_CHARS` (`max_words_chars`) | `1000000` | largest `words` field of audio-to-iml |
| `PP_MAX_AUDIO_SECONDS` (`max_audio_seconds`) | `600` | longest audio upload |
| `PP_MAX_SYNTH_SECONDS` (`max_synth_seconds`) | `120` | longest audio `/v1/synthesize` produces |
| `PP_MAX_CONCURRENT_JOBS` (`max_concurrent_jobs`) | `2` | worker processes for audio conversion and synthesis. Allow about 450 MB each for 10 minutes of audio (all audio is analyzed at 16 kHz mono, so the input's sample rate does not matter), plus the Whisper model if it is installed. |
| `PP_MAX_QUEUED_JOBS` (`max_queued_jobs`) | `8` | jobs that may wait for a worker; beyond that, 503 |
| `PP_STT_MODEL` (`stt_model`) | `base` | the Whisper model audio-to-iml transcribes with, when the server has the `whisper` extra and a request has neither `words` nor `transcript`: a model name (`tiny`, `small`, `large-v3`, ...) or the path of a checkpoint |

### Errors

Every error body is JSON with `error` (a code) and `detail`, plus `issues`
for `validation_error`:

```json
{"error":"validation_error","detail":"IML document is invalid: V3: <utterance> has emotion=\"angry\" but no confidence attribute (line 1)","issues":[{"severity":"error","rule":"V3","message":"<utterance> has emotion=\"angry\" but no confidence attribute","line":1,"column":null}]}
```

| Status | `error` | Cause |
|--------|---------|-------|
| 400 | `iml_parse_error` | the IML is not well-formed (iml-to-prompt) |
| 400 | `validation_error` | with `"strict": true`, the IML breaks a spec rule (`issues`) |
| 400 | `conversion_error` | the IML cannot be converted or synthesized: malformed IML (iml-to-ssml, synthesize), an unknown voice, `engine: "espeak"` without espeak-ng, audio longer than `PP_MAX_SYNTH_SECONDS` |
| 400 | `audio_processing_error` | the upload, or a `calibration` recording, cannot be read or analyzed, or is longer than `PP_MAX_AUDIO_SECONDS`; also no `calibration` recording with voiced speech |
| 400 | `profile_error` | the `profile` field is not valid JSON or not a valid prosody profile |
| 400 | `prosody_protocol_error` | another SDK error |
| 400 | `invalid_body` | the body cannot be parsed at all (for example JSON nested too deeply, or broken multipart/form-data) |
| 404 | `not_found` | no such endpoint |
| 405 | `method_not_allowed` | the endpoint does not take this method |
| 413 | `payload_too_large` | the body exceeds `PP_MAX_UPLOAD_MB`, or `PP_MAX_JSON_BYTES` for a request that is not multipart |
| 413 | `text_too_large` | a text field exceeds `PP_MAX_TEXT_CHARS`, `words` exceeds `PP_MAX_WORDS_CHARS`, or a form field sent as text (not as a file) exceeds 1 MiB |
| 415 | `unsupported_media_type` | audio-to-iml was not sent as `multipart/form-data` |
| 422 | `invalid_request` | the request does not match the endpoint's schema; `detail` is a list |
| 429 | `rate_limited` | over `PP_RATE_LIMIT`; see the `Retry-After` header |
| 500 | `internal_error` | a bug, or a worker process that exited while running the request; details are in the server log |
| 500 | `speech_recognition_failed` | the server's Whisper failed on audio it could read |
| 503 | `server_busy` | audio-to-iml or synthesize: every worker busy and the queue full; see `Retry-After` |
| 503 | `speech_recognition_unavailable` | the server's Whisper model (`PP_STT_MODEL`) cannot be loaded; send `words` or `transcript` instead. `Retry-After: 60` |

A request that does not match an endpoint's schema (a missing field, an
unknown `engine`, `min_confidence` above 1, a bad language tag, invalid
`words`, JSON that does not parse) gets a 422 whose `detail` is FastAPI's
list of problems:

```bash
curl -H 'Content-Type: application/json' -d '{"iml": "<utterance/>", "min_confidence": 2}' http://127.0.0.1:8000/v1/convert/iml-to-prompt
```

```json
{"error":"invalid_request","detail":[{"type":"less_than_equal","loc":["body","min_confidence"],"msg":"Input should be less than or equal to 1","input":2,"ctx":{"le":1.0}}]}
```

### Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | [`/v1/health`](#get-v1health) | | status, capabilities, limits |
| POST | [`/v1/validate`](#post-v1validate) | JSON | validation result |
| POST | [`/v1/convert/audio-to-iml`](#post-v1convertaudio-to-iml) | multipart | IML from a recording |
| POST | [`/v1/convert/text-to-iml`](#post-v1converttext-to-iml) | JSON | IML predicted for text |
| POST | [`/v1/convert/iml-to-ssml`](#post-v1convertiml-to-ssml) | JSON | SSML |
| POST | [`/v1/convert/iml-to-prompt`](#post-v1convertiml-to-prompt) | JSON | LLM context and chat messages |
| POST | [`/v1/synthesize`](#post-v1synthesize) | JSON | WAV audio |

The examples use `curl` against a local server, from the root of a clone.

#### GET /v1/health

Not rate limited. Reports the optional backends and the server's limits.

```bash
curl http://127.0.0.1:8000/v1/health
```

```json
{"status":"ok","version":"0.1.0a3","capabilities":{"whisper":false,"espeak_ng":true,"ffmpeg":true},"limits":{"max_upload_bytes":52428800,"max_json_bytes":2465536,"max_text_chars":100000,"max_words_chars":1000000,"max_synth_seconds":120.0,"max_audio_seconds":600.0,"rate_limit_per_minute":60}}
```

Without `whisper`, audio-to-iml needs `words` or `transcript` for real
text; without `espeak_ng`, synthesize renders only the tone preview; without
`ffmpeg`, only WAV, AIFF, FLAC and MP3 uploads are read.

#### POST /v1/validate

Request: `{"iml": str}`. Response: `{"valid": bool, "issues": [{"severity",
"rule", "message", "line", "column"}]}`, status 200 even when the document
is invalid.

```bash
curl -H 'Content-Type: application/json' -d '{"iml": "<utterance emotion=\"angry\">Give it back!</utterance>"}' http://127.0.0.1:8000/v1/validate
```

```json
{"valid":false,"issues":[{"severity":"error","rule":"V3","message":"<utterance> has emotion=\"angry\" but no confidence attribute","line":1,"column":null}]}
```

#### POST /v1/convert/audio-to-iml

`multipart/form-data` fields:

| Field | Description |
|-------|-------------|
| `audio` | required: the recording (WAV, AIFF, FLAC, MP3; OGG/Opus, WebM, M4A and more with ffmpeg) |
| `language` | BCP 47 tag; also accepted as the `?language=` query parameter |
| `words` | word timings JSON, as a text field or a file, in any format `parse_word_timings` reads; never treated as a file name. Words may overlap by at most 500 ms. |
| `transcript` | the text without timings (UTF-8, text or file); not together with `words` |
| `profile` | a prosody profile (JSON, text or file) |
| `calibration` | optional, repeatable, up to 5: recordings of the same speaker talking as usual, such as their earlier turns, sent as files. They are the speaker baseline (`calibration_audio`); each is limited like `audio`. |

Empty fields count as absent. Without `words` or `transcript`, the server's
Whisper (`PP_STT_MODEL`) transcribes the audio if installed; otherwise the
text is `[speech]` placeholders. A `transcript` without timings adds little
without `calibration` (see [AudioToIML](#audiotoiml)). All fields are
checked before the job waits for a worker. Starlette limits a form field
sent as text to 1 MiB (413 `text_too_large`), so send long word lists as a
file. The `language` field takes the tag form only (`en_US` is a 422).

Response: `{"iml", "plain_text", "transcript_source": "words" |
"transcript" | "whisper" | "none", "warnings": [str], "profile_matches":
[{"utterance", "observed", "pattern", "emotion", "confidence",
"applied"}]}`; `warnings` as in [conversion warnings](#conversion-warnings).

Errors: 400 `audio_processing_error` (unreadable or too long audio or
calibration recording, naming the file; every calibration recording
silent), 400 `profile_error`, 413, 415 (not multipart), 422 (invalid `words`
or `transcript`, both given, bad `language`, `calibration` sent as text or
more than 5 of them), 500 `speech_recognition_failed`, 503 (`server_busy`,
`speech_recognition_unavailable`).

```bash
curl -F audio=@examples/monotone.wav -F words=@examples/monotone.deepgram.json -F profile=@examples/profile.json http://127.0.0.1:8000/v1/convert/audio-to-iml
```

```json
{
  "iml": "<iml version=\"0.1.0\"><utterance emotion=\"calm\" confidence=\"0.61\" x-profile=\"pitch_contour=flat\">I read the list.</utterance> <utterance emotion=\"calm\" confidence=\"0.62\" x-profile=\"pitch_contour=flat\"><pause duration=\"310\"/>The room is booked.</utterance> <utterance emotion=\"calm\" confidence=\"0.61\" x-profile=\"pitch_contour=flat\"><pause duration=\"300\"/>I have the slides.</utterance> <utterance emotion=\"joyful\" confidence=\"0.61\" x-profile=\"pitch_contour=flat rate=fast\"><pause duration=\"290\"/><prosody rate=\"165%\">And we got the grant!</prosody></utterance></iml>",
  "plain_text": "I read the list. The room is booked. I have the slides. And we got the grant!",
  "transcript_source": "words",
  "warnings": [],
  "profile_matches": [
    {
      "utterance": 0,
      "observed": {"pitch": "normal", "pitch_contour": "flat", "volume": "normal", "rate": "normal", "quality": "modal", "pause_frequency": "normal", "emphasis_frequency": "normal"},
      "pattern": {"pitch_contour": "flat"},
      "emotion": "calm",
      "confidence": 0.61,
      "applied": true
    }
  ]
}
```

(Formatted, and the other three matches left out.) Without any words:

```bash
curl -F audio=@examples/speech.wav http://127.0.0.1:8000/v1/convert/audio-to-iml
```

```json
{"iml":"<iml version=\"0.1.0\"><utterance><prosody pitch_contour=\"rise-fall\">[speech]</prosody></utterance> <utterance><pause duration=\"610\"/><prosody pitch_contour=\"rise-fall\">[speech]</prosody></utterance></iml>","plain_text":"[speech] [speech]","transcript_source":"none","warnings":["No transcript (openai-whisper is not installed): each stretch of speech is a '[speech]' placeholder. For real words, give word timings (words) or a transcript, or install 'prosody-protocol[whisper]'.","No speaker baseline: without calibration_audio, the speaker's usual pitch and loudness come from the recording, which needs at least 3 utterances, most of them at a similar level; this one has 2. Pitch, loudness and rate were marked only relative to one another, and no emotion was estimated from them. Pass calibration_audio (recordings of the speaker's usual speech, such as their earlier turns) to assess them."],"profile_matches":[]}
```

Against the speaker's earlier turns: here `turn.wav` is the last sentence
of `examples/monotone.wav` ("And we got the grant!") with its word timings
in `turn.json`, and `earlier1.wav` and `earlier2.wav` are the sentences
before it, cut from the same file (the
[quick start](quickstart.md#compare-with-the-speakers-earlier-turns) shows
how):

```bash
curl -F audio=@turn.wav -F words=@turn.json -F calibration=@earlier1.wav -F calibration=@earlier2.wav http://127.0.0.1:8000/v1/convert/audio-to-iml
```

```json
{"iml":"<iml version=\"0.1.0\"><utterance><prosody rate=\"180%\">And we got the grant!</prosody></utterance></iml>","plain_text":"And we got the grant!","transcript_source":"words","warnings":[],"profile_matches":[]}
```

Without the `calibration` fields the same request gives the plain sentence
and the "No speaker baseline" warning.

#### POST /v1/convert/text-to-iml

Request: `{"text": str, "context": str | null}`. Response: `{"iml",
"plain_text"}` (`TextToIML().predict`).

```bash
curl -H 'Content-Type: application/json' -d '{"text": "The app crashed again.", "context": "the user is frustrated"}' http://127.0.0.1:8000/v1/convert/text-to-iml
```

```json
{"iml":"<iml version=\"0.1.0\"><utterance emotion=\"frustrated\" confidence=\"0.5\">The app crashed again.</utterance></iml>","plain_text":"The app crashed again."}
```

#### POST /v1/convert/iml-to-ssml

Request: `{"iml": str, "strict": false}`. Response: `{"ssml": str}`. By
default invalid attribute values are ignored (spec 6.2), unlike the SDK's
`IMLToSSML`, which is strict by default; with `"strict": true` an invalid
document is a 400 `validation_error`. Malformed XML is a 400
`conversion_error`.

```bash
curl -H 'Content-Type: application/json' -d '{"iml": "<utterance>I <emphasis level=\"strong\">really</emphasis> need this <pause duration=\"500\"/> done today.</utterance>"}' http://127.0.0.1:8000/v1/convert/iml-to-ssml
```

```json
{"ssml":"<speak version=\"1.1\" xmlns=\"http://www.w3.org/2001/10/synthesis\" xml:lang=\"en-US\"><s>I <emphasis level=\"strong\">really</emphasis> need this <break time=\"500ms\"/> done today.</s></speak>"}
```

#### POST /v1/convert/iml-to-prompt

Request: `{"iml": str, "min_confidence": 0.5, "include_numbers": false,
"instruction": null, "strict": false}` (`min_confidence` outside 0-1 is a
422). Response: `{"context": str, "system_prompt": str, "messages":
[{"role", "content"}]}`: the output of `to_llm_context`, `SYSTEM_PROMPT` and
`build_messages`. Malformed XML is a 400 `iml_parse_error`; with
`"strict": true`, an invalid document is a 400 `validation_error`.

```bash
python -c 'import json; print(json.dumps({"iml": open("examples/sarcasm.iml").read(), "instruction": "Reply to the customer."}))' | curl -H 'Content-Type: application/json' -d @- http://127.0.0.1:8000/v1/convert/iml-to-prompt
```

```json
{
  "context": "customer: I've been on hold for {forty minutes} (slightly higher pitch, louder).\nDelivery: sounds frustrated (estimated, 82%).\ncustomer: Oh, that's just wonderful (higher pitch, slower, falling then rising). [pause 0.6s] Really **great** service.\nDelivery: sounds sarcastic (estimated, 74%).",
  "system_prompt": "Some of the input you receive is transcribed speech, given inside <transcript> tags. ...",
  "messages": [
    {"role": "system", "content": "Some of the input you receive is transcribed speech, given inside <transcript> tags. ..."},
    {"role": "user", "content": "<transcript>\ncustomer: I've been on hold for {forty minutes} (slightly higher pitch, louder).\nDelivery: sounds frustrated (estimated, 82%).\ncustomer: Oh, that's just wonderful (higher pitch, slower, falling then rising). [pause 0.6s] Really **great** service.\nDelivery: sounds sarcastic (estimated, 74%).\n</transcript>\n\nReply to the customer."}
  ]
}
```

(Formatted, with the system prompt shortened.)

#### POST /v1/synthesize

Request: `{"iml": str, "voice": null, "engine": "auto", "strict": false}`.
`voice` and `engine` are as for [`IMLToAudio`](#imltoaudio) (`engine` is
`"auto"`, `"espeak"` or `"tones"`; other values are a 422). Response: a WAV
file (`audio/wav`, mono, 16-bit), with the header `X-Prosody-Engine:
espeak` (speech) or `tones` (the preview, not speech). Errors: 400
`conversion_error` (unknown voice, `engine: "espeak"` without espeak-ng,
audio longer than `PP_MAX_SYNTH_SECONDS`, malformed IML), 400
`validation_error` (with `strict`), 503.

```bash
curl -H 'Content-Type: application/json' -d '{"iml": "<utterance>Please <prosody rate=\"slow\">listen carefully</prosody>.</utterance>"}' -D - -o output.wav http://127.0.0.1:8000/v1/synthesize
```

```text
HTTP/1.1 200 OK
...
content-disposition: attachment; filename=output.wav
x-prosody-engine: espeak
content-length: 82070
content-type: audio/wav
```
