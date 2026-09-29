# Changelog

All notable changes to the Prosody Protocol SDK are documented here.

This project follows [Semantic Versioning](https://semver.org/). No version
has been tagged on GitHub or published to PyPI yet; install from GitHub as
the [README](README.md#install) describes.

## [0.1.0a3] - 2026-09-28

This release makes the core installable without the audio stack, fixes how
the audio analysis measures pauses, pitch, loudness and tempo (checked on
synthetic espeak-ng speech), makes emotion labels abstain when unsure, adds
a command line, an LLM formatter and adapters for word timings from any
speech recognizer, measures each speaker against their own voice (or their
earlier turns), says in its warnings what it could not assess, scores
benchmarks without counting abstentions as "neutral", hardens the REST API,
and rewrites the documentation around examples that the test suite runs.
Many outputs change; read **Breaking changes** first.

### Breaking changes

- **The REST API moved** from the top-level `api/` package, which the wheel
  never included, to `prosody_protocol.server`. Run it with
  `prosody-protocol serve`, `python -m prosody_protocol.server` or
  `uvicorn prosody_protocol.server.app:app` instead of
  `uvicorn api.app:app`. It binds to 127.0.0.1 by default (was 0.0.0.0);
  the Docker image sets `PP_HOST=0.0.0.0`.
- **Extras were reworked.** `audio` is now `numpy` + `praat-parselmouth`
  only: `openai-whisper` (and with it PyTorch) moved to the new `whisper`
  extra. `api` and `ml` include `audio`, and `ml` adds `joblib`.
  `ml-neural` and `all-neural` are removed, `all` no longer includes `dev`,
  and `librosa`, `soundfile`, `torch` and `transformers` are no longer
  dependencies. A bare install needs only `lxml`.
- **Training configs were renamed** to `ser_logreg.yaml`,
  `text_prosody_tree.yaml` and `pitch_contour_forest.yaml` (from
  `ser_wav2vec2.yaml`, `text_to_prosody_bert.yaml` and
  `pitch_contour_cnn.yaml`); they always trained scikit-learn baselines.
  Checkpoints from earlier versions were trained on fabricated features
  and must be retrained. `SERModel(regularization=...)` is now `C=` (the
  config key `regularization` still works).
- **Emotion abstains by default.** `AudioToIML` and `IMLAssembler` leave out
  `emotion` and `confidence` below `min_emotion_confidence` (default 0.5).
  Without calibration audio, emotion is reported only for recordings with
  at least three utterances, most of them at the speaker's usual level, so
  single sentences carry no emotion. `RuleBasedEmotionClassifier` labels
  only `neutral`, `calm`, `sad`, `angry`, `joyful` and `fearful`; its
  baseline arguments default to `None`, and without a baseline `classify()`
  returns `("neutral", 0.0)`.
- **Measured values changed.** Pitch and volume in IML are relative to the
  speaker's baseline, not to each utterance's own mean.
  `SpanFeatures.jitter` and `.shimmer` are percentages, as spec Section 4.4
  defines them (they were fractions). `speech_rate` is `None` rather than
  0 when no syllable is found. Voice quality is judged relative to the
  speaker (`modal`, `breathy`, `creaky`, `harsh` or `None`; the analyzer no
  longer produces `tense` or `whispery`). `f0_contour` lists the voiced F0
  every 10 ms. Silence is judged against the speech around each frame, and
  all audio is analyzed as 16 kHz mono, so pauses, rates and levels change
  on some recordings, and on 44.1/48 kHz audio HNR comes out 1-3 dB higher
  and jitter and shimmer change a little.
- **The validator is stricter.** V9, V13, V14, V17 and V18 are errors (were
  warnings), and there are new error rules V19-V31 and warning rules V32
  and V33 (see Added). Numeric attributes must use ASCII digits, floats
  must be finite, and integers must be at most 2147483647. Documents that
  validated before may not.
- **The parser is stricter.** `IMLParser` raises `IMLParseError` for a
  DOCTYPE, a file that is not UTF-8, text or markup outside an utterance,
  nested utterances, and a `<pause>` with content, where it used to drop
  content silently. Comments and processing instructions are never text,
  and unknown elements are transparent (their content is kept).
- **`IMLParser.to_plain_text` normalizes whitespace:** indentation is
  collapsed, utterances are joined by one space, and there is no space
  before punctuation that follows markup or a `<pause/>`.
- **Speech output.** `IMLToAudio()` defaults to `engine="auto"` (espeak-ng
  speech, or a tone preview with a warning when espeak-ng is missing;
  was sine tones) and `voice=None` (follows the document's language; was
  `"en_US-female-medium"`). `"coqui"`, `"piper"` and `"elevenlabs"` raise
  `ConversionError` instead of silently producing tones. `IMLToAudio` and
  `IMLToSSML` validate their input (`strict=True`) and raise
  `IMLValidationError` for invalid IML. `IMLToSSML` writes SSML 1.1 with
  `version`, `xmlns` and `xml:lang`.
- **`TextToIML`** always returns an `<iml version="0.1.0">` document, drops
  labels below `min_confidence` (0.5), and treats `default_confidence` as
  an opt-in floor (default `None`, was 0.6). `!` alone no longer means
  `joyful`.
- **`AudioToIML`** no longer labels every document `en-US`, and raises
  `ValueError` for a `language` that is not a BCP 47 tag (`en_US` is
  accepted as `en-US`). Without words, a transcript or Whisper it emits
  `[speech]` placeholders and says so in its warnings. Whisper load or
  transcription failures raise `SpeechRecognitionError`, a subclass of
  `AudioProcessingError`, instead of falling back silently. Utterances also
  split where the speaker label changes and, in unpunctuated text, at
  pauses of 0.5 s (`UTTERANCE_SPLIT_PAUSE_MS`, was 1 s). A single utterance
  without calibration audio gets no utterance-level pitch, loudness or rate
  and a "No speaker baseline" warning; `convert()` issues these notes as
  `UserWarning`s. `include_extended` adds measurements without changing the
  markup or the text.
- **Datasets and benchmarks.** `DatasetLoader` validates every entry when
  loading and raises `DatasetError` listing the problems
  (`DatasetLoader(strict=False)` skips invalid entries instead); a
  language that is not BCP 47 (D6) is an error; `split()` keeps speakers
  disjoint by default (`group_by="speaker_id"`). `Benchmark` always runs
  the converter (the ground-truth-only mode is removed), `dataset_dir`
  defaults to the directory the dataset was loaded from, several
  `BenchmarkReport` metrics can be `None`, and `check_regression()` fails on
  any failed conversion unless a `failure_rate` threshold allows it.
- **Benchmarks count a missing emotion as an abstention**, no longer as
  the label `neutral`: `emotion_accuracy` is over the entries whose output
  has an emotion (`None` when none has), the new `emotion_coverage` is
  their share, and the per-class F1 counts an abstention as a miss (`emotion_f1_macro` may be `None`). An
  `emotion_accuracy` threshold without an `emotion_coverage` threshold
  counts abstentions as wrong. `Benchmark(abstention_label="neutral")` and
  `benchmark --abstention-label neutral` restore the old scoring. Reports
  saved before load with `abstention_label="neutral"`, and comparing a run
  with one scored the other way is a failure. `Benchmark` now gives
  converters that take them each entry's word timings or transcript
  (`words_from="auto"`), so scores change.
- **Serialized IML.** `IMLParser.to_iml_string` separates the utterances
  inside `<iml>` with a space (`</utterance> <utterance>`), writes parsed
  numbers as they were written, and raises `IMLValidationError` rather than
  write invalid numbers from models built in code.
- **LLM context.** `to_llm_context` merges utterance-level prosody into one
  `Delivery: overall ...` line, accumulates nested offsets, and gives no
  note to the ordinary fall of a statement or rise of a question ("my money
  (falling)." is now "my money."). Speaker-like text in an utterance
  without a speaker is quoted.
- **Profiles and Mavis.** `ProfileLoader` rejects unknown keys and treats
  P5 and P6 as errors. `MavisBridge.phoneme_events_to_entry` records the
  caller's `consent` (default `False`; it was always `True`), and
  `export_dataset` refuses sessions without stated consent. A given
  `emotion_label` is written with `confidence="1.0"`; a guessed one stays in
  the entry's `emotion_label` and is no longer written into the IML with a
  pseudo-confidence from the session's pitch and volume range. Zero offsets
  (`+0%`, `+0dB`) are left out.
- **REST API.** Errors are JSON `{"error": ..., "detail": ...}`, plus
  `issues` for validation errors; that includes 404 (`not_found`), 405
  (`method_not_allowed`), unparseable bodies (400 `invalid_body`) and
  schema errors (422 `invalid_request`, whose `detail` is FastAPI's list).
  `/v1/synthesize` picks a voice for the document's language by default.
  `/v1/convert/audio-to-iml` returns 415 for requests that are not
  multipart, 400 for unreadable audio (was 500), and 500
  `speech_recognition_failed` or 503 `speech_recognition_unavailable` when
  the server's Whisper fails. An empty `PP_HOST` binds 127.0.0.1 (it bound
  every interface).

### Added

- **Command line** `prosody-protocol` with `validate`, `to-text`, `to-ssml`,
  `to-prompt`, `from-text`, `from-audio`, `synthesize`, `benchmark`, `serve`
  and `doctor` (which reports the installed extras and system tools). `-`
  reads stdin. Exit status is 0 on success, 1 for rejected input, 2 for
  usage and I/O errors or a missing extra, which prints a one-line install
  hint. `python -m prosody_protocol` runs the same command where the
  console script is not on `PATH`.
- **`prosody_protocol.llm`:** `to_llm_context()` renders IML as an annotated
  transcript (`*emphasis*`, prosody described in words, `[pause 0.8s]`, a
  `Delivery:` line per utterance that names an emotion only at confidence
  0.5 or above, and always as an estimate). An emotion that a prosody
  profile set (the utterance carries `x-profile`) is marked "interpreted
  with the speaker's prosody profile", as spec Section 7.2 asks.
  `build_messages()` returns chat messages for any chat API, and
  `SYSTEM_PROMPT` explains the notation, tells the model to prefer a
  profile's reading, and cautions against over-trusting prosodic cues and
  against the uses spec Section 8.2 prohibits.
- **`prosody_protocol.alignment`:** adapters for word timings from
  openai-whisper, faster-whisper, WhisperX, the OpenAI transcription API
  (`verbose_json`), Deepgram, AssemblyAI, Google Cloud Speech-to-Text and
  generic records. `load_word_timings()` reads a file or parsed data and
  detects its shape; `parse_word_timings()` parses JSON text and never
  reads files.
- **`AudioToIML`:** `words=` (timings from any recognizer) and `transcript=`
  (text without timings) on `convert()` and `convert_to_doc()`;
  `convert_detailed()` returning `ConversionResult` (`document`, `iml`,
  `transcript_source`, `warnings`, `profile_matches`); and the options
  `stt=`, `calibration_audio=` (recordings of the speaker as the baseline),
  `min_emotion_confidence=`, `max_duration_s=` and `profile=`.
- **Several speakers.** `WordAlignment` has a `speaker` label, which the
  adapters keep from Deepgram (`speaker`), AssemblyAI (`speaker`), Google
  (`speakerTag` in v1, 0 meaning none; `speakerLabel` in v2), WhisperX (the
  word's or its segment's `speaker`) and records (`from_records(...,
  speaker_key="speaker")`). The assembler starts a new utterance where the
  speaker changes, writes the label as `speaker_id`, and measures each
  speaker against their own baseline. Speakers may talk over each other (at
  most `audio_to_iml.MAX_OVERLAPPING_SPEAKERS`, 3, at once). Without labels,
  utterances whose pitch falls into two groups more than 7 semitones apart
  (`assembler.VOICE_SEPARATION_ST`) get no baseline, emotion or profile.
- **Multi-turn calibration.** `calibration_audio` takes one recording or a
  sequence of them, such as a user's earlier turns. The attribute can be
  reassigned between conversions; each file is analyzed once and cached
  (path, modification time and size). A file without voiced speech is
  skipped with a warning; if no file has any, `AudioProcessingError`.
- **Conversion warnings** in `ConversionResult.warnings`: no speaker
  baseline (worded for the classifier in use), two voices without speaker
  labels, calibration audio or a profile not used with several speakers,
  silent calibration files, a transcript without timings (only the whole
  utterance measured), words without sentence punctuation, and word
  timings that do not match the audio (words ending more than 250 ms after
  it, at least 10% of the words over unvoiced audio, or more than half of
  the voiced speech outside the words).
- `SpeechRecognitionError` (a subclass of `AudioProcessingError`, exported
  from the package root) for a Whisper model that cannot be loaded or a
  failed transcription.
- **Prosody profiles in the audio pipeline** (spec Section 7.2):
  `AudioToIML(profile=...)` and `IMLAssembler(profile=...)` apply matching
  mappings, mark affected utterances with `x-profile="<matched pattern>"`
  and report each match as a `ProfileMatch` (exported from the package
  root). `profiles.categorize_features()` turns measured features into the
  categories profiles match on, and `ProfileApplier.match()` returns the
  mapping that applies (the most specific; the first wins a tie).
- **`IMLAssembler`** writes every silence of 200 ms or more between words as
  a `<pause>`, and emits `pitch_contour`, utterance-level `rate` for clear
  tempo changes, and `quality` when a voice is unusual for the speaker.
  `include_extended=True` adds measurements in spec units to the words
  (except an emphasized word inside an utterance-level `<prosody>`, which
  stays two levels deep).
- `SpeakerBaseline`, the `BaselineAwareEmotionClassifier` protocol and
  `RuleBasedEmotionClassifier.classify_relative()`.
- `ProsodyAnalyzer.analyze_recording()`, `ProsodyAnalyzer(max_duration_s=)`
  and `detect_pauses(silence_threshold_db=)`. OGG/Opus, WebM, M4A and other
  formats are decoded through ffmpeg when it is installed.
- `IMLToAudio` speaks with espeak-ng, adapting IML's pitch, volume, rate,
  pauses, emphasis and contours to what espeak-ng renders;
  `engine="tones"` gives a prosody preview; `max_duration_s` caps output.
- `IMLToSSML`: `pitch_contour` becomes an SSML contour, segment `tempo`
  becomes `rate`, `speaker_voices` maps speakers to voices, and
  `vendor="espeak-ng"` writes SSML adapted to espeak-ng.
- `TextToIML` recognizes sarcasm idioms ("yeah right") and deadpan frames
  ("Oh great, another meeting."), and uses cue words in `context`.
- Validator rules V19 (`<iml>` contains only utterances), V20 (no nested
  utterance or iml), V21 (no emphasis directly in emphasis), V22-V26
  (`rate`, `pitch_contour`, `quality`, `tempo`, `rhythm` values), V27
  (extended attribute types), V28 (`version` is SemVer), V29 (`language` is
  BCP 47), V30 (UTF-8), V31 (no DOCTYPE), and the warnings V32 (unknown
  unprefixed or IML-namespaced attributes) and V33 (values implausible for
  a human voice, spec Section 6.4). `ValidationResult.errors`, `.warnings`
  and `.raise_for_errors()`; `IMLValidationError.issues`.
- Spec Sections 2.5 (XML features), 2.6 (attribute value syntax) and 6.4
  (plausible values); requirements M19-M28, S12-S17 and O11-O14.
- `extra_attributes` on every model, so `to_iml_string()` keeps `x-`,
  unknown and namespaced attributes and invalid values.
- `datasets.resolve_audio_path()`, `Dataset.root`,
  `DatasetLoader.load(check_audio=True)`.
- `BenchmarkReport.emotion_f1_macro`, `failure_rate`, `pitch_coverage` and
  `num_entries`; `check_regression(tolerance=)` with per-class F1 checks.
- `Benchmark(words_from=, abstention_label=)`: `"auto"` gives a converter
  that takes them each entry's word timings (`metadata.word_timings`, any
  format `parse_word_timings` reads), else nothing when it can recognize
  speech itself (Whisper installed and `stt` not `"none"`), else the
  transcript; `"timings"`, `"transcript"` and `"stt"` force a source.
  `BenchmarkReport.emotion_coverage`, `num_unaligned` (outputs that are only
  `[speech]` placeholders, left out of the pause and pitch metrics),
  `word_sources` and `abstention_label`; `check_regression(class_tolerance=)`,
  an `emotion_coverage` threshold, and a failure for a metric the baseline
  measured but a run could not. `prosody-protocol benchmark --words-from`
  and `--abstention-label`, and a summary that says where the words came
  from. A committed report for the `training_synthetic` fixture
  (`tests/fixtures/benchmarks/training_synthetic.json`, regenerated by
  `make_baselines.py` there) and a test that fails when a run regresses
  from it.
- `DatasetLoader.split(stratify_by=...)` (e.g. `"emotion_label"`), which
  also works with speaker grouping, and a warning when a label in the
  training split is missing from a non-empty validation or test split.
- `MavisBridge.events_from_entry()`; `export_dataset` can copy session audio
  and accepts numpy event values.
- REST: `POST /v1/convert/iml-to-prompt`; `words`, `transcript` and
  `profile` fields and `transcript_source`, `warnings` and `profile_matches`
  in the response of `/v1/convert/audio-to-iml`; `language` as a form
  field; `engine` and `strict` on `/v1/synthesize` (with an
  `X-Prosody-Engine` response header) and `strict` on
  `/v1/convert/iml-to-ssml`; backends and limits in `/v1/health`;
  `create_app(settings)`; the settings `PP_TRUSTED_PROXIES`,
  `PP_MAX_TEXT_CHARS`, `PP_MAX_WORDS_CHARS`, `PP_MAX_SYNTH_SECONDS`,
  `PP_MAX_AUDIO_SECONDS`, `PP_MAX_CONCURRENT_JOBS`, `PP_MAX_QUEUED_JOBS`,
  `PP_MAX_JSON_BYTES` (largest non-multipart body, default 2465536 bytes:
  24 x `PP_MAX_TEXT_CHARS` + 65536; `max_json_bytes` in `/v1/health`) and
  `PP_STT_MODEL` (the Whisper model name or checkpoint path, default
  `base`); a repeatable `calibration` file field on
  `/v1/convert/audio-to-iml` (up to 5 recordings of the speaker, such as
  earlier turns).
- Training: `export.py --format json` writes a pickle-free `model.json` that
  `training.portable.PortableModel` runs with numpy alone;
  `training.inference.TrainedEmotionClassifier` plugs a trained model into
  `AudioToIML(emotion_classifier=...)`, and abstains on input more than
  `max_feature_z` (default 4) training standard deviations from the
  training data (exports record `feature_stats`);
  `unusual_features()` names such features; `training/README.md`.
- Spec 3.2 defines reference levels: a top-level `<prosody>` is relative to
  the speaker's baseline, a nested one to the `<prosody>` around it, so
  offsets accumulate (dB and semitones add, percentages multiply).
  `SYSTEM_PROMPT` explains this, and `[speech]` placeholders; an utterance
  of only placeholders gets `Delivery: words not transcribed.`
- `prosody_analyzer.MAX_SAMPLE_RATE_HZ` (16000), and the test fixture
  `speech_levels.wav` (one sentence at several levels, with known pauses)
  with its generator `tests/generate_speech_levels_fixture.py`.
- `examples/`: a recording with Whisper-format word timings, a monotone
  speaker with Deepgram-format timings and a matching prosody profile, and a
  sarcastic IML document.
- `tests/test_docs_examples.py` runs the Python, IML and CLI examples of
  README.md, examples/README.md and docs/. CI gained a core-only job and a
  wheel smoke test.
- `PP_JOB_TIMEOUT_S` (default 900): a worker that spends longer on one audio conversion or synthesis is stopped and the request gets a 504 `job_timeout`; `/v1/health` reports the limit.

### Changed

- `import prosody_protocol` needs only lxml. Classes that need numpy or
  parselmouth load on first use and raise `ImportError` naming the extra.
- The rule-based emotion classifier judges an utterance by its deviation
  from a speaker baseline (pitch in semitones, loudness in dB, rate, pitch
  movement), so labels no longer change with recording gain or the
  speaker's natural pitch; confidence stays low when evidence is weak.
- Pitch is tracked in a range fitted to each recording (within 40-1200 Hz)
  instead of a fixed 75-600 Hz, keeping shouts, creak and steep rises.
- `ProsodyAnalyzer` analyzes each file once, in time linear in its length.
- The REST API runs audio conversion and synthesis in worker processes, so
  a long job no longer blocks other requests.
- Training configs are validated: hyperparameters are honored, and unused
  or unknown keys warn.
- The `consent` vocabulary adds `none`, and `processing` adds `hybrid`.
- The `training_synthetic` test dataset is 10 espeak-ng clips whose
  delivery follows their label (was 10 identical tones).
- Silence is judged against the speech around each frame: the loudest
  voiced frame within 0.25 s, capped at the file's 99th percentile, so a
  quieter passage is judged against itself. `silence_threshold_db` also
  sets how far below the file's level voiced sound still counts as speech
  (the threshold plus 5 dB).
- All audio is analyzed as 16 kHz mono: channels are averaged, and faster
  audio is read in blocks and resampled, so memory and time grow with the
  duration only (a 10-minute 192 kHz FLAC: 21 s and 293 MB, was 222 s and
  2.7 GB). A server job needs about 450 MB for 10 minutes of audio.
- One language-tag rule (a 1-8 letter primary subtag, then subtags of 1-8
  letters or digits; the form only, not the IANA registry) for validator
  rule V29, dataset rule D6, `dataset-entry.schema.json`, `IMLToSSML`,
  `AudioToIML`, `IMLAssembler`, `MavisBridge` and the CLI. SDK and CLI
  arguments read `en_US` as `en-US` (`IMLToSSML(default_language=)` too);
  documents and dataset entries are checked as written.
- Install hints in errors and `prosody-protocol doctor` name the GitHub
  install (`pip install "prosody-protocol[audio] @ git+https://github.com/kase1111-hash/Prosody-Protocol"`).
- The REST worker pool runs one process per worker: a worker that dies
  fails only the job it was running, and queued jobs run on a new one.
  `python -m prosody_protocol.server` takes the CLI's `--host` and `--port`
  and reports errors on one line, as `prosody-protocol serve` does.
- `to_llm_context`, `TextToIML` and `parse_word_timings` run in time linear
  in their input (a megabyte in seconds). `IMLToSSML` writes `+0dB` for
  `-0dB`.
- CI lints and type-checks on Python 3.10; the publish workflow runs the
  test suite against the built wheel before uploading. The docs harness
  skips an example only when the extra it needs is really missing, and
  runs only the `$ `-prompted lines of console blocks.
- README, CLAUDE.md, CONTRIBUTING.md, datasets/README.md and the docs were
  rewritten against the current code, with real output.

### Fixed

- Audio analysis: pauses are found in real recordings with a noise floor
  (silence was only detected in exact digital silence); digital silence no
  longer distorts intensity (values like `volume="-83dB"`); shimmer is no
  longer inflated by a window taper; speech rate no longer depends on
  voice pitch; F0 octave errors and missing F0 in low voices; MP3 timing
  about 60 ms late; truncated WAV files padded with silence; the Whisper
  model reloaded on every conversion.
- Every audio read or analysis failure raises `AudioProcessingError`
  (unreadable, empty, too short, sampled below 4 kHz, non-finite samples),
  instead of raw Praat or index errors.
- IML assembly: words glued together next to tags; detected pauses
  discarded; utterances split mid-sentence at long pauses or at "Dr." and
  "U.S."; emphasis marked on words with lowered pitch; a shouted sentence
  measured against itself and left unmarked; `include_extended` adding
  nothing; implausible values in the output.
- Parser and validator: crashes (`AttributeError`) on processing
  instructions and entity references, which gave a 500 from
  `/v1/validate`; comments spoken by synthesis or leaked into plain text;
  lossy `to_iml_string()` (it dropped `consent`/`processing` and custom
  attributes); malformed XML from `to_iml_string()` for control characters;
  quadratic parsing; crashes on numbers with thousands of digits.
- Generators: `TextToIML` moving quotes, merging text, splitting at
  abbreviations and taking seconds to minutes on some 100 KB inputs;
  `IMLToSSML` declaring SSML 1.0 while writing 1.1 values, and leaving out
  `xml:lang`; clipping, spoken
  punctuation and louder reduced emphasis in the tone preview; crashes on
  extreme values such as `volume="+7000dB"` (HTTP 500 from the API).
- Benchmarks counted only successful conversions, pooled pauses across
  entries and reported vacuous scores (1.0 or 0.0) for metrics with nothing
  to compare; `compute_ece` accepted confidences outside [0, 1].
- Training data preparation fabricated features from the labels, so every
  model learned a lookup table and reported fake accuracy; macro-F1
  averaged in absent classes; pitch-contour preparation was not
  reproducible; `evaluate.py` without `--config` crashed.
- `MavisBridge` dropped phoneme events and could write malformed IML. A
  transcript with a character XML does not allow (a terminal escape, say)
  failed the whole session with `ConversionError`; such characters are now
  dropped with a warning. An `emotion_label` containing one raises
  `DatasetError`.
- `ProfileLoader.load` accepts a UTF-8 byte order mark, and raises
  `ProfileError` instead of `UnicodeDecodeError`, `RecursionError`,
  `ValueError` or `OverflowError` for a file that is not UTF-8, JSON nested
  too deeply, and integers too long to convert.
- REST: a JSON body with `NaN` or `Infinity` gave a 500; the `language` form
  field was ignored; unreadable audio gave a 500.
- The Docker image could not start (the `[api]` extra did not install the
  packages the SDK imported); it now installs `.[api]` with espeak-ng and
  ffmpeg and honors `PP_HOST` and `PP_PORT`.
- Quieter speech got false pauses, an inflated speech rate or a wrong level
  when louder speech occurred elsewhere in the file.
- The rule-based classifier's pitch-spread cue depended on utterance length,
  which gave false `joyful` labels when calibration audio was combined with
  `transcript=`.
- `AudioToIML` clamps word timings past the end of the audio, and never
  returns invalid IML (`ConversionError`); `IMLAssembler` and `MavisBridge`
  wrote any `language` they were given, even one that is not a tag.
- Dataset rule D6 required a 2-3 letter primary subtag, so entries whose
  IML validated could be rejected. Dataset files with JSON nested too
  deeply or integers too long to convert raised raw errors, not
  `DatasetError`.
- `from prosody_protocol import *` failed on the core install.
- Pause matching in `Benchmark` hit `RecursionError` on long recordings.
- REST: a text form field over 1 MiB gave a 400; it is a 413
  `text_too_large` that says to send the field as a file.
- Earlier fixes after 0.1.0a2 (February 2026): `ProsodyProfile.mappings` is
  a tuple, matching the frozen dataclass; `MavisBridge` converts volume to
  dB with 20·log10; `BenchmarkReport` gained `num_failures`.
- A client that disconnects no longer leaves its audio conversion or synthesis running: a waiting job is dropped and the worker running a started one is stopped.
- Several silences at one word boundary now add up; the speech between them (an untranscribed "um") is no longer counted as pause.

### Security

- IML documents must not contain a DOCTYPE (spec Section 2.5); the parser
  rejects one, so entity expansion and external entities cannot be used.
- ffmpeg opens only local files in audio container formats, with a
  timeout; an uploaded HLS or concat playlist could make the server read
  other audio files on the machine.
- Audio length limits (`max_duration_s`, `PP_MAX_AUDIO_SECONDS`) are checked
  before full decoding, so a small compressed upload cannot expand into
  hours of audio.
- `PP_MAX_UPLOAD_MB` is enforced on the bytes received, including chunked
  uploads; text fields are capped by `PP_MAX_TEXT_CHARS` and word timings by
  `PP_MAX_WORDS_CHARS`; one speaker's words may overlap by at most 500 ms,
  and at most three speakers' words at once; at most `PP_MAX_QUEUED_JOBS`
  jobs wait for a worker (503 after that).
- `PP_MAX_JSON_BYTES` limits bodies that are not multipart before they are
  parsed, counted as the bytes arrive: a 48 MB JSON body could cost
  gigabytes of memory and stall `/v1/health`.
- A set but empty `PP_HOST` (as a compose file writes for an unset
  variable) bound the server to every interface; it now means 127.0.0.1.
- Install hints named `pip install prosody-protocol[...]`, a PyPI name
  nobody has registered: it fails today and could later install someone
  else's package. They point at the GitHub repository.
- The rate limiter keys on the connecting address and trusts
  `X-Forwarded-For` only from `PP_TRUSTED_PROXIES`.
- Dataset `audio_file` paths must be relative and stay inside the dataset.
- Training checkpoints (`model.joblib`) are documented as pickles that run
  code when loaded; the silent `model.pkl` fallback is removed and prepared
  data loads with `allow_pickle=False`. See the correction under 0.1.0a2.
- Text in an IML document cannot close the `<transcript>` block that
  `build_messages()` writes, and a prosody profile's `user_id` is never
  written into IML. Words in an utterance without a speaker that look like
  another speaker's line are quoted, so they cannot impersonate a
  speaker.

### Deprecated

- `IMLToAudio(engine="builtin")`: use `engine="tones"`.
- `prosody_protocol.assembler.F0_DEVIATION_PCT`, `INTENSITY_DEVIATION_DB`,
  `EMPHASIS_INTENSITY_DB` and `EMPHASIS_F0_PCT` no longer affect output and
  warn when accessed.

### Removed

- The top-level `api/` package (now `prosody_protocol.server`).
- The `ml-neural` and `all-neural` extras.
- The ground-truth-only mode of `Benchmark`.
- The `model.pkl` checkpoint fallback and `train.prepare_and_load()`.
- The training configs `ser_wav2vec2.yaml`, `text_to_prosody_bert.yaml` and
  `pitch_contour_cnn.yaml` (renamed, see Breaking changes).

## [0.1.0a2] - 2026-02-06

### Added

#### Mavis Data Bridge (Phase 16)
- `MavisBridge` for converting Mavis PhonemeEvent streams to prosody_protocol datasets
- `PhonemeEvent` dataclass mirroring Mavis data types
- Feature extraction (7-dimensional) for sklearn training pipelines
- Batch feature extraction and dataset export
- Mavis integration guide (`docs/integrations/mavis.md`)

### Changed

#### Security Hardening (Phase 14)
- XML parser: explicit XXE prevention with `resolve_entities=False, no_network=True`
- Training model serialization: replaced `pickle` with `joblib` (safer deserialization)
  - **Correction (added in 0.1.0a3):** this claim was wrong. A joblib file is a
    pickle, and loading one runs code stored in it, so joblib is not safer than
    pickle. Only load checkpoints you trust. Since 0.1.0a3,
    `training/scripts/export.py --format json` writes a pickle-free `model.json`.
- API upload size enforcement via `UploadSizeLimitMiddleware` (configurable, default 50MB)
- Validator V17/V18: consent and processing attribute validation
- Validator handles non-element nodes (entities, PIs) gracefully
  - **Correction (added in 0.1.0a3):** in 0.1.0a2 the parser and validator still
    raised `AttributeError` on processing instructions and entity references
    (and `/v1/validate` returned 500). Fixed in 0.1.0a3.

#### Dependency & Spec Cleanup (Phase 15)
- Moved `torch` and `transformers` to new `ml-neural` extra (not required for sklearn baselines)
- `ml` extra now only requires `scikit-learn` and `pyyaml`
- Promoted `<segment>` tag to stable status in spec.md and CLAUDE.md

#### API Production Readiness (Phase 17)
- Rate limiting middleware (configurable via `PP_RATE_LIMIT` env var, default 60 req/min)
- CORS restricted to configured origins only (no wildcard default)
- Dockerfile runs as non-root user, includes `api/` directory, env var configuration
  - **Correction (added in 0.1.0a3):** the 0.1.0a2 image could not start: `.[api]`
    did not install numpy or praat-parselmouth, which the package imported at
    startup. Fixed in 0.1.0a3.
- `api/config.py` reads all settings from environment variables

#### Testing & CI (Phase 18)
- Added `test-ml-extras` CI job for sklearn training pipeline
- Added `test-api-extras` CI job for FastAPI endpoints
- Total test count: 526 (from 491)

### Infrastructure
- `pyproject.toml`: new `ml-neural` and `all-neural` extras groups

## [0.1.0a1] - 2026-02-05

### Added

#### Core SDK (Phases 1-5)
- `IMLParser` for parsing IML XML into structured `IMLDocument` objects
- `IMLValidator` with 16 validation rules (V1-V16) covering well-formedness, semantics, and spec compliance
- `IMLAssembler` for programmatic IML document construction
- `ProsodyAnalyzer` for extracting acoustic features (F0, intensity, jitter, shimmer, HNR) via Praat
- `AudioToIML` converter with Whisper STT integration and prosodic feature annotation
- `IMLToAudio` synthesizer producing WAV audio from IML markup
- `IMLToSSML` converter for mapping IML to SSML for TTS engines
- Data models: `IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment`
- `EmotionClassifier` protocol and `RuleBasedEmotionClassifier` baseline

#### Text-to-IML Prediction (Phase 6)
- `TextToIML` rule-based prosody predictor from plain text
- Sentiment lexicon with emotion detection from punctuation, capitalization, and word lists

#### Prosody Profiles (Phase 8)
- `ProfileLoader` for loading and validating JSON prosody profiles
- `ProfileApplier` for applying atypical prosody mappings (accessibility feature)
- Profile validation rules P1-P8

#### REST API (Phase 9)
- FastAPI server with 6 endpoints: audio-to-iml, text-to-iml, iml-to-ssml, synthesize, validate, health
- Error handling middleware for all SDK exception types
- Auto-generated Swagger/OpenAPI docs at `/docs`

#### Dataset Infrastructure (Phase 10)
- `DatasetLoader` with directory loading, entry validation (D1-D8), and lazy iteration
- `DatasetEntry` and `Dataset` dataclasses
- Deterministic train/val/test splitting with seeded randomization
- JSON Schema for dataset entries (`schemas/dataset-entry.schema.json`)

#### Model Training Pipelines (Phase 11)
- Config-driven training with YAML configuration files
- `ModelRegistry` with pluggable model architecture system
- Baseline models: logistic regression (SER), decision tree (text-to-prosody), random forest (pitch contour)
- Unified `train.py`, `evaluate.py`, `export.py`, and `data_prep.py` scripts
- Per-class precision/recall/F1 evaluation metrics

#### Evaluation & Benchmarks (Phase 12)
- `Benchmark` harness for evaluating AudioToIML converters against labelled datasets
- `BenchmarkReport` with 7 metrics: emotion accuracy, per-class F1, confidence ECE, pitch accuracy, pause F1, validity rate
- JSON report persistence for tracking metrics over time
- Regression detection for CI integration with baseline comparison and threshold checks
  - **Correction (added in 0.1.0a3):** the report had the six metrics listed; the
    seventh planned in EXECUTION_GUIDE.md, round-trip (synthesis and
    re-analysis) fidelity, was never implemented, and there is still no such
    metric. `check_regression()` existed, but no CI job ran a benchmark or
    compared one with a baseline. Since 0.1.0a3 a test compares a benchmark of
    the `training_synthetic` fixture with a committed report
    (`tests/fixtures/benchmarks/training_synthetic.json`).

#### Documentation & Adoption (Phase 13)
- Quick Start Guide (`docs/quickstart.md`)
- API Reference (`docs/API.md`)
- Integration guides for Whisper, Claude, ElevenLabs, and Coqui TTS
- Contributing guide (`CONTRIBUTING.md`)
- Dockerfile for API server deployment
- GitHub Actions CI and PyPI publish workflows
- PEP 561 `py.typed` marker

### Infrastructure
- `pyproject.toml` with optional dependency groups: `audio`, `ml`, `api`, `dev`
- Comprehensive test suite (491+ tests)
- IML test fixtures in `tests/fixtures/`
