# CLAUDE.md

Project guidance for Claude Code when working on the Prosody Protocol repository.

## Project Overview

Prosody Protocol defines **Intent Markup Language (IML)** -- an XML-based markup format that preserves prosodic information (pitch, volume, tempo, voice quality, rhythm) alongside transcribed text. The goal is to prevent the loss of emotional intent during speech-to-text conversion so that downstream AI systems can distinguish sarcasm from sincerity, urgency from calm, etc.

**Current status:** alpha. The spec (`spec.md`) is draft 0.1.0-alpha; the Python SDK `prosody_protocol` is 0.1.0a3 (`src/prosody_protocol/_version.py`) and ships a CLI (`prosody-protocol`) and a REST API. It is not on PyPI yet; users install from GitHub or a clone. Emotion labels come from a rule-based heuristic that abstains when unsure, and audio analysis has been validated on synthetic espeak-ng speech only. Docs must say so (see README "What is real and what is heuristic").

## Repository Structure

```
Prosody-Protocol/
├── spec.md                  # IML specification (normative)
├── schemas/                 # iml-1.0.xsd, prosody-profile.schema.json, dataset-entry.schema.json
├── src/prosody_protocol/    # the SDK
│   ├── __init__.py          # public API; lazy-loads the modules that need extras
│   ├── __main__.py          # python -m prosody_protocol (the CLI)
│   ├── parser.py, validator.py, models.py, exceptions.py, _types.py
│   ├── audio_to_iml.py, prosody_analyzer.py    # recordings -> IML (audio extra)
│   ├── assembler.py, emotion_classifier.py     # markup and emotion rules
│   ├── alignment.py         # word timings from Whisper, Deepgram, AssemblyAI, Google, ...
│   ├── llm.py               # to_llm_context, build_messages, SYSTEM_PROMPT
│   ├── iml_to_ssml.py, iml_to_audio.py, text_to_iml.py
│   ├── profiles.py, datasets.py, benchmarks.py, mavis_bridge.py
│   ├── cli.py               # prosody-protocol command
│   └── server/              # FastAPI app (app.py, config.py, routes/, jobs.py, middleware.py)
├── examples/                # speech.wav + word timings, monotone.wav + profile.json, sarcasm.iml
├── training/                # scikit-learn baselines, configs/, scripts/ (source checkout only)
├── tests/                   # pytest; fixtures/ (valid/, invalid/, audio/, datasets/, profiles/, benchmarks/)
├── docs/                    # API.md, cli.md, quickstart.md, integrations/ (speech-to-text, whisper, claude, TTS, mavis)
├── datasets/README.md       # dataset format (no corpora are shipped)
├── Dockerfile               # REST API image
├── EXECUTION_GUIDE.md       # historical build plan, not current usage
└── README.md, CHANGELOG.md, CONTRIBUTING.md, LICENSE
```

## Key Concepts

- **IML:** the XML format. `spec.md` is authoritative; Appendix A (content models) and Appendix D (MUST/SHOULD/MAY table) are normative summaries.
- **Core tags:** `<iml>` (optional wrapper), `<utterance>`, `<prosody>`, `<pause>`, `<emphasis>`, `<segment>` (all stable).
- **Extended attributes:** `f0_mean`, `f0_range`, `f0_contour`, `intensity_mean`, `intensity_range`, `speech_rate`, `duration_ms`, `jitter` and `shimmer` (percent), `hnr` -- experimental, for research use.
- **Speaker baseline:** pitch/volume/rate in IML are relative to the speaker; a nested `<prosody>` is relative to the enclosing one, so offsets accumulate (dB/st add, percentages multiply; spec 3.2, "Reference level"). Each speaker (`WordAlignment.speaker` labels) has their own baseline: `calibration_audio` (one file or several, e.g. the user's earlier turns; single-speaker only) or at least 3 of their typical utterances. Without one: no utterance-level offsets, no emotion, and a "No speaker baseline" warning. Unlabeled utterances whose pitch splits into two groups > 7 st apart get no baseline.
- **Warnings:** `ConversionResult.warnings` (and `UserWarning`s from `convert()`) say what degraded the output: no baseline, placeholders, transcript without timings, unpunctuated words, word timings that do not match the audio, calibration/profile unused with several speakers.
- **Abstention:** utterances below `min_emotion_confidence` (0.5) carry no `emotion`/`confidence`. The rule-based classifier never reports `neutral` or `calm` at 0.5 or above.
- **Prosody profiles:** JSON documents (spec Section 7) that map a speaker's atypical patterns to intended meanings. Profile use is reported downstream (spec 7.2): `x-profile="<matched pattern>"` on the utterance, `ConversionResult.profile_matches`, and "interpreted with the speaker's prosody profile" in `to_llm_context`. The profile's `user_id` is never written into IML.
- **Word timings:** the SDK measures how words were said; the words come from `words=` (any STT, via `prosody_protocol.alignment`; speaker labels kept), `transcript=` (no timings: whole-utterance delivery only, and only against calibration), or Whisper (`whisper` extra). Utterances split at sentence ends, speaker changes, and in unpunctuated text at pauses of 500 ms.
- **Language tags:** one rule everywhere (`_types.LANGUAGE_TAG_RE`: 1-8 letters, then `-` subtags of 1-8 alphanumerics; form only): V29, D6, the dataset schema, `IMLToSSML`, `AudioToIML`, `IMLAssembler`, `MavisBridge`, CLI. SDK/CLI arguments read `en_US` as `en-US`; documents and datasets are checked as written; the REST `language` field takes the tag form only.

## Specification Rules

When editing IML examples or the spec:

- `<utterance>` must include `confidence` whenever `emotion` is present
- Confidence values are floats between 0.0 and 1.0
- `<pause>` is always self-closing: `<pause duration="800"/>`. It has no content (no elements, no text other than whitespace); `duration` is a positive integer of whole milliseconds, at most 2147483647
- `<emphasis>` requires `level` (`strong`, `moderate`, `reduced`) and must not directly contain another `<emphasis>`
- `<segment>` can only be a direct child of `<utterance>`, not nested inside `<prosody>`, `<emphasis>` or another `<segment>`
- `<prosody>` may nest in `<prosody>`, `<emphasis>` in `<prosody>` and vice versa, and `<pause>` may appear in `<emphasis>`; the combined nesting depth of `<prosody>`/`<emphasis>` should not exceed 2 levels
- `<iml>` contains only `<utterance>` elements; `<utterance>` and `<iml>` never nest
- Pitch is a signed relative percentage (`+15%`), signed semitones (`+3st`), or unsigned absolute Hz (`185Hz`)
- Volume is signed relative dB (`+6dB`, `-3dB`)
- Rate is `fast`, `medium`, `slow` or an unsigned percentage (`150%`)
- `consent` is `explicit`, `implicit` or `none`; `processing` is `local`, `remote` or `hybrid`; re-serializers must preserve both
- Documents are UTF-8 and must not contain a DOCTYPE
- Numbers use ASCII digits; floats must be finite (no `NaN`, `1e400`)
- Custom attributes use the `x-` prefix; IML attributes are never namespace-qualified
- Values should be plausible for a human voice (spec 6.4): pitch within +/-24st (-75% to +300%) or 40-1200Hz, volume within +/-40dB, rate 25-400%, pauses at most 60000 ms. The validator warns (V33) outside these

`IMLValidator` implements rules V1-V33 (table in the `validator.py` docstring); MUST violations are errors, SHOULD violations warnings. `tests/test_xsd.py` checks spec examples against the validator and the XSD, so keep them in step.

## Python SDK

Public API (`prosody_protocol.__all__`):

- IML documents: `IMLParser` (`parse`, `parse_file`, `to_plain_text`, `to_iml_string`), `IMLValidator` -> `ValidationResult` (`.valid`, `.errors`/`.warnings` of `ValidationIssue`, `.raise_for_errors()`), models `IMLDocument`, `Utterance`, `Prosody`, `Pause`, `Emphasis`, `Segment`
- Audio to IML: `AudioToIML` (`convert`, `convert_to_doc`, `convert_detailed` -> `ConversionResult`; `words=`, `transcript=`, `calibration_audio=`, `profile=`, `stt=`), `ProsodyAnalyzer`, `IMLAssembler`, `SpanFeatures`, `WordAlignment`, `PauseInterval`, `load_word_timings`, `parse_word_timings` (the latter never reads files; use it on untrusted input)
- Emotion: `EmotionClassifier`, `BaselineAwareEmotionClassifier`, `RuleBasedEmotionClassifier`, `SpeakerBaseline`
- Output: `to_llm_context`, `build_messages`, `IMLToSSML`, `IMLToAudio` (espeak-ng, or a tone preview), `TextToIML` (rule-based)
- Profiles: `ProfileLoader`, `ProfileApplier` (`match`, `apply`), `ProsodyProfile`, `ProsodyMapping`, `ProfileMatch`, `categorize_features`
- Data: `DatasetLoader` (`split(stratify_by=)`), `Dataset`, `DatasetEntry`, `Benchmark` (`words_from=`, `abstention_label=`; entries' `metadata.word_timings`), `BenchmarkReport` (an output without emotion is an abstention: `emotion_accuracy` is over answered entries, `emotion_coverage` their share), `MavisBridge`, `PhonemeEvent`
- Exceptions: `ProsodyProtocolError` and its subclasses `IMLParseError`, `IMLValidationError`, `ProfileError`, `AudioProcessingError` (and its subclass `SpeechRecognitionError`), `ConversionError`, `DatasetError`, `TrainingError`

CLI: `prosody-protocol validate | to-text | to-ssml | to-prompt | from-text | from-audio | synthesize | benchmark | serve | doctor` (`--help` on each; `python -m prosody_protocol` is the same). `-` reads stdin. Exit 0 success, 1 rejected input, 2 usage/I-O error or missing extra.

## REST API

`prosody-protocol serve` or `python -m prosody_protocol.server` (same options; binds 127.0.0.1:8000, also when `PP_HOST` is empty). Settings are `PP_*` env vars (`server/config.py`): `PP_HOST`, `PP_PORT`, `PP_DEBUG`, `PP_CORS_ORIGINS`, `PP_MAX_UPLOAD_MB`, `PP_MAX_JSON_BYTES`, `PP_RATE_LIMIT`, `PP_TRUSTED_PROXIES`, `PP_MAX_TEXT_CHARS`, `PP_MAX_WORDS_CHARS`, `PP_MAX_AUDIO_SECONDS`, `PP_MAX_SYNTH_SECONDS`, `PP_MAX_CONCURRENT_JOBS`, `PP_MAX_QUEUED_JOBS`, `PP_STT_MODEL`.

```
GET  /v1/health
POST /v1/validate
POST /v1/convert/audio-to-iml     # multipart: audio, words, transcript, profile, language, calibration (up to 5)
POST /v1/convert/text-to-iml
POST /v1/convert/iml-to-ssml
POST /v1/convert/iml-to-prompt
POST /v1/synthesize               # -> audio/wav
```

Every error body is `{"error", "detail"}` (422 `invalid_request` has a list as `detail`). Audio conversion and synthesis run in worker processes (about 450 MB per job for 10 minutes of audio); scripts that embed the app need an `if __name__ == "__main__":` guard.

## Build and Test

```bash
pip install -e ".[audio,ml,api,dev]"      # what CI installs
sudo apt install espeak-ng ffmpeg           # optional; tests needing them skip without them
pytest                                      # full suite
pytest -p no:cacheprovider tests/test_docs_examples.py   # documentation examples only
ruff check src/ tests/ training/ examples/
mypy src/                                   # strict; CI lints and type-checks on Python 3.10
```

- **Core-only rule:** a bare `pip install .` has only `lxml`. Every module `__init__.py` imports eagerly (parser, validator, models, exceptions, `_types`, alignment, assembler, emotion_classifier, iml_to_ssml, text_to_iml, llm, profiles, datasets) and `cli.py` must import with the standard library and lxml alone. Modules that need numpy/parselmouth (audio_to_iml, prosody_analyzer, iml_to_audio, benchmarks, mavis_bridge) are listed in `_LAZY` and raise `ImportError` naming the extra. CI's `test-core-only` job enforces this; tests that need an extra use `pytest.importorskip`.
- **Docs harness:** `tests/test_docs_examples.py` runs every `python` block of README.md, examples/README.md and docs/**/*.md in order (cwd is a scratch dir with `examples/` linked), validates `xml` IML blocks, and runs `prosody-protocol ...` lines of `bash` blocks and `$ prosody-protocol ...` lines of `console` blocks. It skips an example only when the extra it needs is really missing. Put `<!-- docs-test: skip -->` above a block only when it calls an external service or needs whisper/network, and `<!-- docs-test: invalid -->` above an IML example that must fail validation. Show real output: run the example and paste it. `tests/test_adoption.py` (TestReadmeOutput) and `tests/test_cli.py` (TestExamples) compare README.md and examples/README.md blocks with real output.
- Audio fixtures are regenerated with `tests/generate_audio_fixtures.py`, `tests/generate_speech_levels_fixture.py` and `tests/generate_training_fixture.py` (need espeak-ng), examples with `examples/make_examples.py`, and the committed benchmark baseline with `tests/fixtures/benchmarks/make_baselines.py` (a test fails when a run regresses from it).
- Training checkpoints (`model.joblib`) are pickles: never load untrusted ones. `export.py --format json` writes a pickle-free model.

## Style Guidelines

- IML examples in documentation should be valid XML and valid IML
- Use realistic prosodic values in examples (not extreme or nonsensical values)
- Always include `confidence` with `emotion` in example `<utterance>` tags
- Prefer the core emotion vocabulary defined in `spec.md` Section 3.1
- Keep the specification language precise: use MUST, SHOULD, MAY, MUST NOT per RFC 2119
- Be honest about capability: do not present heuristic emotion labels as detection, never suggest prosody as the sole basis for a consequential decision, and never suggest the uses spec 8.2 prohibits (deception detection, covert surveillance, profiling in hiring, lending or law enforcement)

## Related Projects

- **Mavis** (github.com/kase1111-hash/Mavis) -- vocal typing instrument that generates IML training data (`MavisBridge`)
- **Intent-Engine** (github.com/kase1111-hash/Intent-Engine) -- prosody-aware AI system that consumes IML
- **Agent-OS** (github.com/kase1111-hash/Agent-OS) -- natural-language operating system for AI agents

## License

- Specification: CC-BY-4.0
- Code: MIT
- Datasets: individual licenses per dataset
