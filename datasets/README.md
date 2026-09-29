# Datasets

This directory documents the dataset format that the SDK's `DatasetLoader`,
`Benchmark`, `MavisBridge` and the `training/` scripts use. **No corpora are
shipped or downloadable yet.** The only datasets in the repository are two
small test fixtures:

- `tests/fixtures/datasets/sample`: 3 entries, for loader tests.
- `tests/fixtures/datasets/training_synthetic`: 10 short espeak-ng clips
  whose speed, pitch and loudness follow their emotion label. It exercises
  the training pipeline; it cannot teach real emotion recognition.

Collecting consented recordings of real speech is on the roadmap (see the
[README](../README.md#roadmap)).

## Layout

A dataset is a directory:

```text
my-dataset/
├── metadata.json    # optional: name, version, license, size, description, ...
├── entries/         # one JSON file per entry (*.json)
├── audio/           # the recordings the entries reference
└── README.md        # optional: how the data was collected, license, consent
```

`metadata.json` is a free-form object. If it declares a `size` that differs
from the number of entries loaded, the loader warns. State the dataset's
license in it or in the dataset's README: each dataset has its own.

## Entries

Each file in `entries/` is one JSON object that follows
[`schemas/dataset-entry.schema.json`](../schemas/dataset-entry.schema.json):

```json
{
  "id": "utt_0001",
  "timestamp": "2026-09-01T12:00:00Z",
  "source": "recorded",
  "language": "en-US",
  "audio_file": "audio/utt_0001.wav",
  "transcript": "Oh, that's great.",
  "iml": "<utterance emotion=\"sarcastic\" confidence=\"0.8\">Oh, that's <emphasis level=\"strong\">great</emphasis>.</utterance>",
  "speaker_id": "spk_01",
  "emotion_label": "sarcastic",
  "annotator": "human",
  "consent": true,
  "metadata": {"session": "s1"}
}
```

| Field | Required | Value |
|-------|----------|-------|
| `id` | yes | Unique within the dataset |
| `timestamp` | yes | When the entry was created, ISO 8601 |
| `source` | yes | `mavis`, `recorded` or `synthetic` |
| `language` | yes | BCP 47 tag, e.g. `en-US`, as written (`en_US` is an error) |
| `audio_file` | yes | Path relative to the dataset directory, with no `..`, no absolute path and no control characters. WAV, 16 kHz mono is recommended |
| `transcript` | yes | Plain text of the recording |
| `iml` | yes | The IML annotation; it must be valid IML |
| `emotion_label` | yes | The entry's primary emotion; prefer the core vocabulary of spec Section 3.1 |
| `annotator` | yes | `human`, `model` or `hybrid` |
| `consent` | yes | Must be `true` (see [Consent](#consent)) |
| `speaker_id` | no | A string or `null`. Used to keep speakers out of more than one split |
| `metadata` | no | An object for anything else; unknown top-level fields are errors. `metadata.word_timings`, if present, holds the recording's word timings (see below) |

## Loading and validation

`DatasetLoader` checks every entry when it loads a dataset:

| Rule | Checks | Severity |
|------|--------|----------|
| D1 | A required string field is missing, empty or not a string | error |
| D2 | `consent` is not `true` | error |
| D3 | `source` is not `mavis`, `recorded` or `synthetic` | error |
| D4 | `annotator` is not `human`, `model` or `hybrid` | error |
| D5 | `timestamp` does not look like ISO 8601 | warning |
| D6 | `language` is not in the form of a BCP 47 tag (1-8 letters, then `-` and subtags of 1-8 letters or digits, as validator rule V29 checks) | error |
| D7 | `iml` fails IML validation | error |
| D8 | The audio file does not exist (only with `check_audio=True`) | error |
| D9 | `audio_file` is absolute, leaves the dataset directory or contains a control character | error |
| D10 | Another entry has the same `id` | error |
| D11 | Unknown top-level field | error |
| D12 | The entry is not an object, or `speaker_id`/`metadata` has the wrong type | error |

By default any error raises `DatasetError` listing every problem.
`DatasetLoader(strict=False)` skips invalid entries with a warning instead.
Entries without consent are never loaded in either mode. For example, a
dataset whose second entry has `"consent": false`, IML without
`confidence`, `"audio_file": "../secret.wav"` and an extra `mood` field:

```python
from prosody_protocol import DatasetLoader

dataset = DatasetLoader().load("my-dataset")
```

```text
DatasetError: 1 of 2 entries in my-dataset failed validation (load with DatasetLoader(strict=False) to skip them):
  utt_0002.json: D2 'consent' must be true
  utt_0002.json: D7 IML validation: <utterance> has emotion="sarcastic" but no confidence attribute
  utt_0002.json: D9 audio_file must be a relative path inside the dataset directory, without control characters, got '../secret.wav'
  utt_0002.json: D11 Unknown field 'mood' (put extra data under 'metadata')
```

`load(path, check_audio=True)` also requires every audio file to exist,
`iter_entries(path)` validates lazily, one entry at a time, and
`validate_entry(entry)` checks a single entry dict. A file that cannot be
read, is not UTF-8 or is not JSON (including JSON nested too deeply)
raises `DatasetError`. The loaded `Dataset`
has `name`, `entries` (`DatasetEntry` objects), `metadata` and `root`, the
directory it was loaded from.

## Splitting

```python
from prosody_protocol import DatasetLoader

loader = DatasetLoader()
dataset = loader.load("tests/fixtures/datasets/sample", check_audio=True)
train, val, test = loader.split(dataset)   # 80/10/10, seed 42
```

`split()` is deterministic for a given seed and keeps each `speaker_id` in
one split (`group_by="speaker_id"`, the default), so no test speaker is
also a training speaker. With fewer speakers than splits it warns and
splits entries individually; `group_by=None` always does.
`loader.split(dataset, stratify_by="emotion_label")` splits each label by
the ratios on its own, so (at 80/10/10) a label with at least three entries
has one in every split; with speaker grouping, whole speakers still stay together, so
the sizes follow the ratios less closely. After any split, a warning names
the labels that are in the training split but missing from a non-empty
validation or test split (among those with enough entries, or speakers, to
be in every split).

## Creating a dataset

- **By hand or from your own pipeline:** write `entries/*.json` and copy
  the audio into `audio/`. `AudioToIML` can produce the `iml` field from a
  recording and its word timings; review it before using it as ground
  truth, since its markup is measured by rules and its emotion labels are
  heuristic.
- **From Mavis sessions:** `MavisBridge().export_dataset(sessions,
  output_dir)` writes a dataset in this format from sessions of the
  [Mavis](https://github.com/kase1111-hash/Mavis) vocal typing instrument,
  copies session audio when a session gives `audio_path`, and refuses any
  session without stated consent. See
  [docs/integrations/mavis.md](../docs/integrations/mavis.md).

## Word timings

An entry may carry the word timings of its recording in
`metadata.word_timings`, in any format `load_word_timings` reads (a
Whisper, Deepgram, AssemblyAI or Google response, or a list of
`{"word", "start_ms", "end_ms"}` records). For example, as the `metadata`
of the entry above:

```json
{"session": "s1", "word_timings": [{"word": "Oh,", "start_ms": 120, "end_ms": 380}, {"word": "that's", "start_ms": 400, "end_ms": 690}, {"word": "great.", "start_ms": 720, "end_ms": 1240}]}
```

`Benchmark` gives them to the converter as `words=`, so pauses, stresses
and word-level pitch can be placed and scored. Without them it gives the
transcript, which places no pauses (or, when Whisper is installed and the
converter may use it, nothing: Whisper finds the words and their timings).
The training scripts do not use them.

## Using a dataset

- `Benchmark(dataset, converter).run()` (or
  `prosody-protocol benchmark my-dataset`) scores a converter's emotion
  labels, pauses, pitch contours and IML validity against the entries. An
  output without an emotion is an abstention: it lowers `emotion_coverage`
  rather than `emotion_accuracy`. See the
  [CLI reference](../docs/cli.md#benchmark).
- `training/` trains scikit-learn baselines on a dataset; see
  [training/README.md](../training/README.md).

## Consent

Prosodic and emotional annotations are personal data (spec Section 8.1).
Only set `"consent": true` when the speaker explicitly agreed to have their
recording and its emotional annotations stored and used for the stated
purpose, and give them a way to review and delete their entries. The
loader, the JSON schema and `MavisBridge` all refuse entries without it.
Datasets must not be used for the purposes spec Section 8.2 prohibits:
deception detection, covert emotional surveillance, or profiling people in
hiring, lending or law enforcement.

## Contributing data

See [CONTRIBUTING.md](../CONTRIBUTING.md#datasets). Entries must pass both
the JSON schema and `DatasetLoader`, and a contributed dataset needs a
license and a description of how consent was collected.
