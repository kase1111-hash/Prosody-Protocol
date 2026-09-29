# Mavis

[Mavis](https://github.com/kase1111-hash/Mavis) is a vocal typing
instrument: typed text becomes sung phonemes, each with its own pitch,
volume, breathiness and vibrato. `MavisBridge` turns a session's phoneme
events into an IML-annotated dataset entry, a feature vector for training,
or a whole dataset that `DatasetLoader` reads. It needs numpy (the `audio`
extra) but not Mavis itself.

## Phoneme events

`PhonemeEvent` mirrors the event Mavis produces:

| Field | Default | Meaning |
|-------|---------|---------|
| `phoneme` | (required) | the phoneme, e.g. `"ah"` |
| `start_ms`, `duration_ms` | 0, 100 | timing |
| `volume` | 0.5 | 0.0-1.0 |
| `pitch_hz` | 220.0 | pitch |
| `vibrato` | `False` | |
| `breathiness` | 0.0 | 0.0-1.0 |
| `harmony_intervals` | `None` | list of semitone intervals |

Timing, volume, pitch and breathiness must be finite numbers (numpy values
are fine).

## A session as a dataset entry

```python
from prosody_protocol import MavisBridge, PhonemeEvent

bridge = MavisBridge(language="en-US")
events = [
    PhonemeEvent(phoneme="dh", start_ms=0, duration_ms=80, volume=0.5, pitch_hz=180.0),
    PhonemeEvent(phoneme="s", start_ms=200, duration_ms=100, volume=0.85, pitch_hz=280.0, vibrato=True),
    PhonemeEvent(phoneme="ah", start_ms=300, duration_ms=120, volume=0.9, pitch_hz=300.0),
]
entry = bridge.phoneme_events_to_entry(
    events,
    transcript="the SUN",
    session_id="session_001",
    emotion_label="joyful",
    consent=True,   # only if the player explicitly agreed to their data being stored
)
print(entry.iml)
print(entry.id, entry.audio_file, entry.annotator, entry.consent)
```

```text
<utterance emotion="joyful" confidence="1.0"><prosody pitch="-29%" volume="-4dB">the</prosody> <prosody pitch="+14%" volume="+1dB">SUN</prosody></utterance>
mavis_session_001 audio/session_001.wav hybrid True
```

How the entry is made:

- **Words.** The events are assigned to the transcript's words at the
  longest silent gaps between events (here the gap after "dh"). When
  Mavis knows which keystrokes form each word, pass
  `phonemes_per_word=[1, 2]` instead.
- **Markup.** A word whose mean pitch differs from the session's mean by
  more than 10%, or whose volume by more than 30%, gets a `<prosody>` with
  its pitch and volume relative to that mean; a value that rounds to the
  session's own (`+0%`, `+0dB`) is left out. Both words differ here, so the
  low "the" is marked as well as "SUN".
- **Emotion.** `emotion_label` is your label, such as the emotion the
  player set out to express, and the IML states it with `confidence="1.0"`:
  it is a label, not a measurement (the annotator is then `"hybrid"`: your
  label, the bridge's markup). Without it the bridge guesses from averages:
  volume above 0.8 is `angry` when the mean pitch is above 300 Hz and
  `joyful` otherwise, breathiness above 0.5 is `sad`, volume below 0.3 is
  `calm`, anything else `neutral`. The guess becomes the entry's
  `emotion_label` (annotator `"model"`) but is not written into the IML,
  whose utterance then has no `emotion` or `confidence`: a threshold rule on
  averages has no measured accuracy to state.
- **Language.** `MavisBridge(language=...)` takes a BCP 47 tag (`en_US` is
  read as `en-US`); anything else raises `ValueError`.
- **Consent.** `consent` defaults to `False`; `DatasetLoader` and
  `export_dataset` refuse entries without it (spec 8.1). Set it only when
  the player agreed.
- `session_id` becomes part of file names, so it must start with a letter
  or digit and contain only letters, digits, `.`, `_` and `-`.
- Characters XML cannot hold (such as a stray terminal escape) are dropped
  from the transcript with a warning. The events are kept in
  `entry.metadata` (`phoneme_events`, plus `mavis_features`).

## Feature vectors

`extract_training_features(events)` returns seven numbers per session;
`batch_extract_features(sessions)` stacks them:

| Feature | Description |
|---------|-------------|
| `mean_pitch_hz` | mean pitch of the events |
| `pitch_range_hz` | highest minus lowest pitch |
| `mean_volume` | mean volume |
| `volume_range` | loudest minus quietest |
| `mean_breathiness` | mean breathiness |
| `speech_rate` | events per second of event time |
| `vibrato_ratio` | share of events with vibrato |

```python
print(entry.metadata["mavis_features"])
print(bridge.batch_extract_features([events, events[:2]]).shape)
```

```text
{'mean_pitch_hz': 253.33333333333334, 'pitch_range_hz': 120.0, 'mean_volume': 0.75, 'volume_range': 0.4, 'mean_breathiness': 0.0, 'speech_rate': 10.0, 'vibrato_ratio': 0.3333333333333333}
(2, 7)
```

These are summaries of the synthesizer's parameters, not acoustic
measurements; with a few sessions they will not teach a classifier much.

## Export a dataset

```python
import tempfile
from pathlib import Path

sessions = [
    {
        "events": events,
        "transcript": "the SUN",
        "session_id": "s001",
        "emotion_label": "joyful",
        "speaker_id": "player_42",
        "consent": True,
        "audio_path": "examples/speech.wav",   # the session's recording, copied into the dataset
    },
    {
        "events": [
            PhonemeEvent("f", 0, 120, 0.3, 160.0, breathiness=0.7),
            PhonemeEvent("ao", 120, 200, 0.25, 150.0, breathiness=0.7),
            PhonemeEvent("l", 450, 100, 0.2, 140.0, breathiness=0.6),
        ],
        "transcript": "falling down",
        "session_id": "s002",
        "emotion_label": "sad",
        "consent": True,
    },
]
out = Path(tempfile.mkdtemp()) / "mavis-corpus"   # an empty directory
dataset = bridge.export_dataset(sessions, out)
print(dataset.name, dataset.size, [e.id for e in dataset.entries])
```

```text
mavis-corpus 2 ['mavis_s001', 'mavis_s002']
```

This writes:

```text
mavis-corpus/
├── metadata.json
├── audio/
│   └── s001.wav
└── entries/
    ├── mavis_s001.json
    └── mavis_s002.json
```

Optional session keys are `emotion_label`, `speaker_id`, `consent`,
`annotator`, `phonemes_per_word` and `audio_path`; `export_dataset(...,
consent=True)` applies to sessions that do not state their own. Every
session is checked before anything is written, and the files are moved
into place only when all of them are written, so an invalid session or a
full disk leaves an earlier export intact. Without consent, nothing is
written:

```python
from prosody_protocol import DatasetError

try:
    bridge.export_dataset([{"events": events, "transcript": "the SUN", "session_id": "s009"}], "other")
except DatasetError as error:
    print(error)
```

```text
Session 's009' has no recorded consent: pass consent=True (or a 'consent' key) only if the speaker explicitly agreed to their data being stored
```

Exporting into a directory that holds an earlier export needs
`overwrite=True`, which replaces it, including the audio files the earlier
entries referenced: re-exporting sessions without `audio_path` deletes the
recordings copied before.

## Load it back

```python
from prosody_protocol import DatasetLoader

loaded = DatasetLoader().load(out)
first = loaded.entries[0]
print(first.id, first.emotion_label, len(MavisBridge.events_from_entry(first)))

try:
    DatasetLoader().load(out, check_audio=True)
except DatasetError as error:
    print(error)
```

```text
mavis_s001 joyful 3
1 of 2 entries in /tmp/tmprviyocsg/mavis-corpus failed validation (load with DatasetLoader(strict=False) to skip them):
  mavis_s002.json: D8 Audio file not found: audio/s002.wav
```

(The temporary directory's name differs on each run.) Entries exported
without `audio_path` point to audio that does not exist: they load
normally, but fail `check_audio=True` and cannot be benchmarked.
`MavisBridge.events_from_entry(entry)` rebuilds the stored events, for
example to extract features again.

## Training on an export

The `training/` scripts of a clone (see
[training/README.md](../../training/README.md)) read such datasets:

- `text_to_prosody` learns from the word-level `<prosody>` markup and needs
  no audio.
- `ser` (emotion from prosody) measures the recordings, so every session
  needs `audio_path`; otherwise data preparation stops and lists the
  missing files.
- `pitch_contour` needs `pitch_contour` annotations, which Mavis exports do
  not contain.

## From Mavis

In a Mavis session, convert each phoneme event Mavis emits to a
`PhonemeEvent` with the same field values and collect them per session;
then call `phoneme_events_to_entry`, or `export_dataset` for many sessions.
If Mavis's own event class is a dataclass with these field names (the
bridge mirrors `mavis.llm_processor.PhonemeEvent`), this is a one-liner:

<!-- docs-test: skip -->
```python
import dataclasses

events = [PhonemeEvent(**dataclasses.asdict(event)) for event in mavis_events]
```
