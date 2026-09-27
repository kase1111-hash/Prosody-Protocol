# Training pipeline

Scripts that train small **scikit-learn baselines** on Prosody Protocol
datasets, evaluate them, and export them for use with the SDK. They live in
the source checkout only; they are not part of the installed
`prosody_protocol` package.

| Task | Config | Model | Input features | Target labels |
|------|--------|-------|----------------|---------------|
| `ser` (speech emotion recognition) | `configs/ser_logreg.yaml` | logistic regression | prosody of each recording, measured by `ProsodyAnalyzer`: F0 mean and range (Hz), intensity (dB), speech rate (syllables/s), jitter and shimmer (%), HNR (dB) | the entry's `emotion_label` |
| `text_to_prosody` | `configs/text_prosody_tree.yaml` | decision tree | per token of text: length, position, capitalisation, punctuation, neighbours' lengths | per token: pitch/volume/rate level read from the IML markup around it |
| `pitch_contour` | `configs/pitch_contour_forest.yaml` | random forest | the recording's F0 track, resampled to 20 points in semitones relative to its median | the `pitch_contour` of a `<prosody>` covering the entry's whole text |

Features always come from the audio or the text, and labels from the
annotations. A model can only learn what the recordings and markup show.

These are baselines: small, fast and easy to inspect. There are no neural
models (wav2vec2, BERT, CNN) here. See [Limits](#limits) before relying on
one.

## Install

From the repository root:

```bash
pip install -e '.[ml]'   # scikit-learn, joblib, pyyaml + the audio extra (numpy, praat-parselmouth)
```

Run the commands below from the repository root.

## Datasets

The scripts read datasets in the format `DatasetLoader` loads: a directory
with `metadata.json`, `entries/*.json` (see
`schemas/dataset-entry.schema.json`) and `audio/`. Every entry is validated,
and entries without `consent: true` are never loaded. `DatasetLoader.split`
splits each dataset 80/10/10 (seed 42) and keeps each `speaker_id` inside
one split. The same entries therefore land in the same split every time,
in `train.py` and in `evaluate.py`.

What each task needs:

- **ser**: an audio file for every entry (a missing file is an error), and
  the label in `emotion_label` (or the entry field named by
  `data.label_field`). Entries without voiced speech are skipped with a
  warning.
- **text_to_prosody**: IML markup that says how words were spoken, e.g.
  `I <prosody pitch="+20%" volume="+6dB">NEVER</prosody> said that.` A
  token is `high`/`low` pitch at +/-5 % (or +/-1 st) and beyond, `loud`/`quiet`
  at +/-2 dB, and `fast`/`slow` for `rate="fast"`/`"slow"`, 110 %/90 % and
  beyond, or `<segment tempo="rushed"/"drawn-out">`. Absolute pitch
  (`185Hz`) counts as `mid`. Audio is not needed.
- **pitch_contour**: audio, and IML in which one `<prosody pitch_contour="...">`
  covers all of the entry's text, e.g. one short phrase per entry. Entries
  have no word timings, so the F0 of a part of the utterance cannot be cut
  out; entries annotated only in part are skipped with a warning.

The repository ships no training corpus. The only dataset in it is the test
fixture `tests/fixtures/datasets/training_synthetic`: 10 entries whose audio
files are the **same 440 Hz tone**. The commands below use it to show that
the pipeline runs. It cannot teach a model anything. Train on your own
recordings, or on a Mavis export (`MavisBridge.export_dataset` with
`audio_path` for each session).

## Commands

The four steps are data preparation, training, evaluation and export.
`train.py --dataset` and `evaluate.py --dataset` run the preparation
themselves, so running `data_prep.py` on its own is optional. The output
below is real output from the fixture dataset, with outputs under `/tmp`.

### 1. Prepare data (optional)

```bash
python training/scripts/data_prep.py \
    --config training/configs/ser_logreg.yaml \
    --dataset tests/fixtures/datasets/training_synthetic \
    --output /tmp/pp/prepared/ser
```

```
UserWarning: Cannot split 10 entries into 3 speaker_id-disjoint sets: they have only 1 distinct speaker_id value(s). ...
Data prepared for task 'ser' in /tmp/pp/prepared/ser:
  train: 8 entries, 8 rows
  val: 1 entries, 1 rows
  test: 1 entries, 1 rows
  features: f0_mean, f0_range, intensity_mean, speech_rate, jitter, shimmer, hnr
```

This writes `train/`, `val/` and `test/` (each with `X.npy` features and
`y.npy` labels) and `prep_metadata.json` (task, feature names, counts, and
skipped entries by reason). The warning comes from the fixture's single
speaker.

### 2. Train

```bash
python training/scripts/train.py \
    --config training/configs/ser_logreg.yaml \
    --prepared-data /tmp/pp/prepared/ser \
    --output /tmp/pp/checkpoints/ser
```

```
Training complete for task 'ser'
  Model type: logistic_regression
  Features: f0_mean, f0_range, intensity_mean, speech_rate, jitter, shimmer, hnr
  Classes: angry, calm, fearful, frustrated, joyful, neutral, sad, sarcastic
  Training samples: 8
  Training time: 0.008s
  Train accuracy: 0.125
  Val accuracy: 0.0000 (1 samples)
  Val macro F1: 0.0000
  Checkpoint saved to: /tmp/pp/checkpoints/ser
```

A train accuracy of 0.125 is correct here: 8 identical clips carry 8
different labels, so no model can do better than one in eight. Use
`--dataset <dir>` instead of `--prepared-data` to prepare and train in one
step. Without `--output`, the checkpoint goes to the config's
`output.checkpoint_dir`.

The checkpoint directory holds `model.joblib` (the model), `metadata.json`
(model class and parameters), `config.yaml` (a copy of the config) and
`training_results.json`.

### 3. Evaluate

```bash
python training/scripts/evaluate.py \
    --checkpoint /tmp/pp/checkpoints/ser \
    --dataset tests/fixtures/datasets/training_synthetic \
    --split test --output /tmp/pp/reports/ser_test.json
```

```
Label                 Precision     Recall         F1  Support
--------------------------------------------------------------
angry                    1.0000     1.0000     1.0000        1
calm                        n/a        n/a        n/a        0
fearful                     n/a        n/a        n/a        0
...
sarcastic                   n/a        n/a        n/a        0
--------------------------------------------------------------
macro avg                1.0000     1.0000     1.0000        1
accuracy                                       1.0000        1

Report saved to: /tmp/pp/reports/ser_test.json
```

`--dataset` prepares the split again, using the config saved in the
checkpoint (`--config` overrides it); `--prepared-data <dir>` evaluates
prepared data instead. The macro average covers the classes present in the
split, in its true labels or its predictions, as scikit-learn's does.
Classes the model knows but the split lacks are listed as `n/a`, and in the
JSON report they have `null` scores.

The 1.0 above is **chance**. The model cannot tell the identical clips
apart, so it gives every input the same label. The single test entry
happens to have that label. A test split of one entry measures nothing:
evaluate on dozens of entries per class at least.

### 4. Export

```bash
python training/scripts/export.py \
    --checkpoint /tmp/pp/checkpoints/ser \
    --output /tmp/pp/exports/ser
```

```
Model exported successfully:
  Model class: SERModel
  Export path: /tmp/pp/exports/ser
  Format: json (config.json, model.json)
```

`--format json` (the default, or the config's `output.export_format`) writes
`model.json`, which holds the fitted parameters as plain JSON: class names,
feature names, the imputation and scaling values, and coefficients or tree
nodes. `training/portable.py` documents the format. `PortableModel` runs it
with numpy alone, and loading it never executes code. This is the format to
share. `--format pickle` writes `model.joblib` for `BaseModel.load`; use it
only yourself.

## Using a trained emotion model in the SDK

`training.inference.TrainedEmotionClassifier` implements the SDK's
`EmotionClassifier` protocol, so an exported SER model can label the
utterances `AudioToIML` produces:

```python
from prosody_protocol import AudioToIML
from training.inference import TrainedEmotionClassifier

classifier = TrainedEmotionClassifier("/tmp/pp/exports/ser")
converter = AudioToIML(emotion_classifier=classifier)
print(converter.convert(
    "tests/fixtures/datasets/training_synthetic/audio/synth_002.wav",
    transcript="I CANNOT believe you did that!",
))
```

```
<iml version="0.1.0"><utterance>I CANNOT believe you did that!</utterance></iml>
```

The classifier summarises the utterance's word features with the same code
data preparation uses (`training/features.py`), so the model sees the kind
of input it was trained on. Its confidence is the model's probability for
the label. `AudioToIML` leaves out the emotion when that is below 0.5, as
here: the fixture model gives every label 1/8. Utterances without voiced
speech get `("neutral", 0.0)`, which means no emotion.
`TrainedEmotionClassifier.from_checkpoint(dir)` builds the classifier from
a checkpoint instead (a pickle; see below).

To run any exported model on feature rows directly:

```python
from training.portable import PortableModel

model = PortableModel.load("/tmp/pp/exports/ser")    # directory or model.json
model.feature_names                                  # ('f0_mean', 'f0_range', ...)
model.predict_labels([[180.0, 40.0, 65.0, 4.0, 1.0, 3.0, 18.0]])
```

Text-prosody and pitch-contour models export the same way. They are not
wired into `TextToIML` or `AudioToIML`.

## Checkpoints are pickles

`model.joblib`, the file in checkpoints and `--format pickle` exports, is a
pickle. Loading it with `BaseModel.load`, `evaluate.py`, `export.py` or
`TrainedEmotionClassifier.from_checkpoint` **runs code stored in the file**,
before anything can check what it contains. joblib is no safer than pickle
in this respect. Only load checkpoints you created or otherwise trust, and
share models as JSON exports. Prepared data is loaded with
`allow_pickle=False`.

## Configs

A config is YAML with these sections:

```yaml
task: ser                        # ser | text_to_prosody | pitch_contour
description: "..."
model:
  type: logistic_regression      # logistic_regression | decision_tree | random_forest
  labels: [neutral, angry, ...]  # optional: data with other labels is an error
data: {...}                      # per task, below
training: {...}                  # hyperparameters
evaluation:
  metrics: [precision, recall, f1]
  average: macro                 # the only average implemented
output:
  checkpoint_dir: training/checkpoints/ser_logreg   # default for train.py --output
  export_format: json                               # default for export.py --format
```

**Hyperparameters** can go in `model` or `training`, but not both. They
have scikit-learn's meaning:

| Model type | Hyperparameters |
|------------|-----------------|
| `logistic_regression` | `C` (inverse regularisation strength), `max_iter`, `solver` (`lbfgs`, `newton-cg`, `newton-cholesky`, `sag`, `saga`; `liblinear` for two classes), `class_weight` (`balanced` or null), `random_state` |
| `decision_tree` | `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`, `random_state` |
| `random_forest` | the `decision_tree` ones and `n_estimators` |

The old names `regularization` (for `C`) and `optimizer` (for `solver`)
are still accepted.

**`data`** per task:

- `ser`: `features` (any of `f0_mean`, `f0_range`, `intensity_mean`,
  `speech_rate`, `jitter`, `shimmer`, `hnr`, in the order you list them), and
  `label_field` (default `emotion_label`)
- `text_to_prosody`: `text_features` (any of `word_length`,
  `position_ratio`, `is_capitalized`, `has_punctuation`,
  `sentence_position`, `prev_word_length`, `next_word_length`), and
  `labels`, which maps the label dimensions to use to their values:
  `pitch_level: [high, mid, low]`, `volume_level: [loud, normal, quiet]`,
  `rate_level: [fast, normal, slow]`, `emphasis_level: [strong, moderate,
  reduced, none]`
- `pitch_contour`: `sequence_length` (points per F0 track) and
  `contour_classes` (from `rise`, `fall`, `rise-fall`, `fall-rise`,
  `rise-sharp`, `fall-sharp`, `flat`; entries with other contours are
  skipped)

An invalid value raises an error that names the key. A key the pipeline
does not use produces a warning and is ignored, so a setting never silently
does nothing. This covers unknown keys and keys the scikit-learn baselines
cannot honour (`epochs`, `learning_rate`, `batch_size`, `num_classes`,
`output.format`).

Earlier versions named the configs `ser_wav2vec2.yaml`,
`text_to_prosody_bert.yaml` and `pitch_contour_cnn.yaml`, although they
always trained these scikit-learn baselines.

## Limits

- **Absolute features.** The SER features are absolute: Hz and dB. They
  depend on the speaker's voice and on the microphone and gain as much as
  on emotion, so a model trained on one set of speakers or recording
  conditions transfers poorly to others. There is no speaker normalisation.
  The SDK's rule-based `RuleBasedEmotionClassifier` judges deviations from
  a speaker baseline instead.
- **Uncalibrated confidence.** The confidence is a raw logistic-regression
  probability. A model trained on little data can be confidently wrong,
  especially on recordings unlike its training data.
- **Utterance level only.** Dataset entries have no word timings, so SER
  and pitch-contour features describe whole recordings.
- **Shallow text features.** The text-to-prosody features do not look at
  what the words mean. Expect the model to learn patterns such as "the
  capitalised word is stressed", not real prosody prediction.
- **Small splits are noise.** Metrics from a handful of samples (as with
  the fixture's 1-entry val and test splits) are noise. Check the support
  column.
- **Synthetic fixture.** The fixture dataset's clips are identical. Poor
  scores on it are the correct result.
