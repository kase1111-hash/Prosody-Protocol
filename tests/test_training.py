"""Tests for Phase 11: Model Training Pipelines.

Covers acceptance criteria:
- Training script runs end-to-end on a small synthetic dataset (10 samples)
- Evaluation script produces precision/recall/F1 per emotion class
- Exported model loads and runs inference via the SDK classes
- Configs are YAML; no hardcoded hyperparameters in scripts

and that the pipeline is honest: features come from the audio and text
(never from the labels), so a model only learns what the recordings show.
Most tests use small datasets generated here, whose clips differ by class.
The ``training_synthetic`` fixture holds 10 short espeak-ng clips whose
delivery follows their label (tests/generate_training_fixture.py).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("yaml")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

import joblib
import numpy as np
import yaml
from sklearn.metrics import f1_score

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from prosody_protocol import IMLParser, SpanFeatures
from prosody_protocol.exceptions import DatasetError, TrainingError
from training.config import load_config
from training.features import (
    feature_vector,
    recording_features,
    token_prosody_labels,
    utterance_contour_label,
    utterance_features,
)
from training.metrics import ClassMetrics, EvaluationReport, compute_metrics
from training.models import ModelRegistry, PitchContourModel, SERModel, TextProsodyModel
from training.models.base import BaseModel
from training.models.pitch_contour import resample_f0
from training.models.text_prosody import extract_text_features
from training.portable import MODEL_FILE, PortableModel

FIXTURES = Path(__file__).parent / "fixtures"
CONFIGS_DIR = _PROJECT_ROOT / "training" / "configs"
SCRIPTS_DIR = _PROJECT_ROOT / "training" / "scripts"
SYNTHETIC_DATASET = FIXTURES / "datasets" / "training_synthetic"
SER_CONFIG = CONFIGS_DIR / "ser_logreg.yaml"
TEXT_CONFIG = CONFIGS_DIR / "text_prosody_tree.yaml"
CONTOUR_CONFIG = CONFIGS_DIR / "pitch_contour_forest.yaml"

needs_audio = pytest.mark.skipif(
    importlib.util.find_spec("parselmouth") is None,
    reason="audio analysis needs praat-parselmouth (the audio extra)",
)

# The fixture dataset has a single speaker, which DatasetLoader.split warns about.
single_speaker = pytest.mark.filterwarnings("ignore:Cannot split.*speaker_id")


# ---------------------------------------------------------------------------
# Generated datasets
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16_000

# Acoustically distinct emotion classes: F0 glide (Hz) and amplitude.
_EMOTIONS: dict[str, tuple[tuple[float, float], float]] = {
    "angry": ((240.0, 270.0), 0.7),
    "sad": ((120.0, 100.0), 0.08),
    "calm": ((170.0, 170.0), 0.25),
}

# Pitch contour shapes, in semitones over the clip.
_CONTOURS: dict[str, tuple[float, ...]] = {
    "rise": (0.0, 4.0),
    "fall": (4.0, 0.0),
    "rise-fall": (0.0, 4.0, 0.0),
    "flat": (0.0, 0.0),
}

_SENTENCES = (
    "I did not expect that at all.",
    "We are leaving in five minutes.",
    "The report is on your desk.",
    "Nobody told me about the meeting.",
)


def _write_clip(
    path: Path, f0_points: Sequence[float], amplitude: float, *, seconds: float = 0.8, seed: int = 0
) -> None:
    """A vowel-like harmonic tone following *f0_points* (Hz), with 100 ms of silence around it."""
    rng = np.random.default_rng(seed)
    n = int(_SAMPLE_RATE * seconds)
    t = np.arange(n) / _SAMPLE_RATE
    f0 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(f0_points)), f0_points)
    phase = 2 * np.pi * np.cumsum(f0) / _SAMPLE_RATE
    signal = sum((0.6**k) * np.sin(k * phase) for k in range(1, 12))
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.04)
    signal = amplitude * envelope * signal / np.max(np.abs(signal))
    signal = signal + 0.001 * rng.standard_normal(n)
    pad = np.zeros(int(0.1 * _SAMPLE_RATE))
    samples = np.clip(np.concatenate([pad, signal, pad]), -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes((samples * 32767).astype("<i2").tobytes())


def _write_silence(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(_SAMPLE_RATE * seconds))


def _write_dataset(root: Path, clips: list[dict[str, Any]]) -> Path:
    """Write a dataset directory; clips with an ``f0`` key get a generated WAV."""
    (root / "entries").mkdir(parents=True, exist_ok=True)
    for clip in clips:
        audio_file = f"audio/{clip['id']}.wav"
        if "f0" in clip:
            _write_clip(root / audio_file, clip["f0"], clip["amplitude"], seed=clip["seed"])
        entry = {
            "id": clip["id"],
            "timestamp": "2026-01-01T12:00:00Z",
            "source": "synthetic",
            "language": "en-US",
            "audio_file": audio_file,
            "transcript": clip["transcript"],
            "iml": clip["iml"],
            "emotion_label": clip["label"],
            "annotator": "model",
            "consent": True,
        }
        (root / "entries" / f"{clip['id']}.json").write_text(json.dumps(entry), encoding="utf-8")
    meta = {"name": root.name, "version": "0.1.0", "size": len(clips)}
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return root


def _emotion_clips(seed: int, per_class: int) -> list[dict[str, Any]]:
    """Clips whose pitch and loudness differ by emotion, varied per clip."""
    rng = np.random.default_rng(seed)
    clips = []
    for label, ((f0_start, f0_end), amplitude) in _EMOTIONS.items():
        for i in range(per_class):
            shift, gain = rng.uniform(0.92, 1.08), rng.uniform(0.8, 1.2)
            text = _SENTENCES[i % len(_SENTENCES)]
            clips.append({
                "id": f"{label}_{i:02d}",
                "label": label,
                "f0": [f0_start * shift, f0_end * shift],
                "amplitude": amplitude * gain,
                "seed": int(rng.integers(1 << 30)),
                "transcript": text,
                "iml": f'<utterance emotion="{label}" confidence="0.9">{text}</utterance>',
            })
    return clips


def _contour_clips(seed: int, per_class: int) -> list[dict[str, Any]]:
    """Clips with a pitch_contour annotation over the whole utterance, at varied pitch."""
    rng = np.random.default_rng(seed)
    clips = []
    for contour, semitones in _CONTOURS.items():
        for i in range(per_class):
            base = rng.uniform(110.0, 240.0)
            text = _SENTENCES[i % len(_SENTENCES)]
            clips.append({
                "id": f"{contour}_{i:02d}",
                "label": "neutral",
                "f0": [base * 2 ** (st / 12) for st in semitones],
                "amplitude": rng.uniform(0.1, 0.6),
                "seed": int(rng.integers(1 << 30)),
                "transcript": text,
                "iml": (
                    f'<utterance><prosody pitch_contour="{contour}">{text}</prosody></utterance>'
                ),
            })
    return clips


def _text_clips(n: int) -> list[dict[str, Any]]:
    """Entries whose IML marks the capitalised word as high and loud (no audio)."""
    words = ["never", "always", "really", "only", "still"]
    clips = []
    for i in range(n):
        word = words[i % len(words)]
        text = f"I {word.upper()} said that to you"
        iml = (
            f'<utterance>I <prosody pitch="+20%" volume="+6dB">{word.upper()}</prosody> '
            "said that to you</utterance>"
        )
        clips.append({
            "id": f"text_{i:02d}", "label": "neutral", "transcript": text, "iml": iml,
        })
    return clips


def _relabel(src: Path, dst: Path, labels: dict[str, str]) -> Path:
    """Copy dataset *src* to *dst*, changing entries' emotion_label (the audio is unchanged)."""
    shutil.copytree(src, dst)
    for entry_id, label in labels.items():
        path = dst / "entries" / f"{entry_id}.json"
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry["emotion_label"] = label
        path.write_text(json.dumps(entry), encoding="utf-8")
    return dst


def _shuffled_labels(dataset: Path, seed: int) -> dict[str, str]:
    """A random permutation of the dataset's emotion labels over its entries."""
    entries = sorted((dataset / "entries").glob("*.json"))
    ids = [p.stem for p in entries]
    labels = [json.loads(p.read_text(encoding="utf-8"))["emotion_label"] for p in entries]
    permuted = np.random.default_rng(seed).permutation(labels)
    return dict(zip(ids, (str(label) for label in permuted), strict=True))


@pytest.fixture(scope="module")
def emotion_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_dataset(tmp_path_factory.mktemp("emotion_train"), _emotion_clips(1, 8))


@pytest.fixture(scope="module")
def emotion_heldout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Independently generated clips of the same classes, for held-out testing."""
    return _write_dataset(tmp_path_factory.mktemp("emotion_heldout"), _emotion_clips(2, 6))


@pytest.fixture(scope="module")
def contour_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_dataset(tmp_path_factory.mktemp("contour_train"), _contour_clips(3, 6))


@pytest.fixture(scope="module")
def text_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_dataset(tmp_path_factory.mktemp("text"), _text_clips(10))


def _prepare(preparer_name: str, config_path: Path, dataset: Path, out: Path) -> dict[str, int]:
    from training.scripts import data_prep

    config = load_config(config_path)
    stats: dict[str, int] = getattr(data_prep, preparer_name)(dataset, config.data, out)
    return stats


def _all_splits(prepared: Path) -> tuple[np.ndarray, list[str]]:
    X = np.vstack([np.load(prepared / s / "X.npy") for s in ("train", "val", "test")])
    y = [str(v) for s in ("train", "val", "test") for v in np.load(prepared / s / "y.npy")]
    return X, y


def _write_config(path: Path, base: Path, **changes: dict[str, Any]) -> Path:
    """A copy of config *base* with the given sections updated."""
    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    for section, values in changes.items():
        raw[section].update(values)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Acceptance Criteria Tests
# ---------------------------------------------------------------------------


@needs_audio
class TestAcceptanceCriteria:
    """Test the four acceptance criteria from the execution guide."""

    @single_speaker
    def test_training_runs_end_to_end_on_synthetic_dataset(self, tmp_path):
        """AC1: Training script runs end-to-end on a small synthetic dataset (10 samples)."""
        from training.scripts.train import train

        results = train(
            config_path=SER_CONFIG,
            dataset_dir=SYNTHETIC_DATASET,
            output_dir=tmp_path / "checkpoint",
        )

        assert results["task"] == "ser"
        assert results["train_samples"] > 0
        assert "accuracy" in results["train_metrics"]
        assert (tmp_path / "checkpoint" / "model.joblib").exists()
        assert (tmp_path / "checkpoint" / "metadata.json").exists()
        assert (tmp_path / "checkpoint" / "config.yaml").exists()

    def test_identical_clips_cannot_be_told_apart(self, tmp_path, emotion_dataset):
        """With the same recording behind every label, no model can separate the labels:
        training must not report the perfect scores that features derived from the
        labels used to give (the 10 clips of the training_synthetic fixture were all
        the same tone, and training on them reported 100 % accuracy)."""
        from training.scripts.train import train

        same = tmp_path / "same"
        shutil.copytree(emotion_dataset, same)
        for wav in (same / "audio").glob("*.wav"):
            shutil.copyfile(emotion_dataset / "audio" / "calm_00.wav", wav)
        results = train(config_path=SER_CONFIG, dataset_dir=same)
        assert results["train_metrics"]["accuracy"] < 0.5

    def test_evaluation_produces_precision_recall_f1(self, tmp_path, emotion_dataset):
        """AC2: Evaluation script produces precision/recall/F1 per emotion class."""
        from training.scripts.evaluate import evaluate
        from training.scripts.train import train

        train(config_path=SER_CONFIG, dataset_dir=emotion_dataset, output_dir=tmp_path / "ckpt")
        report = evaluate(
            checkpoint_path=tmp_path / "ckpt",
            dataset_dir=emotion_dataset,
            split="train",
            config_path=SER_CONFIG,
        )

        assert isinstance(report, EvaluationReport)
        assert {m.label for m in report.per_class} == set(_EMOTIONS)
        for m in report.per_class:
            assert isinstance(m, ClassMetrics)
            assert 0.0 <= m.precision <= 1.0
            assert 0.0 <= m.recall <= 1.0
            assert 0.0 <= m.f1 <= 1.0
            assert m.support > 0
        assert 0.0 <= report.macro_f1 <= 1.0
        assert report.total_samples > 0

    def test_exported_model_loads_and_runs_inference(self, tmp_path, emotion_dataset):
        """AC3: Exported model loads and runs inference via the SDK classes."""
        from training.inference import TrainedEmotionClassifier
        from training.scripts.export import export_model
        from training.scripts.train import train

        train(config_path=SER_CONFIG, dataset_dir=emotion_dataset, output_dir=tmp_path / "ckpt")
        export_model(checkpoint_path=tmp_path / "ckpt", output_path=tmp_path / "export")

        assert (tmp_path / "export" / "config.json").exists()
        assert (tmp_path / "export" / MODEL_FILE).exists()
        assert (tmp_path / "export" / "export_metadata.json").exists()
        assert not (tmp_path / "export" / "model.joblib").exists()

        classifier = TrainedEmotionClassifier(tmp_path / "export")
        span = recording_features(emotion_dataset / "audio" / "angry_00.wav")
        label, confidence = classifier.classify([span])
        assert label in _EMOTIONS
        assert 0.0 < confidence <= 1.0

    def test_configs_are_yaml_no_hardcoded_hyperparams(self, recwarn):
        """AC4: Configs are YAML, valid, and use only keys the pipeline honours."""
        configs = sorted(CONFIGS_DIR.glob("*.yaml"))
        assert [c.name for c in configs] == [
            "pitch_contour_forest.yaml", "ser_logreg.yaml", "text_prosody_tree.yaml",
        ]
        for config_file in configs:
            config = load_config(config_file)
            assert config.model_params()["type"] == config.model_type
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


# ---------------------------------------------------------------------------
# Config Loading Tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Test YAML config loading and validation."""

    def test_load_ser_config(self):
        config = load_config(SER_CONFIG)
        assert config.task == "ser"
        assert config.model_type == "logistic_regression"
        assert "labels" in config.model
        assert len(config.labels) == 8

    def test_load_text_prosody_config(self):
        config = load_config(TEXT_CONFIG)
        assert config.task == "text_to_prosody"
        assert config.model_type == "decision_tree"
        assert "max_depth" in config.model

    def test_load_pitch_contour_config(self):
        config = load_config(CONTOUR_CONFIG)
        assert config.task == "pitch_contour"
        assert config.model_type == "random_forest"
        assert "contour_classes" in config.data

    def test_missing_config_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_invalid_config_missing_keys(self, tmp_path):
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("task: ser\nmodel: {}\n")
        with pytest.raises(ValueError, match="missing required keys"):
            load_config(bad_config)

    def test_config_properties(self):
        config = load_config(SER_CONFIG)
        assert "precision" in config.metrics
        assert "recall" in config.metrics
        assert "f1" in config.metrics
        assert config.average == "macro"

    def test_unused_keys_warn(self, tmp_path):
        """Keys the sklearn baselines cannot honour are reported, not silently ignored."""
        path = _write_config(
            tmp_path / "c.yaml", SER_CONFIG,
            training={"epochs": 100, "learning_rate": 0.01, "batch_size": 32, "momentum": 0.9},
        )
        with pytest.warns(UserWarning) as caught:
            config = load_config(path)
        messages = " ".join(str(w.message) for w in caught)
        for key in ("epochs", "learning_rate", "batch_size", "momentum"):
            assert f"'{key}'" in messages
        assert "not trained in epochs" in messages
        assert "unknown key" in messages
        assert set(config.model_params()) == {
            "type", "C", "max_iter", "solver", "class_weight", "random_state",
        }

    def test_hyperparameters_reach_the_estimator(self, tmp_path):
        path = _write_config(
            tmp_path / "c.yaml", SER_CONFIG,
            training={"C": 0.25, "max_iter": 77, "solver": "newton-cg", "random_state": 3},
        )
        model = ModelRegistry.create(load_config(path).model_params())
        params = model._classifier.get_params()
        assert (params["C"], params["max_iter"], params["solver"], params["random_state"]) == (
            0.25, 77, "newton-cg", 3,
        )
        assert params["class_weight"] == "balanced"

        forest = ModelRegistry.create(load_config(_write_config(
            tmp_path / "f.yaml", CONTOUR_CONFIG,
            model={"n_estimators": 7, "max_depth": 3}, training={"min_samples_leaf": 2},
        )).model_params())
        forest_params = forest._classifier.get_params()
        assert (forest_params["n_estimators"], forest_params["max_depth"]) == (7, 3)
        assert forest_params["min_samples_leaf"] == 2

    def test_old_hyperparameter_names_are_honoured(self, tmp_path):
        raw = yaml.safe_load(SER_CONFIG.read_text(encoding="utf-8"))
        raw["training"] = {"regularization": 0.5, "optimizer": "saga", "max_iter": 50}
        path = tmp_path / "old.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        params = load_config(path).model_params()
        assert (params["C"], params["solver"]) == (0.5, "saga")

    @pytest.mark.parametrize(
        ("section", "values", "message"),
        [
            ("training", {"solver": "sgd"}, "solver must be one of"),
            ("training", {"C": -1}, "C must be a positive number"),
            ("training", {"max_iter": 0}, "max_iter must be an integer"),
            ("training", {"class_weight": "heavy"}, "class_weight must be one of"),
            ("model", {"type": "wav2vec2"}, "Unknown model type 'wav2vec2'"),
            ("data", {"features": ["f0_mean", "loudness"]}, "unknown name"),
            ("data", {"label_field": "mood"}, "label_field"),
            ("evaluation", {"average": "weighted"}, "only 'macro'"),
            ("output", {"export_format": "onnx"}, "export_format"),
        ],
    )
    def test_invalid_values_raise(self, tmp_path, section, values, message):
        path = _write_config(tmp_path / "c.yaml", SER_CONFIG, **{section: values})
        with pytest.raises(ValueError, match=message):
            load_config(path)

    def test_hyperparameter_set_twice_raises(self, tmp_path):
        path = _write_config(tmp_path / "c.yaml", SER_CONFIG, model={"C": 2.0})
        with pytest.raises(ValueError, match="already sets"):
            load_config(path)

    def test_text_label_values_must_match_vocabulary(self, tmp_path):
        path = _write_config(
            tmp_path / "c.yaml", TEXT_CONFIG, data={"labels": {"pitch_level": ["up", "down"]}},
        )
        with pytest.raises(ValueError, match="pitch_level must list the values"):
            load_config(path)


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------


class TestMetrics:
    """Test evaluation metric computation."""

    def test_perfect_predictions(self):
        y_true = ["angry", "sad", "joyful", "angry", "sad"]
        y_pred = ["angry", "sad", "joyful", "angry", "sad"]
        report = compute_metrics(y_true, y_pred)

        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0
        assert report.macro_precision == 1.0
        assert report.macro_recall == 1.0

    def test_imperfect_predictions(self):
        y_true = ["angry", "sad", "joyful", "angry"]
        y_pred = ["angry", "sad", "angry", "angry"]
        report = compute_metrics(y_true, y_pred)

        assert 0.0 < report.accuracy < 1.0
        assert report.total_samples == 4

    def test_per_class_metrics(self):
        y_true = ["a", "a", "b", "b", "c"]
        y_pred = ["a", "b", "b", "b", "c"]
        report = compute_metrics(y_true, y_pred, labels=["a", "b", "c"])

        assert len(report.per_class) == 3
        assert report.per_class[0].label == "a"
        assert report.per_class[1].label == "b"
        assert report.per_class[2].label == "c"

    def test_macro_average_ignores_classes_absent_from_split(self):
        """A perfect prediction scores macro F1 1.0 even when the config lists more classes."""
        labels = ["neutral", "angry", "frustrated", "joyful", "sad", "fearful", "sarcastic", "calm"]
        report = compute_metrics(["angry"], ["angry"], labels=labels)
        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0
        # The listed classes still get a row, marked as not evaluated.
        assert [m.label for m in report.per_class] == labels
        assert [m.evaluated for m in report.per_class] == [label == "angry" for label in labels]
        row = report.to_dict()["per_class"][0]
        assert row == {"label": "neutral", "precision": None, "recall": None, "f1": None,
                       "support": 0}
        assert "n/a" in report.format_table()

    def test_macro_f1_matches_sklearn(self):
        y_true = ["a", "a", "b", "c", "c", "c"]
        y_pred = ["a", "b", "b", "c", "a", "d"]
        report = compute_metrics(y_true, y_pred, labels=["a", "b", "c", "e"])
        assert report.macro_f1 == pytest.approx(f1_score(y_true, y_pred, average="macro"))
        # A predicted class that is not in the list still gets a row.
        assert [m.label for m in report.per_class] == ["a", "b", "c", "e", "d"]

    def test_report_to_dict(self):
        y_true = ["a", "b", "a"]
        y_pred = ["a", "b", "b"]
        report = compute_metrics(y_true, y_pred)
        d = report.to_dict()

        assert "per_class" in d
        assert "macro" in d
        assert "accuracy" in d
        assert "total_samples" in d

    def test_report_format_table(self):
        y_true = ["angry", "sad", "joyful"]
        y_pred = ["angry", "sad", "joyful"]
        report = compute_metrics(y_true, y_pred)
        table = report.format_table()

        assert "Label" in table
        assert "Precision" in table
        assert "Recall" in table
        assert "F1" in table
        assert "angry" in table

    def test_empty_labels_derived_from_data(self):
        y_true = ["x", "y"]
        y_pred = ["x", "y"]
        report = compute_metrics(y_true, y_pred)
        labels_in_report = {m.label for m in report.per_class}
        assert labels_in_report == {"x", "y"}


# ---------------------------------------------------------------------------
# Model Registry Tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Test the model registry and creation."""

    def test_available_models(self):
        available = ModelRegistry.available()
        assert "logistic_regression" in available
        assert "decision_tree" in available
        assert "random_forest" in available

    def test_create_ser_model(self):
        model = ModelRegistry.create({"type": "logistic_regression", "num_classes": 8})
        assert isinstance(model, SERModel)

    def test_create_text_prosody_model(self):
        model = ModelRegistry.create({"type": "decision_tree", "max_depth": 5})
        assert isinstance(model, TextProsodyModel)

    def test_create_pitch_contour_model(self):
        model = ModelRegistry.create({"type": "random_forest", "n_estimators": 10})
        assert isinstance(model, PitchContourModel)

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelRegistry.create({"type": "nonexistent"})

    def test_unknown_parameter_raises(self):
        """A misspelt hyperparameter is an error, not silently dropped."""
        with pytest.raises(ValueError, match="max_iterations"):
            ModelRegistry.create({"type": "logistic_regression", "max_iterations": 10})


# ---------------------------------------------------------------------------
# SER Model Tests
# ---------------------------------------------------------------------------


class TestSERModel:
    """Test Speech Emotion Recognition model."""

    @pytest.fixture()
    def trained_ser(self):
        rng = np.random.RandomState(42)
        model = SERModel(num_classes=3, labels=["angry", "sad", "neutral"])
        X = rng.randn(30, 7)
        y = np.array(["angry"] * 10 + ["sad"] * 10 + ["neutral"] * 10)
        model.train(X, y)
        return model

    def test_train_and_predict(self, trained_ser):
        X_test = np.random.RandomState(99).randn(5, 7)
        labels = trained_ser.predict_labels(X_test)
        assert len(labels) == 5
        assert all(label in ("angry", "sad", "neutral") for label in labels)

    def test_predict_before_training_raises(self):
        model = SERModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(np.zeros((1, 7)))

    def test_predict_proba(self, trained_ser):
        X = np.zeros((2, 7))
        proba = trained_ser.predict_proba(X)
        assert proba.shape == (2, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_unmeasured_features_are_filled_with_training_medians(self):
        X = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0], [7.0, np.nan]])
        y = np.array(["a", "a", "b", "b"])
        model = SERModel(feature_names=["f0_mean", "hnr"])
        model.train(X, y)
        filled = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0], [7.0, 20.0]])
        np.testing.assert_allclose(model.predict_proba(X), model.predict_proba(filled))

    def test_save_and_load(self, trained_ser, tmp_path):
        trained_ser.save(tmp_path / "ser_ckpt")
        loaded = BaseModel.load(tmp_path / "ser_ckpt")
        assert isinstance(loaded, SERModel)

        X = np.zeros((2, 7))
        orig = trained_ser.predict_labels(X)
        reloaded = loaded.predict_labels(X)
        assert orig == reloaded

    def test_get_params(self, trained_ser):
        params = trained_ser.get_params()
        assert params["type"] == "logistic_regression"
        assert params["trained"] is True
        assert len(params["classes"]) == 3
        assert params["feature_names"] == [f"x{i}" for i in range(7)]


# ---------------------------------------------------------------------------
# Text Prosody Model Tests
# ---------------------------------------------------------------------------


class TestTextProsodyModel:
    """Test text-to-prosody prediction model."""

    @pytest.fixture()
    def trained_text_prosody(self):
        model = TextProsodyModel(max_depth=5)
        words = ["Hello", "world", "this", "is", "great!"]
        X = extract_text_features(words)
        y = np.array(["mid_normal_normal", "mid_normal_normal", "mid_normal_normal",
                       "mid_normal_normal", "high_loud_fast"])
        # Duplicate to have enough data
        X = np.vstack([X] * 4)
        y = np.concatenate([y] * 4)
        model.train(X, y)
        return model

    def test_train_and_predict(self, trained_text_prosody):
        words = ["Another", "test"]
        X = extract_text_features(words)
        labels = trained_text_prosody.predict_labels(X)
        assert len(labels) == 2

    def test_extract_text_features(self):
        words = ["Hello", "world!"]
        features = extract_text_features(words)
        assert features.shape == (2, 7)
        # First word: length=5, position_ratio=0.0, is_capitalized=1.0
        assert features[0, 0] == 5.0  # word_length
        assert features[0, 1] == 0.0  # position_ratio
        assert features[0, 2] == 1.0  # is_capitalized

    def test_extract_selected_text_features(self):
        words = ["Hello", "big", "world!"]
        selected = extract_text_features(words, ["has_punctuation", "word_length"])
        full = extract_text_features(words)
        np.testing.assert_array_equal(selected, full[:, [3, 0]])

    def test_get_params(self, trained_text_prosody):
        params = trained_text_prosody.get_params()
        assert params["type"] == "decision_tree"
        assert params["trained"] is True


# ---------------------------------------------------------------------------
# Pitch Contour Model Tests
# ---------------------------------------------------------------------------


class TestPitchContourModel:
    """Test pitch contour classification model."""

    @pytest.fixture()
    def trained_contour(self):
        rng = np.random.RandomState(42)
        model = PitchContourModel(n_estimators=10, max_depth=4, sequence_length=20)
        # Create synthetic contours
        X_list = []
        y_list = []
        for _ in range(20):
            X_list.append(np.linspace(100, 200, 20) + rng.normal(0, 5, 20))
            y_list.append("rise")
        for _ in range(20):
            X_list.append(np.linspace(200, 100, 20) + rng.normal(0, 5, 20))
            y_list.append("fall")
        X = np.array(X_list)
        y = np.array(y_list)
        model.train(X, y)
        return model

    def test_train_and_predict(self, trained_contour):
        X_test = np.linspace(100, 200, 20).reshape(1, -1)
        labels = trained_contour.predict_labels(X_test)
        assert labels == ["rise"]

    def test_resample_f0(self):
        f0 = [100.0, 150.0, 200.0]
        resampled = resample_f0(f0, target_length=5)
        assert len(resampled) == 5
        assert resampled[0] == pytest.approx(100.0)
        assert resampled[-1] == pytest.approx(200.0)

    def test_resample_f0_empty(self):
        resampled = resample_f0([], target_length=5)
        assert len(resampled) == 5
        assert all(v == 0.0 for v in resampled)

    def test_resample_f0_single(self):
        resampled = resample_f0([180.0], target_length=5)
        assert len(resampled) == 5
        assert all(v == 180.0 for v in resampled)

    def test_get_params(self, trained_contour):
        params = trained_contour.get_params()
        assert params["type"] == "random_forest"
        assert params["trained"] is True


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _span(
    f0: float | None = None,
    intensity: float | None = None,
    *,
    start_ms: int = 0,
    end_ms: int = 500,
    contour: list[float] | None = None,
    rate: float | None = 4.0,
) -> SpanFeatures:
    voiced = f0 is not None
    return SpanFeatures(
        start_ms=start_ms,
        end_ms=end_ms,
        text="w",
        f0_mean=f0,
        f0_range=(f0 * 0.9, f0 * 1.1) if voiced else None,
        f0_contour=contour if contour is not None else ([f0] * 10 if voiced else None),
        intensity_mean=intensity,
        speech_rate=rate,
        jitter=1.0 if voiced else None,
        shimmer=3.0 if voiced else None,
        hnr=15.0 if voiced else None,
    )


class TestFeatures:
    """The feature functions shared by data preparation and inference."""

    def test_word_spans_summarise_like_one_span(self):
        words = [
            _span(100.0, 60.0, start_ms=0, end_ms=500, contour=[100.0] * 10),
            _span(200.0, 70.0, start_ms=500, end_ms=1000, contour=[200.0] * 30),
        ]
        summary = utterance_features(words)
        # F0 is the mean over all voiced samples: (10*100 + 30*200) / 40.
        assert summary["f0_mean"] == pytest.approx(175.0)
        assert summary["f0_range"] == pytest.approx(220.0 - 90.0)
        # Intensity is averaged as power: 60 and 70 dB for equal durations.
        assert summary["intensity_mean"] == pytest.approx(10 * np.log10((1e6 + 1e7) / 2))

    def test_unmeasured_features_are_nan(self):
        vector = feature_vector([_span(None, 55.0)], ["f0_mean", "intensity_mean", "hnr"])
        assert np.isnan(vector[0]) and np.isnan(vector[2])
        assert vector[1] == pytest.approx(55.0)

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown SER feature"):
            feature_vector([_span(100.0, 60.0)], ["loudness"])

    def test_token_labels_come_from_markup(self):
        iml = (
            '<utterance emotion="angry" confidence="0.8">I '
            '<prosody pitch="+20%" volume="+6dB">NEVER</prosody> said '
            '<emphasis level="strong">that</emphasis>, '
            '<prosody rate="slow" pitch="-2st">okay</prosody>?</utterance>'
        )
        labels = token_prosody_labels(
            iml, ["pitch_level", "volume_level", "rate_level", "emphasis_level"]
        )
        assert labels == [
            ("I", "mid_normal_normal_none"),
            ("NEVER", "high_loud_normal_none"),
            ("said", "mid_normal_normal_none"),
            ("that,", "mid_normal_normal_strong"),
            ("okay?", "low_normal_slow_none"),
        ]

    def test_small_or_absolute_pitch_is_mid(self):
        iml = (
            '<utterance><prosody pitch="+3%">a</prosody> <prosody pitch="185Hz">b</prosody> '
            '<segment tempo="rushed">c</segment></utterance>'
        )
        assert token_prosody_labels(iml, ["pitch_level", "rate_level"]) == [
            ("a", "mid_normal"), ("b", "mid_normal"), ("c", "mid_fast"),
        ]

    def test_contour_label_must_cover_the_utterance(self):
        whole = '<utterance><prosody pitch_contour="rise">Really, now?</prosody></utterance>'
        part = '<utterance>Really, <prosody pitch_contour="rise">now?</prosody></utterance>'
        assert utterance_contour_label(whole) == "rise"
        assert utterance_contour_label(part) is None
        assert utterance_contour_label("<utterance>Really.</utterance>") is None


# ---------------------------------------------------------------------------
# The training_synthetic fixture
# ---------------------------------------------------------------------------


class TestSyntheticFixture:
    """The fixture's clips were 10 identical 440 Hz tones behind 8 labels, so
    nothing could be learnt from it. They are now espeak-ng speech whose
    speed, pitch, pitch range and loudness follow the label."""

    def test_clips_are_distinct_short_speech_clips(self):
        digests = set()
        for path in sorted((SYNTHETIC_DATASET / "audio").glob("*.wav")):
            with wave.open(str(path)) as w:
                assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16_000, 1, 2)
                assert w.getnframes() <= 2 * 16_000
            digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
        assert len(digests) == 10

    def test_iml_holds_the_spoken_words(self):
        parser = IMLParser()
        for path in sorted((SYNTHETIC_DATASET / "entries").glob("*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            doc = parser.parse(entry["iml"])
            assert parser.to_plain_text(doc) == entry["transcript"]
            assert doc.utterances[0].emotion == entry["emotion_label"]

    @needs_audio
    def test_measured_delivery_follows_the_labels(self):
        by_label: dict[str, list[SpanFeatures]] = {}
        for path in sorted((SYNTHETIC_DATASET / "entries").glob("*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            features = recording_features(SYNTHETIC_DATASET / entry["audio_file"])
            by_label.setdefault(entry["emotion_label"], []).append(features)

        def mean(label: str, name: str) -> float:
            return float(np.mean([getattr(f, name) for f in by_label[label]]))

        def f0_span(label: str) -> float:
            return float(np.mean([f.f0_range[1] - f.f0_range[0] for f in by_label[label]]))

        assert mean("angry", "f0_mean") > 1.5 * mean("sad", "f0_mean")
        assert mean("angry", "intensity_mean") > mean("sad", "intensity_mean") + 10
        assert min(f.speech_rate for f in by_label["angry"]) > by_label["sad"][0].speech_rate
        assert f0_span("joyful") > 1.5 * f0_span("neutral")
        for label in ("calm", "sad"):
            assert mean(label, "intensity_mean") < mean("neutral", "intensity_mean") - 3
            assert mean(label, "f0_mean") < mean("neutral", "f0_mean")
        assert max(by_label, key=lambda label: mean(label, "intensity_mean")) == "angry"


# ---------------------------------------------------------------------------
# Data Preparation Tests
# ---------------------------------------------------------------------------


@needs_audio
class TestDataPrep:
    """Data preparation derives features from the recordings and text, never from labels."""

    @single_speaker
    def test_prepare_ser_data(self, tmp_path):
        stats = _prepare("prepare_ser_data", SER_CONFIG, SYNTHETIC_DATASET, tmp_path)

        assert "train" in stats
        assert stats["train"] > 0
        assert (tmp_path / "train" / "X.npy").exists()
        assert (tmp_path / "train" / "y.npy").exists()

        X = np.load(tmp_path / "train" / "X.npy")
        y = np.load(tmp_path / "train" / "y.npy")
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 7  # 7 features
        meta = json.loads((tmp_path / "prep_metadata.json").read_text(encoding="utf-8"))
        assert meta["task"] == "ser"
        assert len(meta["feature_names"]) == 7

    def test_ser_features_do_not_depend_on_labels(self, tmp_path, emotion_dataset):
        relabelled = _relabel(
            emotion_dataset, tmp_path / "relabelled", _shuffled_labels(emotion_dataset, 0)
        )
        _prepare("prepare_ser_data", SER_CONFIG, emotion_dataset, tmp_path / "a")
        _prepare("prepare_ser_data", SER_CONFIG, relabelled, tmp_path / "b")
        for split in ("train", "val", "test"):
            np.testing.assert_array_equal(
                np.load(tmp_path / "a" / split / "X.npy"), np.load(tmp_path / "b" / split / "X.npy")
            )
        assert _all_splits(tmp_path / "a")[1] != _all_splits(tmp_path / "b")[1]

    def test_ser_features_come_from_the_audio(self, tmp_path, emotion_dataset):
        """Features follow the audio: a louder, higher clip gives a louder, higher row."""
        changed = tmp_path / "changed"
        shutil.copytree(emotion_dataset, changed)
        for wav in (changed / "audio").glob("sad_*.wav"):
            _write_clip(wav, [300.0, 300.0], 0.9)
        _prepare("prepare_ser_data", SER_CONFIG, emotion_dataset, tmp_path / "a")
        _prepare("prepare_ser_data", SER_CONFIG, changed, tmp_path / "b")
        X_a, y_a = _all_splits(tmp_path / "a")
        X_b, y_b = _all_splits(tmp_path / "b")
        assert y_a == y_b
        sad = np.array([label == "sad" for label in y_a])
        # f0_mean and intensity_mean columns
        assert np.all(X_b[sad, 0] > 280) and np.all(X_a[sad, 0] < 140)
        assert np.all(X_b[sad, 2] > X_a[sad, 2] + 10)
        np.testing.assert_array_equal(X_a[~sad], X_b[~sad])

    def test_missing_audio_is_an_error(self, tmp_path, emotion_dataset):
        broken = tmp_path / "broken"
        shutil.copytree(emotion_dataset, broken)
        (broken / "audio" / "calm_00.wav").unlink()
        with pytest.raises(DatasetError, match="calm_00"):
            _prepare("prepare_ser_data", SER_CONFIG, broken, tmp_path / "out")

    def test_entries_without_speech_are_skipped(self, tmp_path, emotion_dataset):
        silent = tmp_path / "silent"
        shutil.copytree(emotion_dataset, silent)
        _write_silence(silent / "audio" / "angry_03.wav")
        with pytest.warns(UserWarning, match="Skipped 1 entries: no voiced speech"):
            stats = _prepare("prepare_ser_data", SER_CONFIG, silent, tmp_path / "out")
        assert sum(stats.values()) == len(_emotion_clips(1, 8)) - 1
        meta = json.loads((tmp_path / "out" / "prep_metadata.json").read_text(encoding="utf-8"))
        assert meta["skipped"] == {"no voiced speech in the audio": ["angry_03"]}

    def test_configured_features_select_columns(self, tmp_path, emotion_dataset):
        config = _write_config(
            tmp_path / "c.yaml", SER_CONFIG, data={"features": ["hnr", "f0_mean"]}
        )
        _prepare("prepare_ser_data", SER_CONFIG, emotion_dataset, tmp_path / "all")
        _prepare("prepare_ser_data", config, emotion_dataset, tmp_path / "two")
        X_all, _ = _all_splits(tmp_path / "all")
        X_two, _ = _all_splits(tmp_path / "two")
        np.testing.assert_array_equal(X_two, X_all[:, [6, 0]])

    def test_prepare_text_prosody_data(self, tmp_path, text_dataset):
        stats = _prepare("prepare_text_prosody_data", TEXT_CONFIG, text_dataset, tmp_path)

        assert sum(stats.values()) == 10
        X, y = _all_splits(tmp_path)
        assert X.shape == (10 * 6, 7)  # 6 tokens per entry, 7 text features
        # The labels come from the markup: only the marked word is high and loud.
        assert y.count("high_loud_normal") == 10
        assert y.count("mid_normal_normal") == 50

    def test_text_labels_do_not_depend_on_emotion(self, tmp_path, text_dataset):
        relabelled = _relabel(
            text_dataset, tmp_path / "relabelled", {f"text_{i:02d}": "angry" for i in range(10)}
        )
        _prepare("prepare_text_prosody_data", TEXT_CONFIG, text_dataset, tmp_path / "a")
        _prepare("prepare_text_prosody_data", TEXT_CONFIG, relabelled, tmp_path / "b")
        assert _all_splits(tmp_path / "a")[1] == _all_splits(tmp_path / "b")[1]

    def test_prepare_pitch_contour_data(self, tmp_path, contour_dataset):
        stats = _prepare("prepare_pitch_contour_data", CONTOUR_CONFIG, contour_dataset, tmp_path)

        assert sum(stats.values()) == 4 * 6
        X, y = _all_splits(tmp_path)
        assert X.shape == (24, 20)  # sequence_length
        # Semitones relative to the clip's median: a rise goes from below 0 to above.
        rises = X[[label == "rise" for label in y]]
        assert np.all(rises[:, 0] < -1.0) and np.all(rises[:, -1] > 1.0)

    @single_speaker
    def test_unannotated_contours_are_skipped(self, tmp_path):
        with pytest.warns(UserWarning, match="Skipped 10 entries: no pitch_contour"):
            stats = _prepare(
                "prepare_pitch_contour_data", CONTOUR_CONFIG, SYNTHETIC_DATASET, tmp_path
            )
        assert sum(stats.values()) == 0

    def test_data_prep_is_reproducible_across_processes(self, tmp_path, contour_dataset):
        """No per-process randomness (such as Python's salted hash()) leaks into the data."""
        outputs = []
        for hash_seed in ("1", "2"):
            out = tmp_path / f"run{hash_seed}"
            env = {**os.environ, "PYTHONHASHSEED": hash_seed}
            subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "data_prep.py"), "--config",
                 str(CONTOUR_CONFIG), "--dataset", str(contour_dataset), "--output", str(out)],
                check=True, capture_output=True, env=env,
            )
            outputs.append((out / "train" / "X.npy").read_bytes())
        assert outputs[0] == outputs[1]


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------


def _heldout_accuracy(checkpoint: Path, prepared: Path) -> float:
    model = BaseModel.load(checkpoint)
    X, y = _all_splits(prepared)
    return float(np.mean([p == t for p, t in zip(model.predict_labels(X), y, strict=True)]))


@needs_audio
class TestEndToEnd:
    """Test full training pipeline for each task."""

    def test_ser_pipeline(self, tmp_path, emotion_dataset):
        from training.scripts.evaluate import evaluate
        from training.scripts.export import export_model
        from training.scripts.train import train

        # Train
        results = train(
            config_path=SER_CONFIG, dataset_dir=emotion_dataset, output_dir=tmp_path / "ckpt"
        )
        assert results["train_metrics"]["accuracy"] > 0.9
        assert results["classes"] == sorted(_EMOTIONS)

        # Evaluate, with the config saved in the checkpoint
        report = evaluate(checkpoint_path=tmp_path / "ckpt", dataset_dir=emotion_dataset,
                          split="test")
        assert report.total_samples > 0

        # Export
        meta = export_model(tmp_path / "ckpt", tmp_path / "export")
        assert meta["model_class"] == "SERModel"
        assert meta["files"] == ["config.json", MODEL_FILE]

        # Load and infer
        model = PortableModel.load(tmp_path / "export")
        labels = model.predict_labels(np.zeros((1, 7)))
        assert len(labels) == 1

    def test_ser_model_learns_from_audio_not_labels(self, tmp_path, emotion_dataset,
                                                    emotion_heldout):
        """Trained on the real labels, the model labels unseen clips correctly; trained on
        shuffled labels (same audio), it cannot."""
        from training.scripts.train import train

        _prepare("prepare_ser_data", SER_CONFIG, emotion_heldout, tmp_path / "heldout")
        train(config_path=SER_CONFIG, dataset_dir=emotion_dataset, output_dir=tmp_path / "real")
        assert _heldout_accuracy(tmp_path / "real", tmp_path / "heldout") >= 0.9

        # Chance is 1/3; a single shuffle can land well above it, so average a few.
        shuffled_accuracies = []
        for seed in (0, 1, 2):
            labels = _shuffled_labels(emotion_dataset, seed)
            shuffled = _relabel(emotion_dataset, tmp_path / f"shuffled{seed}", labels)
            ckpt = tmp_path / f"noise{seed}"
            train(config_path=SER_CONFIG, dataset_dir=shuffled, output_dir=ckpt)
            shuffled_accuracies.append(_heldout_accuracy(ckpt, tmp_path / "heldout"))
        assert np.mean(shuffled_accuracies) <= 0.6

    def test_text_prosody_pipeline(self, tmp_path, text_dataset):
        from training.scripts.train import train

        results = train(
            config_path=TEXT_CONFIG, dataset_dir=text_dataset, output_dir=tmp_path / "ckpt"
        )
        assert results["task"] == "text_to_prosody"
        assert results["train_samples"] > 0
        assert results["classes"] == ["high_loud_normal", "mid_normal_normal"]

        # Load and infer
        model = BaseModel.load(tmp_path / "ckpt")
        X = extract_text_features(["I", "NEVER", "said", "that", "to", "you"])
        labels = model.predict_labels(X)
        assert labels[1] == "high_loud_normal"

    def test_pitch_contour_pipeline(self, tmp_path, contour_dataset):
        from training.scripts.train import train

        results = train(
            config_path=CONTOUR_CONFIG, dataset_dir=contour_dataset, output_dir=tmp_path / "ckpt"
        )
        assert results["task"] == "pitch_contour"
        assert results["train_samples"] > 0

        heldout = _write_dataset(tmp_path / "heldout", _contour_clips(4, 3))
        _prepare("prepare_pitch_contour_data", CONTOUR_CONFIG, heldout, tmp_path / "prep")
        assert _heldout_accuracy(tmp_path / "ckpt", tmp_path / "prep") >= 0.9

    def test_training_results_saved(self, tmp_path, emotion_dataset):
        """Verify training results are persisted alongside checkpoint."""
        from training.scripts.train import train

        train(config_path=SER_CONFIG, dataset_dir=emotion_dataset, output_dir=tmp_path / "ckpt")

        results_file = tmp_path / "ckpt" / "training_results.json"
        assert results_file.exists()
        with open(results_file) as f:
            saved = json.load(f)
        assert saved["task"] == "ser"
        assert "train_metrics" in saved
        assert saved["feature_names"] == list(load_config(SER_CONFIG).data["features"])

    def test_prepared_data_for_another_task_is_rejected(self, tmp_path, text_dataset):
        from training.scripts.train import train

        _prepare("prepare_text_prosody_data", TEXT_CONFIG, text_dataset, tmp_path / "prep")
        with pytest.raises(ValueError, match="prepared for task 'text_to_prosody'"):
            train(config_path=SER_CONFIG, prepared_data=tmp_path / "prep")

    def test_labels_outside_the_config_are_rejected(self, tmp_path, emotion_dataset):
        from training.scripts.train import train

        config = _write_config(tmp_path / "c.yaml", SER_CONFIG, model={"labels": ["angry", "sad"]})
        with pytest.raises(TrainingError, match=r"labels the config does not list: \['calm'\]"):
            train(config_path=config, dataset_dir=emotion_dataset)


# ---------------------------------------------------------------------------
# Portable (pickle-free) export
# ---------------------------------------------------------------------------


class TestPortableExport:
    """JSON exports reproduce the scikit-learn models without unpickling anything."""

    @pytest.mark.parametrize("n_classes", [2, 3])
    def test_logistic_regression_matches_sklearn(self, n_classes):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 4)) * [50.0, 5.0, 1.0, 0.1] + [200.0, 70.0, 4.0, 1.0]
        y = np.array((["angry", "calm", "sad"][:n_classes] * 30)[:60])
        X[::7, 1] = np.nan
        model = SERModel(feature_names=["f0_mean", "intensity_mean", "speech_rate", "jitter"])
        model.train(X, y)
        portable = PortableModel(model.portable_params())
        np.testing.assert_allclose(portable.predict_proba(X), model.predict_proba(X), atol=1e-12)
        assert portable.predict_labels(X) == model.predict_labels(X)

    @pytest.mark.parametrize("model_class", [TextProsodyModel, PitchContourModel])
    def test_trees_match_sklearn(self, model_class):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 5))
        y = np.where(X[:, 0] + X[:, 1] ** 2 > 0.5, "high", np.where(X[:, 2] > 0, "mid", "low"))
        model = model_class()
        model.train(X, y)
        portable = PortableModel(model.portable_params())
        X_new = rng.normal(size=(200, 5))
        np.testing.assert_allclose(
            portable.predict_proba(X_new), model._classifier.predict_proba(X_new), atol=1e-12
        )
        assert portable.predict_labels(X_new) == model.predict_labels(X_new)

    def test_json_and_pickle_exports_differ(self, tmp_path):
        model = SERModel()
        model.train(np.random.default_rng(2).normal(size=(20, 7)), np.array(["a", "b"] * 10))
        json_files = model.export(tmp_path / "json", "json")
        pickle_files = model.export(tmp_path / "pickle", "pickle")
        assert json_files == ["config.json", MODEL_FILE]
        assert pickle_files == ["config.json", "model.joblib"]
        assert not (tmp_path / "json" / "model.joblib").exists()
        # The JSON file holds the fitted parameters, not a pickle.
        params = json.loads((tmp_path / "json" / MODEL_FILE).read_text(encoding="utf-8"))
        assert len(params["estimator"]["coef"][0]) == 7
        assert isinstance(BaseModel.load(tmp_path / "pickle"), SERModel)

    def test_loading_json_never_unpickles(self, tmp_path, monkeypatch):
        model = SERModel()
        model.train(np.random.default_rng(3).normal(size=(20, 7)), np.array(["a", "b"] * 10))
        model.export(tmp_path, "json")

        def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError("joblib.load called")

        monkeypatch.setattr(joblib, "load", refuse)
        assert PortableModel.load(tmp_path).classes == ("a", "b")

    @pytest.mark.parametrize(
        "change",
        [
            {"format": "pickle"},
            {"classes": []},
            {"estimator": {"kind": "linear", "coef": [[1.0]], "intercept": [0.0]}},
            {"estimator": {"kind": "code", "source": "print('hi')"}},
        ],
    )
    def test_malformed_model_file_raises(self, tmp_path, change):
        model = SERModel()
        model.train(np.random.default_rng(4).normal(size=(20, 7)), np.array(["a", "b"] * 10))
        (tmp_path / MODEL_FILE).write_text(
            json.dumps({**model.portable_params(), **change}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Invalid portable model"):
            PortableModel.load(tmp_path)

    def test_checkpoint_load_has_no_pickle_fallback(self, tmp_path):
        """Only model.joblib is loaded; a stray legacy model.pkl is not unpickled."""
        joblib.dump(SERModel(), tmp_path / "model.pkl")
        with pytest.raises(FileNotFoundError, match=r"model\.joblib"):
            BaseModel.load(tmp_path)

    @pytest.mark.parametrize("model_class", [SERModel, TextProsodyModel, PitchContourModel])
    def test_exports_record_training_feature_stats(self, model_class):
        X = np.array([[100.0, 1.0, 5.0], [120.0, np.nan, 5.0], [140.0, 3.0, 5.0]])
        model = model_class(feature_names=["f0_mean", "jitter", "hnr"])
        model.train(X, np.array(["a", "b", "a"]))
        params = model.portable_params()
        stats = params["feature_stats"]
        np.testing.assert_allclose(stats["mean"], [120.0, 2.0, 5.0])
        # NaN (unmeasured) values are left out; a constant feature has no spread.
        np.testing.assert_allclose(stats["std"], [np.std([100.0, 120.0, 140.0]), 1.0, 0.0])
        portable = PortableModel(params)
        assert portable.has_feature_stats
        z = portable.feature_z_scores([[160.0, np.nan, 50.0]])[0]
        assert z[0] == pytest.approx(40.0 / np.std([100.0, 120.0, 140.0]))
        assert np.isnan(z[1]) and np.isnan(z[2])

    def test_older_exports_fall_back_to_the_scaler(self):
        model = SERModel(feature_names=["f0_mean", "jitter"])
        model.train(np.array([[100.0, 1.0], [140.0, 3.0]]), np.array(["a", "b"]))
        params = model.portable_params()
        del params["feature_stats"]
        portable = PortableModel(params)
        assert portable.has_feature_stats
        np.testing.assert_allclose(portable.feature_z_scores([[180.0, 2.0]])[0], [3.0, 0.0])

    def test_models_without_statistics_check_nothing(self):
        model = TextProsodyModel(feature_names=["f0_mean"])
        model.train(np.array([[1.0], [5.0]]), np.array(["a", "b"]))
        params = model.portable_params()
        del params["feature_stats"]
        portable = PortableModel(params)
        assert not portable.has_feature_stats
        assert np.isnan(portable.feature_z_scores([[1000.0]])).all()

    @pytest.mark.parametrize(
        "stats",
        [[1.0], {"mean": [1.0, 2.0]}, {"mean": [1.0], "std": [1.0]},
         {"mean": [1.0, 2.0], "std": [1.0, -1.0]}],
    )
    def test_malformed_feature_stats_raise(self, stats):
        model = SERModel(feature_names=["f0_mean", "jitter"])
        model.train(np.array([[100.0, 1.0], [140.0, 3.0]]), np.array(["a", "b"]))
        with pytest.raises(ValueError, match="feature_stats"):
            PortableModel({**model.portable_params(), "feature_stats": stats})


# ---------------------------------------------------------------------------
# Using a trained model from the SDK
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory: pytest.TempPathFactory, emotion_dataset: Path) -> Path:
    """A JSON export of an SER model trained on *emotion_dataset* (checkpoint next to it)."""
    from training.scripts.export import export_model
    from training.scripts.train import train

    root = tmp_path_factory.mktemp("ser_model")
    train(config_path=SER_CONFIG, dataset_dir=emotion_dataset, output_dir=root / "ckpt")
    export_model(root / "ckpt", root / "export")
    return root / "export"


@needs_audio
class TestTrainedEmotionClassifier:
    """A trained SER model plugged into AudioToIML."""

    def test_matches_the_trained_model(self, export_dir, emotion_heldout):
        from training.features import recording_features
        from training.inference import TrainedEmotionClassifier

        classifier = TrainedEmotionClassifier(export_dir)
        checkpoint = BaseModel.load(export_dir.parent / "ckpt")
        assert isinstance(checkpoint, SERModel)
        span = recording_features(emotion_heldout / "audio" / "sad_00.wav")
        expected = checkpoint.predict_proba(
            feature_vector([span], checkpoint.feature_names)[None, :]
        )[0]
        probabilities = classifier.probabilities([span])
        assert list(probabilities) == list(checkpoint.get_params()["classes"])
        np.testing.assert_allclose(list(probabilities.values()), expected, atol=1e-12)
        label, confidence = classifier.classify([span])
        assert label == "sad"
        assert confidence == round(float(expected.max()), 2)

    def test_from_checkpoint(self, export_dir):
        from training.inference import TrainedEmotionClassifier

        classifier = TrainedEmotionClassifier.from_checkpoint(export_dir.parent / "ckpt")
        assert classifier.labels == tuple(sorted(_EMOTIONS))

    def test_no_speech_gives_no_emotion(self, export_dir):
        from training.inference import TrainedEmotionClassifier

        classifier = TrainedEmotionClassifier(export_dir)
        assert classifier.classify([_span(None, None, rate=None)]) == ("neutral", 0.0)
        assert classifier.classify([]) == ("neutral", 0.0)

    def test_audio_to_iml_uses_the_trained_model(self, export_dir, emotion_heldout):
        from prosody_protocol import AudioToIML, IMLParser
        from training.inference import TrainedEmotionClassifier

        converter = AudioToIML(emotion_classifier=TrainedEmotionClassifier(export_dir))
        for label in _EMOTIONS:
            iml = converter.convert(
                emotion_heldout / "audio" / f"{label}_01.wav", transcript="I see what you mean"
            )
            utterance = IMLParser().parse(iml).utterances[0]
            assert utterance.emotion == label
            assert utterance.confidence is not None and utterance.confidence >= 0.5

    def test_rejects_models_of_other_tasks(self, tmp_path):
        from training.inference import TrainedEmotionClassifier

        model = TextProsodyModel(feature_names=["word_length", "position_ratio"])
        model.train(np.array([[1.0, 0.0], [5.0, 1.0]]), np.array(["a", "b"]))
        model.export(tmp_path, "json")
        with pytest.raises(ValueError, match="not trained on SER features"):
            TrainedEmotionClassifier(tmp_path)

    @pytest.mark.parametrize("tone", ["tone_gap_tone.wav", "tone_440hz.wav", "rising_pitch.wav"])
    def test_abstains_on_input_unlike_the_training_data(self, export_dir, tone):
        """Sine tones used to be labeled with near certainty (joyful 0.99)."""
        from training.inference import TrainedEmotionClassifier

        span = recording_features(FIXTURES / "audio" / tone)
        classifier = TrainedEmotionClassifier(export_dir)
        unusual = classifier.unusual_features([span])
        assert unusual and all(abs(z) > 4.0 for z in unusual.values())
        assert classifier.classify([span]) == ("neutral", 0.0)
        # The check can be turned off, which gives the model's raw answer.
        label, confidence = TrainedEmotionClassifier(export_dir, max_feature_z=None).classify(
            [span]
        )
        assert label in _EMOTIONS and confidence > 0.0

    def test_audio_to_iml_leaves_out_emotion_for_non_speech(self, export_dir):
        from prosody_protocol import AudioToIML, IMLParser
        from training.inference import TrainedEmotionClassifier

        converter = AudioToIML(emotion_classifier=TrainedEmotionClassifier(export_dir))
        iml = converter.convert(FIXTURES / "audio" / "tone_gap_tone.wav", transcript="Hello there")
        utterance = IMLParser().parse(iml).utterances[0]
        assert utterance.emotion is None and utterance.confidence is None

    def test_in_distribution_input_is_not_unusual(self, export_dir, emotion_heldout):
        from training.inference import TrainedEmotionClassifier

        classifier = TrainedEmotionClassifier(export_dir)
        for label in _EMOTIONS:
            span = recording_features(emotion_heldout / "audio" / f"{label}_00.wav")
            assert classifier.unusual_features([span]) == {}

    @pytest.mark.parametrize("value", [0, -1.0, float("inf"), float("nan"), True, "4"])
    def test_invalid_max_feature_z(self, export_dir, value):
        from training.inference import TrainedEmotionClassifier

        with pytest.raises(ValueError, match="max_feature_z"):
            TrainedEmotionClassifier(export_dir, max_feature_z=value)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / name), *args],
        capture_output=True, text=True, check=False,
    )


@needs_audio
class TestCommandLine:
    """The documented commands work and fail with a message, not a traceback."""

    def test_documented_commands(self, tmp_path, emotion_dataset):
        prepared, ckpt, export = tmp_path / "prepared", tmp_path / "ckpt", tmp_path / "export"
        steps = [
            ("data_prep.py", "--config", str(SER_CONFIG), "--dataset", str(emotion_dataset),
             "--output", str(prepared)),
            ("train.py", "--config", str(SER_CONFIG), "--prepared-data", str(prepared),
             "--output", str(ckpt)),
            ("evaluate.py", "--checkpoint", str(ckpt), "--dataset", str(emotion_dataset),
             "--split", "test"),
            ("export.py", "--checkpoint", str(ckpt), "--output", str(export)),
        ]
        outputs = []
        for script, *args in steps:
            result = _run_script(script, *args)
            assert result.returncode == 0, result.stderr
            outputs.append(result.stdout)
        assert "macro avg" in outputs[2]
        assert "Format: json" in outputs[3]
        assert (export / MODEL_FILE).exists()

    def test_unknown_model_type_is_a_one_line_error(self, tmp_path, emotion_dataset):
        config = tmp_path / "nn.yaml"
        config.write_text(
            SER_CONFIG.read_text(encoding="utf-8").replace(
                "type: logistic_regression", "type: wav2vec2"
            ),
            encoding="utf-8",
        )
        result = _run_script(
            "train.py", "--config", str(config), "--dataset", str(emotion_dataset),
            "--output", str(tmp_path / "ckpt"),
        )
        assert result.returncode == 1
        assert result.stderr.startswith("Error: Unknown model type 'wav2vec2'")
        assert "Traceback" not in result.stderr

    def test_malformed_yaml_is_a_one_line_error(self, tmp_path, emotion_dataset):
        """A YAML syntax error used to print a traceback."""
        config = tmp_path / "bad.yaml"
        config.write_text("task: [unclosed\n", encoding="utf-8")
        result = _run_script(
            "train.py", "--config", str(config), "--dataset", str(emotion_dataset),
            "--output", str(tmp_path / "ckpt"),
        )
        assert result.returncode == 1
        assert result.stderr.startswith(f"Error: {config} is not valid YAML: ")
        assert "(line 2, column 1)" in result.stderr
        assert result.stderr.count("\n") == 1, result.stderr

    def test_malformed_checkpoint_config_is_a_one_line_error(self, tmp_path, emotion_dataset):
        """evaluate.py and export.py read the checkpoint's copy of the config."""
        from training.scripts.train import train

        ckpt = tmp_path / "ckpt"
        train(config_path=SER_CONFIG, dataset_dir=emotion_dataset, output_dir=ckpt)
        (ckpt / "config.yaml").write_text("task: [unclosed\n", encoding="utf-8")
        for script, *args in [
            ("evaluate.py", "--checkpoint", str(ckpt), "--dataset", str(emotion_dataset)),
            ("export.py", "--checkpoint", str(ckpt), "--output", str(tmp_path / "export")),
        ]:
            result = _run_script(script, *args)
            assert result.returncode == 1, script
            assert result.stderr.startswith("Error: "), (script, result.stderr)
            assert "is not valid YAML" in result.stderr and "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_training_set_raises(self, tmp_path):
        """Training on empty data raises TrainingError."""
        from training.scripts.train import train

        empty_dataset = tmp_path / "empty_ds"
        (empty_dataset / "entries").mkdir(parents=True)
        (empty_dataset / "metadata.json").write_text('{"name":"empty","version":"0.1.0","size":0}')

        with pytest.raises(TrainingError, match="empty"):
            train(
                config_path=TEXT_CONFIG,
                dataset_dir=empty_dataset,
                output_dir=tmp_path / "ckpt",
            )

    def test_load_nonexistent_checkpoint_raises(self):
        with pytest.raises(FileNotFoundError):
            BaseModel.load("/nonexistent/checkpoint")

    def test_model_save_creates_directory(self, tmp_path):
        model = SERModel(num_classes=2)
        X = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]], dtype=np.float64)
        y = np.array(["a", "b"])
        model.train(X, y)

        deep_path = tmp_path / "a" / "b" / "c"
        model.save(deep_path)
        assert (deep_path / "model.joblib").exists()

    def test_evaluation_report_dict_serializable(self):
        report = compute_metrics(["a", "b"], ["a", "b"])
        d = report.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_evaluate_without_saved_config_needs_config(self, tmp_path):
        from training.scripts.evaluate import evaluate

        model = SERModel()
        model.train(np.random.default_rng(5).normal(size=(10, 7)), np.array(["a", "b"] * 5))
        model.save(tmp_path / "ckpt")
        with pytest.raises(ValueError, match="pass the training config with --config"):
            evaluate(tmp_path / "ckpt", dataset_dir=tmp_path)
