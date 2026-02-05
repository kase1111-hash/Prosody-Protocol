"""Tests for Phase 11: Model Training Pipelines.

Covers acceptance criteria:
- Training script runs end-to-end on a small synthetic dataset (10 samples)
- Evaluation script produces precision/recall/F1 per emotion class
- Exported model loads and runs inference via the SDK classes
- Configs are YAML; no hardcoded hyperparameters in scripts
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from training.config import TrainingConfig, load_config
from training.metrics import ClassMetrics, EvaluationReport, compute_metrics
from training.models import ModelRegistry, SERModel, TextProsodyModel, PitchContourModel
from training.models.base import BaseModel
from training.models.text_prosody import extract_text_features
from training.models.pitch_contour import resample_f0

FIXTURES = Path(__file__).parent / "fixtures"
CONFIGS_DIR = _PROJECT_ROOT / "training" / "configs"
SYNTHETIC_DATASET = FIXTURES / "datasets" / "training_synthetic"


# ---------------------------------------------------------------------------
# Acceptance Criteria Tests
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """Test the four acceptance criteria from the execution guide."""

    def test_training_runs_end_to_end_on_synthetic_dataset(self, tmp_path):
        """AC1: Training script runs end-to-end on a small synthetic dataset (10 samples)."""
        from training.scripts.train import train

        results = train(
            config_path=CONFIGS_DIR / "ser_wav2vec2.yaml",
            dataset_dir=SYNTHETIC_DATASET,
            output_dir=tmp_path / "checkpoint",
        )

        assert results["task"] == "ser"
        assert results["train_samples"] > 0
        assert "accuracy" in results["train_metrics"]
        assert (tmp_path / "checkpoint" / "model.pkl").exists()
        assert (tmp_path / "checkpoint" / "metadata.json").exists()

    def test_evaluation_produces_precision_recall_f1(self, tmp_path):
        """AC2: Evaluation script produces precision/recall/F1 per emotion class."""
        from training.scripts.train import train
        from training.scripts.evaluate import evaluate

        # Train first
        train(
            config_path=CONFIGS_DIR / "ser_wav2vec2.yaml",
            dataset_dir=SYNTHETIC_DATASET,
            output_dir=tmp_path / "checkpoint",
        )

        # Evaluate
        report = evaluate(
            checkpoint_path=tmp_path / "checkpoint",
            dataset_dir=SYNTHETIC_DATASET,
            split="test",
            config_path=CONFIGS_DIR / "ser_wav2vec2.yaml",
        )

        assert isinstance(report, EvaluationReport)
        assert len(report.per_class) > 0
        for m in report.per_class:
            assert isinstance(m, ClassMetrics)
            assert 0.0 <= m.precision <= 1.0
            assert 0.0 <= m.recall <= 1.0
            assert 0.0 <= m.f1 <= 1.0
            assert m.support >= 0

        assert 0.0 <= report.macro_precision <= 1.0
        assert 0.0 <= report.macro_recall <= 1.0
        assert 0.0 <= report.macro_f1 <= 1.0
        assert report.total_samples > 0

    def test_exported_model_loads_and_runs_inference(self, tmp_path):
        """AC3: Exported model loads and runs inference via the SDK classes."""
        from training.scripts.train import train
        from training.scripts.export import export_model

        # Train
        train(
            config_path=CONFIGS_DIR / "ser_wav2vec2.yaml",
            dataset_dir=SYNTHETIC_DATASET,
            output_dir=tmp_path / "checkpoint",
        )

        # Export
        export_model(
            checkpoint_path=tmp_path / "checkpoint",
            output_path=tmp_path / "export",
        )

        assert (tmp_path / "export" / "config.json").exists()
        assert (tmp_path / "export" / "model.pkl").exists()
        assert (tmp_path / "export" / "export_metadata.json").exists()

        # Load exported model and run inference
        model = BaseModel.load(tmp_path / "export")
        X = np.random.RandomState(42).randn(3, 7)
        labels = model.predict_labels(X)
        assert len(labels) == 3
        assert all(isinstance(l, str) for l in labels)

    def test_configs_are_yaml_no_hardcoded_hyperparams(self):
        """AC4: Configs are YAML; no hardcoded hyperparameters in scripts."""
        # Verify all configs are valid YAML
        for config_file in CONFIGS_DIR.glob("*.yaml"):
            with open(config_file) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{config_file.name} is not a valid YAML mapping"
            assert "task" in data, f"{config_file.name} missing 'task' key"
            assert "model" in data, f"{config_file.name} missing 'model' key"
            assert "training" in data, f"{config_file.name} missing 'training' key"

        # Verify scripts read hyperparameters from config, not hardcoded
        scripts_dir = _PROJECT_ROOT / "training" / "scripts"
        for script_file in scripts_dir.glob("*.py"):
            content = script_file.read_text()
            # Scripts should use load_config or config objects, not hardcode learning rates etc.
            # Check that no raw float hyperparameters are assigned directly
            assert "learning_rate = 0." not in content, (
                f"{script_file.name} has hardcoded learning_rate"
            )
            assert "max_iter = " not in content or "config" in content, (
                f"{script_file.name} may have hardcoded max_iter"
            )


# ---------------------------------------------------------------------------
# Config Loading Tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Test YAML config loading and validation."""

    def test_load_ser_config(self):
        config = load_config(CONFIGS_DIR / "ser_wav2vec2.yaml")
        assert config.task == "ser"
        assert config.model_type == "logistic_regression"
        assert "labels" in config.model
        assert len(config.labels) == 8

    def test_load_text_prosody_config(self):
        config = load_config(CONFIGS_DIR / "text_to_prosody_bert.yaml")
        assert config.task == "text_to_prosody"
        assert config.model_type == "decision_tree"
        assert "max_depth" in config.model

    def test_load_pitch_contour_config(self):
        config = load_config(CONFIGS_DIR / "pitch_contour_cnn.yaml")
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
        config = load_config(CONFIGS_DIR / "ser_wav2vec2.yaml")
        assert "precision" in config.metrics
        assert "recall" in config.metrics
        assert "f1" in config.metrics
        assert config.average == "macro"


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
        assert all(l in ("angry", "sad", "neutral") for l in labels)

    def test_predict_before_training_raises(self):
        model = SERModel()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(np.zeros((1, 7)))

    def test_predict_proba(self, trained_ser):
        X = np.zeros((2, 7))
        proba = trained_ser.predict_proba(X)
        assert proba.shape == (2, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

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
        assert len(labels) == 1
        assert labels[0] in ("rise", "fall")

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
# Data Preparation Tests
# ---------------------------------------------------------------------------


class TestDataPrep:
    """Test data preparation scripts."""

    def test_prepare_ser_data(self, tmp_path):
        from training.scripts.data_prep import prepare_ser_data
        from training.config import load_config

        config = load_config(CONFIGS_DIR / "ser_wav2vec2.yaml")
        stats = prepare_ser_data(SYNTHETIC_DATASET, config.data, tmp_path)

        assert "train" in stats
        assert stats["train"] > 0
        assert (tmp_path / "train" / "X.npy").exists()
        assert (tmp_path / "train" / "y.npy").exists()

        X = np.load(tmp_path / "train" / "X.npy")
        y = np.load(tmp_path / "train" / "y.npy")
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 7  # 7 features

    def test_prepare_text_prosody_data(self, tmp_path):
        from training.scripts.data_prep import prepare_text_prosody_data
        from training.config import load_config

        config = load_config(CONFIGS_DIR / "text_to_prosody_bert.yaml")
        stats = prepare_text_prosody_data(SYNTHETIC_DATASET, config.data, tmp_path)

        assert "train" in stats
        assert (tmp_path / "train" / "X.npy").exists()
        X = np.load(tmp_path / "train" / "X.npy")
        assert X.shape[1] == 7  # text features

    def test_prepare_pitch_contour_data(self, tmp_path):
        from training.scripts.data_prep import prepare_pitch_contour_data
        from training.config import load_config

        config = load_config(CONFIGS_DIR / "pitch_contour_cnn.yaml")
        stats = prepare_pitch_contour_data(SYNTHETIC_DATASET, config.data, tmp_path)

        assert "train" in stats
        assert (tmp_path / "train" / "X.npy").exists()
        X = np.load(tmp_path / "train" / "X.npy")
        assert X.shape[1] == 20  # sequence_length


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Test full training pipeline for each task."""

    def test_ser_pipeline(self, tmp_path):
        from training.scripts.train import train
        from training.scripts.evaluate import evaluate
        from training.scripts.export import export_model

        # Train
        results = train(
            config_path=CONFIGS_DIR / "ser_wav2vec2.yaml",
            dataset_dir=SYNTHETIC_DATASET,
            output_dir=tmp_path / "ckpt",
        )
        assert results["train_metrics"]["accuracy"] > 0

        # Evaluate
        report = evaluate(
            checkpoint_path=tmp_path / "ckpt",
            dataset_dir=SYNTHETIC_DATASET,
            split="train",
            config_path=CONFIGS_DIR / "ser_wav2vec2.yaml",
        )
        assert report.total_samples > 0

        # Export
        meta = export_model(tmp_path / "ckpt", tmp_path / "export")
        assert meta["model_class"] == "SERModel"

        # Load and infer
        model = BaseModel.load(tmp_path / "export")
        labels = model.predict_labels(np.zeros((1, 7)))
        assert len(labels) == 1

    def test_text_prosody_pipeline(self, tmp_path):
        from training.scripts.train import train

        results = train(
            config_path=CONFIGS_DIR / "text_to_prosody_bert.yaml",
            dataset_dir=SYNTHETIC_DATASET,
            output_dir=tmp_path / "ckpt",
        )
        assert results["task"] == "text_to_prosody"
        assert results["train_samples"] > 0

        # Load and infer
        model = BaseModel.load(tmp_path / "ckpt")
        X = extract_text_features(["Hello", "world"])
        labels = model.predict_labels(X)
        assert len(labels) == 2

    def test_pitch_contour_pipeline(self, tmp_path):
        from training.scripts.train import train

        results = train(
            config_path=CONFIGS_DIR / "pitch_contour_cnn.yaml",
            dataset_dir=SYNTHETIC_DATASET,
            output_dir=tmp_path / "ckpt",
        )
        assert results["task"] == "pitch_contour"
        assert results["train_samples"] > 0

        # Load and infer
        model = BaseModel.load(tmp_path / "ckpt")
        X = resample_f0([100, 150, 200], target_length=20).reshape(1, -1)
        labels = model.predict_labels(X)
        assert len(labels) == 1

    def test_training_results_saved(self, tmp_path):
        """Verify training results are persisted alongside checkpoint."""
        from training.scripts.train import train

        train(
            config_path=CONFIGS_DIR / "ser_wav2vec2.yaml",
            dataset_dir=SYNTHETIC_DATASET,
            output_dir=tmp_path / "ckpt",
        )

        results_file = tmp_path / "ckpt" / "training_results.json"
        assert results_file.exists()
        with open(results_file) as f:
            saved = json.load(f)
        assert saved["task"] == "ser"
        assert "train_metrics" in saved


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_training_set_raises(self, tmp_path):
        """Training on empty data should raise ValueError."""
        from training.scripts.train import train

        empty_dataset = tmp_path / "empty_ds"
        (empty_dataset / "entries").mkdir(parents=True)
        (empty_dataset / "metadata.json").write_text('{"name":"empty","version":"0.1.0","size":0}')

        with pytest.raises(ValueError, match="empty"):
            train(
                config_path=CONFIGS_DIR / "ser_wav2vec2.yaml",
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
        assert (deep_path / "model.pkl").exists()

    def test_evaluation_report_dict_serializable(self):
        report = compute_metrics(["a", "b"], ["a", "b"])
        d = report.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
