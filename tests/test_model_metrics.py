"""
Unit tests for candidate model training + metric thresholding.
"""
from pathlib import Path
import yaml
import pytest

from src.model.train import train_model
from src.model.evaluate import compute_metrics, check_thresholds

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_loan_data.csv"
THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"


@pytest.fixture(scope="module")
def trained_model():
    model, (X_test, y_test) = train_model(DATA_PATH)
    return model, X_test, y_test


def test_model_trains_successfully(trained_model):
    model, X_test, y_test = trained_model
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_metrics_are_computed(trained_model):
    model, X_test, y_test = trained_model
    metrics = compute_metrics(model, X_test, y_test)
    for key in ["f1_score", "precision", "recall", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_metrics_meet_regulatory_thresholds(trained_model):
    model, X_test, y_test = trained_model
    metrics = compute_metrics(model, X_test, y_test)

    with open(THRESHOLDS_PATH) as f:
        thresholds = yaml.safe_load(f)["model_metrics"]

    passed, failures = check_thresholds(metrics, thresholds)
    assert passed, f"Model failed regulatory thresholds: {failures}"


def test_check_thresholds_blocks_bad_metrics():
    bad_metrics = {"f1_score": 0.1, "precision": 0.1, "recall": 0.1, "roc_auc": 0.5}
    thresholds = {"min_f1_score": 0.75, "min_precision": 0.70, "min_recall": 0.65, "min_roc_auc": 0.75}
    passed, failures = check_thresholds(bad_metrics, thresholds)
    assert not passed
    assert len(failures) == 4
