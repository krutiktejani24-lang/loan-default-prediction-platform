"""
Computes evaluation metrics for the candidate model and checks them
against the thresholds defined in config/thresholds.yaml.
"""
from __future__ import annotations
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def check_thresholds(metrics: dict, thresholds: dict) -> tuple[bool, list[str]]:
    """
    thresholds is the `model_metrics` block from config/thresholds.yaml, e.g.:
    {"min_f1_score": 0.75, "min_precision": 0.70, "min_recall": 0.65, "min_roc_auc": 0.75}
    """
    failures = []
    checks = [
        ("f1_score", "min_f1_score"),
        ("precision", "min_precision"),
        ("recall", "min_recall"),
        ("roc_auc", "min_roc_auc"),
    ]
    for metric_key, threshold_key in checks:
        if threshold_key not in thresholds:
            continue
        if metrics[metric_key] < thresholds[threshold_key]:
            failures.append(
                f"{metric_key}={metrics[metric_key]:.4f} < required {thresholds[threshold_key]:.4f}"
            )
    return len(failures) == 0, failures
