"""
CI/CD MODEL VALIDATION GATE
============================
This is the single entrypoint the GitHub Actions workflow calls.

It runs, in order:
  1. Data schema / data-quality validation  (Great Expectations)
  2. Candidate model training
  3. Metric evaluation against config/thresholds.yaml
  4. (on success) registration + promotion in the MLflow Model Registry

Exit code 0  -> all checks passed, PR / deployment may proceed.
Exit code 1  -> a check failed, PR is BLOCKED (script exits non-zero, CI job fails).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_data
from src.data.schema import validate_dataframe
from src.model.train import train_model, save_model
from src.model.evaluate import compute_metrics, check_thresholds


def log_step(msg: str):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def main() -> int:
    data_path = REPO_ROOT / "data" / "sample_loan_data.csv"
    thresholds_path = REPO_ROOT / "config" / "thresholds.yaml"

    with open(thresholds_path) as f:
        config = yaml.safe_load(f)

    # ---------- STEP 1: Data schema / quality validation ----------
    log_step("STEP 1/4: Validating data schema & quality (Great Expectations)")
    df = load_data(data_path)

    if len(df) < config["data_quality"]["min_row_count"]:
        print(f"BLOCKED: dataset has {len(df)} rows, "
              f"minimum required is {config['data_quality']['min_row_count']}")
        return 1

    success, summary = validate_dataframe(
        df, max_null_fraction=config["data_quality"]["max_null_fraction"]
    )
    print(json.dumps(summary, indent=2))
    if not success:
        print("BLOCKED: data failed schema/quality validation.")
        return 1
    print("PASSED: data schema & quality checks.")

    # ---------- STEP 2: Train candidate model ----------
    log_step("STEP 2/4: Training candidate model")
    model, (X_test, y_test) = train_model(data_path)
    artifact_path = REPO_ROOT / "artifacts" / "candidate_model.joblib"
    save_model(model, artifact_path)
    print(f"Candidate model trained and saved to {artifact_path}")

    # ---------- STEP 3: Evaluate metrics against thresholds ----------
    log_step("STEP 3/4: Evaluating candidate model metrics")
    metrics = compute_metrics(model, X_test, y_test)
    print(json.dumps(metrics, indent=2))

    passed, failures = check_thresholds(metrics, config["model_metrics"])
    if not passed:
        print("BLOCKED: candidate model did not meet regulatory thresholds:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED: candidate model meets all regulatory thresholds.")

    # ---------- STEP 4: Register + promote in MLflow (best-effort) ----------
    log_step("STEP 4/4: Registering model in MLflow Model Registry")
    try:
        from src.registry.mlflow_registry import register_model
        result = register_model(
            model=model,
            metrics=metrics,
            params=model.get_params(),
            model_name=config["registry"]["model_name"],
            stage=config["registry"]["gate_stage"],
        )
        print(json.dumps(result, indent=2))
        print("PASSED: model registered and promoted.")
    except Exception as e:
        # In CI without a running MLflow tracking server, this step is
        # best-effort — the PR gate itself (steps 1-3) is what blocks merges.
        print(f"WARNING: could not register model in MLflow ({e}). "
              f"Set MLFLOW_TRACKING_URI to enable registry integration.")

    log_step("ALL CHECKS PASSED — deployment gate open.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
