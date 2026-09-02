"""
Registers the candidate model + its metrics into the MLflow Model Registry,
and promotes it to the configured stage (e.g. "Staging") once it has
passed the CI gate.
"""
from __future__ import annotations
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def register_model(
    model,
    metrics: dict,
    params: dict,
    model_name: str,
    stage: str = "Staging",
    tracking_uri: str | None = None,
    run_name: str = "ci-candidate-model",
):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("loan-default-prediction")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        run_id = run.info.run_id

    client = MlflowClient()
    # Find the version that was just registered from this run
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise RuntimeError("Model registration did not produce a model version.")
    version = versions[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=False,
    )

    return {"run_id": run_id, "model_name": model_name, "version": version, "stage": stage}
