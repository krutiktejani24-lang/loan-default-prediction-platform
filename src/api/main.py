"""
FastAPI inference service for the loan-default classifier.
Loads the latest "Staging"/"Production" model from the MLflow Model Registry
(falls back to a local artifact for local/dev use).
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Loan Default Prediction API",
    description="Serves predictions from the CI/CD-gated loan-default classifier.",
    version="1.0.0",
)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "loan-default-classifier")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Staging")
LOCAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "candidate_model.joblib"

_model = None


class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    debt_to_income: float = Field(..., ge=0, le=5)
    employment_years: int = Field(..., ge=0)
    num_open_accounts: int = Field(..., ge=0)
    delinquencies_2yrs: int = Field(..., ge=0)
    loan_term_months: int
    home_ownership: Literal["RENT", "OWN", "MORTGAGE"]


class PredictionResponse(BaseModel):
    default_probability: float
    default_prediction: int
    model_source: str


def _load_model():
    global _model
    if _model is not None:
        return _model

    try:
        import mlflow.sklearn
        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        return _model
    except Exception:
        pass

    if LOCAL_MODEL_PATH.exists():
        _model = joblib.load(LOCAL_MODEL_PATH)
        return _model

    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    model = _load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model available. Train a candidate model or register one in MLflow first.",
        )

    row = pd.DataFrame([application.model_dump()])
    row = pd.get_dummies(row, columns=["home_ownership"], drop_first=True)

    # Align columns with what the model was trained on
    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is not None:
        for col in expected_cols:
            if col not in row.columns:
                row[col] = 0
        row = row[expected_cols]

    proba = float(model.predict_proba(row)[0][1])
    pred = int(proba >= 0.5)

    source = "mlflow_registry" if not LOCAL_MODEL_PATH.exists() else "local_or_registry"
    return PredictionResponse(default_probability=proba, default_prediction=pred, model_source=source)
