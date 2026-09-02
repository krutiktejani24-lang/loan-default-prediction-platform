"""
Unit tests for the FastAPI inference service.
"""
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_endpoint_valid_input():
    payload = {
        "age": 35,
        "annual_income": 60000,
        "loan_amount": 15000,
        "credit_score": 680,
        "debt_to_income": 0.3,
        "employment_years": 8,
        "num_open_accounts": 4,
        "delinquencies_2yrs": 0,
        "loan_term_months": 36,
        "home_ownership": "MORTGAGE",
    }
    resp = client.post("/predict", json=payload)
    # 503 is acceptable if no model has been trained/registered yet in this test env
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert 0.0 <= body["default_probability"] <= 1.0
        assert body["default_prediction"] in (0, 1)


def test_predict_endpoint_invalid_input():
    payload = {"age": 15, "annual_income": -100}  # invalid: underage, negative income
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
