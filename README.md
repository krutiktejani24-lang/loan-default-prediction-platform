# Financial Loan Default Prediction Platform — CI/CD Model Validation Gate

Automated PR gate that blocks a credit-scoring model from being registered/deployed
unless it passes:

1. **Unit tests** (`pytest`)
2. **Data schema / data-quality validation** (Great Expectations)
3. **Candidate model training** on the PR's data/code
4. **Metric thresholds** (F1, Precision, Recall, ROC-AUC — see `config/thresholds.yaml`)

If any step fails, the GitHub Actions workflow fails and the PR is blocked from merging
(enforce this with a branch protection rule requiring the `model-validation-gate` check).
On success, the candidate model is registered in the MLflow Model Registry and promoted
to the `Staging` stage automatically.

## Folder structure

```
loan-default-prediction-platform/
├── .github/workflows/model-validation-gate.yml   # CI/CD gate (GitHub Actions)
├── config/thresholds.yaml                        # metric + data-quality thresholds
├── data/sample_loan_data.csv                      # sample dataset (generated)
├── great_expectations/expectations/               # GE expectation suite (JSON)
├── scripts/
│   ├── generate_sample_data.py                    # creates synthetic loan dataset
│   └── validate_and_gate.py                       # orchestrates the whole gate, used by CI
├── src/
│   ├── data/
│   │   ├── loader.py                               # load + split data
│   │   └── schema.py                                # Great Expectations suite (code-based)
│   ├── model/
│   │   ├── train.py                                 # trains candidate model
│   │   └── evaluate.py                              # computes F1/Precision/Recall/AUC
│   ├── registry/
│   │   └── mlflow_registry.py                       # register + promote model in MLflow
│   └── api/
│       └── main.py                                   # FastAPI inference service
├── tests/
│   ├── test_data_schema.py
│   ├── test_model_metrics.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml                              # spins up MLflow server + API
└── requirements.txt
```

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. generate sample data
python scripts/generate_sample_data.py

# 2. run the full gate locally (same thing CI runs)
python scripts/validate_and_gate.py

# 3. run unit tests only
pytest -v

# 4. run the API
uvicorn src.api.main:app --reload
```

## Docker

```bash
docker compose up --build
# MLflow UI      -> http://localhost:5000
# FastAPI docs   -> http://localhost:8000/docs
```

## How the gate is wired into GitHub

1. Push this repo to GitHub.
2. In **Settings → Branches → Branch protection rules**, require the status check
   named `model-validation-gate` on `main` before merging.
3. Any PR that touches `src/`, `data/`, or `config/` will trigger
   `.github/workflows/model-validation-gate.yml`, which:
   - installs deps
   - runs `pytest`
   - runs Great Expectations validation against `data/sample_loan_data.csv`
   - trains a candidate model
   - evaluates metrics against `config/thresholds.yaml`
   - **fails the job (blocking the PR)** if any check fails
   - on success, registers + promotes the model in MLflow Model Registry
