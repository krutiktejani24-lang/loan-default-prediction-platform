# Financial Loan Default Prediction Platform — CI/CD Model Validation Gate

Automated PR gate that blocks a credit-scoring model from being registered/deployed
unless it passes:

1. **Unit tests** (`pytest`)
2. **Data schema / data-quality validation** (Great Expectations)
3. **Candidate model training** on the PR's data/code
4. **Metric thresholds** (F1, Precision, Recall, ROC-AUC — see `config/thresholds.yaml`)

## Docker
```bash
docker compose up --build
# MLflow UI      -> http://localhost:5000
# FastAPI docs   -> http://localhost:8000/docs
```

