"""
Trains the candidate loan-default classifier.
"""
from __future__ import annotations
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.data.loader import load_data, split_data


def train_model(data_path: str | Path, random_state: int = 42) -> tuple[RandomForestClassifier, tuple]:
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, random_state=random_state)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, (X_test, y_test)


def save_model(model, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    model, (X_test, y_test) = train_model(repo_root / "data" / "sample_loan_data.csv")
    save_model(model, repo_root / "artifacts" / "candidate_model.joblib")
    print("Candidate model trained and saved to artifacts/candidate_model.joblib")
