"""
Data loading + train/test split utilities.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "default"
CATEGORICAL_COLUMNS = ["home_ownership"]
NUMERIC_COLUMNS = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "num_open_accounts",
    "delinquencies_2yrs",
    "loan_term_months",
]


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
