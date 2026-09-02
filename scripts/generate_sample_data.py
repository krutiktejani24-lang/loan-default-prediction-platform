"""
Generates a synthetic loan-default dataset for local dev / CI runs.
Run: python scripts/generate_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 3000

def generate(n_rows: int = N) -> pd.DataFrame:
    age = RNG.integers(21, 70, n_rows)
    annual_income = RNG.normal(55000, 20000, n_rows).clip(12000, 250000)
    loan_amount = RNG.normal(15000, 8000, n_rows).clip(1000, 60000)
    credit_score = RNG.normal(650, 80, n_rows).clip(300, 850)
    debt_to_income = RNG.normal(0.35, 0.15, n_rows).clip(0.0, 1.2)
    employment_years = RNG.integers(0, 40, n_rows)
    num_open_accounts = RNG.integers(0, 20, n_rows)
    delinquencies_2yrs = RNG.poisson(0.4, n_rows)
    loan_term_months = RNG.choice([12, 24, 36, 48, 60], n_rows)
    home_ownership = RNG.choice(["RENT", "OWN", "MORTGAGE"], n_rows)

    # synthetic default probability driven by a few realistic signals
    # (coefficients tuned so a RandomForest baseline clears the default
    #  regulatory thresholds in config/thresholds.yaml on this sample data)
    risk_score = (
        -0.035 * (credit_score - 650)
        + 6.0 * debt_to_income
        + 0.00006 * loan_amount
        - 0.00004 * annual_income
        + 0.6 * delinquencies_2yrs
        - 0.06 * employment_years
    )
    prob_default = 1 / (1 + np.exp(-(risk_score - 1.0)))
    default = RNG.binomial(1, prob_default)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "loan_amount": loan_amount.round(2),
        "credit_score": credit_score.round(0).astype(int),
        "debt_to_income": debt_to_income.round(3),
        "employment_years": employment_years,
        "num_open_accounts": num_open_accounts,
        "delinquencies_2yrs": delinquencies_2yrs,
        "loan_term_months": loan_term_months,
        "home_ownership": home_ownership,
        "default": default,
    })
    return df


if __name__ == "__main__":
    df = generate()
    out_path = Path(__file__).resolve().parents[1] / "data" / "sample_loan_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Default rate: {df['default'].mean():.3f}")
