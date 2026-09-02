"""
Unit tests for the Great Expectations data-quality gate.
"""
import pandas as pd
import pytest
from pathlib import Path

from src.data.schema import validate_dataframe, EXPECTED_COLUMNS

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_loan_data.csv"


@pytest.fixture(scope="module")
def sample_df():
    return pd.read_csv(DATA_PATH)


def test_sample_data_exists():
    assert DATA_PATH.exists(), "Run scripts/generate_sample_data.py first"


def test_sample_data_has_expected_columns(sample_df):
    assert set(sample_df.columns) == set(EXPECTED_COLUMNS)


def test_sample_data_passes_ge_suite(sample_df):
    success, summary = validate_dataframe(sample_df)
    assert success, f"Data validation failed: {summary['failed_expectations']}"


def test_bad_data_fails_ge_suite(sample_df):
    bad_df = sample_df.copy()
    bad_df.loc[0, "credit_score"] = 9999  # out of valid range
    bad_df.loc[1, "home_ownership"] = "UNKNOWN_CATEGORY"
    success, summary = validate_dataframe(bad_df)
    assert not success
    assert len(summary["failed_expectations"]) > 0
