"""
Great Expectations-based schema & data-quality validation for the
loan-default dataset. Used both by the CI gate and by unit tests.
"""
from __future__ import annotations
import pandas as pd
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest

EXPECTED_COLUMNS = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "num_open_accounts",
    "delinquencies_2yrs",
    "loan_term_months",
    "home_ownership",
    "default",
]

VALID_HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE"]


def build_expectation_suite(df: pd.DataFrame, max_null_fraction: float = 0.02):
    """
    Builds an in-memory GE validator with the expectation suite for the
    loan dataset and returns the validation result object.
    """
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name="loan_pandas_source")
    data_asset = data_source.add_dataframe_asset(name="loan_dataframe")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("loan_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(gx.ExpectationSuite(name="loan_default_suite"))

    # --- Schema expectations ---
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=EXPECTED_COLUMNS, exact_match=True)
    )

    # --- Value / range expectations ---
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="age", min_value=18, max_value=100))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="credit_score", min_value=300, max_value=850))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="annual_income", min_value=0, max_value=10_000_000))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="loan_amount", min_value=0, max_value=1_000_000))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="debt_to_income", min_value=0, max_value=5))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="home_ownership", value_set=VALID_HOME_OWNERSHIP))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="default", value_set=[0, 1]))

    # --- Null / completeness expectations ---
    for col in EXPECTED_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col, mostly=1 - max_null_fraction)
        )

    validation_result = batch.validate(suite)
    return validation_result


def validate_dataframe(df: pd.DataFrame, max_null_fraction: float = 0.02) -> tuple[bool, dict]:
    """
    Returns (success: bool, summary: dict) for use by the CI gate script.
    """
    result = build_expectation_suite(df, max_null_fraction=max_null_fraction)
    success = bool(result.success)
    failed = [
        r["expectation_config"]["type"]
        for r in result.results
        if not r["success"]
    ]
    summary = {
        "success": success,
        "total_expectations": len(result.results),
        "failed_expectations": failed,
    }
    return success, summary
