"""Testes da validacao do dataset bruto (Pydantic)."""

import pandas as pd
import pytest

from churn.schema import validate_raw_churn_schema


def test_valid_sample_passes_and_accepts_target_alias(raw_df):
    validated = validate_raw_churn_schema(raw_df.rename(columns={"Churn": "Churn?"}))

    assert len(validated) == len(raw_df)
    assert validated["Churn"].isin(["Yes", "No"]).all()


def test_invalid_value_raises(raw_df):
    raw_df["HandsetPrice"] = raw_df["HandsetPrice"].astype(object)
    raw_df.loc[0, "HandsetPrice"] = "abc"

    with pytest.raises(ValueError, match="HandsetPrice"):
        validate_raw_churn_schema(raw_df)


def test_cross_column_rule_raises(raw_df):
    raw_df.loc[0, ["UniqueSubs", "ActiveSubs"]] = [1, 2]

    with pytest.raises(ValueError, match="ActiveSubs"):
        validate_raw_churn_schema(raw_df)


def test_duplicated_customer_id_raises(raw_df):
    duplicated = pd.concat([raw_df, raw_df.head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="CustomerID duplicados"):
        validate_raw_churn_schema(duplicated)
