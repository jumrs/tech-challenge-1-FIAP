"""Testes do feature engineering e do pre-processamento."""

import numpy as np
import pandas as pd

from churn.features import LEAKAGE_COLUMNS, build_preprocessor, engineer_features


def test_encodings_and_removed_columns(raw_row):
    raw_row.loc[0, ["ChildrenInHH", "CreditRating", "HandsetPrice"]] = ["Yes", "5-Low", "Unknown"]
    original = raw_row.copy()

    result = engineer_features(raw_row)

    pd.testing.assert_frame_equal(raw_row, original)  # nao altera a entrada
    assert not {"CustomerID", "ServiceArea", *LEAKAGE_COLUMNS} & set(result.columns)
    assert result.loc[0, "ChildrenInHH"] == 1
    assert result.loc[0, "CreditRating"] == 5
    assert np.isnan(result.loc[0, "HandsetPrice"])
    assert result.loc[0, "HandsetPrice_Unknown"] == 1


def test_contract_phase(raw_row):
    rows = pd.concat([raw_row] * 3, ignore_index=True)
    rows["MonthsInService"] = [10, 12, 13]

    result = engineer_features(rows)

    assert result["Contract_Locked"].tolist() == [1, 0, 0]
    assert result["Contract_EndWindow"].tolist() == [0, 1, 0]


def test_disguised_missing_values(raw_row):
    raw_row.loc[0, ["AgeHH1", "AgeHH2", "IncomeGroup", "CurrentEquipmentDays"]] = [0, 0, 0, -3]

    result = engineer_features(raw_row)

    assert result.loc[0, ["AgeHH1", "AgeHH2", "IncomeGroup"]].isna().all()
    assert result.loc[0, "Demographics_Missing"] == 1
    assert result.loc[0, "CurrentEquipmentDays"] == 0


def test_preprocessor_without_nan_and_with_unseen_region(xy_train, raw_row):
    x, _ = xy_train
    preprocessor = build_preprocessor(x).fit(x)
    raw_row.loc[0, "ServiceArea"] = "ZZZABC123"

    transformed = preprocessor.transform(pd.concat([x, raw_row.drop(columns=["Churn"])]))

    assert not np.isnan(transformed).any()
    assert transformed.shape[1] == len(preprocessor.get_feature_names_out())
