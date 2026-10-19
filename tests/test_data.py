"""Testes de leitura, limpeza e divisao dos dados."""

import pytest

from churn.data import MAX_UNIQUE_SUBS, clean_churn_dataframe, load_churn_csv, split_train_test


def test_load_churn_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_churn_csv(tmp_path / "nao_existe.csv")


def test_clean_removes_inconsistent_records(raw_df):
    raw_df.loc[0, "Handsets"] = None
    raw_df.loc[1, "UniqueSubs"] = MAX_UNIQUE_SUBS + 96

    cleaned = clean_churn_dataframe(raw_df)

    assert len(cleaned) == len(raw_df) - 2
    assert cleaned["Handsets"].notna().all()


def test_features_target_and_stratified_split(xy_train):
    x, y = xy_train

    x_train, x_test, y_train, y_test = split_train_test(x, y)

    assert not {"CustomerID", "Churn"} & set(x.columns)
    assert set(y.unique()) <= {0, 1}
    assert len(x_test) == pytest.approx(0.2 * len(x), abs=1)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.02)
    assert x_train.index.equals(split_train_test(x, y)[0].index)  # reproduzivel (seed fixa)
