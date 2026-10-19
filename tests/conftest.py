"""Fixtures compartilhadas: amostras pequenas do CSV bruto para testes rapidos."""

from pathlib import Path

import pandas as pd
import pytest

RAW_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "churn.csv"
SAMPLE_ROWS = 600


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """Primeiras linhas do dataset bruto (com valores ausentes e categorias reais)."""

    if not RAW_CSV.exists():
        pytest.skip(f"Dataset bruto nao encontrado em {RAW_CSV}")
    return pd.read_csv(RAW_CSV, nrows=SAMPLE_ROWS)


@pytest.fixture
def raw_df(raw_sample: pd.DataFrame) -> pd.DataFrame:
    """Copia da amostra, para cada teste poder alterar livremente."""

    return raw_sample.copy()


@pytest.fixture
def raw_row(raw_sample: pd.DataFrame) -> pd.DataFrame:
    """Um unico cliente valido (sem ausentes), para montar casos especificos."""

    complete = raw_sample.dropna()
    return complete.iloc[[0]].reset_index(drop=True).copy()


@pytest.fixture(scope="session")
def xy_train(raw_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features brutas e alvo 0/1, prontos para os pipelines."""

    from churn.data import clean_churn_dataframe, normalize_churn_dataframe, split_features_target
    from churn.schema import validate_raw_churn_schema

    df = normalize_churn_dataframe(clean_churn_dataframe(validate_raw_churn_schema(raw_sample)))
    return split_features_target(df)
