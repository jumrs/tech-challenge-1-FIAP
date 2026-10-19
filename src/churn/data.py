

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import ID_COLUMNS, RANDOM_SEED, TARGET_COLUMN
from churn.logger import get_logger
from churn.schema import validate_raw_churn_schema

logger = get_logger(__name__)

# UniqueSubs = 196 e erro de carga (o segundo maior valor da base e 18).
MAX_UNIQUE_SUBS = 100

# O teste fica reservado: a comparacao entre modelos usa so a CV no treino.
TEST_SIZE = 0.2


def load_churn_csv(path: str | Path) -> pd.DataFrame:
    """Carrega o CSV bruto, valida o schema, remove registros invalidos e normaliza o alvo."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {csv_path}")

    # O schema ja aceita os aliases do alvo (Churn ou Churn?) e renomeia para Churn.
    df = validate_raw_churn_schema(pd.read_csv(csv_path))
    return normalize_churn_dataframe(clean_churn_dataframe(df))


def clean_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove os registros inconsistentes identificados na EDA."""

    cleaned = df.dropna(subset=["Handsets"])  # cliente sem dados de aparelho
    cleaned = cleaned[cleaned["UniqueSubs"] < MAX_UNIQUE_SUBS]  # erro de carga
    removed = len(df) - len(cleaned)
    logger.info(f"Limpeza do dataset: {len(cleaned)} linhas ({removed} removidas).")
    return cleaned.reset_index(drop=True)


def normalize_churn_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Converte o alvo Yes/No para 1/0."""

    normalized = df.copy()
    normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    return normalized


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    # Remove alvo e identificadores que nao devem entrar no modelo.
    ignored = set(ID_COLUMNS + [TARGET_COLUMN])
    return [column for column in df.columns if column not in ignored]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e coluna alvo para treino."""

    return df[infer_feature_columns(df)], df[TARGET_COLUMN]


def split_train_test(x: pd.DataFrame, y: pd.Series):
    """Divide em treino (80%) e teste (20%), estratificado para manter a proporcao de churn."""

    return train_test_split(x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
