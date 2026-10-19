
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.config import ID_COLUMNS

# Variaveis de retencao removidas por vazamento de informacao (target leakage).
LEAKAGE_COLUMNS = ["RetentionCalls", "RetentionOffersAccepted", "MadeCallToRetentionTeam"]

YES_NO_COLUMNS = [
    "ChildrenInHH",
    "HandsetRefurbished",
    "HandsetWebCapable",
    "TruckOwner",
    "RVOwner",
    "BuysViaMailOrder",
    "RespondsToMailOffers",
    "OptOutMailings",
    "NonUSTravel",
    "OwnsComputer",
    "HasCreditCard",
    "NewCellphoneUser",
    "NotNewCellphoneUser",
    "OwnsMotorcycle",
]

REGION_COLUMN = "ServiceRegion"

# No notebook, regioes com menos de 500 clientes na base completa viram uma unica categoria;
# como o encoder e ajustado so no treino (80%), o limiar equivalente e 400.
REGION_MIN_FREQUENCY = 400


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as transformacoes linha a linha do notebook (sem estatisticas da base)."""

    df = df.drop(columns=ID_COLUMNS + LEAKAGE_COLUMNS, errors="ignore").copy()

    # Codificacao binaria.
    df[YES_NO_COLUMNS] = df[YES_NO_COLUMNS].replace({"Yes": 1, "No": 0}).astype(int)
    df["Homeownership"] = df["Homeownership"].map({"Known": 1, "Unknown": 0})

    # CreditRating ordinal: "2-High" -> 2.
    df["CreditRating"] = df["CreditRating"].str[0].astype(int)

    # HandsetPrice numerico + flag de preco desconhecido.
    df["HandsetPrice_Unknown"] = (df["HandsetPrice"] == "Unknown").astype(int)
    df["HandsetPrice"] = pd.to_numeric(df["HandsetPrice"], errors="coerce")

    # ServiceArea -> regiao (3 primeiras letras); areas ausentes vao para "Other".
    df[REGION_COLUMN] = df["ServiceArea"].str[:3].fillna("Other")
    df = df.drop(columns=["ServiceArea"])

    # Fase da fidelidade contratual.
    df["Contract_Locked"] = (df["MonthsInService"] <= 10).astype(int)
    df["Contract_EndWindow"] = df["MonthsInService"].between(11, 12).astype(int)

    # Flags de ausencia (a ausencia de informacao e um sinal de risco).
    df["Usage_Missing"] = df["MonthlyRevenue"].isna().astype(int)
    df["PercChange_Missing"] = df["PercChangeMinutes"].isna().astype(int)
    df["Demographics_Missing"] = (df["AgeHH1"].isna() | (df["AgeHH1"] == 0)).astype(int)

    # Valores disfarcados de ausente viram NaN (imputados depois, dentro do pipeline).
    df.loc[df["AgeHH1"] == 0, ["AgeHH1", "AgeHH2"]] = np.nan
    df.loc[df["IncomeGroup"] == 0, "IncomeGroup"] = np.nan
    df["CurrentEquipmentDays"] = df["CurrentEquipmentDays"].clip(lower=0)

    return df


def engineered_feature_names(transformer, input_features) -> list[str]:
    """Nomes das colunas geradas por engineer_features (habilita get_feature_names_out)."""

    empty = pd.DataFrame(columns=list(input_features))
    return engineer_features(empty).columns.tolist()


def infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Tipos inferidos apos o feature engineering: o que sobra como texto e nominal.
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    return categorical_columns, numeric_columns


def build_preprocessor(df: pd.DataFrame) -> Pipeline:
    """Cria o pipeline de feature engineering, imputacao, escala e one-hot encoding."""

    categorical_columns, numeric_columns = infer_column_types(engineer_features(df.head()))
    nominal_columns = [column for column in categorical_columns if column != REGION_COLUMN]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    # Regioes raras sao agrupadas numa unica categoria, como no notebook.
    region_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    min_frequency=REGION_MIN_FREQUENCY,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, nominal_columns),
            ("region", region_pipeline, [REGION_COLUMN]),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            (
                "features",
                FunctionTransformer(
                    engineer_features, feature_names_out=engineered_feature_names
                ),
            ),
            ("columns", columns),
        ]
    )
