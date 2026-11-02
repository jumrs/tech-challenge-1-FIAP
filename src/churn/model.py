"""Modelos da Etapa 2 e pipeline unico: pre-processamento -> selecao de features -> modelo."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import PREDICTION_THRESHOLD, RANDOM_SEED
from churn.features import build_preprocessor

# (classe, parametros fixos), como no notebook. n_jobs=1: o paralelismo fica nos folds da CV.
MODELS = {
    "Random Forest": (
        RandomForestClassifier,
        {
            "class_weight": "balanced_subsample",
            "min_samples_leaf": 20,
            "n_jobs": 1,
            "random_state": RANDOM_SEED,
        },
    ),
    "LightGBM": (
        LGBMClassifier,
        {"class_weight": "balanced", "n_jobs": 1, "verbose": -1, "random_state": RANDOM_SEED},
    ),
    "XGBoost": (
        XGBClassifier,
        {"tree_method": "hist", "n_jobs": 1, "random_state": RANDOM_SEED},
    ),
    "MLP": (
        MLPClassifier,
        {
            "hidden_layer_sizes": (64, 32),
            "early_stopping": True,
            "max_iter": 200,
            "random_state": RANDOM_SEED,
        },
    ),
}


# Tuning do campeao (Optuna, ver notebook). A comparacao usa os parametros de MODELS.
TUNED_PARAMS = {
    "LightGBM": {
        "n_estimators": 506,
        "learning_rate": 0.010801671793357538,
        "num_leaves": 113,
        "min_child_samples": 23,
        "subsample": 0.6023864197911676,
        "subsample_freq": 1,
        "colsample_bytree": 0.9157841081557863,
        "reg_lambda": 0.5184997421242614,
    },
}

# O class_weight infla as probabilidades. A calibracao corrige, e o corte e escolhido por CV.
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_FOLDS = 5
THRESHOLD_SCORING = "f1"


def scale_pos_weight(y_train: pd.Series) -> float:
    """Peso da classe churn para o XGBoost (RF e LightGBM usam class_weight)."""

    y = np.asarray(y_train)
    return float((y == 0).sum() / (y == 1).sum())


def build_feature_selector(threshold: str | float = "median") -> SelectFromModel:
    """Seletor com um LightGBM auxiliar: mantem as features com importancia acima do limiar."""

    return SelectFromModel(
        LGBMClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=1,
            verbose=-1,
            random_state=RANDOM_SEED,
        ),
        threshold=threshold,
    )


def build_model_pipeline(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    select_features: bool = True,
    threshold: str | float = "median",
    **params,
) -> Pipeline:
    """Pre-processamento -> seletor de features (opcional) -> modelo."""

    model_class, fixed_params = MODELS[name]
    fixed_params = dict(fixed_params)
    if model_class is XGBClassifier:
        fixed_params["scale_pos_weight"] = scale_pos_weight(y_train)

    selector = build_feature_selector(threshold) if select_features else "passthrough"
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("selector", selector),
            ("classifier", model_class(**(fixed_params | params))),
        ]
    ).set_output(transform="pandas")  # mantem os nomes das colunas ate o seletor


def build_champion(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    select_features: bool = True,
    threshold: str | float = "median",
    **params,
) -> TunedThresholdClassifierCV:
    """Pipeline do campeao com tuning, calibracao e corte escolhido por CV (best_threshold_)."""

    pipeline = build_model_pipeline(
        name,
        x_train,
        y_train,
        select_features=select_features,
        threshold=threshold,
        **(TUNED_PARAMS.get(name, {}) | params),
    )
    classifier = pipeline.steps[-1][1]
    pipeline.steps[-1] = (
        "classifier",
        CalibratedClassifierCV(classifier, method=CALIBRATION_METHOD, cv=CALIBRATION_FOLDS),
    )
    return TunedThresholdClassifierCV(
        pipeline,
        scoring=THRESHOLD_SCORING,
        cv=StratifiedKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        random_state=RANDOM_SEED,
    )


def get_pipeline(model) -> Pipeline:

    return model.estimator_ if isinstance(model, TunedThresholdClassifierCV) else model


def clf_name(model) -> str:
    classifier = get_pipeline(model).named_steps["classifier"]
    if isinstance(classifier, CalibratedClassifierCV):
        classifier = classifier.estimator
    return type(classifier).__name__


def get_threshold(model) -> float:

    return float(getattr(model, "best_threshold_", PREDICTION_THRESHOLD))
