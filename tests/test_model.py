"""Testes dos modelos da Etapa 2 e da escolha do campeao."""

import pandas as pd

from churn.compare_models import select_champion
from churn.model import (
    MODELS,
    build_champion,
    build_model_pipeline,
    clf_name,
    get_threshold,
)


def test_all_models_fit_and_predict_with_feature_selection(xy_train):
    x, y = xy_train

    for model_name in MODELS:
        params = {"max_iter": 50} if model_name == "MLP" else {"n_estimators": 20}
        pipeline = build_model_pipeline(model_name, x, y, select_features=True, **params)
        proba = pipeline.fit(x, y).predict_proba(x.head(5))[:, 1]

        assert ((proba >= 0) & (proba <= 1)).all(), model_name
        n_total = len(pipeline.named_steps["preprocessor"].get_feature_names_out())
        assert len(pipeline[:-1].get_feature_names_out()) < n_total, model_name


def test_select_champion_ignores_the_baseline():
    comparison = pd.DataFrame(
        {
            "model": ["Regressao Logistica (baseline)", "Random Forest", "LightGBM"],
            "selecao": ["nao", "sim", "nao"],
            "baseline": [True, False, False],
            "cv_roc_auc_mean": [0.99, 0.67, 0.68],
            "cv_roc_auc_std": [0.004, 0.002, 0.003],
        }
    )

    assert select_champion(comparison) == ("LightGBM", False)


def test_champion_calibrated(xy_train):
    x, y = xy_train

    model = build_champion("LightGBM", x, y, select_features=False, n_estimators=20)
    proba = model.fit(x, y).predict_proba(x)[:, 1]

    assert clf_name(model) == "LGBMClassifier"
    threshold = get_threshold(model)
    assert 0 < threshold < 1 and threshold != 0.5
    assert abs(proba.mean() - y.mean()) < 0.05
    assert (model.predict(x) == (proba >= threshold)).all()
