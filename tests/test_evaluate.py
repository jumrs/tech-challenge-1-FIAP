"""Testes da avaliacao final no conjunto de teste."""

from sklearn.linear_model import LogisticRegression

from churn.baseline import build_baseline_pipeline
from churn.data import split_train_test
from churn.evaluate import evaluate_models
from churn.model import build_model_pipeline


def test_evaluate_models_table(xy_train):
    x, y = xy_train
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    models = {
        "baseline": build_baseline_pipeline(x_train, LogisticRegression(max_iter=500)),
        "campeao": build_model_pipeline(
            "LightGBM", x_train, y_train, select_features=False, n_estimators=20
        ),
    }
    for model in models.values():
        model.fit(x_train, y_train)

    table, probabilities = evaluate_models(models, x_test, y_test, threshold=0.5)

    assert list(table.columns) == ["baseline", "campeao"]
    assert {"roc_auc", "top10_lift", "net_avoided_churn_value"} <= set(table.index)
    # A matriz de confusao soma o total de clientes do teste.
    confusion = ["true_negatives", "false_positives", "false_negatives", "true_positives"]
    assert (table.loc[confusion].sum() == len(y_test)).all()
    assert probabilities["campeao"].between(0, 1).all()
