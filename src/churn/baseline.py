

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from churn.config import (
    BASELINE_MODEL_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
)
from churn.data import load_churn_csv, split_features_target, split_train_test
from churn.features import build_preprocessor
from churn.logger import get_logger
from churn.metrics import CV_SCORING, binary_classification_metrics

logger = get_logger(__name__)

CV_FOLDS = 5


def build_baseline_models() -> dict[str, ClassifierMixin]:
    """Dummy (piso) e Regressão Logística (referência a ser superada)."""

    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
    }


def build_baseline_pipeline(x_train, classifier: ClassifierMixin) -> Pipeline:
    """Pré-processamento do notebook seguido do classificador."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("classifier", classifier),
        ]
    )


def _log_logistic_diagnostics(pipeline: Pipeline, x_train, y_train, cv) -> None:
    """Matriz de confusao (CV) e coeficientes da Regressao Logistica, como no notebook."""

    # Previsoes fora da amostra: cada cliente e previsto pelo modelo que nao o viu no treino.
    y_pred_cv = cross_val_predict(clone(pipeline), x_train, y_train, cv=cv)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred_cv, labels=[0, 1]).ravel()
    mlflow.log_dict(
        {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "cv_confusion_matrix.json",
    )
    report = classification_report(
        y_train, y_pred_cv, target_names=["Nao churn", "Churn"], digits=3
    )
    mlflow.log_text(report, "cv_classification_report.txt")

    # Coeficientes padronizados (coef x desvio da feature), comparaveis entre numericas e one-hot,
    # como no notebook (que padroniza tudo). Positivos aumentam o risco de churn.
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_std = preprocessor.transform(x_train).std(axis=0)
    coefficients = pd.Series(
        pipeline.named_steps["classifier"].coef_[0] * feature_std,
        index=preprocessor.get_feature_names_out(),
    ).sort_values()
    mlflow.log_dict(coefficients.round(4).to_dict(), "coefficients.json")
    logger.info(
        f"Coeficientes LR: maiores riscos={coefficients.tail(3).index.tolist()}, "
        f"menores riscos={coefficients.head(3).index.tolist()}"
    )


def train_baseline(data_path: str | Path) -> dict[str, float]:
    """Avalia os baselines por CV, mede no teste e registra tudo no MLflow."""

    df = load_churn_csv(data_path)
    x, y = split_features_target(df)
    # Mantém a proporção de churn (~29%) nas bases de treino e teste.
    x_train, x_test, y_train, y_test = split_train_test(x, y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    test_metrics = {}
    for model_name, classifier in build_baseline_models().items():
        pipeline = build_baseline_pipeline(x_train, classifier)

        with mlflow.start_run(run_name=f"baseline_{model_name}"):
            mlflow.set_tags({"etapa": "baseline"})
            mlflow.log_params({"model_type": model_name, "cv_folds": CV_FOLDS})

            # A validação cruzada roda somente dentro dos 80% de treino.
            scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=CV_SCORING)
            for metric_name in CV_SCORING:
                values = scores[f"test_{metric_name}"]
                mlflow.log_metric(f"cv_{metric_name}_mean", float(values.mean()))
                mlflow.log_metric(f"cv_{metric_name}_std", float(values.std()))

            pipeline.fit(x_train, y_train)
            y_prob = pipeline.predict_proba(x_test)[:, 1]
            test_metrics = binary_classification_metrics(y_test, y_prob)
            mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

            if model_name == "logistic_regression":
                _log_logistic_diagnostics(pipeline, x_train, y_train, cv)
                BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline, BASELINE_MODEL_PATH)
                mlflow.log_artifact(str(BASELINE_MODEL_PATH))

        cv_auc = scores["test_roc_auc"]
        logger.info(
            f"Baseline {model_name}: CV AUC-ROC={cv_auc.mean():.3f} ± {cv_auc.std():.3f} | "
            f"teste: {', '.join(f'{k}={v:.3f}' for k, v in test_metrics.items())}"
        )

    logger.info(f"Modelo baseline salvo em {BASELINE_MODEL_PATH}.")
    # Retorna as métricas de teste da Regressão Logística (último modelo do loop).
    return test_metrics


def main() -> None:
    """Entrada de linha de comando para treinar a baseline."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/churn.csv")
    args = parser.parse_args()
    train_baseline(args.data)


if __name__ == "__main__":
    main()
