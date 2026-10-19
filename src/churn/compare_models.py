"""Etapa 2: compara os modelos por validacao cruzada no treino e escolhe o campeao."""

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn.baseline import build_baseline_models, build_baseline_pipeline
from churn.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, RANDOM_SEED
from churn.data import load_churn_csv, split_features_target, split_train_test
from churn.logger import get_logger
from churn.metrics import CV_SCORING
from churn.model import MODELS, build_model_pipeline
from churn.train import train_champion

logger = get_logger(__name__)

CV_FOLDS = 5
# Regressao Logistica avaliada nos mesmos folds: referencia (linear) que os modelos devem superar.
BASELINE_NAME = "Regressao Logistica (baseline)"
REPORTS_DIR = Path("docs") / "reports"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
COMPARISON_MD = REPORTS_DIR / "model_comparison.md"
SELECTION_GAIN_MD = REPORTS_DIR / "feature_selection_gain.md"


def _cross_validate_logged(
    run_name: str,
    pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    params: dict,
) -> dict[str, float]:
    """Validacao cruzada (5 folds) + registro de parametros e metricas no MLflow."""

    with mlflow.start_run(run_name=run_name):
        scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        mlflow.set_tags({"etapa": "comparacao_cv"})
        mlflow.log_params(params)
        results = {}
        for metric_name in CV_SCORING:
            values = scores[f"test_{metric_name}"]
            results[f"cv_{metric_name}_mean"] = float(values.mean())
            results[f"cv_{metric_name}_std"] = float(values.std())
        results["fit_time_mean_s"] = float(scores["fit_time"].mean())
        mlflow.log_metrics(results)
    return results


def compare_models(data_path: str | Path) -> pd.DataFrame:
    """Avalia cada modelo com e sem selecao de features e gera os relatorios."""

    df = load_churn_csv(data_path)
    x, y = split_features_target(df)
    # O teste nao e usado aqui: fica reservado para a avaliacao final do campeao.
    x_train, _, y_train, _ = split_train_test(x, y)
    # Mesmos folds do baseline.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Baseline linear na mesma CV, para a tabela comparar Linear x Arvores x MLP.
    baseline_pipeline = build_baseline_pipeline(
        x_train, build_baseline_models()["logistic_regression"]
    )
    baseline_results = _cross_validate_logged(
        f"{BASELINE_NAME} | selecao=nao",
        baseline_pipeline,
        x_train,
        y_train,
        cv,
        {"modelo": BASELINE_NAME, "selecao_features": False},
    )
    rows = [{"model": BASELINE_NAME, "selecao": "nao", "baseline": True} | baseline_results]
    logger.info(
        f"{BASELINE_NAME}: CV AUC-ROC={baseline_results['cv_roc_auc_mean']:.3f} ± "
        f"{baseline_results['cv_roc_auc_std']:.3f}"
    )

    for model_name in MODELS:
        for select_features in (False, True):
            selection_label = "sim" if select_features else "nao"
            pipeline = build_model_pipeline(
                model_name, x_train, y_train, select_features=select_features
            )
            results = _cross_validate_logged(
                f"{model_name} | selecao={selection_label}",
                pipeline,
                x_train,
                y_train,
                cv,
                {"modelo": model_name, "selecao_features": select_features},
            )
            rows.append(
                {"model": model_name, "selecao": selection_label, "baseline": False} | results
            )
            logger.info(
                f"{model_name} (selecao={selection_label}): "
                f"CV AUC-ROC={results['cv_roc_auc_mean']:.3f} ± {results['cv_roc_auc_std']:.3f}"
            )

    comparison = (
        pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)
    )

    # Efeito da selecao de features no AUC-ROC, por modelo.
    candidates = comparison[~comparison["baseline"]]
    selection_gain = candidates.pivot(index="model", columns="selecao", values="cv_roc_auc_mean")
    selection_gain["ganho_selecao"] = selection_gain["sim"] - selection_gain["nao"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_CSV, index=False)
    COMPARISON_MD.write_text(to_markdown_table(comparison), encoding="utf-8")
    SELECTION_GAIN_MD.write_text(
        to_markdown_table(selection_gain.reset_index()), encoding="utf-8"
    )
    logger.info(f"Comparacao de {len(comparison)} runs salva em {COMPARISON_CSV} e {COMPARISON_MD}.")
    return comparison


def select_champion(comparison: pd.DataFrame) -> tuple[str, bool]:
    """Criterio do notebook: maior AUC-ROC medio na validacao cruzada (o baseline nao concorre)."""

    candidates = comparison[~comparison["baseline"]]
    champion = candidates.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]
    message = (
        f"Campeao: {champion['model']} (selecao de features: {champion['selecao']}) | "
        f"CV AUC-ROC={champion['cv_roc_auc_mean']:.3f} ± {champion['cv_roc_auc_std']:.3f}"
    )
    baseline = comparison[comparison["baseline"]]
    if not baseline.empty:
        baseline_auc = baseline["cv_roc_auc_mean"].iloc[0]
        gain = champion["cv_roc_auc_mean"] - baseline_auc
        message += f" | baseline={baseline_auc:.3f} (ganho de {gain:+.3f})"
    logger.info(message)
    return champion["model"], champion["selecao"] == "sim"


def to_markdown_table(df: pd.DataFrame) -> str:
    """Gera tabela Markdown sem depender de bibliotecas extras."""

    columns = list(df.columns)
    header = "| " + " | ".join(map(str, columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in df.round(4).to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows]) + "\n"


def main() -> None:
    """Entrada de linha de comando: compara os modelos e treina o campeao."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/churn.csv")
    args = parser.parse_args()
    comparison = compare_models(args.data)
    model_name, select_features = select_champion(comparison)
    train_champion(args.data, model_name, select_features=select_features)


if __name__ == "__main__":
    main()
