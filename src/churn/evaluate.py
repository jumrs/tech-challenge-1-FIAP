"""Avaliacao final: campeao x baseline no conjunto de teste, usado uma unica vez."""

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.compare_models import REPORTS_DIR, to_markdown_table
from churn.config import (
    BASELINE_MODEL_PATH,
    CHAMPION_MODEL_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PREDICTION_THRESHOLD,
)
from churn.data import load_churn_csv, split_features_target, split_train_test
from churn.logger import get_logger
from churn.metrics import (
    binary_classification_metrics,
    bootstrap_auc_difference,
    cost_tradeoff_metrics,
    top_k_metrics,
)

logger = get_logger(__name__)

TEST_REPORT_CSV = REPORTS_DIR / "test_evaluation.csv"
TEST_REPORT_MD = REPORTS_DIR / "test_evaluation.md"
# Fracoes da base abordadas numa campanha de retencao (clientes de maior risco primeiro).
TOP_FRACTIONS = (0.10, 0.20)
# Premissas ilustrativas da metrica de negocio: devem ser validadas com a area de negocio.
BUSINESS_ASSUMPTIONS = {
    "customer_value": 1000.0,
    "retention_success_rate": 0.30,
    "action_cost": 50.0,
}


def _load_model(path: Path) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {path}. Rode `uv run python -m churn.main` antes."
        )
    return joblib.load(path)


def evaluate_models(
    models: dict[str, Pipeline],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = PREDICTION_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Metricas tecnicas, de campanha (top X%) e de custo de cada modelo no teste."""

    columns = {}
    probabilities = {}
    for name, model in models.items():
        y_prob = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index)
        probabilities[name] = y_prob
        metrics = binary_classification_metrics(y_test, y_prob, threshold=threshold)
        for fraction in TOP_FRACTIONS:
            label = f"top{int(fraction * 100)}"
            top = top_k_metrics(y_test, y_prob, fraction)
            metrics |= {f"{label}_{key}": value for key, value in top.items()}
        metrics |= cost_tradeoff_metrics(
            y_test, y_prob, threshold=threshold, **BUSINESS_ASSUMPTIONS
        )
        columns[name] = metrics
    return pd.DataFrame(columns), probabilities


def _to_markdown_report(
    table: pd.DataFrame, auc_difference: dict[str, float], n_test: int, churn_rate: float
) -> str:
    """Relatorio legivel da avaliacao final, com as premissas usadas."""

    assumptions = ", ".join(f"{key}={value}" for key, value in BUSINESS_ASSUMPTIONS.items())
    return "\n".join(
        [
            "# Avaliacao final no conjunto de teste",
            "",
            f"- Conjunto de teste: {n_test} clientes (20% reservados), churn de {churn_rate:.1%}.",
            f"- Corte de classificacao: {PREDICTION_THRESHOLD}.",
            f"- `topN_*`: abordando so os N% de clientes de maior risco.",
            f"- Metricas de custo com premissas ilustrativas: {assumptions}.",
            (
                f"- Diferenca de AUC-ROC (campeao - baseline): {auc_difference['auc_difference']:+.4f} "
                f"(IC 95% bootstrap: {auc_difference['ci_low']:+.4f} a "
                f"{auc_difference['ci_high']:+.4f})."
            ),
            "",
            to_markdown_table(table.reset_index(names="metrica")),
        ]
    )


def evaluate_on_test(data_path: str | Path) -> pd.DataFrame:
    """Avalia o baseline e o campeao salvos no teste e registra no MLflow e em docs/reports."""

    baseline = _load_model(BASELINE_MODEL_PATH)
    champion = _load_model(CHAMPION_MODEL_PATH)
    champion_name = f"Campeao ({type(champion.named_steps['classifier']).__name__})"
    baseline_name = "Regressao Logistica (baseline)"

    df = load_churn_csv(data_path)
    x, y = split_features_target(df)
    # Mesma divisao (seed fixa) do treino: os modelos nunca viram estes clientes.
    _, x_test, _, y_test = split_train_test(x, y)

    table, probabilities = evaluate_models(
        {baseline_name: baseline, champion_name: champion}, x_test, y_test
    )
    table["ganho_campeao"] = table[champion_name] - table[baseline_name]
    auc_difference = bootstrap_auc_difference(
        y_test, probabilities[champion_name], probabilities[baseline_name]
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="AVALIACAO FINAL | teste"):
        mlflow.set_tags({"etapa": "avaliacao_final"})
        mlflow.log_params(
            {"campeao": champion_name, "threshold": PREDICTION_THRESHOLD, **BUSINESS_ASSUMPTIONS}
        )
        for column, prefix in [(champion_name, "champion"), (baseline_name, "baseline")]:
            mlflow.log_metrics({f"test_{prefix}_{key}": value for key, value in table[column].items()})
        mlflow.log_metrics({f"test_{key}": value for key, value in auc_difference.items()})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TEST_REPORT_CSV, index_label="metrica")
    TEST_REPORT_MD.write_text(
        _to_markdown_report(table, auc_difference, len(y_test), float(y_test.mean())),
        encoding="utf-8",
    )

    logger.info(
        f"Teste: AUC-ROC campeao={table.loc['roc_auc', champion_name]:.3f} x "
        f"baseline={table.loc['roc_auc', baseline_name]:.3f} "
        f"(diferenca {auc_difference['auc_difference']:+.3f}, IC 95% "
        f"{auc_difference['ci_low']:+.3f} a {auc_difference['ci_high']:+.3f}). "
        f"Relatorio em {TEST_REPORT_MD}."
    )
    return table


def main() -> None:
    """Entrada de linha de comando para a avaliacao final."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/churn.csv")
    args = parser.parse_args()
    evaluate_on_test(args.data)


if __name__ == "__main__":
    main()
