"""Executa o fluxo completo: dados -> baseline -> comparacao -> campeao -> avaliacao no teste."""

import argparse
import time
from pathlib import Path

from churn.baseline import train_baseline
from churn.compare_models import compare_models, select_champion
from churn.config import BASELINE_MODEL_PATH, CHAMPION_MODEL_PATH, DATA_RAW_DIR
from churn.download_data import download_churn_dataset
from churn.evaluate import evaluate_on_test
from churn.logger import get_logger
from churn.train import train_champion

logger = get_logger(__name__)

DEFAULT_DATA_PATH = DATA_RAW_DIR / "churn.csv"


def _run_step(step: str, function, *args, **kwargs):
    """Executa uma etapa do fluxo registrando inicio, fim e duracao."""

    logger.info(f"[{step}] iniciando...")
    start = time.perf_counter()
    result = function(*args, **kwargs)
    logger.info(f"[{step}] concluido em {time.perf_counter() - start:.1f}s.")
    return result


def run_pipeline(data_path: str | Path = DEFAULT_DATA_PATH, *, download: bool = False) -> dict:
    """Roda todas as etapas em sequencia e retorna um resumo do resultado."""

    data_path = Path(data_path)
    # O dataset ja esta versionado; o download so roda se pedido ou se o arquivo nao existir.
    if download or not data_path.exists():
        _run_step("1/5 download dos dados", download_churn_dataset, output=data_path)
    else:
        logger.info(f"[1/5 dados] usando o arquivo existente {data_path}.")

    _run_step("2/5 baseline", train_baseline, data_path)
    comparison = _run_step("3/5 comparacao dos modelos", compare_models, data_path)
    model_name, select_features = select_champion(comparison)
    _run_step(
        "4/5 treino do campeao",
        train_champion,
        data_path,
        model_name,
        select_features=select_features,
    )
    test_table = _run_step("5/5 avaliacao final no teste", evaluate_on_test, data_path)
    test_auc = test_table.loc["roc_auc"]

    candidates = comparison[~comparison["baseline"]]
    baseline = comparison[comparison["baseline"]]
    summary = {
        "campeao": model_name,
        "selecao_features": select_features,
        "campeao_cv_roc_auc": round(float(candidates["cv_roc_auc_mean"].max()), 4),
        "baseline_cv_roc_auc": round(float(baseline["cv_roc_auc_mean"].iloc[0]), 4),
        "campeao_test_roc_auc": round(float(test_auc.iloc[1]), 4),
        "baseline_test_roc_auc": round(float(test_auc.iloc[0]), 4),
        "baseline_model": str(BASELINE_MODEL_PATH),
        "champion_model": str(CHAMPION_MODEL_PATH),
    }
    logger.info(f"Fluxo completo finalizado: {summary}")
    return summary


def main() -> None:
    """Entrada de linha de comando para o fluxo completo."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="caminho do CSV bruto")
    parser.add_argument(
        "--download",
        action="store_true",
        help="baixa o dataset do Kaggle antes de treinar (sobrescreve o CSV)",
    )
    args = parser.parse_args()
    run_pipeline(args.data, download=args.download)


if __name__ == "__main__":
    main()
