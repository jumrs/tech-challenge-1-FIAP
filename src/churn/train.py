"""Treina o modelo campeao no treino completo e registra no MLflow."""

import argparse
from pathlib import Path

import joblib
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn.config import (
    CHAMPION_MODEL_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
)
from churn.data import load_churn_csv, split_features_target, split_train_test
from churn.logger import get_logger
from churn.model import (
    CALIBRATION_METHOD,
    MODELS,
    THRESHOLD_SCORING,
    TUNED_PARAMS,
    build_champion,
    get_pipeline,
    get_threshold,
)

logger = get_logger(__name__)


def train_champion(
    data_path: str | Path,
    model_name: str,
    *,
    select_features: bool = True,
    threshold: str | float = "median",
):
    """Treina o campeao em todo o treino (o teste segue reservado)."""

    df = load_churn_csv(data_path)
    x, y = split_features_target(df)
    x_train, _, y_train, _ = split_train_test(x, y)

    model = build_champion(
        model_name, x_train, y_train, select_features=select_features, threshold=threshold
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"CAMPEAO | {model_name}"):
        mlflow.set_tags({"etapa": "campeao"})
        mlflow.log_params(
            {
                "modelo": model_name,
                "selecao_features": select_features,
                "limiar_seletor": threshold,
                "calibracao": CALIBRATION_METHOD,
                "criterio_corte": THRESHOLD_SCORING,
                **TUNED_PARAMS.get(model_name, {}),
            }
        )
        # Mesmos folds da comparacao. O corte nao altera o AUC.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_auc = cross_val_score(
            model.estimator, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        mlflow.log_metrics({"cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std()})

        model.fit(x_train, y_train)
        pipeline = get_pipeline(model)
        cutoff = get_threshold(model)
        mlflow.log_metrics({"corte": cutoff, f"cv_{THRESHOLD_SCORING}_no_corte": model.best_score_})

        # Features mantidas pelo seletor (ou todas, sem selecao).
        kept_features = pipeline[:-1].get_feature_names_out().tolist()
        all_features = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        removed_features = sorted(set(all_features) - set(kept_features))
        mlflow.log_metric("n_features", len(kept_features))
        mlflow.log_dict({"mantidas": kept_features, "removidas": removed_features}, "features.json")

        CHAMPION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, CHAMPION_MODEL_PATH)
        mlflow.sklearn.log_model(model, name="modelo", serialization_format="cloudpickle")

    logger.info(
        f"Campeao {model_name} treinado com {len(kept_features)} de {len(all_features)} features "
        f"(CV AUC-ROC={cv_auc.mean():.4f} ± {cv_auc.std():.4f}, calibracao {CALIBRATION_METHOD}, "
        f"corte {cutoff:.3f}) e salvo em {CHAMPION_MODEL_PATH}."
    )
    return model


def main() -> None:
    """Entrada de linha de comando para treinar um modelo como campeao."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/churn.csv")
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--no-selection", action="store_true")
    args = parser.parse_args()
    train_champion(args.data, args.model, select_features=not args.no_selection)


if __name__ == "__main__":
    main()
