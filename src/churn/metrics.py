
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import RANDOM_SEED

# Scorers da validacao cruzada, as mesmas metricas do notebook (principal: AUC-ROC).
CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def binary_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Calcula metricas principais de classificacao binaria."""

    probabilities = np.asarray(y_prob)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def cost_tradeoff_metrics(
    y_true,
    y_prob,
    *,
    threshold: float,
    customer_value: float = 1000.0,
    retention_success_rate: float = 0.30,
    action_cost: float = 50.0,
) -> dict[str, float]:
    """Estima impacto financeiro simples de falsos positivos e negativos."""

    probabilities = np.asarray(y_prob)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    avoided_churn_value = tp * customer_value * retention_success_rate
    action_total_cost = (tp + fp) * action_cost
    missed_churn_cost = fn * customer_value * retention_success_rate
    return {
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
        "retention_action_cost": float(action_total_cost),
        "missed_churn_cost": float(missed_churn_cost),
        "avoided_churn_value": float(avoided_churn_value),
        "net_avoided_churn_value": float(avoided_churn_value - action_total_cost),
    }


def top_k_metrics(y_true, y_prob, fraction: float) -> dict[str, float]:
    """Desempenho ao abordar so a fracao de clientes com maior risco (ex.: 10% da base).

    E a visao da campanha de retencao: quantos dos abordados cancelariam (precisao), quanto do
    churn total e alcancado (recall) e quantas vezes isso supera abordar ao acaso (lift).
    """

    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_prob), kind="stable")
    k = max(1, int(round(len(y_true) * fraction)))
    top = y_true[order[:k]]
    precision = float(top.mean())
    return {
        "precision": precision,
        "recall": float(top.sum() / y_true.sum()),
        "lift": precision / float(y_true.mean()),
    }


def bootstrap_auc_difference(
    y_true,
    y_prob_a,
    y_prob_b,
    *,
    n_resamples: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Diferenca de AUC-ROC (a - b) com intervalo de confianca de 95% por bootstrap pareado."""

    y_true, y_prob_a, y_prob_b = map(np.asarray, (y_true, y_prob_a, y_prob_b))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_resamples):
        # Mesma reamostragem para os dois modelos: a comparacao e pareada.
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        differences.append(
            roc_auc_score(y_true[idx], y_prob_a[idx]) - roc_auc_score(y_true[idx], y_prob_b[idx])
        )
    return {
        "auc_difference": float(roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b)),
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
    }
