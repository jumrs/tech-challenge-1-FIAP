"""Testes das metricas tecnicas e de negocio."""

import numpy as np
import pytest

from churn.metrics import (
    binary_classification_metrics,
    bootstrap_auc_difference,
    cost_tradeoff_metrics,
    top_k_metrics,
)

Y_TRUE = np.array([0, 0, 1, 1])
Y_PROB = np.array([0.1, 0.4, 0.35, 0.8])


def test_threshold_changes_class_metrics_but_not_auc():
    default = binary_classification_metrics(Y_TRUE, Y_PROB, threshold=0.5)
    low = binary_classification_metrics(Y_TRUE, Y_PROB, threshold=0.3)

    assert default["recall"] == pytest.approx(0.5)
    assert low["recall"] == pytest.approx(1.0)
    assert default["roc_auc"] == low["roc_auc"] == pytest.approx(0.75)


def test_cost_tradeoff_metrics():
    result = cost_tradeoff_metrics(
        Y_TRUE, Y_PROB, threshold=0.5, customer_value=1000, retention_success_rate=0.5,
        action_cost=100,
    )

    # Threshold 0,5: TP=1, FN=1, FP=0 -> valor evitado 500, custo da acao 100.
    assert result["net_avoided_churn_value"] == pytest.approx(400)
    assert result["missed_churn_cost"] == pytest.approx(500)


def test_top_k_metrics():
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.05])

    result = top_k_metrics(y_true, y_prob, fraction=0.3)

    # Top 3 (0,9; 0,8; 0,7): 2 churns de 3 abordados, 2 dos 3 churns da base.
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["lift"] == pytest.approx((2 / 3) / 0.3)


def test_bootstrap_auc_difference():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    better = y_true * 0.3 + rng.random(300)

    result = bootstrap_auc_difference(y_true, better, rng.random(300), n_resamples=200)

    assert result["ci_low"] <= result["auc_difference"] <= result["ci_high"]
    assert result["ci_low"] > 0
