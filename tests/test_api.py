"""Testes da API FastAPI (/health, /predict e /predict/batch)."""

import joblib
import pytest
from fastapi.testclient import TestClient

import churn.api as api
from churn.api import EXAMPLE_CUSTOMER
from churn.model import build_model_pipeline


@pytest.fixture
def client(xy_train, tmp_path, monkeypatch):
    """API servindo um pipeline pequeno treinado na amostra."""

    x, y = xy_train
    pipeline = build_model_pipeline("LightGBM", x, y, select_features=False, n_estimators=20)
    joblib.dump(pipeline.fit(x, y), tmp_path / "champion.joblib")
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "champion.joblib")
    with TestClient(api.app) as test_client:  # o "with" executa o lifespan (carga do modelo)
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model": "LGBMClassifier"}


def test_predict(client):
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["CustomerID"] == EXAMPLE_CUSTOMER["CustomerID"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_prediction"] == int(body["churn_probability"] >= body["threshold"])
    assert "X-Process-Time-ms" in response.headers


def test_predict_rejects_invalid_payload(client):
    # Variavel de retencao (vazamento) nao faz parte da entrada do modelo.
    response = client.post("/predict", json=EXAMPLE_CUSTOMER | {"RetentionCalls": 2})

    assert response.status_code == 422
    assert "RetentionCalls" in response.text


def test_predict_batch(client):
    customers = [EXAMPLE_CUSTOMER, EXAMPLE_CUSTOMER | {"CustomerID": 1, "MonthsInService": 11}]

    response = client.post("/predict/batch", json={"customers": customers})

    assert response.status_code == 200
    assert [p["CustomerID"] for p in response.json()["predictions"]] == [3000002, 1]


def test_without_model_health_is_degraded_and_predict_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "nao_existe.joblib")

    with TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/predict", json=EXAMPLE_CUSTOMER).status_code == 503
