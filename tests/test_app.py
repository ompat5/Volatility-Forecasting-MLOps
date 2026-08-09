import pytest
from fastapi.testclient import TestClient

import src.serving.app as app_module
from src.config import load_config
from src.serving.app import app


class _StubModel:
    """Stands in for the loaded pyfunc so tests need no MLflow registry / mlflow.db (CI-safe)."""

    def predict(self, prices):
        return 0.25


class _RaisingModel:
    """Simulates a model that blows up mid-predict, to exercise the error path."""

    def predict(self, prices):
        raise ValueError("bad input")


def _valid_prices(n: int = 120) -> list[float]:
    return [100.0 + i * 0.1 for i in range(n)]


@pytest.fixture
def client(monkeypatch):
    # lifespan calls mlflow.pyfunc.load_model at startup — patch it to return a stub so the
    # tests never touch the real registry. Using TestClient as a context manager runs lifespan.
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: _StubModel())
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_forecast_and_horizon(client):
    resp = client.post("/predict", json={"prices": _valid_prices()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["forecast"] == 0.25
    assert body["horizon"] == load_config().data.horizon


def test_predict_rejects_too_few_prices(client):
    # The pydantic min_length guardrail fires before the model is ever called.
    resp = client.post("/predict", json={"prices": [1.0, 2.0, 3.0]})
    assert resp.status_code == 422


def test_predict_handles_model_error(client, monkeypatch):
    # A model failure should surface as a clean 422, not an opaque 500.
    monkeypatch.setitem(app_module._state, "model", _RaisingModel())
    resp = client.post("/predict", json={"prices": _valid_prices()})
    assert resp.status_code == 422
    assert "Could not produce a forecast" in resp.json()["detail"]
