from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.serving.app as app_module
from src.config import load_global_config
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe
from src.serving.app import app


class _StubGlobalPythonModel:
    def __init__(self):
        universe = load_universe(DEFAULT_TICKERS_CONFIG)
        self.target_symbols = universe.target_symbols
        self.context_symbol = universe.context_symbol("implied_volatility")
        self.horizon = load_global_config().data.horizon


class _StubGlobalModel:
    """Inspectably compatible global pyfunc stand-in."""

    def __init__(self):
        universe = load_universe(DEFAULT_TICKERS_CONFIG)
        self.metadata = SimpleNamespace(
            metadata={
                "artifact_role": "global_candidate",
                "target_count": len(universe.target_symbols),
                "context_symbols": list(universe.context_symbols),
            }
        )
        self.python_model = _StubGlobalPythonModel()

    def unwrap_python_model(self):
        return self.python_model

    def predict(self, frame):
        supplied = set(frame["ticker"])
        requested = [
            ticker for ticker in self.python_model.target_symbols if ticker in supplied
        ]
        return pd.DataFrame(
            {
                "ticker": requested,
                "as_of_date": pd.to_datetime(["2026-06-26"] * len(requested)),
                "horizon_sessions": [self.python_model.horizon] * len(requested),
                "forecast": [0.2] * len(requested),
            }
        )


class _IncompleteGlobalModel(_StubGlobalModel):
    def predict(self, frame):
        return super().predict(frame).iloc[:1]


class _RaisingGlobalModel(_StubGlobalModel):
    def predict(self, frame):
        raise ValueError("bad input")


def _observations(tickers: tuple[str, ...]) -> list[dict]:
    dates = pd.bdate_range("2026-01-02", periods=2)
    return [
        {
            "date": date.date().isoformat(),
            "ticker": ticker,
            "adjusted_close": 100.0 + index,
        }
        for index, ticker in enumerate(tickers)
        for date in dates
    ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: _StubGlobalModel())
    monkeypatch.setenv("MODEL_BACKEND", "mlflow")
    monkeypatch.setenv("MODEL_URI", "/models/global")
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_global_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_scope": "global_34_target",
        "ready": True,
    }


def test_predict_serves_all_targets(client):
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    response = client.post(
        "/predict",
        json={
            "observations": _observations(
                universe.target_symbols + universe.context_symbols
            )
        },
    )

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [row["ticker"] for row in predictions] == list(universe.target_symbols)
    assert {row["horizon_sessions"] for row in predictions} == {5}
    assert all(row["forecast"] > 0 for row in predictions)


def test_predict_preserves_known_target_subset(client):
    response = client.post(
        "/predict",
        json={"observations": _observations(("SPY", "AAPL", "^VIX"))},
    )

    assert response.status_code == 200
    assert [row["ticker"] for row in response.json()["predictions"]] == ["AAPL", "SPY"]


def test_predict_surfaces_model_errors(client, monkeypatch):
    monkeypatch.setitem(app_module._state, "model", _RaisingGlobalModel())
    response = client.post(
        "/predict", json={"observations": _observations(("AAPL", "^VIX"))}
    )
    assert response.status_code == 422
    assert "Could not produce global forecasts" in response.json()["detail"]


def test_predict_rejects_invalid_artifact_output(client, monkeypatch):
    monkeypatch.setitem(app_module._state, "model", _IncompleteGlobalModel())
    response = client.post(
        "/predict", json={"observations": _observations(("AAPL", "SPY", "^VIX"))}
    )
    assert response.status_code == 500
    assert "invalid response" in response.json()["detail"]


def test_load_forecaster_selects_global_onnx_backend(monkeypatch):
    sentinel = _StubGlobalModel()
    captured = {}

    def fake_onnx_forecaster(
        model_path,
        scaler_path,
        artifact_manifest_path,
        export_manifest_path,
        *,
        threads,
    ):
        captured.update(
            model_path=model_path,
            scaler_path=scaler_path,
            artifact_manifest_path=artifact_manifest_path,
            export_manifest_path=export_manifest_path,
            threads=threads,
        )
        return sentinel

    monkeypatch.setenv("MODEL_BACKEND", "onnx")
    monkeypatch.setenv("MODEL_DIR", "/tmp/global-model")
    monkeypatch.setenv("ONNX_NUM_THREADS", "2")
    monkeypatch.setattr(
        app_module, "GlobalONNXVolatilityForecaster", fake_onnx_forecaster
    )

    assert app_module._load_forecaster() is sentinel
    assert (
        str(captured["model_path"])
        == "/tmp/global-model/optimized/global_volatility_lstm_fp32.onnx"
    )
    assert str(captured["scaler_path"]) == "/tmp/global-model/artifacts/scaler.pkl"
    assert (
        str(captured["artifact_manifest_path"])
        == "/tmp/global-model/artifacts/manifest.json"
    )
    assert (
        str(captured["export_manifest_path"])
        == "/tmp/global-model/optimized/global_volatility_lstm_fp32.json"
    )
    assert captured["threads"] == 2


def test_load_forecaster_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "unknown")
    with pytest.raises(ValueError, match="Unsupported MODEL_BACKEND"):
        app_module._load_forecaster()


def test_load_forecaster_rejects_floating_global_registry_uri(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "mlflow")
    monkeypatch.setenv("MODEL_URI", "models:/global-volatility-lstm/latest")
    with pytest.raises(ValueError, match="explicit numeric"):
        app_module._load_forecaster()
