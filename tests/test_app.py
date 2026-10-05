from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.serving.app as app_module
from src.config import load_config, load_global_config
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe
from src.serving.app import app


class _StubModel:
    """Stands in for the loaded pyfunc so tests need no MLflow registry / mlflow.db (CI-safe)."""

    def predict(self, prices):
        return 0.25


class _RaisingModel:
    """Simulates a model that blows up mid-predict, to exercise the error path."""

    def predict(self, prices):
        raise ValueError("bad input")


class _StubGlobalPythonModel:
    def __init__(self):
        universe = load_universe(DEFAULT_TICKERS_CONFIG)
        self.target_symbols = universe.target_symbols
        self.context_symbol = universe.context_symbol("implied_volatility")
        self.horizon = load_global_config().data.horizon


class _StubGlobalModel:
    """Global pyfunc stand-in with the same inspectable system contract."""

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
        output = super().predict(frame)
        return output.iloc[:1]


def _valid_prices(n: int = 120) -> list[float]:
    return [100.0 + i * 0.1 for i in range(n)]


def _global_observations(tickers: tuple[str, ...]) -> list[dict]:
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
    # lifespan calls mlflow.pyfunc.load_model at startup — patch it to return a stub so the
    # tests never touch the real registry. Using TestClient as a context manager runs lifespan.
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: _StubModel())
    monkeypatch.setenv("MODEL_BACKEND", "mlflow")
    monkeypatch.delenv("GLOBAL_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("GLOBAL_MODEL_URI", raising=False)
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_global_health_is_separate_and_disabled_by_default(client):
    resp = client.get("/health/global")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "disabled",
        "configured": False,
        "ready": False,
    }


def test_global_predict_is_unavailable_without_opt_in(client):
    resp = client.post("/predict/global", json={"observations": []})

    assert resp.status_code == 422

    observations = _global_observations(("AAPL", "^VIX"))
    resp = client.post("/predict/global", json={"observations": observations})
    assert resp.status_code == 503
    assert "GLOBAL_MODEL_URI" in resp.json()["detail"]


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


def test_load_forecaster_selects_onnx_backend(monkeypatch):
    sentinel = _StubModel()
    captured = {}

    def fake_onnx_forecaster(model_path, scaler_path, *, seq_len, threads):
        captured.update(
            model_path=model_path,
            scaler_path=scaler_path,
            seq_len=seq_len,
            threads=threads,
        )
        return sentinel

    monkeypatch.setenv("MODEL_BACKEND", "onnx")
    monkeypatch.setenv("ONNX_MODEL_PATH", "/tmp/model.onnx")
    monkeypatch.setenv("ONNX_SCALER_PATH", "/tmp/scaler.pkl")
    monkeypatch.setenv("ONNX_NUM_THREADS", "2")
    monkeypatch.setattr(
        app_module, "ONNXVolatilityForecaster", fake_onnx_forecaster
    )

    assert app_module._load_forecaster() is sentinel
    assert str(captured["model_path"]) == "/tmp/model.onnx"
    assert str(captured["scaler_path"]) == "/tmp/scaler.pkl"
    assert captured["seq_len"] == load_config().model.seq_len
    assert captured["threads"] == 2


def test_load_forecaster_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "unknown")

    with pytest.raises(ValueError, match="Unsupported MODEL_BACKEND"):
        app_module._load_forecaster()


def test_global_endpoint_serves_all_targets_without_changing_aapl(monkeypatch):
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    legacy_model = _StubModel()
    global_model = _StubGlobalModel()

    def fake_load_model(uri):
        return global_model if uri == "/models/global" else legacy_model

    monkeypatch.setattr("mlflow.pyfunc.load_model", fake_load_model)
    monkeypatch.setenv("MODEL_BACKEND", "mlflow")
    monkeypatch.setenv("MODEL_URI", "/models/aapl")
    monkeypatch.setenv("GLOBAL_MODEL_URI", "/models/global")
    with TestClient(app) as configured_client:
        health = configured_client.get("/health/global")
        legacy = configured_client.post(
            "/predict", json={"prices": _valid_prices()}
        )
        observations = _global_observations(
            universe.target_symbols + universe.context_symbols
        )
        response = configured_client.post(
            "/predict/global",
            json={"observations": observations},
        )

    assert health.json() == {"status": "ok", "configured": True, "ready": True}
    assert legacy.json() == {"forecast": 0.25, "horizon": 5}
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [row["ticker"] for row in predictions] == list(universe.target_symbols)
    assert len(predictions) == 34
    assert {row["horizon_sessions"] for row in predictions} == {5}
    assert all(row["forecast"] > 0 for row in predictions)


def test_global_endpoint_preserves_known_target_subset(monkeypatch):
    legacy_model = _StubModel()
    global_model = _StubGlobalModel()
    monkeypatch.setattr(
        "mlflow.pyfunc.load_model",
        lambda uri: global_model if uri == "/models/global" else legacy_model,
    )
    monkeypatch.setenv("MODEL_URI", "/models/aapl")
    monkeypatch.setenv("GLOBAL_MODEL_URI", "/models/global")
    observations = _global_observations(("SPY", "AAPL", "^VIX"))

    with TestClient(app) as configured_client:
        response = configured_client.post(
            "/predict/global", json={"observations": observations}
        )

    assert response.status_code == 200
    assert [
        row["ticker"] for row in response.json()["predictions"]
    ] == ["AAPL", "SPY"]


def test_global_endpoint_rejects_invalid_observation_before_model(client):
    observation = _global_observations(("AAPL",))[0]
    observation["adjusted_close"] = 0

    response = client.post(
        "/predict/global", json={"observations": [observation]}
    )

    assert response.status_code == 422


def test_global_endpoint_rejects_incomplete_model_output(monkeypatch):
    legacy_model = _StubModel()
    global_model = _IncompleteGlobalModel()
    monkeypatch.setattr(
        "mlflow.pyfunc.load_model",
        lambda uri: global_model if uri == "/models/global" else legacy_model,
    )
    monkeypatch.setenv("MODEL_URI", "/models/aapl")
    monkeypatch.setenv("GLOBAL_MODEL_URI", "/models/global")
    observations = _global_observations(("AAPL", "SPY", "^VIX"))

    with TestClient(app) as configured_client:
        response = configured_client.post(
            "/predict/global", json={"observations": observations}
        )

    assert response.status_code == 500
    assert "each requested target exactly once" in response.json()["detail"]


def test_global_loader_rejects_stale_target_universe(monkeypatch):
    stale_model = _StubGlobalModel()
    stale_model.python_model.target_symbols = ("AAPL",)
    monkeypatch.setenv("GLOBAL_MODEL_URI", "/models/stale-global")
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda _uri: stale_model)

    with pytest.raises(ValueError, match="target universe"):
        app_module._load_global_forecaster()


def test_global_loader_rejects_aapl_artifact(monkeypatch):
    monkeypatch.setenv("GLOBAL_MODEL_URI", "/models/aapl")
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda _uri: _StubModel())

    with pytest.raises(ValueError, match="compatible global candidate"):
        app_module._load_global_forecaster()


def test_global_loader_rejects_floating_registry_version(monkeypatch):
    monkeypatch.setenv(
        "GLOBAL_MODEL_URI", "models:/global-volatility-lstm/latest"
    )

    with pytest.raises(ValueError, match="explicit numeric"):
        app_module._load_global_forecaster()


def test_global_loader_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("GLOBAL_MODEL_BACKEND", "unknown")

    with pytest.raises(ValueError, match="Unsupported GLOBAL_MODEL_BACKEND"):
        app_module._load_global_forecaster()


def test_global_onnx_backend_rejects_ambiguous_mlflow_uri(monkeypatch):
    monkeypatch.setenv("GLOBAL_MODEL_BACKEND", "onnx")
    monkeypatch.setenv("GLOBAL_MODEL_URI", "/models/global")

    with pytest.raises(ValueError, match="must be unset"):
        app_module._load_global_forecaster()
