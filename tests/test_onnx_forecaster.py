from pathlib import Path

import joblib
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from src.features.realized_vol import FEATURE_COLS, build_inference_features
from src.models.lstm import VolatilityLSTM
from src.optimization.onnx_export import export_lstm_to_onnx
import src.serving.app as app_module
from src.serving.model_wrapper import VolatilityForecaster
from src.serving.onnx_forecaster import ONNXVolatilityForecaster

SEQ_LEN = 30


def _prices(n: int = 150) -> pd.Series:
    returns = np.random.default_rng(0).normal(0, 0.01, n)
    return pd.Series(100 * np.exp(np.cumsum(returns)))


@pytest.fixture(scope="module")
def forecasters(tmp_path_factory):
    tmp_path: Path = tmp_path_factory.mktemp("onnx_forecaster")
    torch.manual_seed(0)
    prices = _prices()
    features = build_inference_features(prices)
    scaler = StandardScaler().fit(features[FEATURE_COLS])
    model = VolatilityLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=8,
        num_layers=1,
        dropout=0.2,
    )
    model.eval()

    model_path = tmp_path / "model.pt"
    scaler_path = tmp_path / "scaler.pkl"
    onnx_path = tmp_path / "model.onnx"
    torch.save(model, model_path)
    joblib.dump(scaler, scaler_path)
    export_lstm_to_onnx(
        model_path,
        onnx_path,
        seq_len=SEQ_LEN,
        input_size=len(FEATURE_COLS),
    )

    pytorch = VolatilityForecaster(seq_len=SEQ_LEN)
    pytorch.model = model
    pytorch.scaler = scaler
    onnx = ONNXVolatilityForecaster(
        onnx_path, scaler_path, seq_len=SEQ_LEN
    )
    return prices, pytorch, onnx, onnx_path, scaler_path


def test_onnx_forecaster_matches_pytorch(forecasters):
    prices, pytorch, onnx, _, _ = forecasters

    assert onnx.predict(prices) == pytest.approx(
        pytorch.predict(None, prices), rel=1e-4, abs=1e-6
    )


def test_onnx_forecaster_is_deterministic(forecasters):
    prices, _, onnx, _, _ = forecasters

    assert onnx.predict(prices) == onnx.predict(prices)


def test_onnx_forecaster_rejects_short_price_history(forecasters):
    prices, _, onnx, _, _ = forecasters

    with pytest.raises(ValueError, match="complete feature rows"):
        onnx.predict(prices.iloc[:89])


def test_fastapi_serves_with_onnx_backend(forecasters, monkeypatch):
    prices, _, onnx, onnx_path, scaler_path = forecasters
    monkeypatch.setenv("MODEL_BACKEND", "onnx")
    monkeypatch.setenv("ONNX_MODEL_PATH", str(onnx_path))
    monkeypatch.setenv("ONNX_SCALER_PATH", str(scaler_path))
    monkeypatch.setenv("ONNX_NUM_THREADS", "1")

    with TestClient(app_module.app) as client:
        response = client.post("/predict", json={"prices": prices.tolist()})

    assert response.status_code == 200
    assert response.json()["forecast"] == pytest.approx(onnx.predict(prices))
    assert response.json()["horizon"] == 5
