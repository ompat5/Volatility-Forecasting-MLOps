from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard.data import (
    build_dashboard_snapshot,
    load_model_results,
    status_label,
)
import src.dashboard.runtime as runtime
from src.features.realized_vol import FEATURE_COLS, build_inference_features
from src.monitoring.drift import build_reference
from src.monitoring.pipeline import MonitoringConfig


class _ConstantModel:
    def predict(self, _prices):
        return 0.2


def _prices(n: int = 400) -> pd.Series:
    returns = np.random.default_rng(8).normal(0.0, 0.01, n)
    return pd.Series(
        100 * np.exp(np.cumsum(returns)),
        index=pd.bdate_range("2023-01-02", periods=n),
    )


def test_load_model_results_has_expected_models():
    path = Path("dashboard/assets/model_results.csv")

    results = load_model_results(path)

    assert results["model"].tolist() == ["Naive", "EWMA", "GARCH(1,1)", "LSTM"]
    assert results.loc[results["model"] == "LSTM", "rmse"].item() == 0.2061


def test_build_dashboard_snapshot_contains_all_views():
    prices = _prices()
    reference_features = build_inference_features(prices.iloc[:300])[FEATURE_COLS]
    reference = build_reference(reference_features, ticker="AAPL")
    config = MonitoringConfig(
        ticker="AAPL",
        current_window=30,
        backtest_window=12,
        recent_error_window=4,
        psi_warning=0.1,
        psi_critical=0.25,
        error_ratio_warning=1.5,
        error_ratio_critical=2.0,
    )

    snapshot = build_dashboard_snapshot(
        _ConstantModel(),
        prices,
        reference,
        config,
        horizon=5,
        model_version="test-v1",
        chart_window=90,
    )

    assert snapshot.report["ticker"] == "AAPL"
    assert len(snapshot.recent_volatility) == 90
    assert list(snapshot.recent_volatility) == ["rv_5d", "rv_20d", "rv_60d"]
    assert len(snapshot.predictions) == 13


def test_status_label_is_explicit():
    assert status_label("ok") == "🟢 OK"
    assert status_label("warning") == "🟠 WARNING"
    assert status_label("critical") == "🔴 CRITICAL"


def test_dashboard_runtime_reuses_existing_artifacts(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    artifacts = model_dir / "artifacts"
    optimized = model_dir / "optimized"
    artifacts.mkdir(parents=True)
    optimized.mkdir()
    (artifacts / "model.pt").write_bytes(b"model")
    (artifacts / "scaler.pkl").write_bytes(b"scaler")
    (optimized / "volatility_lstm_fp32.onnx").write_bytes(b"onnx")
    sentinel = object()

    monkeypatch.setattr(runtime, "ONNXVolatilityForecaster", lambda *a, **k: sentinel)
    monkeypatch.setattr(
        runtime,
        "download_model",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    assert runtime.load_dashboard_forecaster(model_dir) is sentinel


def test_dashboard_runtime_downloads_and_exports_when_missing(tmp_path, monkeypatch):
    model_dir = tmp_path / "downloaded-model"
    sentinel = object()

    def fake_download(_config_path, output):
        artifacts = output / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "model.pt").write_bytes(b"model")
        (artifacts / "scaler.pkl").write_bytes(b"scaler")

    def fake_export(_model_path, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"onnx")

    monkeypatch.setattr(runtime, "download_model", fake_download)
    monkeypatch.setattr(runtime, "export_lstm_to_onnx", fake_export)
    monkeypatch.setattr(runtime, "ONNXVolatilityForecaster", lambda *a, **k: sentinel)

    assert runtime.load_dashboard_forecaster(model_dir) is sentinel
    assert (model_dir / "optimized" / "volatility_lstm_fp32.onnx").is_file()
