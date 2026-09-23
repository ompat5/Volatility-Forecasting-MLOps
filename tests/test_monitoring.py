import numpy as np
import pandas as pd
import pytest

from src.features.realized_vol import FEATURE_COLS, build_inference_features
from src.monitoring.drift import (
    build_reference,
    feature_drift_scores,
    population_stability_index,
)
from src.monitoring.pipeline import MonitoringConfig, run_monitoring, write_outputs


class _ConstantModel:
    def predict(self, _prices):
        return 0.2


def _feature_frame(n: int = 1_000) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            column: rng.normal(index, 1.0, n)
            for index, column in enumerate(FEATURE_COLS)
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )


def _prices(n: int = 400) -> pd.Series:
    rng = np.random.default_rng(8)
    returns = rng.normal(0.0, 0.01, n)
    return pd.Series(
        100 * np.exp(np.cumsum(returns)),
        index=pd.bdate_range("2023-01-02", periods=n),
    )


def test_identical_distribution_has_zero_psi():
    features = _feature_frame()
    reference = build_reference(features, ticker="TEST")

    score = population_stability_index(
        features["log_returns"], reference["features"]["log_returns"]
    )

    assert score == pytest.approx(0.0)


def test_shifted_distribution_has_material_drift():
    features = _feature_frame()
    reference = build_reference(features, ticker="TEST")
    shifted = features + 5.0

    scores = feature_drift_scores(shifted, reference)

    assert all(score > 0.25 for score in scores.values())


def test_run_monitoring_writes_complete_artifacts(tmp_path):
    prices = _prices()
    reference_features = build_inference_features(prices.iloc[:300])[FEATURE_COLS]
    reference = build_reference(reference_features, ticker="TEST")
    config = MonitoringConfig(
        ticker="TEST",
        current_window=30,
        backtest_window=12,
        recent_error_window=4,
        psi_warning=0.1,
        psi_critical=0.25,
        error_ratio_warning=1.5,
        error_ratio_critical=2.0,
    )

    report, predictions = run_monitoring(
        _ConstantModel(),
        prices,
        reference,
        config,
        horizon=5,
        model_version="test-v1",
    )
    write_outputs(tmp_path, report, predictions)

    assert report["ticker"] == "TEST"
    assert report["model_version"] == "test-v1"
    assert set(report["feature_drift"]["psi"]) == set(FEATURE_COLS)
    assert len(predictions) == 13
    assert predictions.iloc[-1]["record_type"] == "live"
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "predictions.csv").is_file()
