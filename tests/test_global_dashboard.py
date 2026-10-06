import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import REPO_ROOT
from src.dashboard.global_data import (
    GlobalDashboardSnapshot,
    build_global_dashboard_snapshot,
    group_model_comparison,
    load_global_evaluation_tables,
    portfolio_model_comparison,
    select_global_ticker_view,
    ticker_model_comparison,
)
from src.dashboard.global_runtime import load_global_dashboard_forecaster
from src.data.global_prices import long_prices_from_series
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import ContextSeries, TargetAsset, Universe, load_universe
from src.monitoring.global_pipeline import GlobalMonitoringConfig


@pytest.fixture
def small_universe() -> Universe:
    return Universe(
        schema_version=1,
        targets=(
            TargetAsset("AAA", "group_one"),
            TargetAsset("BBB", "group_two"),
        ),
        context=(ContextSeries("^VIX", "implied_volatility"),),
    )


def _small_prices(universe: Universe) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=140)
    return long_prices_from_series(
        {
            symbol: pd.Series(
                (80 + 10 * index)
                * np.exp(np.cumsum(0.001 + 0.005 * np.sin(np.arange(140) / 7))),
                index=dates,
            )
            for index, symbol in enumerate(universe.all_symbols)
        },
        universe,
    )


def _monitoring_config() -> GlobalMonitoringConfig:
    return GlobalMonitoringConfig(
        current_window=20,
        backtest_window=12,
        recent_error_window=4,
        history_observations=120,
        max_stale_sessions=0,
        max_calendar_age_days=7,
        psi_warning=0.1,
        psi_critical=0.25,
        error_ratio_warning=1.5,
        error_ratio_critical=2.0,
        fleet_warning_fraction=0.1,
        fleet_critical_fraction=0.25,
    )


def test_global_evaluation_tables_cover_every_target_and_slice():
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    tables = load_global_evaluation_tables(
        REPO_ROOT / "benchmarks" / "global_model.json",
        universe,
    )

    assert len(tables.aggregate) == 4 * (2 + 6)
    assert len(tables.per_ticker) == 4 * 34
    assert ticker_model_comparison(tables, "AAPL").shape == (4, 3)
    assert group_model_comparison(tables, "technology").shape == (4, 3)
    assert portfolio_model_comparison(tables, "micro").shape == (4, 3)


def test_global_evaluation_rejects_partial_universe(tmp_path: Path):
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    payload = json.loads(
        (REPO_ROOT / "benchmarks" / "global_model.json").read_text()
    )
    payload["universe"]["targets"] = payload["universe"]["targets"][:-1]
    benchmark_path = tmp_path / "partial.json"
    benchmark_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="target universe"):
        load_global_evaluation_tables(benchmark_path, universe)


def test_global_price_frame_preserves_authoritative_ticker_order(
    small_universe: Universe,
):
    prices = _small_prices(small_universe)
    first_date = prices["date"].min()

    assert prices.loc[prices["date"] == first_date, "ticker"].tolist() == [
        "AAA",
        "BBB",
        "^VIX",
    ]
    with pytest.raises(ValueError, match="complete global universe"):
        long_prices_from_series(
            {"AAA": pd.Series([1.0], index=[first_date])},
            small_universe,
        )


def test_dashboard_snapshot_prepares_every_ticker_once(
    monkeypatch: pytest.MonkeyPatch,
    small_universe: Universe,
):
    prices = _small_prices(small_universe)
    predictions = pd.DataFrame(
        {
            "as_of": [pd.Timestamp("2024-07-01")] * 2,
            "ticker": ["AAA", "BBB"],
            "group": ["group_one", "group_two"],
            "forecast": [0.2, 0.3],
            "actual": [0.21, 0.29],
            "record_type": ["backtest", "backtest"],
            "model_version": ["candidate", "candidate"],
        }
    )
    report = {"as_of": "2024-07-01"}
    calls = []

    def fake_monitor(*args, **kwargs):
        calls.append((args, kwargs))
        return report, predictions

    monkeypatch.setattr(
        "src.dashboard.global_data.run_global_monitoring",
        fake_monitor,
    )
    snapshot = build_global_dashboard_snapshot(
        object(),
        prices,
        {},
        small_universe,
        _monitoring_config(),
        horizon=5,
        model_version="candidate",
        artifact_manifest={},
        run_date=None,
        chart_window=20,
    )

    assert len(calls) == 1
    assert set(snapshot.recent_volatility["ticker"]) == {"AAA", "BBB"}
    assert snapshot.recent_volatility.groupby("ticker").size().to_dict() == {
        "AAA": 20,
        "BBB": 20,
    }


def test_ticker_selection_is_a_slice_and_vix_is_not_a_target(
    small_universe: Universe,
):
    ticker_monitor = {
        "group": "group_one",
        "status": "ok",
        "psi": {"rv_5d": 0.02},
        "latest_rv_20d": 0.2,
        "reference_p95": 0.4,
        "recent_to_baseline_rmse_ratio": 1.1,
    }
    second_monitor = ticker_monitor | {"group": "group_two"}
    group_monitor = {"status": "ok"}
    fleet_monitor = {"status": "ok"}
    report = {
        "status": "ok",
        "as_of": "2024-07-01",
        "component_status": {
            "data_quality": "ok",
            "feature_drift": "ok",
            "volatility_regime": "ok",
            "forecast_error": "ok",
        },
        "data_quality": {"status": "ok"},
        "latest_forecasts": [
            {"ticker": "AAA", "group": "group_one", "forecast": 0.2},
            {"ticker": "BBB", "group": "group_two", "forecast": 0.3},
        ],
        "feature_drift": {
            "aggregate": fleet_monitor,
            "groups": {"group_one": group_monitor, "group_two": group_monitor},
            "tickers": {"AAA": ticker_monitor, "BBB": second_monitor},
        },
        "volatility_regime": {
            "aggregate": fleet_monitor,
            "groups": {"group_one": group_monitor, "group_two": group_monitor},
            "tickers": {"AAA": ticker_monitor, "BBB": second_monitor},
        },
        "forecast_error": {
            "aggregate": fleet_monitor,
            "groups": {"group_one": group_monitor, "group_two": group_monitor},
            "tickers": {"AAA": ticker_monitor, "BBB": second_monitor},
            "micro": ticker_monitor,
            "macro": {"rmse": 0.1},
        },
    }
    predictions = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2024-06-28", "2024-06-28"]),
            "ticker": ["AAA", "BBB"],
            "forecast": [0.19, 0.29],
            "actual": [0.2, 0.3],
            "record_type": ["backtest", "backtest"],
        }
    )
    volatility = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-06-28", "2024-06-28"]),
            "ticker": ["AAA", "BBB"],
            "rv_5d": [0.1, 0.2],
            "rv_20d": [0.2, 0.3],
            "rv_60d": [0.3, 0.4],
        }
    )
    snapshot = GlobalDashboardSnapshot(report, predictions, volatility)

    view = select_global_ticker_view(snapshot, small_universe, "AAA")

    assert view.ticker == "AAA"
    assert view.group == "group_one"
    assert view.forecast == 0.2
    assert len(view.history) == 1
    with pytest.raises(ValueError, match="not a global forecast target"):
        select_global_ticker_view(snapshot, small_universe, "^VIX")


def _candidate_files(model_dir: Path) -> None:
    artifacts = model_dir / "artifacts"
    artifacts.mkdir(parents=True)
    for name in ("model_state.pt", "scaler.pkl", "manifest.json"):
        (artifacts / name).touch()


def test_global_runtime_exports_complete_missing_onnx_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _candidate_files(tmp_path)
    sentinel = object()
    exported = []
    validated = []

    def fake_export(**kwargs):
        kwargs["output_path"].parent.mkdir(parents=True)
        kwargs["output_path"].touch()
        kwargs["export_manifest_path"].touch()
        exported.append(kwargs)

    monkeypatch.setattr(
        "src.dashboard.global_runtime.export_global_lstm_to_onnx",
        fake_export,
    )
    monkeypatch.setattr(
        "src.dashboard.global_runtime.GlobalONNXVolatilityForecaster",
        lambda *args, **kwargs: sentinel,
    )
    monkeypatch.setattr(
        "src.dashboard.global_runtime.validate_global_forecaster",
        validated.append,
    )

    result = load_global_dashboard_forecaster(tmp_path)

    assert result is sentinel
    assert len(exported) == 1
    assert validated == [sentinel]


def test_global_runtime_rejects_partial_onnx_bundle(tmp_path: Path):
    _candidate_files(tmp_path)
    optimized = tmp_path / "optimized"
    optimized.mkdir()
    (optimized / "global_volatility_lstm_fp32.onnx").touch()

    with pytest.raises(FileNotFoundError, match="Incomplete global ONNX bundle"):
        load_global_dashboard_forecaster(tmp_path)
