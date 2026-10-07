import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.panel import (
    PANEL_COLUMNS,
    VIX_LEVEL_COL,
    VIX_RETURN_COL,
)
from src.data.universe import ContextSeries, TargetAsset, Universe, load_universe
from src.features.realized_vol import build_features, log_returns
from src.monitoring.drift import (
    build_global_reference,
    validate_global_reference,
)
from src.monitoring.global_pipeline import (
    GlobalMonitoringConfig,
    aggregate_fleet_status,
    run_global_monitoring,
    validate_global_price_frame,
)
from src.monitoring.global_reporting import write_global_outputs
from src.data.ingest import DEFAULT_TICKERS_CONFIG


@pytest.fixture
def small_universe() -> Universe:
    return Universe(
        schema_version=1,
        targets=(
            TargetAsset("AAA", "group_one"),
            TargetAsset("BBB", "group_one"),
            TargetAsset("CCC", "group_two"),
            TargetAsset("DDD", "group_two"),
        ),
        context=(ContextSeries("^VIX", "implied_volatility"),),
    )


@pytest.fixture
def global_prices(small_universe: Universe) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=240)
    records: list[tuple[pd.Timestamp, str, float]] = []
    for ticker_number, ticker in enumerate(small_universe.target_symbols):
        returns = 0.0005 + 0.008 * np.sin(
            np.arange(len(dates)) / 9 + ticker_number / 3
        )
        prices = (80 + 10 * ticker_number) * np.exp(np.cumsum(returns))
        records.extend(zip(dates, [ticker] * len(dates), prices, strict=True))
    vix = 18 + 2 * np.sin(np.arange(len(dates)) / 11)
    records.extend(zip(dates, ["^VIX"] * len(dates), vix, strict=True))
    return pd.DataFrame(
        records,
        columns=["date", "ticker", "adjusted_close"],
    )


def _panel_from_prices(
    prices: pd.DataFrame,
    universe: Universe,
    horizon: int = 5,
) -> pd.DataFrame:
    by_symbol = {
        symbol: (
            prices.loc[prices["ticker"] == symbol]
            .set_index("date")["adjusted_close"]
            .sort_index()
        )
        for symbol in universe.all_symbols
    }
    context_prices = by_symbol[universe.context_symbol("implied_volatility")]
    context = pd.DataFrame(
        {
            VIX_LEVEL_COL: context_prices / 100,
            VIX_RETURN_COL: log_returns(context_prices),
        }
    )
    frames = {
        ticker: build_features(by_symbol[ticker], horizon)
        .join(context, how="inner")
        .loc[:, PANEL_COLUMNS]
        .dropna()
        for ticker in universe.target_symbols
    }
    common = next(iter(frames.values())).index
    for frame in frames.values():
        common = common.intersection(frame.index)
    panel = pd.concat(
        {ticker: frame.loc[common] for ticker, frame in frames.items()},
        names=["ticker", "date"],
    )
    return panel.swaplevel("ticker", "date").sort_index()


class _RecordingGlobalModel:
    def __init__(self, universe: Universe) -> None:
        self.universe = universe
        self.input_max_dates: list[pd.Timestamp] = []

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        as_of = pd.Timestamp(frame["date"].max())
        self.input_max_dates.append(as_of)
        return pd.DataFrame(
            {
                "ticker": self.universe.target_symbols,
                "as_of_date": [as_of] * len(self.universe.target_symbols),
                "horizon_sessions": [5] * len(self.universe.target_symbols),
                "forecast": [
                    0.18 + index * 0.01
                    for index in range(len(self.universe.target_symbols))
                ],
            }
        )


def _config() -> GlobalMonitoringConfig:
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
        fleet_warning_fraction=0.25,
        fleet_critical_fraction=0.5,
    )


def test_global_reference_covers_authoritative_universe():
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    reference = json.loads(Path("monitoring/global_reference.json").read_text())

    validate_global_reference(reference, universe)

    assert set(reference["tickers"]) == set(universe.target_symbols)
    assert reference["context"]["ticker"] == "^VIX"
    assert reference["training_rows"] == 118_354


def test_global_monitoring_covers_all_slices_without_future_inputs(
    tmp_path: Path,
    small_universe: Universe,
    global_prices: pd.DataFrame,
):
    panel = _panel_from_prices(global_prices, small_universe)
    reference = build_global_reference(
        panel,
        small_universe,
        panel_sha256="test-panel-sha",
    )
    model = _RecordingGlobalModel(small_universe)
    manifest = {"training": {"panel": {"sha256": "test-panel-sha"}}}

    report, predictions = run_global_monitoring(
        model,
        global_prices,
        reference,
        small_universe,
        _config(),
        horizon=5,
        model_version="test-global-v1",
        artifact_manifest=manifest,
    )
    write_global_outputs(tmp_path, report, predictions)

    assert report["universe"]["target_count"] == 4
    assert report["universe"]["context"] == ["^VIX"]
    assert set(report["feature_drift"]["tickers"]) == set(
        small_universe.target_symbols
    )
    assert set(report["feature_drift"]["groups"]) == {
        "group_one",
        "group_two",
    }
    assert set(report["forecast_error"]["tickers"]) == set(
        small_universe.target_symbols
    )
    assert len(predictions) == 12 * 4 + 4
    assert not predictions.loc[
        predictions["record_type"] == "backtest"
    ].duplicated(["as_of", "ticker"]).any()
    assert all(
        input_date.date().isoformat() == prediction_date
        for input_date, prediction_date in zip(
            model.input_max_dates,
            [report["as_of"]]
            + sorted(
                predictions.loc[
                    predictions["record_type"] == "backtest", "as_of"
                ].dt.date.astype(str).unique()
            ),
            strict=True,
        )
    )
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "predictions.csv").is_file()


def test_packaged_global_pyfunc_runs_complete_monitoring_contract(
    global_ci_fixture,
):
    _, model, prices = global_ci_fixture
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    reference = build_global_reference(
        _panel_from_prices(prices, universe),
        universe,
        panel_sha256="ci-fixture",
    )
    config = GlobalMonitoringConfig(
        current_window=10,
        backtest_window=4,
        recent_error_window=2,
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

    report, predictions = run_global_monitoring(
        model,
        prices,
        reference,
        universe,
        config,
        horizon=5,
        model_version="ci-global-fixture",
        artifact_manifest=model.unwrap_python_model().manifest,
    )

    assert report["universe"]["target_count"] == 34
    assert len(report["latest_forecasts"]) == 34
    assert len(predictions) == 4 * 34 + 34
    assert set(predictions["ticker"]) == set(universe.target_symbols)


def test_one_critical_ticker_produces_one_fleet_warning():
    result = aggregate_fleet_status(
        {"AAA": "critical", "BBB": "ok", "CCC": "ok", "DDD": "ok"},
        warning_fraction=0.25,
        critical_fraction=0.5,
    )

    assert result["status"] == "warning"
    assert result["affected_tickers"] == ["AAA"]


def test_widespread_critical_state_produces_critical_fleet_alert():
    result = aggregate_fleet_status(
        {
            "AAA": "critical",
            "BBB": "critical",
            "CCC": "ok",
            "DDD": "ok",
        },
        warning_fraction=0.25,
        critical_fraction=0.5,
    )

    assert result["status"] == "critical"


def test_global_monitoring_rejects_partial_or_stale_universe(
    small_universe: Universe,
    global_prices: pd.DataFrame,
):
    partial = global_prices.loc[global_prices["ticker"] != "DDD"]
    with pytest.raises(ValueError, match="coverage is incomplete"):
        validate_global_price_frame(
            partial,
            small_universe,
            max_stale_sessions=0,
        )

    stale = global_prices.loc[
        ~(
            (global_prices["ticker"] == "DDD")
            & (global_prices["date"] == global_prices["date"].max())
        )
    ]
    with pytest.raises(ValueError, match="stale series"):
        validate_global_price_frame(
            stale,
            small_universe,
            max_stale_sessions=0,
        )

    with pytest.raises(ValueError, match="stale relative to the run date"):
        validate_global_price_frame(
            global_prices,
            small_universe,
            max_stale_sessions=0,
            max_calendar_age_days=7,
            run_date=global_prices["date"].max() + pd.Timedelta(days=30),
        )


def test_reference_must_match_candidate_training_panel(
    small_universe: Universe,
    global_prices: pd.DataFrame,
):
    reference = build_global_reference(
        _panel_from_prices(global_prices, small_universe),
        small_universe,
        panel_sha256="accepted-panel",
    )

    with pytest.raises(ValueError, match="model training panel"):
        validate_global_reference(
            reference,
            small_universe,
            artifact_manifest={"training": {"panel": {"sha256": "other-panel"}}},
        )
