"""Prepare complete-universe model, monitoring, and evaluation dashboard data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.universe import Universe
from src.features.global_inference import DATE_COL, PRICE_COL, TICKER_COL
from src.features.realized_vol import build_inference_features
from src.monitoring.global_pipeline import (
    GlobalMonitoringConfig,
    run_global_monitoring,
    validate_global_price_frame,
)

MODEL_IDS = ("naive", "ewma", "garch", "global_lstm")
MODEL_LABELS = {
    "naive": "Naive",
    "ewma": "EWMA",
    "garch": "GARCH(1,1)",
    "global_lstm": "Global LSTM",
}
METRIC_COLUMNS = ("rmse", "mae", "qlike")
VOLATILITY_COLUMNS = ("rv_5d", "rv_20d", "rv_60d")


@dataclass(frozen=True)
class GlobalDashboardSnapshot:
    """All dynamic data calculated once for the complete global universe."""

    report: dict[str, Any]
    predictions: pd.DataFrame
    recent_volatility: pd.DataFrame


@dataclass(frozen=True)
class GlobalEvaluationTables:
    """Validated final-holdout evidence at portfolio and ticker levels."""

    aggregate: pd.DataFrame
    per_ticker: pd.DataFrame


@dataclass(frozen=True)
class GlobalTickerView:
    """A fast slice of the complete snapshot for one forecast target."""

    ticker: str
    group: str
    forecast: float
    as_of: pd.Timestamp
    recent_volatility: pd.DataFrame
    history: pd.DataFrame
    monitoring: dict[str, Any]


def _validate_metric_rows(frame: pd.DataFrame, label: str) -> None:
    required = {*METRIC_COLUMNS, "observations"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} metrics are missing columns: {sorted(missing)}")
    metrics = frame.loc[:, [*METRIC_COLUMNS, "observations"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(metrics.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} metrics must be finite")
    if (metrics.loc[:, METRIC_COLUMNS] < 0).any().any() or (
        metrics["observations"] <= 0
    ).any():
        raise ValueError(f"{label} metrics contain invalid values")


def load_global_evaluation_tables(
    path: Path,
    universe: Universe,
) -> GlobalEvaluationTables:
    """Load the evidence-gated final holdout and enforce full coverage."""
    with path.open() as benchmark_file:
        payload = json.load(benchmark_file)
    if payload.get("schema_version") != 1:
        raise ValueError("Global benchmark schema_version must be 1")
    benchmark_universe = payload.get("universe", {})
    if benchmark_universe.get("targets") != list(universe.target_symbols):
        raise ValueError("Global benchmark target universe is incompatible")
    if benchmark_universe.get("context") != list(universe.context_symbols):
        raise ValueError("Global benchmark context universe is incompatible")

    aggregate = pd.DataFrame(payload.get("aggregate_metrics", []))
    per_ticker = pd.DataFrame(
        payload.get("final_holdout_per_ticker_metrics", [])
    )
    aggregate = aggregate.loc[aggregate["split"] == "final_holdout"].copy()
    per_ticker = per_ticker.loc[
        per_ticker["split"] == "final_holdout"
    ].copy()
    _validate_metric_rows(aggregate, "Aggregate")
    _validate_metric_rows(per_ticker, "Per-ticker")

    expected_groups = {asset.group for asset in universe.targets}
    expected_aggregate = {
        (model, scope, group)
        for model in MODEL_IDS
        for scope, groups in (
            ("macro", (None,)),
            ("micro", (None,)),
            ("group", tuple(expected_groups)),
        )
        for group in groups
    }
    actual_aggregate = {
        (
            row.model,
            row.scope,
            None if pd.isna(row.group) else row.group,
        )
        for row in aggregate.itertuples(index=False)
    }
    if actual_aggregate != expected_aggregate or len(aggregate) != len(
        expected_aggregate
    ):
        raise ValueError("Global aggregate benchmark coverage is incomplete")

    expected_ticker_rows = {
        (model, ticker)
        for model in MODEL_IDS
        for ticker in universe.target_symbols
    }
    actual_ticker_rows = set(
        per_ticker.loc[:, ["model", "ticker"]].itertuples(
            index=False,
            name=None,
        )
    )
    if actual_ticker_rows != expected_ticker_rows or len(per_ticker) != len(
        expected_ticker_rows
    ):
        raise ValueError("Global per-ticker benchmark coverage is incomplete")

    group_by_ticker = {asset.symbol: asset.group for asset in universe.targets}
    per_ticker["group"] = per_ticker["ticker"].map(group_by_ticker)
    return GlobalEvaluationTables(
        aggregate=aggregate.reset_index(drop=True),
        per_ticker=per_ticker.reset_index(drop=True),
    )


def _comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Format one complete four-model slice in a stable display order."""
    if set(frame["model"]) != set(MODEL_IDS) or len(frame) != len(MODEL_IDS):
        raise ValueError("Comparison must contain each required model exactly once")
    result = frame.set_index("model").loc[list(MODEL_IDS), [*METRIC_COLUMNS]]
    result.index = [MODEL_LABELS[model] for model in result.index]
    result.index.name = "model"
    return result


def ticker_model_comparison(
    tables: GlobalEvaluationTables,
    ticker: str,
) -> pd.DataFrame:
    return _comparison_table(tables.per_ticker.loc[tables.per_ticker["ticker"] == ticker])


def group_model_comparison(
    tables: GlobalEvaluationTables,
    group: str,
) -> pd.DataFrame:
    rows = tables.aggregate.loc[
        (tables.aggregate["scope"] == "group")
        & (tables.aggregate["group"] == group)
    ]
    return _comparison_table(rows)


def portfolio_model_comparison(
    tables: GlobalEvaluationTables,
    scope: str,
) -> pd.DataFrame:
    if scope not in {"macro", "micro"}:
        raise ValueError("Portfolio scope must be 'macro' or 'micro'")
    rows = tables.aggregate.loc[tables.aggregate["scope"] == scope]
    return _comparison_table(rows)


def build_global_dashboard_snapshot(
    model,
    prices: pd.DataFrame,
    reference: dict[str, Any],
    universe: Universe,
    monitoring_config: GlobalMonitoringConfig,
    *,
    horizon: int,
    model_version: str,
    artifact_manifest: dict[str, Any],
    run_date: pd.Timestamp | None,
    chart_window: int = 120,
) -> GlobalDashboardSnapshot:
    """Run the fleet monitor once and prepare every ticker's chart series."""
    report, predictions = run_global_monitoring(
        model,
        prices,
        reference,
        universe,
        monitoring_config,
        horizon=horizon,
        model_version=model_version,
        artifact_manifest=artifact_manifest,
        run_date=run_date,
    )
    normalized, _ = validate_global_price_frame(
        prices,
        universe,
        max_stale_sessions=monitoring_config.max_stale_sessions,
        max_calendar_age_days=monitoring_config.max_calendar_age_days,
        run_date=run_date,
    )
    volatility_frames: list[pd.DataFrame] = []
    for ticker in universe.target_symbols:
        series = normalized.loc[
            normalized[TICKER_COL] == ticker,
            [DATE_COL, PRICE_COL],
        ].set_index(DATE_COL)[PRICE_COL]
        features = build_inference_features(series).loc[
            :, list(VOLATILITY_COLUMNS)
        ].tail(chart_window)
        features.insert(0, TICKER_COL, ticker)
        volatility_frames.append(features.reset_index())
    recent_volatility = pd.concat(volatility_frames, ignore_index=True)
    return GlobalDashboardSnapshot(report, predictions, recent_volatility)


def select_global_ticker_view(
    snapshot: GlobalDashboardSnapshot,
    universe: Universe,
    ticker: str,
) -> GlobalTickerView:
    """Select one target without recalculating the all-universe monitor."""
    if ticker not in universe.target_symbols:
        raise ValueError(f"Ticker {ticker!r} is not a global forecast target")
    group = next(asset.group for asset in universe.targets if asset.symbol == ticker)
    latest = {
        item["ticker"]: item for item in snapshot.report["latest_forecasts"]
    }
    if set(latest) != set(universe.target_symbols):
        raise ValueError("Dashboard snapshot does not cover the complete universe")
    history = (
        snapshot.predictions.loc[
            (snapshot.predictions["ticker"] == ticker)
            & (snapshot.predictions["record_type"] == "backtest")
        ]
        .sort_values("as_of")
        .set_index("as_of")
    )
    recent_volatility = (
        snapshot.recent_volatility.loc[
            snapshot.recent_volatility[TICKER_COL] == ticker
        ]
        .drop(columns=TICKER_COL)
        .set_index(DATE_COL)
        .sort_index()
    )
    monitoring = {
        "fleet_status": snapshot.report["status"],
        "component_status": snapshot.report["component_status"],
        "data_quality": snapshot.report["data_quality"],
        "feature_drift": {
            "ticker": snapshot.report["feature_drift"]["tickers"][ticker],
            "group": snapshot.report["feature_drift"]["groups"][group],
            "fleet": snapshot.report["feature_drift"]["aggregate"],
        },
        "volatility_regime": {
            "ticker": snapshot.report["volatility_regime"]["tickers"][ticker],
            "group": snapshot.report["volatility_regime"]["groups"][group],
            "fleet": snapshot.report["volatility_regime"]["aggregate"],
        },
        "forecast_error": {
            "ticker": snapshot.report["forecast_error"]["tickers"][ticker],
            "group": snapshot.report["forecast_error"]["groups"][group],
            "fleet": snapshot.report["forecast_error"]["aggregate"],
            "micro": snapshot.report["forecast_error"]["micro"],
            "macro": snapshot.report["forecast_error"]["macro"],
        },
    }
    return GlobalTickerView(
        ticker=ticker,
        group=group,
        forecast=float(latest[ticker]["forecast"]),
        as_of=pd.Timestamp(snapshot.report["as_of"]),
        recent_volatility=recent_volatility,
        history=history,
        monitoring=monitoring,
    )
