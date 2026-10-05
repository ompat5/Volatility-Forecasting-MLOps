"""Full-universe drift, regime, and delayed-error monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.panel import (
    GLOBAL_FEATURE_COLS,
    TARGET_COL,
    VIX_LEVEL_COL,
    VIX_RETURN_COL,
)
from src.data.universe import Universe
from src.eval.metrics import mae, qlike, rmse
from src.features.global_inference import (
    DATE_COL,
    GLOBAL_INFERENCE_COLUMNS,
    PRICE_COL,
    TICKER_COL,
)
from src.features.realized_vol import (
    FEATURE_COLS,
    build_features,
    build_inference_features,
    log_returns,
)
from src.monitoring.drift import (
    feature_drift_scores,
    validate_global_reference,
)
from src.monitoring.pipeline import summarize_forecast_error


@dataclass(frozen=True)
class GlobalMonitoringConfig:
    """Thresholds shared by all global monitoring slices."""

    current_window: int
    backtest_window: int
    recent_error_window: int
    history_observations: int
    max_stale_sessions: int
    max_calendar_age_days: int
    psi_warning: float
    psi_critical: float
    error_ratio_warning: float
    error_ratio_critical: float
    fleet_warning_fraction: float
    fleet_critical_fraction: float

    def __post_init__(self) -> None:
        positive = (
            self.current_window,
            self.backtest_window,
            self.recent_error_window,
            self.history_observations,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Global monitoring windows must be positive")
        if self.backtest_window <= self.recent_error_window:
            raise ValueError(
                "Global backtest window must exceed recent error window"
            )
        if self.history_observations < 90:
            raise ValueError("Global monitoring needs at least 90 raw observations")
        if self.max_stale_sessions < 0 or self.max_calendar_age_days < 0:
            raise ValueError("Global monitoring freshness limits cannot be negative")
        if not 0 < self.psi_warning < self.psi_critical:
            raise ValueError("PSI thresholds must be positive and ordered")
        if not 0 < self.error_ratio_warning < self.error_ratio_critical:
            raise ValueError("Error-ratio thresholds must be positive and ordered")
        if not (
            0 < self.fleet_warning_fraction <= self.fleet_critical_fraction <= 1
        ):
            raise ValueError("Fleet alert fractions must be ordered within (0, 1]")


def load_global_monitoring_config(
    path: Path,
) -> tuple[GlobalMonitoringConfig, dict[str, str]]:
    """Load global thresholds and candidate identity metadata."""
    with path.open() as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError("Global monitoring config must contain a mapping")
    return GlobalMonitoringConfig(**raw["monitoring"]), raw["model"]


def _severity(value: float, warning: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "ok"


def _overall_status(statuses: list[str]) -> str:
    if "critical" in statuses:
        return "critical"
    if "warning" in statuses:
        return "warning"
    return "ok"


def aggregate_fleet_status(
    statuses: dict[str, str],
    *,
    warning_fraction: float,
    critical_fraction: float,
) -> dict[str, Any]:
    """Collapse ticker states without turning one outlier into 34 alerts."""
    if not statuses:
        raise ValueError("Fleet status requires at least one ticker")
    if not 0 < warning_fraction <= critical_fraction <= 1:
        raise ValueError("Fleet alert fractions must be ordered within (0, 1]")
    invalid = set(statuses.values()) - {"ok", "warning", "critical"}
    if invalid:
        raise ValueError(f"Unknown monitoring statuses: {sorted(invalid)}")

    total = len(statuses)
    critical_count = sum(status == "critical" for status in statuses.values())
    warning_count = sum(status == "warning" for status in statuses.values())
    non_ok_count = critical_count + warning_count
    critical_share = critical_count / total
    non_ok_share = non_ok_count / total
    if critical_share >= critical_fraction:
        status = "critical"
    elif non_ok_share >= warning_fraction or critical_count:
        status = "warning"
    else:
        status = "ok"
    return {
        "status": status,
        "total": total,
        "ok": total - non_ok_count,
        "warning": warning_count,
        "critical": critical_count,
        "non_ok_fraction": non_ok_share,
        "critical_fraction": critical_share,
        "affected_tickers": [
            ticker for ticker, ticker_status in statuses.items() if ticker_status != "ok"
        ],
    }


def _group_fleet_summaries(
    ticker_statuses: dict[str, str],
    universe: Universe,
    config: GlobalMonitoringConfig,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, str]] = {}
    for asset in universe.targets:
        groups.setdefault(asset.group, {})[asset.symbol] = ticker_statuses[asset.symbol]
    return {
        group: aggregate_fleet_status(
            statuses,
            warning_fraction=config.fleet_warning_fraction,
            critical_fraction=config.fleet_critical_fraction,
        )
        for group, statuses in groups.items()
    }


def validate_global_price_frame(
    model_input: pd.DataFrame,
    universe: Universe,
    *,
    max_stale_sessions: int,
    max_calendar_age_days: int = 7,
    run_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require complete, finite, sufficiently current histories for 35 series."""
    if not isinstance(model_input, pd.DataFrame):
        raise ValueError("Global monitoring prices must be a pandas DataFrame")
    if set(model_input.columns) != set(GLOBAL_INFERENCE_COLUMNS):
        raise ValueError(
            "Global monitoring input columns must be exactly "
            f"{list(GLOBAL_INFERENCE_COLUMNS)}"
        )
    frame = model_input.loc[:, list(GLOBAL_INFERENCE_COLUMNS)].copy()
    if frame.empty:
        raise ValueError("Global monitoring prices cannot be empty")
    parsed_dates = pd.DatetimeIndex(
        pd.to_datetime(frame[DATE_COL], errors="raise")
    )
    if parsed_dates.hasnans:
        raise ValueError("Global monitoring dates cannot be missing")
    frame[DATE_COL] = pd.DatetimeIndex(
        [timestamp.date() for timestamp in parsed_dates],
        name=DATE_COL,
    )
    frame[TICKER_COL] = frame[TICKER_COL].astype(str).str.strip()
    frame[PRICE_COL] = pd.to_numeric(frame[PRICE_COL], errors="coerce")
    if frame.duplicated([DATE_COL, TICKER_COL]).any():
        raise ValueError("Global monitoring prices contain duplicate keys")
    values = frame[PRICE_COL].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Global monitoring prices must be finite and positive")

    supplied = set(frame[TICKER_COL])
    expected = set(universe.all_symbols)
    if supplied != expected:
        raise ValueError(
            "Global monitoring universe coverage is incomplete: "
            f"missing={sorted(expected - supplied)}, unknown={sorted(supplied - expected)}"
        )
    ordered_dates = pd.DatetimeIndex(frame[DATE_COL].unique()).sort_values()
    latest_by_symbol = frame.groupby(TICKER_COL)[DATE_COL].max()
    stale_sessions = {
        symbol: int((ordered_dates > latest_by_symbol.loc[symbol]).sum())
        for symbol in universe.all_symbols
    }
    stale = {
        symbol: count
        for symbol, count in stale_sessions.items()
        if count > max_stale_sessions
    }
    if stale:
        raise ValueError(f"Global monitoring found stale series: {stale}")
    calendar_age_days: int | None = None
    if run_date is not None:
        normalized_run_date = pd.Timestamp(run_date)
        if normalized_run_date.tzinfo is not None:
            normalized_run_date = normalized_run_date.tz_localize(None)
        normalized_run_date = normalized_run_date.normalize()
        calendar_age_days = int((normalized_run_date - ordered_dates.max()).days)
        if calendar_age_days < 0:
            raise ValueError("Global monitoring inputs are dated in the future")
        if calendar_age_days > max_calendar_age_days:
            raise ValueError(
                "Global monitoring basket is stale relative to the run date: "
                f"age={calendar_age_days} days, limit={max_calendar_age_days}"
            )
    rows_by_symbol = frame.groupby(TICKER_COL).size()
    too_short = {
        symbol: int(rows_by_symbol.loc[symbol])
        for symbol in universe.all_symbols
        if rows_by_symbol.loc[symbol] < 90
    }
    if too_short:
        raise ValueError(f"Global monitoring histories are too short: {too_short}")

    frame = frame.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    return frame, {
        "status": "ok",
        "input_rows": len(frame),
        "target_count": len(universe.target_symbols),
        "context_count": len(universe.context_symbols),
        "latest_date": ordered_dates.max().date().isoformat(),
        "max_stale_sessions": max(stale_sessions.values()),
        "wall_clock_freshness_enforced": run_date is not None,
        "calendar_age_days": calendar_age_days,
        "stale_sessions": stale_sessions,
        "rows_by_symbol": {
            symbol: int(rows_by_symbol.loc[symbol])
            for symbol in universe.all_symbols
        },
    }


def _feature_frames(
    frame: pd.DataFrame,
    universe: Universe,
    *,
    horizon: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    context_symbol = universe.context_symbol("implied_volatility")
    context_prices = (
        frame.loc[frame[TICKER_COL] == context_symbol]
        .set_index(DATE_COL)[PRICE_COL]
        .sort_index()
    )
    context = pd.DataFrame(
        {
            VIX_LEVEL_COL: context_prices / 100.0,
            VIX_RETURN_COL: log_returns(context_prices),
        }
    )
    inference: dict[str, pd.DataFrame] = {}
    labeled: dict[str, pd.DataFrame] = {}
    for ticker in universe.target_symbols:
        prices = (
            frame.loc[frame[TICKER_COL] == ticker]
            .set_index(DATE_COL)[PRICE_COL]
            .sort_index()
        )
        inference[ticker] = (
            build_inference_features(prices)
            .join(context, how="inner")
            .loc[:, GLOBAL_FEATURE_COLS]
            .dropna()
        )
        labeled[ticker] = (
            build_features(prices, horizon)
            .join(context, how="inner")
            .loc[:, [*GLOBAL_FEATURE_COLS, TARGET_COL]]
            .dropna()
        )
    return inference, labeled


def _common_dates(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    common = next(iter(frames.values())).index
    for frame in frames.values():
        common = common.intersection(frame.index, sort=False)
    return pd.DatetimeIndex(common).sort_values()


def _prediction_snapshot(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    observations: int,
) -> pd.DataFrame:
    available = frame.loc[frame[DATE_COL] <= as_of]
    snapshot = available.groupby(TICKER_COL, sort=False).tail(observations)
    return snapshot.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)


def _validated_prediction(
    model,
    snapshot: pd.DataFrame,
    universe: Universe,
    *,
    as_of: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    output = model.predict(snapshot)
    if not isinstance(output, pd.DataFrame):
        raise ValueError("Global monitor requires tabular model predictions")
    required = {"ticker", "as_of_date", "horizon_sessions", "forecast"}
    if set(output.columns) != required:
        raise ValueError("Global monitor received an incompatible model output")
    if output["ticker"].tolist() != list(universe.target_symbols):
        raise ValueError("Global monitor prediction coverage or order is incomplete")
    if output["ticker"].duplicated().any():
        raise ValueError("Global monitor received duplicate ticker predictions")
    forecast = pd.to_numeric(output["forecast"], errors="coerce").to_numpy()
    if not np.isfinite(forecast).all() or (forecast <= 0).any():
        raise ValueError("Global monitor received invalid volatility forecasts")
    if set(output["horizon_sessions"]) != {horizon}:
        raise ValueError("Global monitor received an incompatible horizon")
    output_dates = pd.DatetimeIndex(pd.to_datetime(output["as_of_date"]))
    if any(date != as_of for date in output_dates):
        raise ValueError("Global monitor prediction dates are not synchronized")
    return output.copy()


def historical_global_predictions(
    model,
    frame: pd.DataFrame,
    labeled_features: dict[str, pd.DataFrame],
    universe: Universe,
    config: GlobalMonitoringConfig,
    *,
    horizon: int,
) -> pd.DataFrame:
    """Recreate recent synchronized forecasts using only each as-of history."""
    eligible_dates = _common_dates(labeled_features)
    if len(eligible_dates) < config.backtest_window:
        raise ValueError(
            "Global monitoring lacks enough common target-observable dates"
        )
    dates = eligible_dates[-config.backtest_window :]
    groups = {asset.symbol: asset.group for asset in universe.targets}
    records: list[dict[str, Any]] = []
    for as_of in dates:
        snapshot = _prediction_snapshot(
            frame,
            as_of=as_of,
            observations=config.history_observations,
        )
        predictions = _validated_prediction(
            model,
            snapshot,
            universe,
            as_of=as_of,
            horizon=horizon,
        )
        for row in predictions.itertuples(index=False):
            records.append(
                {
                    "as_of": as_of,
                    "ticker": row.ticker,
                    "group": groups[row.ticker],
                    "forecast": float(row.forecast),
                    "actual": float(
                        labeled_features[row.ticker].loc[as_of, TARGET_COL]
                    ),
                    "record_type": "backtest",
                }
            )
    history = pd.DataFrame.from_records(records)
    expected_rows = config.backtest_window * len(universe.target_symbols)
    if len(history) != expected_rows or history.duplicated(
        ["as_of", "ticker"]
    ).any():
        raise ValueError("Global monitoring history coverage is incomplete")
    return history


def _pooled_error_summary(
    history: pd.DataFrame,
    config: GlobalMonitoringConfig,
) -> dict[str, Any]:
    dates = pd.DatetimeIndex(history["as_of"].unique()).sort_values()
    if len(dates) <= config.recent_error_window:
        raise ValueError("Global error history has too few dates")
    cutoff = dates[-config.recent_error_window]
    baseline = history.loc[history["as_of"] < cutoff]
    recent = history.loc[history["as_of"] >= cutoff]
    baseline_rmse = rmse(baseline["actual"], baseline["forecast"])
    recent_rmse = rmse(recent["actual"], recent["forecast"])
    ratio = recent_rmse / max(baseline_rmse, 1e-12)
    return {
        "dates": len(dates),
        "observations": len(history),
        "rmse": rmse(history["actual"], history["forecast"]),
        "mae": mae(history["actual"], history["forecast"]),
        "qlike": qlike(history["actual"], history["forecast"]),
        "baseline_rmse": baseline_rmse,
        "recent_rmse": recent_rmse,
        "recent_to_baseline_rmse_ratio": ratio,
        "status": _severity(
            ratio,
            config.error_ratio_warning,
            config.error_ratio_critical,
        ),
    }


def _drift_report(
    inference_features: dict[str, pd.DataFrame],
    reference: dict[str, Any],
    universe: Universe,
    config: GlobalMonitoringConfig,
) -> dict[str, Any]:
    ticker_results: dict[str, Any] = {}
    ticker_statuses: dict[str, str] = {}
    for ticker in universe.target_symbols:
        current = inference_features[ticker].iloc[-config.current_window :]
        if len(current) < config.current_window:
            raise ValueError(f"Global drift window is incomplete for {ticker}")
        scores = feature_drift_scores(current[FEATURE_COLS], reference["tickers"][ticker])
        feature_status = {
            feature: _severity(score, config.psi_warning, config.psi_critical)
            for feature, score in scores.items()
        }
        status = _overall_status(list(feature_status.values()))
        ticker_statuses[ticker] = status
        ticker_results[ticker] = {
            "group": reference["tickers"][ticker]["group"],
            "psi": scores,
            "feature_status": feature_status,
            "status": status,
        }

    first_target = universe.target_symbols[0]
    current_context = inference_features[first_target].iloc[-config.current_window :]
    context_scores = feature_drift_scores(
        current_context[[VIX_LEVEL_COL, VIX_RETURN_COL]], reference["context"]
    )
    context_feature_status = {
        feature: _severity(score, config.psi_warning, config.psi_critical)
        for feature, score in context_scores.items()
    }
    context_status = _overall_status(list(context_feature_status.values()))
    aggregate = aggregate_fleet_status(
        ticker_statuses,
        warning_fraction=config.fleet_warning_fraction,
        critical_fraction=config.fleet_critical_fraction,
    )
    return {
        "window": config.current_window,
        "status": _overall_status([aggregate["status"], context_status]),
        "context": {
            "ticker": universe.context_symbol("implied_volatility"),
            "psi": context_scores,
            "feature_status": context_feature_status,
            "status": context_status,
        },
        "aggregate": aggregate,
        "groups": _group_fleet_summaries(ticker_statuses, universe, config),
        "tickers": ticker_results,
    }


def _regime_report(
    inference_features: dict[str, pd.DataFrame],
    reference: dict[str, Any],
    universe: Universe,
    config: GlobalMonitoringConfig,
) -> dict[str, Any]:
    tickers: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for asset in universe.targets:
        latest = float(inference_features[asset.symbol]["rv_20d"].iloc[-1])
        ticker_reference = reference["tickers"][asset.symbol]
        p95 = float(ticker_reference["rv_20d_p95"])
        p99 = float(ticker_reference["rv_20d_p99"])
        status = "critical" if latest > p99 else "warning" if latest > p95 else "ok"
        statuses[asset.symbol] = status
        tickers[asset.symbol] = {
            "group": asset.group,
            "latest_rv_20d": latest,
            "reference_p95": p95,
            "reference_p99": p99,
            "status": status,
        }
    aggregate = aggregate_fleet_status(
        statuses,
        warning_fraction=config.fleet_warning_fraction,
        critical_fraction=config.fleet_critical_fraction,
    )
    return {
        "status": aggregate["status"],
        "aggregate": aggregate,
        "groups": _group_fleet_summaries(statuses, universe, config),
        "tickers": tickers,
    }


def _error_report(
    history: pd.DataFrame,
    universe: Universe,
    config: GlobalMonitoringConfig,
) -> dict[str, Any]:
    ticker_results: dict[str, Any] = {}
    ticker_statuses: dict[str, str] = {}
    for asset in universe.targets:
        ticker_history = (
            history.loc[history["ticker"] == asset.symbol]
            .sort_values("as_of")
            .set_index("as_of")
        )
        summary = summarize_forecast_error(
            ticker_history,
            config.recent_error_window,
            config.error_ratio_warning,
            config.error_ratio_critical,
        )
        summary["group"] = asset.group
        ticker_results[asset.symbol] = summary
        ticker_statuses[asset.symbol] = summary["status"]

    group_results = {
        group: _pooled_error_summary(group_history, config)
        for group, group_history in history.groupby("group", sort=False)
    }
    micro = _pooled_error_summary(history, config)
    macro = {
        metric: float(
            np.mean([ticker_results[ticker][metric] for ticker in universe.target_symbols])
        )
        for metric in ("rmse", "mae", "qlike", "recent_to_baseline_rmse_ratio")
    }
    aggregate = aggregate_fleet_status(
        ticker_statuses,
        warning_fraction=config.fleet_warning_fraction,
        critical_fraction=config.fleet_critical_fraction,
    )
    return {
        "status": _overall_status([micro["status"], aggregate["status"]]),
        "micro": micro,
        "macro": macro,
        "aggregate": aggregate,
        "groups": group_results,
        "tickers": ticker_results,
    }


def run_global_monitoring(
    model,
    model_input: pd.DataFrame,
    reference: dict[str, Any],
    universe: Universe,
    config: GlobalMonitoringConfig,
    *,
    horizon: int,
    model_version: str,
    artifact_manifest: dict | None = None,
    run_date: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Produce one global report and one complete prediction ledger."""
    if horizon <= 0:
        raise ValueError("Global monitoring horizon must be positive")
    validate_global_reference(
        reference,
        universe,
        artifact_manifest=artifact_manifest,
    )
    frame, data_quality = validate_global_price_frame(
        model_input,
        universe,
        max_stale_sessions=config.max_stale_sessions,
        max_calendar_age_days=config.max_calendar_age_days,
        run_date=run_date,
    )
    inference_features, labeled_features = _feature_frames(
        frame,
        universe,
        horizon=horizon,
    )
    common_inference_dates = _common_dates(inference_features)
    if common_inference_dates.empty:
        raise ValueError("Global monitoring has no common inference date")
    latest_as_of = common_inference_dates[-1]
    latest_snapshot = _prediction_snapshot(
        frame,
        as_of=latest_as_of,
        observations=config.history_observations,
    )
    latest = _validated_prediction(
        model,
        latest_snapshot,
        universe,
        as_of=latest_as_of,
        horizon=horizon,
    )
    groups = {asset.symbol: asset.group for asset in universe.targets}
    latest_records = [
        {
            "as_of": latest_as_of,
            "ticker": row.ticker,
            "group": groups[row.ticker],
            "forecast": float(row.forecast),
            "actual": np.nan,
            "record_type": "live",
            "model_version": model_version,
        }
        for row in latest.itertuples(index=False)
    ]

    history = historical_global_predictions(
        model,
        frame,
        labeled_features,
        universe,
        config,
        horizon=horizon,
    ).assign(model_version=model_version)
    drift = _drift_report(
        inference_features,
        reference,
        universe,
        config,
    )
    regime = _regime_report(
        inference_features,
        reference,
        universe,
        config,
    )
    forecast_error = _error_report(history, universe, config)
    component_status = {
        "data_quality": data_quality["status"],
        "feature_drift": drift["status"],
        "volatility_regime": regime["status"],
        "forecast_error": forecast_error["status"],
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_role": "global_candidate_monitoring",
        "status": _overall_status(list(component_status.values())),
        "component_status": component_status,
        "as_of": latest_as_of.date().isoformat(),
        "horizon_sessions": horizon,
        "model_version": model_version,
        "reference_training_end": reference["training_end"],
        "universe": {
            "target_count": len(universe.target_symbols),
            "targets": list(universe.target_symbols),
            "context": list(universe.context_symbols),
            "groups": sorted({asset.group for asset in universe.targets}),
        },
        "data_quality": data_quality,
        "latest_forecasts": [
            {
                "ticker": row.ticker,
                "group": groups[row.ticker],
                "forecast": float(row.forecast),
            }
            for row in latest.itertuples(index=False)
        ],
        "feature_drift": drift,
        "volatility_regime": regime,
        "forecast_error": forecast_error,
    }
    ledger = pd.concat(
        [history, pd.DataFrame.from_records(latest_records)],
        ignore_index=True,
    )
    return report, ledger
