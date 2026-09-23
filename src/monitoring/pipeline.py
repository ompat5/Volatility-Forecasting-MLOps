"""Run a daily forecast plus feature, regime, and delayed-error monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.eval.metrics import mae, qlike, rmse
from src.features.realized_vol import (
    FEATURE_COLS,
    TARGET_COL,
    build_features,
    build_inference_features,
)
from src.monitoring.drift import feature_drift_scores


@dataclass(frozen=True)
class MonitoringConfig:
    ticker: str
    current_window: int
    backtest_window: int
    recent_error_window: int
    psi_warning: float
    psi_critical: float
    error_ratio_warning: float
    error_ratio_critical: float


def load_monitoring_config(path: Path) -> tuple[MonitoringConfig, dict[str, str]]:
    """Load monitoring thresholds and immutable model-release metadata."""
    with path.open() as config_file:
        raw = yaml.safe_load(config_file)
    return MonitoringConfig(**raw["monitoring"]), raw["model"]


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


def historical_predictions(
    model,
    prices: pd.Series,
    *,
    horizon: int,
    window: int,
) -> pd.DataFrame:
    """Recreate recent as-of forecasts, then join targets now observable."""
    features = build_features(prices, horizon)
    dates = features.index[-window:]
    records = []
    for as_of in dates:
        forecast = float(model.predict(prices.loc[:as_of].tail(120)))
        records.append(
            {
                "as_of": as_of,
                "forecast": forecast,
                "actual": float(features.loc[as_of, TARGET_COL]),
            }
        )
    return pd.DataFrame.from_records(records).set_index("as_of")


def _error_summary(
    history: pd.DataFrame,
    recent_window: int,
    warning: float,
    critical: float,
) -> dict[str, Any]:
    if len(history) <= recent_window:
        raise ValueError("Backtest window must exceed the recent error window")

    baseline = history.iloc[:-recent_window]
    recent = history.iloc[-recent_window:]
    baseline_rmse = rmse(baseline["actual"], baseline["forecast"])
    recent_rmse = rmse(recent["actual"], recent["forecast"])
    ratio = recent_rmse / max(baseline_rmse, 1e-12)

    summary: dict[str, Any] = {
        "observations": len(history),
        "rmse": rmse(history["actual"], history["forecast"]),
        "mae": mae(history["actual"], history["forecast"]),
        "baseline_rmse": baseline_rmse,
        "recent_rmse": recent_rmse,
        "recent_to_baseline_rmse_ratio": ratio,
        "status": _severity(ratio, warning, critical),
    }
    if (history["forecast"] > 0).all():
        summary["qlike"] = qlike(history["actual"], history["forecast"])
    else:
        summary["qlike"] = None
        summary["non_positive_predictions"] = int(
            (history["forecast"] <= 0).sum()
        )
        summary["status"] = "critical"
    return summary


def run_monitoring(
    model,
    prices: pd.Series,
    reference: dict[str, Any],
    config: MonitoringConfig,
    *,
    horizon: int,
    model_version: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Produce the latest forecast, backtest, drift scores, and alert state."""
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    inference_features = build_inference_features(prices)
    current_features = inference_features.iloc[-config.current_window:]

    psi_scores = feature_drift_scores(current_features[FEATURE_COLS], reference)
    drift_statuses = {
        feature: _severity(
            score,
            config.psi_warning,
            config.psi_critical,
        )
        for feature, score in psi_scores.items()
    }
    drift_status = _overall_status(list(drift_statuses.values()))

    latest_rv_20d = float(inference_features["rv_20d"].iloc[-1])
    reference_p95 = float(reference["rv_20d_p95"])
    regime_status = "warning" if latest_rv_20d > reference_p95 else "ok"

    history = historical_predictions(
        model,
        prices,
        horizon=horizon,
        window=config.backtest_window,
    )
    error_summary = _error_summary(
        history,
        config.recent_error_window,
        config.error_ratio_warning,
        config.error_ratio_critical,
    )

    latest_forecast = float(model.predict(prices.tail(120)))
    as_of = prices.index.max()
    report: dict[str, Any] = {
        "ticker": config.ticker,
        "as_of": as_of.isoformat(),
        "horizon": horizon,
        "model_version": model_version,
        "forecast": latest_forecast,
        "reference_training_end": reference["training_end"],
        "feature_drift": {
            "window": config.current_window,
            "psi": psi_scores,
            "feature_status": drift_statuses,
            "status": drift_status,
        },
        "volatility_regime": {
            "latest_rv_20d": latest_rv_20d,
            "reference_p95": reference_p95,
            "status": regime_status,
        },
        "forecast_error": error_summary,
    }
    report["status"] = _overall_status(
        [drift_status, regime_status, error_summary["status"]]
    )

    latest_row = pd.DataFrame(
        [
            {
                "as_of": as_of,
                "forecast": latest_forecast,
                "actual": np.nan,
                "record_type": "live",
                "model_version": model_version,
            }
        ]
    ).set_index("as_of")
    history = history.assign(
        record_type="backtest",
        model_version=model_version,
    )
    return report, pd.concat([history, latest_row])


def _report_markdown(report: dict[str, Any]) -> str:
    drift = report["feature_drift"]
    regime = report["volatility_regime"]
    error = report["forecast_error"]
    lines = [
        f"# Volatility monitoring — {report['ticker']}",
        "",
        f"**Overall status:** `{report['status'].upper()}`  ",
        f"**As of:** {report['as_of']}  ",
        f"**Model:** `{report['model_version']}`  ",
        f"**{report['horizon']}-day forecast:** {report['forecast']:.4f}",
        "",
        "## Feature drift",
        "",
        "| Feature | PSI | Status |",
        "|---|---:|---|",
    ]
    for feature, score in drift["psi"].items():
        lines.append(
            f"| {feature} | {score:.4f} | {drift['feature_status'][feature]} |"
        )
    lines.extend(
        [
            "",
            "## Volatility regime",
            "",
            f"Current RV20 is **{regime['latest_rv_20d']:.4f}**; the training "
            f"95th percentile is **{regime['reference_p95']:.4f}** "
            f"(`{regime['status']}`).",
            "",
            "## Delayed forecast error",
            "",
            f"- RMSE: {error['rmse']:.4f}",
            f"- MAE: {error['mae']:.4f}",
            f"- QLIKE: {error['qlike'] if error['qlike'] is not None else 'unavailable'}",
            f"- Recent/baseline RMSE ratio: "
            f"{error['recent_to_baseline_rmse_ratio']:.3f} (`{error['status']}`)",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    predictions: pd.DataFrame,
) -> None:
    """Persist machine-readable and human-readable monitoring artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "report.md").write_text(_report_markdown(report))
    predictions.to_csv(output_dir / "predictions.csv", index_label="as_of")
