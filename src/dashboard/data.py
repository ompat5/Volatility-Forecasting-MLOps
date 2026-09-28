"""Prepare model, monitoring, and evaluation data for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.realized_vol import build_inference_features
from src.monitoring.pipeline import MonitoringConfig, run_monitoring

RESULT_COLUMNS = ["model", "rmse", "mae", "qlike"]
VOLATILITY_COLUMNS = ["rv_5d", "rv_20d", "rv_60d"]


@dataclass
class DashboardSnapshot:
    """All dynamic values needed to render one dashboard refresh."""

    report: dict[str, Any]
    predictions: pd.DataFrame
    recent_volatility: pd.DataFrame


def load_model_results(path: Path) -> pd.DataFrame:
    """Load and validate the stored walk-forward comparison table."""
    results = pd.read_csv(path)
    missing = set(RESULT_COLUMNS) - set(results.columns)
    if missing:
        raise ValueError(f"Model results are missing columns: {sorted(missing)}")
    return results[RESULT_COLUMNS]


def build_dashboard_snapshot(
    model,
    prices: pd.Series,
    reference: dict[str, Any],
    monitoring_config: MonitoringConfig,
    *,
    horizon: int,
    model_version: str,
    chart_window: int = 120,
) -> DashboardSnapshot:
    """Run monitoring and prepare recent annualized-volatility series."""
    report, predictions = run_monitoring(
        model,
        prices,
        reference,
        monitoring_config,
        horizon=horizon,
        model_version=model_version,
    )
    recent_volatility = build_inference_features(prices)[
        VOLATILITY_COLUMNS
    ].tail(chart_window)
    return DashboardSnapshot(report, predictions, recent_volatility)


def status_label(status: str) -> str:
    """Turn a monitoring status into a compact UI label."""
    icons = {"ok": "🟢", "warning": "🟠", "critical": "🔴"}
    return f"{icons.get(status, '⚪')} {status.upper()}"
