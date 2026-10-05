"""Small, dependency-free drift checks for volatility features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.panel import (
    GLOBAL_FEATURE_COLS,
    VIX_LEVEL_COL,
    VIX_RETURN_COL,
    validate_global_panel,
)
from src.data.universe import Universe
from src.features.realized_vol import FEATURE_COLS


def _bin_proportions(values: pd.Series, edges: list[float]) -> np.ndarray:
    bins = np.array([-np.inf, *edges, np.inf], dtype=float)
    counts, _ = np.histogram(values.dropna().to_numpy(dtype=float), bins=bins)
    if counts.sum() == 0:
        raise ValueError("Cannot calculate drift from an empty feature series")
    return counts / counts.sum()


def build_reference(
    features: pd.DataFrame,
    *,
    ticker: str,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Capture quantile bins and distributions from the model's training data."""
    if features.empty:
        raise ValueError("Reference features cannot be empty")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    feature_reference: dict[str, Any] = {}
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for column in features.columns:
        values = features[column].dropna().astype(float)
        edges = np.unique(values.quantile(quantiles).to_numpy()).tolist()
        proportions = _bin_proportions(values, edges)
        feature_reference[column] = {
            "bin_edges": edges,
            "proportions": proportions.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    rv_20d = features["rv_20d"].dropna()
    return {
        "ticker": ticker,
        "training_start": features.index.min().isoformat(),
        "training_end": features.index.max().isoformat(),
        "n_observations": len(features),
        "features": feature_reference,
        "rv_20d_p95": float(rv_20d.quantile(0.95)),
    }


def build_global_reference(
    panel: pd.DataFrame,
    universe: Universe,
    *,
    panel_sha256: str,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Capture fixed per-target and shared-context training distributions."""
    validate_global_panel(panel, universe)
    if not panel_sha256:
        raise ValueError("Global monitoring reference requires a panel SHA-256")

    per_ticker: dict[str, Any] = {}
    for asset in universe.targets:
        features = panel.xs(asset.symbol, level="ticker")
        reference = build_reference(
            features[FEATURE_COLS],
            ticker=asset.symbol,
            n_bins=n_bins,
        )
        rv_20d = features["rv_20d"]
        reference["group"] = asset.group
        reference["rv_20d_p99"] = float(rv_20d.quantile(0.99))
        per_ticker[asset.symbol] = reference

    # VIX inputs are identical on every balanced-panel ticker row. Store them
    # once so a shared market-context shift is not counted as 34 ticker drifts.
    first_target = universe.target_symbols[0]
    context_features = panel.xs(first_target, level="ticker")[
        [VIX_LEVEL_COL, VIX_RETURN_COL]
    ]
    context_reference = build_reference(
        context_features.assign(rv_20d=context_features[VIX_LEVEL_COL]),
        ticker=universe.context_symbol("implied_volatility"),
        n_bins=n_bins,
    )
    context_reference["features"] = {
        column: context_reference["features"][column]
        for column in (VIX_LEVEL_COL, VIX_RETURN_COL)
    }
    context_reference.pop("rv_20d_p95")

    dates = panel.index.get_level_values("date").unique().sort_values()
    return {
        "schema_version": 1,
        "artifact_role": "global_monitoring_reference",
        "panel_sha256": panel_sha256,
        "training_start": dates.min().date().isoformat(),
        "training_end": dates.max().date().isoformat(),
        "training_dates": len(dates),
        "training_rows": len(panel),
        "feature_columns": GLOBAL_FEATURE_COLS,
        "universe": {
            "targets": [
                {"symbol": asset.symbol, "group": asset.group}
                for asset in universe.targets
            ],
            "context": [
                {"symbol": series.symbol, "role": series.role}
                for series in universe.context
            ],
        },
        "context": context_reference,
        "tickers": per_ticker,
    }


def validate_global_reference(
    reference: dict[str, Any],
    universe: Universe,
    *,
    artifact_manifest: dict | None = None,
) -> None:
    """Require monitoring metadata to match the model and repository universe."""
    if reference.get("schema_version") != 1:
        raise ValueError("Global monitoring reference schema_version must be 1")
    if reference.get("artifact_role") != "global_monitoring_reference":
        raise ValueError("Global monitoring reference role is incompatible")
    expected_targets = [
        {"symbol": asset.symbol, "group": asset.group}
        for asset in universe.targets
    ]
    expected_context = [
        {"symbol": series.symbol, "role": series.role}
        for series in universe.context
    ]
    if reference.get("universe", {}).get("targets") != expected_targets:
        raise ValueError("Global monitoring target universe is incompatible")
    if reference.get("universe", {}).get("context") != expected_context:
        raise ValueError("Global monitoring context universe is incompatible")
    if reference.get("feature_columns") != GLOBAL_FEATURE_COLS:
        raise ValueError("Global monitoring feature schema is incompatible")
    if set(reference.get("tickers", {})) != set(universe.target_symbols):
        raise ValueError("Global monitoring reference lacks complete target coverage")
    if reference.get("context", {}).get("ticker") != universe.context_symbol(
        "implied_volatility"
    ):
        raise ValueError("Global monitoring context reference is incompatible")
    for asset in universe.targets:
        ticker_reference = reference["tickers"][asset.symbol]
        if ticker_reference.get("group") != asset.group:
            raise ValueError(
                f"Global monitoring group is incompatible for {asset.symbol}"
            )
        if set(ticker_reference.get("features", {})) != set(FEATURE_COLS):
            raise ValueError(
                f"Global monitoring feature reference is incomplete for {asset.symbol}"
            )
        if not 0 < ticker_reference.get("rv_20d_p95", 0) <= ticker_reference.get(
            "rv_20d_p99", 0
        ):
            raise ValueError(
                f"Global monitoring regime reference is invalid for {asset.symbol}"
            )
    if set(reference["context"].get("features", {})) != {
        VIX_LEVEL_COL,
        VIX_RETURN_COL,
    }:
        raise ValueError("Global monitoring context features are incomplete")

    if artifact_manifest is not None:
        panel_hash = (
            artifact_manifest.get("training", {})
            .get("panel", {})
            .get("sha256")
        )
        if reference.get("panel_sha256") != panel_hash:
            raise ValueError(
                "Global monitoring reference does not match the model training panel"
            )


def population_stability_index(
    current: pd.Series,
    reference: dict[str, Any],
    *,
    epsilon: float = 1e-6,
) -> float:
    """Compare a current feature distribution with fixed reference bins."""
    current_proportions = _bin_proportions(current, reference["bin_edges"])
    reference_proportions = np.asarray(reference["proportions"], dtype=float)
    if len(current_proportions) != len(reference_proportions):
        raise ValueError("Reference bin metadata is inconsistent")

    current_safe = np.clip(current_proportions, epsilon, None)
    reference_safe = np.clip(reference_proportions, epsilon, None)
    return float(
        np.sum(
            (current_safe - reference_safe)
            * np.log(current_safe / reference_safe)
        )
    )


def feature_drift_scores(
    current_features: pd.DataFrame,
    reference: dict[str, Any],
) -> dict[str, float]:
    """Calculate PSI for every feature in the stored reference."""
    missing = set(reference["features"]) - set(current_features.columns)
    if missing:
        raise ValueError(f"Current features are missing columns: {sorted(missing)}")
    return {
        column: population_stability_index(
            current_features[column], metadata
        )
        for column, metadata in reference["features"].items()
    }
