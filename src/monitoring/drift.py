"""Small, dependency-free drift checks for volatility features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
