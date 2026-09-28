"""Shared preprocessing helpers for optimized inference paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.realized_vol import FEATURE_COLS, build_inference_features


def load_adjusted_close(path: Path, observations: int | None = None) -> pd.Series:
    """Load sorted adjusted-close prices from a cached ingestion artifact."""
    frame = pd.read_parquet(path)
    if "Adj Close" not in frame:
        raise ValueError(f"{path} does not contain an 'Adj Close' column")
    prices = frame["Adj Close"].dropna().sort_index()
    if observations is not None:
        prices = prices.tail(observations)
        if len(prices) < observations:
            raise ValueError(
                f"Need {observations} prices, found {len(prices)}"
            )
    return prices


def build_scaled_window(prices: pd.Series, scaler, seq_len: int) -> np.ndarray:
    """Convert raw prices into one contiguous FP32 model input window."""
    features = build_inference_features(prices)
    if len(features) < seq_len:
        raise ValueError(
            f"Need {seq_len} complete feature rows, found {len(features)}"
        )
    scaled = scaler.transform(features[FEATURE_COLS])
    window = np.asarray(scaled[-seq_len:], dtype=np.float32)
    return np.ascontiguousarray(window[None, :, :])
