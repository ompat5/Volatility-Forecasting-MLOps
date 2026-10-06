"""Shared long-form raw-price loading for global monitoring and dashboards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.ingest import DEFAULT_RAW_DIR, fetch_ticker
from src.data.panel import load_adjusted_close, normalize_session_index
from src.data.universe import Universe
from src.features.global_inference import DATE_COL, PRICE_COL, TICKER_COL


def long_prices_from_series(
    prices: dict[str, pd.Series],
    universe: Universe,
) -> pd.DataFrame:
    """Combine exact universe histories into the serving long-form contract."""
    if set(prices) != set(universe.all_symbols):
        raise ValueError("Price series must cover the complete global universe")
    frames = [
        pd.DataFrame(
            {
                DATE_COL: prices[symbol].index,
                TICKER_COL: symbol,
                PRICE_COL: prices[symbol].to_numpy(dtype=float),
            }
        )
        for symbol in universe.all_symbols
    ]
    frame = pd.concat(frames, ignore_index=True)
    ticker_order = {
        symbol: index for index, symbol in enumerate(universe.all_symbols)
    }
    frame["_ticker_order"] = frame[TICKER_COL].map(ticker_order)
    return (
        frame.sort_values([DATE_COL, "_ticker_order"])
        .drop(columns="_ticker_order")
        .reset_index(drop=True)
    )


def load_cached_global_prices(
    universe: Universe,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> pd.DataFrame:
    """Load every configured series from the local ingestion cache."""
    return long_prices_from_series(
        {
            symbol: load_adjusted_close(symbol, raw_dir)
            for symbol in universe.all_symbols
        },
        universe,
    )


def fetch_global_prices(
    universe: Universe,
    *,
    attempts: int = 2,
) -> pd.DataFrame:
    """Refresh every series, retrying transient empties before failing closed."""
    if attempts <= 0:
        raise ValueError("Global price refresh attempts must be positive")
    prices: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}
    for symbol in universe.all_symbols:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                frame = fetch_ticker(symbol)
                series = pd.to_numeric(
                    frame["Adj Close"], errors="coerce"
                ).dropna()
                if series.empty:
                    raise ValueError(
                        f"Price refresh returned no adjusted closes for {symbol}"
                    )
                series.index = normalize_session_index(series.index)
                prices[symbol] = series.sort_index().astype(float)
                break
            except Exception as exc:  # noqa: BLE001 - retry/report every symbol
                last_error = exc
        if symbol not in prices:
            failures[symbol] = f"after {attempts} attempts: {last_error}"
    if failures:
        raise RuntimeError(f"Failed to refresh global price inputs: {failures}")
    return long_prices_from_series(prices, universe)
