"""Build the balanced, leakage-safe panel used by the global model."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.ingest import DEFAULT_RAW_DIR, REPO_ROOT, ticker_cache_path
from src.data.universe import Universe
from src.features.realized_vol import (
    FEATURE_COLS,
    TARGET_COL,
    build_features,
    log_returns,
)

VIX_LEVEL_COL = "vix_level"
VIX_RETURN_COL = "vix_log_return"
CONTEXT_FEATURE_COLS = [VIX_LEVEL_COL, VIX_RETURN_COL]
GLOBAL_FEATURE_COLS = [*FEATURE_COLS, *CONTEXT_FEATURE_COLS]
PANEL_COLUMNS = [*GLOBAL_FEATURE_COLS, TARGET_COL]


def normalize_session_index(index: pd.Index) -> pd.DatetimeIndex:
    """Convert exchange-local timestamps to timezone-free trading dates.

    yfinance stores equities at New York midnight and VIX at Chicago midnight.
    Converting those instants between timezones changes their clock time and makes
    direct joins fail. The financial key is the local session date, so preserve
    each timestamp's displayed date and deliberately discard its timezone.
    """
    timestamps = pd.DatetimeIndex(index)
    if timestamps.hasnans:
        raise ValueError("Price index contains missing timestamps")
    sessions = pd.DatetimeIndex(
        [timestamp.date() for timestamp in timestamps],
        name="date",
    )
    if sessions.has_duplicates:
        duplicates = sessions[sessions.duplicated()].unique()
        raise ValueError(
            "Multiple observations map to the same trading session: "
            f"{[date.date().isoformat() for date in duplicates]}"
        )
    return sessions


def load_adjusted_close(symbol: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.Series:
    """Load one validated adjusted-close series keyed by trading session."""
    path = ticker_cache_path(symbol, raw_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw data for {symbol}: {path}")

    frame = pd.read_parquet(path)
    if "Adj Close" not in frame:
        raise ValueError(f"{path} does not contain an 'Adj Close' column")

    prices = pd.to_numeric(frame["Adj Close"], errors="coerce").dropna().astype(float)
    prices.index = normalize_session_index(prices.index)
    prices = prices.sort_index()
    if prices.empty:
        raise ValueError(f"Adjusted-close history is empty for {symbol}")
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError(f"Adjusted-close history contains non-finite values for {symbol}")
    if (prices <= 0).any():
        raise ValueError(f"Adjusted-close history contains non-positive values for {symbol}")
    prices.name = symbol
    return prices


def _vix_context(prices: pd.Series) -> pd.DataFrame:
    """Create close-of-session VIX inputs known at the prediction date."""
    return pd.DataFrame(
        {
            # VIX is quoted in percentage points; divide by 100 so it is on the
            # same decimal-volatility scale as the realized-volatility features.
            VIX_LEVEL_COL: prices / 100.0,
            VIX_RETURN_COL: log_returns(prices),
        }
    ).dropna()


def build_global_panel(
    universe: Universe,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    horizon: int = 5,
) -> pd.DataFrame:
    """Build one balanced panel for every target using same-session VIX context."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    vix_symbol = universe.context_symbol("implied_volatility")
    context = _vix_context(load_adjusted_close(vix_symbol, raw_dir))

    by_ticker: dict[str, pd.DataFrame] = {}
    for symbol in universe.target_symbols:
        prices = load_adjusted_close(symbol, raw_dir)
        target_features = build_features(prices, horizon)
        # Inner join only: never fill an unavailable VIX observation across a
        # market session, and never use a later observation for an earlier date.
        by_ticker[symbol] = target_features.join(context, how="inner")[PANEL_COLUMNS]

    common_dates = next(iter(by_ticker.values())).index
    for frame in by_ticker.values():
        common_dates = common_dates.intersection(frame.index, sort=False)
    common_dates = common_dates.sort_values()
    if common_dates.empty:
        raise ValueError("Targets and context have no common complete trading sessions")

    balanced = {
        symbol: frame.loc[common_dates, PANEL_COLUMNS]
        for symbol, frame in by_ticker.items()
    }
    panel = pd.concat(balanced, names=["ticker", "date"])
    panel = panel.swaplevel("ticker", "date").sort_index()
    validate_global_panel(panel, universe)
    return panel


def validate_global_panel(panel: pd.DataFrame, universe: Universe) -> None:
    """Fail fast when the panel violates the global model's data contract."""
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Global panel must use a MultiIndex")
    if panel.index.names != ["date", "ticker"]:
        raise ValueError("Global panel index must be named ['date', 'ticker']")
    if list(panel.columns) != PANEL_COLUMNS:
        raise ValueError(f"Global panel columns must be {PANEL_COLUMNS}")
    if panel.empty:
        raise ValueError("Global panel cannot be empty")
    if panel.index.has_duplicates:
        raise ValueError("Global panel contains duplicate date/ticker rows")
    if panel.isna().any().any():
        raise ValueError("Global panel contains missing values")
    if not np.isfinite(panel.to_numpy(dtype=float)).all():
        raise ValueError("Global panel contains non-finite values")

    actual_tickers = set(panel.index.get_level_values("ticker"))
    expected_tickers = set(universe.target_symbols)
    if actual_tickers != expected_tickers:
        raise ValueError(
            "Global panel ticker coverage differs from the universe: "
            f"expected={sorted(expected_tickers)}, actual={sorted(actual_tickers)}"
        )

    counts = panel.groupby(level="ticker", sort=False).size()
    if counts.nunique() != 1:
        raise ValueError("Global panel must contain the same dates for every ticker")
    date_sets = [
        tuple(panel.xs(symbol, level="ticker").index)
        for symbol in universe.target_symbols
    ]
    if any(dates != date_sets[0] for dates in date_sets[1:]):
        raise ValueError("Global panel target date sets are not identical")
    if (panel[VIX_LEVEL_COL] <= 0).any():
        raise ValueError("VIX level must be positive")
    if (panel[TARGET_COL] <= 0).any():
        raise ValueError("Realized-volatility target must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_panel_manifest(
    panel: pd.DataFrame,
    universe: Universe,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    horizon: int = 5,
) -> dict[str, Any]:
    """Describe the exact inputs and balanced panel used for model training."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    validate_global_panel(panel, universe)
    dates = panel.index.get_level_values("date").unique().sort_values()
    rows_per_ticker = panel.groupby(level="ticker", sort=False).size()

    sources: dict[str, dict[str, Any]] = {}
    for symbol in universe.all_symbols:
        path = ticker_cache_path(symbol, raw_dir)
        prices = load_adjusted_close(symbol, raw_dir)
        try:
            portable_path = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            portable_path = path.resolve()
        sources[symbol] = {
            "path": str(portable_path),
            "sha256": _sha256(path),
            "rows": len(prices),
            "start": prices.index.min().date().isoformat(),
            "end": prices.index.max().date().isoformat(),
        }

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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
        "panel": {
            "start": dates.min().date().isoformat(),
            "end": dates.max().date().isoformat(),
            "dates": len(dates),
            "rows": len(panel),
            "horizon": horizon,
            "balanced": True,
            "index": ["date", "ticker"],
            "features": GLOBAL_FEATURE_COLS,
            "target": TARGET_COL,
            "rows_per_ticker": {
                symbol: int(rows_per_ticker.loc[symbol])
                for symbol in universe.target_symbols
            },
            "columns": PANEL_COLUMNS,
        },
        "sources": sources,
    }
