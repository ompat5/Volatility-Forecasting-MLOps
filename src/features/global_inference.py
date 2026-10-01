"""Raw-price inference preparation for the global volatility model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.panel import GLOBAL_FEATURE_COLS, VIX_LEVEL_COL, VIX_RETURN_COL
from src.features.realized_vol import build_inference_features, log_returns

DATE_COL = "date"
TICKER_COL = "ticker"
PRICE_COL = "adjusted_close"
GLOBAL_INFERENCE_COLUMNS = (DATE_COL, TICKER_COL, PRICE_COL)


@dataclass(frozen=True)
class GlobalInferenceBatch:
    """One latest feature window for every requested target."""

    features: np.ndarray
    ticker_ids: np.ndarray
    tickers: tuple[str, ...]
    as_of_dates: tuple[pd.Timestamp, ...]


def _validate_raw_prices(
    model_input: pd.DataFrame,
    *,
    target_symbols: tuple[str, ...],
    context_symbol: str,
) -> pd.DataFrame:
    if not isinstance(model_input, pd.DataFrame):
        raise ValueError("Global inference input must be a pandas DataFrame")
    if len(model_input.columns) != len(GLOBAL_INFERENCE_COLUMNS) or set(
        model_input.columns
    ) != set(GLOBAL_INFERENCE_COLUMNS):
        raise ValueError(
            "Global inference input columns must be exactly "
            f"{list(GLOBAL_INFERENCE_COLUMNS)}"
        )
    if model_input.empty:
        raise ValueError("Global inference input cannot be empty")

    frame = model_input.loc[:, list(GLOBAL_INFERENCE_COLUMNS)].copy()
    parsed_dates = pd.to_datetime(frame[DATE_COL], errors="raise")
    timestamps = pd.DatetimeIndex(parsed_dates)
    if timestamps.hasnans:
        raise ValueError("Inference dates cannot be missing")
    frame[DATE_COL] = pd.DatetimeIndex(
        [timestamp.date() for timestamp in timestamps],
        name=DATE_COL,
    )
    frame[TICKER_COL] = frame[TICKER_COL].astype(str).str.strip()
    frame[PRICE_COL] = pd.to_numeric(frame[PRICE_COL], errors="coerce")
    if (frame[TICKER_COL] == "").any():
        raise ValueError("Ticker values must be non-empty")
    if frame.duplicated([DATE_COL, TICKER_COL]).any():
        raise ValueError("Global inference input contains duplicate date/ticker rows")
    if not np.isfinite(frame[PRICE_COL].to_numpy(dtype=float)).all():
        raise ValueError("Adjusted-close values must be finite")
    if (frame[PRICE_COL] <= 0).any():
        raise ValueError("Adjusted-close values must be positive")

    supported = set(target_symbols) | {context_symbol}
    supplied = set(frame[TICKER_COL])
    unknown = supplied - supported
    if unknown:
        raise ValueError(f"Unsupported inference tickers: {sorted(unknown)}")
    if context_symbol not in supplied:
        raise ValueError(
            f"Inference input must include context ticker {context_symbol}"
        )
    if not supplied.intersection(target_symbols):
        raise ValueError("Inference input must include at least one forecast target")
    return frame.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)


def build_global_inference_batch(
    model_input: pd.DataFrame,
    *,
    target_symbols: tuple[str, ...],
    context_symbol: str,
    ticker_to_id: dict[str, int],
    seq_len: int,
) -> GlobalInferenceBatch:
    """Build ticker-isolated latest windows from long-form adjusted closes."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if list(ticker_to_id.values()) != list(range(len(ticker_to_id))):
        raise ValueError("Ticker IDs must be contiguous and start at zero")
    if tuple(sorted(ticker_to_id, key=ticker_to_id.get)) != target_symbols:
        raise ValueError("Ticker vocabulary and target order do not match")

    frame = _validate_raw_prices(
        model_input,
        target_symbols=target_symbols,
        context_symbol=context_symbol,
    )
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

    requested = set(frame[TICKER_COL]).intersection(target_symbols)
    ordered_targets = tuple(ticker for ticker in target_symbols if ticker in requested)
    windows: list[np.ndarray] = []
    as_of_dates: list[pd.Timestamp] = []
    for ticker in ordered_targets:
        prices = (
            frame.loc[frame[TICKER_COL] == ticker]
            .set_index(DATE_COL)[PRICE_COL]
            .sort_index()
        )
        features = build_inference_features(prices).join(context, how="inner")
        features = features.loc[:, GLOBAL_FEATURE_COLS].dropna()
        if len(features) < seq_len:
            raise ValueError(
                f"Ticker {ticker} has {len(features)} complete feature rows; "
                f"at least {seq_len} are required"
            )
        window = features.iloc[-seq_len:].to_numpy(dtype=np.float32)
        if not np.isfinite(window).all():
            raise ValueError(f"Ticker {ticker} produced non-finite inference features")
        windows.append(window)
        as_of_dates.append(pd.Timestamp(features.index[-1]))

    return GlobalInferenceBatch(
        features=np.stack(windows),
        ticker_ids=np.asarray(
            [ticker_to_id[ticker] for ticker in ordered_targets],
            dtype=np.int64,
        ),
        tickers=ordered_targets,
        as_of_dates=tuple(as_of_dates),
    )
