"""Ticker-isolated sequence Dataset for the global volatility model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PanelVolatilityDataset(Dataset):
    """Build sequences within tickers while selecting anchors by calendar date.

    The full panel is retained as feature context. Only `anchor_dates` determine
    which target rows become samples, so validation and test sequences may use
    legally available earlier features without admitting earlier target labels.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
        anchor_dates: pd.DatetimeIndex,
        ticker_to_id: dict[str, int],
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError("Panel must use a MultiIndex")
        if panel.index.names != ["date", "ticker"]:
            raise ValueError("Panel index must be named ['date', 'ticker']")
        required = set(feature_cols) | {target_col}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(f"Panel is missing columns: {sorted(missing)}")
        if not ticker_to_id:
            raise ValueError("ticker_to_id cannot be empty")
        if set(ticker_to_id.values()) != set(range(len(ticker_to_id))):
            raise ValueError("ticker IDs must be contiguous and start at zero")

        panel_tickers = set(panel.index.get_level_values("ticker"))
        if set(ticker_to_id) != panel_tickers:
            raise ValueError(
                "ticker_to_id must contain every panel ticker exactly once"
            )

        panel_dates = (
            panel.index.get_level_values("date").unique().sort_values()
        )
        anchors = pd.DatetimeIndex(anchor_dates).unique().sort_values()
        unknown_dates = anchors[~anchors.isin(panel_dates)]
        if not unknown_dates.empty:
            raise ValueError(
                f"Anchor dates are absent from the panel: {unknown_dates.tolist()}"
            )

        self.seq_len = seq_len
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.ticker_to_id = dict(ticker_to_id)
        self._features: dict[str, np.ndarray] = {}
        self._targets: dict[str, np.ndarray] = {}
        self._samples: list[tuple[str, int]] = []

        eligible_anchors = anchors[anchors.isin(panel_dates[seq_len - 1:])]
        self.dropped_anchor_dates = anchors.difference(eligible_anchors)

        # Sample order is calendar-first and then vocabulary order. This keeps
        # predictions easy to align and makes coverage deterministic.
        position_by_date = {
            date: position for position, date in enumerate(panel_dates)
        }
        ordered_tickers = sorted(ticker_to_id, key=ticker_to_id.get)
        for ticker in ordered_tickers:
            frame = panel.xs(ticker, level="ticker").sort_index()
            if not frame.index.equals(panel_dates):
                raise ValueError(
                    f"Ticker {ticker} does not share the balanced panel calendar"
                )
            self._features[ticker] = frame[feature_cols].to_numpy(dtype=np.float32)
            self._targets[ticker] = frame[target_col].to_numpy(dtype=np.float32)

        sample_index: list[tuple[pd.Timestamp, str]] = []
        for date in eligible_anchors:
            for ticker in ordered_tickers:
                position = position_by_date[date]
                self._samples.append((ticker, position))
                sample_index.append((date, ticker))
        self.sample_index = pd.MultiIndex.from_tuples(
            sample_index,
            names=["date", "ticker"],
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ticker, position = self._samples[index]
        start = position - self.seq_len + 1
        features = self._features[ticker][start : position + 1]
        target = self._targets[ticker][position]
        ticker_id = self.ticker_to_id[ticker]
        return (
            torch.from_numpy(features.copy()),
            torch.tensor(ticker_id, dtype=torch.long),
            torch.tensor(float(target), dtype=torch.float32),
        )
