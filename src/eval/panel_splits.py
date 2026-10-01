"""Calendar-wide, purged evaluation splits for the global ticker panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class PanelSplit:
    """One train/validation/test partition shared by every target ticker."""

    name: str
    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    def to_dict(self) -> dict[str, Any]:
        """Return stable boundary metadata suitable for MLflow artifacts."""
        return {
            "name": self.name,
            "train": _date_summary(self.train_dates),
            "validation": _date_summary(self.validation_dates),
            "test": _date_summary(self.test_dates),
        }


@dataclass(frozen=True)
class PanelHoldout:
    """Final refit/test partition used after cross-validation is complete."""

    fit_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "final_holdout",
            "fit": _date_summary(self.fit_dates),
            "test": _date_summary(self.test_dates),
        }


@dataclass(frozen=True)
class PanelSplitPlan:
    """Cross-validation folds plus the untouched final holdout split."""

    folds: tuple[PanelSplit, ...]
    holdout: PanelHoldout
    horizon: int
    min_train_size: int
    val_frac: float
    holdout_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "horizon": self.horizon,
            "min_train_size": self.min_train_size,
            "val_frac": self.val_frac,
            "holdout_size": self.holdout_size,
            "folds": [fold.to_dict() for fold in self.folds],
            "holdout": self.holdout.to_dict(),
        }


def _date_summary(dates: pd.DatetimeIndex) -> dict[str, Any]:
    return {
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "count": len(dates),
    }


def _balanced_panel_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Panel must use a MultiIndex")
    if panel.index.names != ["date", "ticker"]:
        raise ValueError("Panel index must be named ['date', 'ticker']")
    if panel.index.has_duplicates:
        raise ValueError("Panel contains duplicate date/ticker rows")

    tickers = panel.index.get_level_values("ticker").unique()
    if len(tickers) == 0:
        raise ValueError("Panel contains no tickers")
    date_sets = [
        panel.xs(ticker, level="ticker").index.unique().sort_values()
        for ticker in tickers
    ]
    if any(not dates.equals(date_sets[0]) for dates in date_sets[1:]):
        raise ValueError("Panel must contain identical dates for every ticker")
    return pd.DatetimeIndex(date_sets[0], name="date")


def _inner_train_validation_split(
    outer_train_dates: pd.DatetimeIndex,
    *,
    val_frac: float,
    horizon: int,
    min_train_size: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    n_validation = max(1, int(len(outer_train_dates) * val_frac))
    validation_start = len(outer_train_dates) - n_validation
    train_end = validation_start - horizon
    if train_end < min_train_size:
        raise ValueError(
            "Insufficient dates for the requested training minimum, validation "
            "fraction, and embargo: "
            f"train={train_end}, minimum={min_train_size}"
        )
    train_dates = outer_train_dates[:train_end]
    validation_dates = outer_train_dates[validation_start:]
    return train_dates, validation_dates


def _validate_split(
    all_dates: pd.DatetimeIndex,
    split: PanelSplit,
    horizon: int,
) -> None:
    partitions = [split.train_dates, split.validation_dates, split.test_dates]
    if any(partition.empty for partition in partitions):
        raise ValueError(f"{split.name} contains an empty partition")
    if not (
        split.train_dates.max()
        < split.validation_dates.min()
        < split.test_dates.min()
    ):
        raise ValueError(f"{split.name} partitions are not chronological")

    train_validation_gap = all_dates[
        (all_dates > split.train_dates.max())
        & (all_dates < split.validation_dates.min())
    ]
    validation_test_gap = all_dates[
        (all_dates > split.validation_dates.max())
        & (all_dates < split.test_dates.min())
    ]
    if len(train_validation_gap) < horizon:
        raise ValueError(f"{split.name} train/validation embargo is too short")
    if len(validation_test_gap) < horizon:
        raise ValueError(f"{split.name} validation/test embargo is too short")

    if any(
        not partition.isin(all_dates).all()
        for partition in partitions
    ):
        raise ValueError(f"{split.name} contains dates outside the panel")


def _validate_holdout(
    all_dates: pd.DatetimeIndex,
    holdout: PanelHoldout,
    horizon: int,
) -> None:
    if holdout.fit_dates.empty or holdout.test_dates.empty:
        raise ValueError("Final holdout contains an empty partition")
    if holdout.fit_dates.max() >= holdout.test_dates.min():
        raise ValueError("Final holdout partitions are not chronological")
    gap = all_dates[
        (all_dates > holdout.fit_dates.max())
        & (all_dates < holdout.test_dates.min())
    ]
    if len(gap) < horizon:
        raise ValueError("Final holdout embargo is too short")
    if not holdout.fit_dates.isin(all_dates).all():
        raise ValueError("Final holdout fit dates are outside the panel")
    if not holdout.test_dates.isin(all_dates).all():
        raise ValueError("Final holdout test dates are outside the panel")


def build_panel_split_plan(
    panel: pd.DataFrame,
    *,
    n_splits: int,
    horizon: int,
    min_train_size: int,
    val_frac: float,
    holdout_size: int,
) -> PanelSplitPlan:
    """Build expanding folds and a final holdout with horizon-sized embargoes.

    Splits operate on unique calendar dates, never flattened panel rows. A date
    therefore belongs to the same partition for every ticker, preventing one
    asset's future from entering another asset's training set.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if horizon <= 0 or min_train_size <= 0 or holdout_size <= 0:
        raise ValueError("horizon, min_train_size, and holdout_size must be positive")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")

    all_dates = _balanced_panel_dates(panel)
    development_end = len(all_dates) - holdout_size - horizon
    if development_end <= 0:
        raise ValueError("Panel is too short for the holdout and its embargo")
    development_dates = all_dates[:development_end]
    holdout_dates = all_dates[-holdout_size:]

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    folds: list[PanelSplit] = []
    try:
        raw_splits = splitter.split(development_dates)
        for fold_number, (outer_train_index, test_index) in enumerate(
            raw_splits,
            start=1,
        ):
            outer_train_dates = development_dates[outer_train_index]
            train_dates, validation_dates = _inner_train_validation_split(
                outer_train_dates,
                val_frac=val_frac,
                horizon=horizon,
                min_train_size=min_train_size,
            )
            split = PanelSplit(
                name=f"fold_{fold_number}",
                train_dates=train_dates,
                validation_dates=validation_dates,
                test_dates=development_dates[test_index],
            )
            _validate_split(all_dates, split, horizon)
            folds.append(split)
    except ValueError as exc:
        raise ValueError(f"Could not build global walk-forward folds: {exc}") from exc

    # Hyperparameters and the fixed epoch count are selected from the CV folds.
    # The final evaluation model then refits on every pre-holdout date rather
    # than discarding the most recent years into another validation slice.
    holdout = PanelHoldout(
        fit_dates=development_dates,
        test_dates=holdout_dates,
    )
    _validate_holdout(all_dates, holdout, horizon)
    return PanelSplitPlan(
        tuple(folds),
        holdout,
        horizon,
        min_train_size,
        val_frac,
        holdout_size,
    )
