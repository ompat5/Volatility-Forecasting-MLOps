import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int, horizon: int, min_train_size: int) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate walk-forward splits for time series cross-validation."""
    tss = TimeSeriesSplit(n_splits=n_splits, gap=horizon)

    return [
        (index[train_index], index[test_index]) 
        for train_index, test_index in tss.split(index) 
        if len(train_index) >= min_train_size
    ]


def validate_splits(index: pd.DatetimeIndex, splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]], horizon: int) -> None:
    """Validate the generated walk-forward splits."""
    for train_dates, test_dates in splits:
        if test_dates[0] <= train_dates[-1]:
            raise ValueError(f"test starts at {test_dates[0]} but train ends at {train_dates[-1]}")

        gap_rows = index[(index > train_dates[-1]) & (index < test_dates[0])]
        if len(gap_rows) < horizon:
            raise ValueError(f"gap is {len(gap_rows)} rows, need {horizon}.")