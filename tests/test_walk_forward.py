import pandas as pd
import pytest

from src.eval.walk_forward import _validate_splits, walk_forward_splits


INDEX_50 = pd.bdate_range("2020-01-01", periods=50)


def test_returns_correct_number_of_splits():
    result = walk_forward_splits(INDEX_50, n_splits=3, horizon=5, min_train_size=1)
    assert len(result) == 3


def test_train_always_precedes_test():
    result = walk_forward_splits(INDEX_50, n_splits=3, horizon=5, min_train_size=1)
    for train, test in result:
        assert train[-1] < test[0]


def test_gap_between_train_and_test():
    horizon = 5
    result = walk_forward_splits(INDEX_50, n_splits=3, horizon=horizon, min_train_size=1)
    for train, test in result:
        gap_rows = INDEX_50[(INDEX_50 > train[-1]) & (INDEX_50 < test[0])]
        assert len(gap_rows) >= horizon


def test_min_train_size_filters_small_folds():
    small_index = pd.bdate_range("2020-01-01", periods=20)
    result = walk_forward_splits(small_index, n_splits=5, horizon=2, min_train_size=10)
    assert len(result) < 5


def test_train_set_grows_each_fold():
    result = walk_forward_splits(INDEX_50, n_splits=3, horizon=5, min_train_size=1)
    train_sizes = [len(train) for train, _ in result]
    assert all(train_sizes[i] < train_sizes[i + 1] for i in range(len(train_sizes) - 1))


def test_validate_passes_on_valid_splits():
    splits = walk_forward_splits(INDEX_50, n_splits=3, horizon=5, min_train_size=1)
    _validate_splits(INDEX_50, splits, horizon=5)


def test_validate_raises_on_overlapping_dates():
    dates = pd.bdate_range("2020-01-01", periods=20)
    bad_splits = [(dates[10:15], dates[12:18])]
    with pytest.raises(ValueError, match="test starts at"):
        _validate_splits(dates, bad_splits, horizon=2)


def test_validate_raises_on_insufficient_gap():
    dates = pd.bdate_range("2020-01-01", periods=20)
    bad_splits = [(dates[:10], dates[11:15])]
    with pytest.raises(ValueError, match="gap is"):
        _validate_splits(dates, bad_splits, horizon=5)
