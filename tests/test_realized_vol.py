import numpy as np
import pandas as pd

from src.features.realized_vol import (
    ANNUALIZED_FACTOR,
    log_returns,
    realized_vol,
    realized_vol_target,
)


def test_log_returns_formula():
    prices = pd.Series([1.0, 2.0, 4.0, 8.0])
    result = log_returns(prices)
    assert pd.isna(result.iloc[0])
    assert np.allclose(result.iloc[1:], np.log(2))


def test_log_returns_preserves_length():
    prices = pd.Series([10.0, 11.0, 12.0])
    assert len(log_returns(prices)) == len(prices)


def test_realized_vol_first_rows_are_nan():
    returns = pd.Series([0.01, -0.02, 0.015, 0.0, -0.01])
    vol = realized_vol(returns, window=3, annualize=False)
    assert pd.isna(vol.iloc[0])
    assert pd.isna(vol.iloc[1])
    assert not pd.isna(vol.iloc[2:]).any()


def test_realized_vol_annualize_scales_by_factor():
    returns = pd.Series([0.01, -0.02, 0.015, 0.0, -0.01, 0.02])
    raw = realized_vol(returns, window=3, annualize=False)
    ann = realized_vol(returns, window=3, annualize=True)
    assert np.allclose(raw.dropna() * ANNUALIZED_FACTOR, ann.dropna())


def test_target_matches_forward_window():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, size=20))
    h = 5
    target = realized_vol_target(returns, horizon=h, annualize=True)
    t = 10
    forward_window = returns.iloc[t + 1 : t + 1 + h]
    manual = forward_window.std() * ANNUALIZED_FACTOR
    assert np.isclose(target.iloc[t], manual)


def test_target_does_not_leak_day_t():
    """target[t] must depend only on days t+1..t+h, never day t itself."""
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, size=20))
    h = 5
    t = 8
    target = realized_vol_target(returns, horizon=h, annualize=True)
    corrupted = returns.copy()
    corrupted.iloc[t] = 999.0  # poison ONLY day t
    target_corrupted = realized_vol_target(corrupted, horizon=h, annualize=True)
    assert np.isclose(target.iloc[t], target_corrupted.iloc[t])


def test_target_tail_is_nan():
    returns = pd.Series(np.arange(10, dtype=float))
    h = 5
    target = realized_vol_target(returns, horizon=h, annualize=True)
    assert target.iloc[-h:].isna().all()