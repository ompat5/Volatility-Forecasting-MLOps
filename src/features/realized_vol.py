import numpy as np
import pandas as pd

ANNUALIZED_FACTOR = np.sqrt(252)


def log_returns(prices: pd.Series) -> pd.Series:
    """Calculate the log returns of a ticker's price series."""
    return np.log(prices / prices.shift(1))


def realized_vol(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Calculate the realized volatility of a ticker's return series."""
    vol = returns.rolling(window).std()
    if annualize:
        vol *= ANNUALIZED_FACTOR
    return vol


def realized_vol_target(returns: pd.Series, horizon: int, annualize: bool = True) -> pd.Series:
    """Calculate the realized volatility target of a ticker's return series."""
    return realized_vol(returns, horizon, annualize).shift(-horizon)


def build_features(prices: pd.Series, horizon: int = 5) -> pd.DataFrame:
    """Build features for volatility forecasting"""
    returns = log_returns(prices)

    realized_volatility_5d = realized_vol(returns, window=5)
    realized_volatility_20d = realized_vol(returns, window=20)
    realized_volatility_60d = realized_vol(returns, window=60)
    realized_volatility_target = realized_vol_target(returns, horizon)

    return pd.DataFrame({
        "log_returns": returns,
        "rv_5d": realized_volatility_5d,
        "rv_20d": realized_volatility_20d,
        "rv_60d": realized_volatility_60d,
        "rv_target": realized_volatility_target
    }).dropna()