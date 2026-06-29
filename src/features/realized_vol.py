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