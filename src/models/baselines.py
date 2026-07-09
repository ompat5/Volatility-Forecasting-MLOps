import pandas as pd
import numpy as np

from arch import arch_model
from src.features.realized_vol import realized_vol, ANNUALIZED_FACTOR


def naive_forecast(returns: pd.Series, horizon: int, annualize: bool = True) -> pd.Series:
    """Calculate the naive forecast of a ticker's return series."""
    return realized_vol(returns, horizon, annualize)


def ewma_forecast(returns: pd.Series, span: float, annualize: bool = True) -> pd.Series:
    """Calculate the EWMA forecast of a ticker's return series."""
    ewma_vol = returns.ewm(span=span).std()
    if annualize:
        ewma_vol *= ANNUALIZED_FACTOR
    return ewma_vol


def garch_forecast(train_returns: pd.Series, test_index: pd.DatetimeIndex, horizon: int, annualize: bool = True) -> pd.Series:
    """Calculate the GARCH forecast of a ticker's return series."""
    model = arch_model(train_returns * 100, vol='Garch', p=1, q=1, dist='normal')
    result = model.fit(disp='off')

    forecasts = result.forecast(horizon=horizon, reindex=False)

    h_steps_variances = forecasts.variance.values[0]
    mean_var = h_steps_variances.mean() / (100 ** 2)
    vol = np.sqrt(mean_var)
    if annualize:
        vol *= ANNUALIZED_FACTOR
    return pd.Series(vol, index=test_index)