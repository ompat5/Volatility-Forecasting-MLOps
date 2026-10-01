"""Same-date classical baselines for the global panel evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from src.features.realized_vol import ANNUALIZED_FACTOR


def rolling_garch_forecast(
    returns: pd.Series,
    train_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    *,
    horizon: int,
    annualize: bool = True,
) -> pd.Series:
    """Fit parameters on training only, then update GARCH state as time advances.

    Parameters remain fixed throughout the fold, but every forecast incorporates
    returns observed through its own as-of date. This is both leakage-safe and a
    stronger comparator than repeating one fold-start forecast over the test.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    train_dates = pd.DatetimeIndex(train_dates).sort_values()
    forecast_dates = pd.DatetimeIndex(forecast_dates).sort_values()
    if train_dates.empty or forecast_dates.empty:
        raise ValueError("Training and forecast dates must not be empty")
    if train_dates.max() >= forecast_dates.min():
        raise ValueError("GARCH training dates must precede forecast dates")
    if not train_dates.isin(returns.index).all():
        raise ValueError("GARCH training returns are incomplete")
    if not forecast_dates.isin(returns.index).all():
        raise ValueError("GARCH forecast returns are incomplete")

    train_returns = returns.loc[train_dates].dropna().astype(float) * 100.0
    if len(train_returns) != len(train_dates):
        raise ValueError("GARCH training returns contain missing values")
    fitted = arch_model(
        train_returns,
        mean="Constant",
        vol="Garch",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    ).fit(disp="off")
    if fitted.convergence_flag != 0:
        raise RuntimeError(
            f"GARCH optimization did not converge: flag={fitted.convergence_flag}"
        )

    parameters = fitted.params
    omega = float(parameters["omega"])
    alpha = float(parameters["alpha[1]"])
    beta = float(parameters["beta[1]"])
    mean = float(parameters.get("mu", 0.0))
    previous_variance = float(fitted.conditional_volatility.iloc[-1] ** 2)
    previous_residual = float(train_returns.iloc[-1] - mean)

    subsequent = returns.loc[
        (returns.index > train_dates.max()) & (returns.index <= forecast_dates.max())
    ].dropna()
    forecast_set = set(forecast_dates)
    predictions: dict[pd.Timestamp, float] = {}
    persistence = alpha + beta
    for date, raw_return in subsequent.items():
        current_variance = (
            omega + alpha * previous_residual**2 + beta * previous_variance
        )
        current_residual = float(raw_return * 100.0 - mean)
        if date in forecast_set:
            future_variance = (
                omega + alpha * current_residual**2 + beta * current_variance
            )
            variance_path = [future_variance]
            for _ in range(1, horizon):
                future_variance = omega + persistence * future_variance
                variance_path.append(future_variance)
            mean_variance = float(np.mean(variance_path)) / (100.0**2)
            forecast = np.sqrt(mean_variance)
            if annualize:
                forecast *= ANNUALIZED_FACTOR
            predictions[date] = float(forecast)
        previous_variance = current_variance
        previous_residual = current_residual

    missing = forecast_dates.difference(pd.DatetimeIndex(predictions))
    if not missing.empty:
        raise ValueError(f"Could not produce GARCH forecasts for {missing.tolist()}")
    result = pd.Series(predictions, dtype=float).reindex(forecast_dates)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("GARCH produced invalid volatility forecasts")
    result.name = "forecast"
    return result
