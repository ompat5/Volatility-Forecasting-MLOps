import numpy as np
import pandas as pd

import src.models.global_baselines as global_baselines
from src.models.global_baselines import rolling_garch_forecast


def test_rolling_garch_is_positive_and_updates_across_as_of_dates(monkeypatch):
    class FakeResult:
        convergence_flag = 0
        params = pd.Series({"mu": 0.0, "omega": 0.01, "alpha[1]": 0.2, "beta[1]": 0.7})
        conditional_volatility = pd.Series([1.0])

    class FakeModel:
        def fit(self, *, disp):
            assert disp == "off"
            return FakeResult()

    monkeypatch.setattr(
        global_baselines, "arch_model", lambda *args, **kwargs: FakeModel()
    )

    dates = pd.bdate_range("2020-01-02", periods=12)
    returns = pd.Series(
        [0.01, -0.01, 0.01, -0.01, 0.01, 0.005, 0.03, -0.01, 0.04, 0.0, -0.02, 0.01],
        index=dates,
    )
    train_dates = dates[:5]
    forecast_dates = dates[6:11]

    forecast = rolling_garch_forecast(
        returns,
        train_dates,
        forecast_dates,
        horizon=5,
    )

    assert forecast.index.equals(forecast_dates)
    assert np.isfinite(forecast).all()
    assert (forecast > 0).all()
    assert forecast.nunique() > 1
