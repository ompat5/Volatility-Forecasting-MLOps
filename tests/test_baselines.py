import numpy as np
import pandas as pd

from src.models.baselines import ewma_forecast, garch_forecast, naive_forecast
from src.features.realized_vol import realized_vol


def test_naive_forecast_matches_realized_vol():
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    horizon = 10
    annualize = True

    naive_result = naive_forecast(returns, horizon, annualize)
    realized_result = realized_vol(returns, horizon, annualize)

    assert np.isclose(naive_result, realized_result, equal_nan=True).all()


def test_naive_forecast_length_matches_input():
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    result = naive_forecast(returns, horizon=5)
    assert len(result) == len(returns)


def test_naive_forecast_first_rows_are_nan():
    returns = pd.Series(np.random.normal(0, 0.01, 50))
    horizon = 10
    result = naive_forecast(returns, horizon=horizon)
    assert result.iloc[:horizon - 1].isna().all()
    assert not result.iloc[horizon - 1:].isna().all()


def test_ewma_forecast_length_matches_input():
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    result = ewma_forecast(returns, span=20)
    assert len(result) == len(returns)


def test_ewma_forecast_only_first_row_is_nan():
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    result = ewma_forecast(returns, span=20)
    assert pd.isna(result.iloc[0])
    assert not result.iloc[1:].isna().any()


def test_ewma_forecast_is_positive():
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    result = ewma_forecast(returns, span=20)
    assert result.iloc[1:].gt(0).all()


TRAIN_RETURNS = pd.Series(
    np.random.normal(0, 0.01, 500),
    index=pd.bdate_range("2018-01-01", periods=500),
)
TEST_INDEX = pd.bdate_range("2020-01-01", periods=10)


def test_garch_forecast_length_matches_test_index():
    result = garch_forecast(TRAIN_RETURNS, TEST_INDEX, horizon=5)
    assert len(result) == len(TEST_INDEX)


def test_garch_forecast_index_matches_test_index():
    result = garch_forecast(TRAIN_RETURNS, TEST_INDEX, horizon=5)
    assert (result.index == TEST_INDEX).all()


def test_garch_forecast_is_positive():
    result = garch_forecast(TRAIN_RETURNS, TEST_INDEX, horizon=5)
    assert result.gt(0).all()


def test_garch_forecast_has_no_nans():
    result = garch_forecast(TRAIN_RETURNS, TEST_INDEX, horizon=5)
    assert not result.isna().any()
