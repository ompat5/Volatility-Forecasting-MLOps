import numpy as np
import pandas as pd

from src.eval.evaluate import evaluate_baselines


RETURNS = pd.Series(
    np.random.normal(0, 0.01, 400),
    index=pd.bdate_range("2018-01-01", periods=400),
)

RESULT = evaluate_baselines(RETURNS, horizon=5, n_splits=3, min_train_size=100)


def test_evaluate_returns_dataframe():
    assert isinstance(RESULT, pd.DataFrame)


def test_evaluate_shape():
    assert RESULT.shape == (3, 3)


def test_evaluate_index():
    assert list(RESULT.index) == ["naive", "ewma", "garch"]


def test_evaluate_columns():
    assert set(RESULT.columns) == {"rmse", "mae", "qlike"}


def test_evaluate_all_positive():
    assert (RESULT > 0).all().all()


def test_evaluate_no_nans():
    assert not RESULT.isna().any().any()
