import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import mae, qlike, rmse


def test_rmse_perfect_prediction_is_zero():
    s = pd.Series([0.1, 0.2, 0.3])

    assert rmse(s, s) == 0.0


def test_rmse_known_value():
    actual = pd.Series([0.10, 0.20, 0.30])
    pred = pd.Series([0.12, 0.18, 0.33])
    expected = np.sqrt((0.0004 + 0.0004 + 0.0009) / 3)

    assert np.isclose(rmse(actual, pred), expected)


def test_mae_perfect_prediction_is_zero():
    s = pd.Series([0.1, 0.2, 0.3])

    assert mae(s, s) == 0.0


def test_mae_known_value():
    actual = pd.Series([0.10, 0.20, 0.30])
    pred = pd.Series([0.12, 0.18, 0.33])
    expected = (0.02 + 0.02 + 0.03) / 3

    assert np.isclose(mae(actual, pred), expected)


def test_rmse_penalizes_large_errors_more_than_mae():
    actual = pd.Series([0.20, 0.20, 0.20, 0.20])
    pred = pd.Series([0.20, 0.20, 0.20, 0.40])

    assert rmse(actual, pred) > mae(actual, pred)


def test_qlike_perfect_prediction_is_zero():
    s = pd.Series([0.1, 0.2, 0.3])

    assert np.isclose(qlike(s, s), 0.0)


def test_qlike_known_value():
    actual = pd.Series([0.20])
    pred = pd.Series([0.10])
    expected = 4 - np.log(4) - 1

    assert np.isclose(qlike(actual, pred), expected)


def test_qlike_penalizes_underprediction_more_than_overprediction():
    actual = pd.Series([0.20, 0.20, 0.20])
    under = pd.Series([0.10, 0.10, 0.10]) 
    over = pd.Series([0.30, 0.30, 0.30])

    assert qlike(actual, under) > qlike(actual, over)
    assert np.isclose(rmse(actual, under), rmse(actual, over))
    assert np.isclose(mae(actual, under), mae(actual, over))


def test_qlike_raises_on_nonpositive_pred():
    actual = pd.Series([0.2, 0.2])
    pred = pd.Series([0.2, 0.0]) 
    
    with pytest.raises(ValueError, match="strictly positive"):
        qlike(actual, pred)


def test_qlike_raises_on_negative_actual():
    actual = pd.Series([0.2, -0.1])
    pred = pd.Series([0.2, 0.2])

    with pytest.raises(ValueError, match="strictly positive"):
        qlike(actual, pred)
