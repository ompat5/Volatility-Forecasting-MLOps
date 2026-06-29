import numpy as np
import pandas as pd


def rmse(actual: pd.Series, pred: pd.Series) -> float:
    """Calculate the root mean squared error between two series."""
    return float(np.sqrt(((actual - pred) ** 2).mean()))


def mae(actual: pd.Series, pred: pd.Series) -> float:
    """Calculate the mean absolute error between two series."""
    return float((actual - pred).abs().mean())


def qlike(actual: pd.Series, pred: pd.Series) -> float:
    """Calculate the QLIKE loss between two series."""
    if (actual <= 0).any() or (pred <= 0).any():
        raise ValueError("QLIKE requires strictly positive volatilities")

    actual_var = actual ** 2
    pred_var = pred ** 2

    return float((actual_var / pred_var - np.log(actual_var / pred_var) - 1).mean())
