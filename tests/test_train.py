import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.config import Config, DataConfig, EvalConfig, ModelConfig, TrainConfig
from src.features.realized_vol import build_features
from src.models.lstm import VolatilityLSTM
from src.models.train import predict, set_seeds, train_final_model


_FEATURE_COLS = ["log_returns", "rv_5d", "rv_20d", "rv_60d"]
_TARGET_COL = "rv_target"


def _make_prices(n: int = 500) -> pd.Series:
    """A deterministic synthetic price series (geometric random walk)."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    index = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(prices, index=index)


def _small_config() -> Config:
    """Tiny config so the smoke test trains in a fraction of a second."""
    return Config(
        data=DataConfig(horizon=5),
        eval=EvalConfig(n_splits=3, min_train_size=100, val_frac=0.2),
        model=ModelConfig(seq_len=10, hidden_size=8, num_layers=1, dropout=0.0),
        train=TrainConfig(epochs=2, lr=1e-3, patience=2, batch_size=16, seed=0),
    )


_PRICES = _make_prices()
_CFG = _small_config()
_MODEL, _SCALER = train_final_model(_PRICES, _CFG)


def test_returns_model_and_scaler():
    assert isinstance(_MODEL, VolatilityLSTM)
    assert isinstance(_SCALER, StandardScaler)


def test_scaler_is_fitted_on_all_features():
    # A model without a correctly-fitted scaler is useless at serving time.
    assert hasattr(_SCALER, "mean_")
    assert _SCALER.n_features_in_ == len(_FEATURE_COLS)


def test_model_and_scaler_produce_finite_predictions():
    features = build_features(_PRICES, _CFG.data.horizon).copy()
    features[_FEATURE_COLS] = _SCALER.transform(features[_FEATURE_COLS])
    preds = predict(_MODEL, features, _FEATURE_COLS, _TARGET_COL, _CFG.model.seq_len)
    assert preds.notna().all()
    assert np.isfinite(preds).all()


def test_set_seeds_makes_torch_reproducible():
    set_seeds(0)
    a = torch.rand(4)
    set_seeds(0)
    b = torch.rand(4)
    assert torch.equal(a, b)
