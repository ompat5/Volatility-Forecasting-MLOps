import numpy as np
import pandas as pd

from src.config import Config, DataConfig, EvalConfig, ModelConfig, TrainConfig
from src.models.train import train_final_model
from src.serving.model_wrapper import VolatilityForecaster


def _make_prices(n: int = 500) -> pd.Series:
    """Deterministic synthetic price series (geometric random walk)."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    index = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(prices, index=index)


def _small_config() -> Config:
    return Config(
        data=DataConfig(horizon=5),
        eval=EvalConfig(n_splits=3, min_train_size=100, val_frac=0.2),
        model=ModelConfig(seq_len=10, hidden_size=8, num_layers=1, dropout=0.0),
        train=TrainConfig(epochs=2, lr=1e-3, patience=2, batch_size=16, seed=0),
    )


_PRICES = _make_prices()
_CFG = _small_config()
_MODEL, _SCALER = train_final_model(_PRICES, _CFG)


def _wrapper() -> VolatilityForecaster:
    """A ready-to-predict wrapper, wired the way load_context would wire it."""
    w = VolatilityForecaster(seq_len=_CFG.model.seq_len)
    w.model = _MODEL
    w.model.eval()  # what load_context does — deterministic inference
    w.scaler = _SCALER
    return w


def test_predict_returns_a_python_float():
    forecast = _wrapper().predict(None, _PRICES)
    assert isinstance(forecast, float)


def test_predict_is_finite_and_plausible():
    # A realized-vol forecast should be finite and of a sane order of magnitude
    # (~1% daily vol annualized ≈ 0.16 for this synthetic series).
    forecast = _wrapper().predict(None, _PRICES)
    assert np.isfinite(forecast)
    assert 0.0 < forecast < 2.0


def test_predict_is_deterministic():
    # eval() must disable dropout — same input twice must give the same forecast.
    w = _wrapper()
    assert w.predict(None, _PRICES) == w.predict(None, _PRICES)


def test_predict_only_depends_on_recent_window():
    # Serving property: only the last (warm-up + seq_len) rows matter, so a caller
    # can send just a recent slice and get the same forecast as sending full history.
    w = _wrapper()
    full = w.predict(None, _PRICES)
    recent = w.predict(None, _PRICES.iloc[-120:])
    assert np.isclose(full, recent)
