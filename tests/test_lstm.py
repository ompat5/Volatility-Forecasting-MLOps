import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.features.dataset import VolatilityDataset
from src.models.lstm import VolatilityLSTM
from src.models.train import predict, train_model

_BATCH_SIZE = 8
_SEQ_LEN = 10
_INPUT_SIZE = 4
_HIDDEN_SIZE = 16
_FEATURE_COLS = ["f1", "f2", "f3", "f4"]
_TARGET_COL = "target"

_MODEL = VolatilityLSTM(input_size=_INPUT_SIZE, hidden_size=_HIDDEN_SIZE)


def _make_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n).astype(np.float32),
            "f2": rng.normal(0, 1, n).astype(np.float32),
            "f3": rng.normal(0, 1, n).astype(np.float32),
            "f4": rng.normal(0, 1, n).astype(np.float32),
            "target": rng.uniform(0.05, 0.5, n).astype(np.float32),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )


# --- VolatilityLSTM ---

def test_lstm_forward_output_shape():
    x = torch.randn(_BATCH_SIZE, _SEQ_LEN, _INPUT_SIZE)
    out = _MODEL(x)
    assert out.shape == (_BATCH_SIZE,)


def test_lstm_forward_is_1d():
    x = torch.randn(_BATCH_SIZE, _SEQ_LEN, _INPUT_SIZE)
    out = _MODEL(x)
    assert out.ndim == 1


# --- train_model ---

def test_train_model_returns_lstm():
    df = _make_df()
    train_df, val_df = df.iloc[:-20], df.iloc[-20:]
    train_loader = DataLoader(
        VolatilityDataset(train_df, _FEATURE_COLS, _TARGET_COL, _SEQ_LEN),
        batch_size=16, shuffle=True,
    )
    val_loader = DataLoader(
        VolatilityDataset(val_df, _FEATURE_COLS, _TARGET_COL, _SEQ_LEN),
        batch_size=16, shuffle=False,
    )
    model = VolatilityLSTM(input_size=_INPUT_SIZE, hidden_size=_HIDDEN_SIZE)
    result = train_model(model, train_loader, val_loader, epochs=2, patience=5)
    assert isinstance(result, VolatilityLSTM)


# --- predict ---

def test_predict_returns_series():
    df = _make_df()
    result = predict(_MODEL, df, _FEATURE_COLS, _TARGET_COL, _SEQ_LEN)
    assert isinstance(result, pd.Series)


def test_predict_correct_length():
    df = _make_df()
    result = predict(_MODEL, df, _FEATURE_COLS, _TARGET_COL, _SEQ_LEN)
    assert len(result) == len(df) - _SEQ_LEN + 1


def test_predict_index_starts_at_seq_len_minus_one():
    df = _make_df()
    result = predict(_MODEL, df, _FEATURE_COLS, _TARGET_COL, _SEQ_LEN)
    assert result.index[0] == df.index[_SEQ_LEN - 1]


def test_predict_index_ends_at_last_row():
    df = _make_df()
    result = predict(_MODEL, df, _FEATURE_COLS, _TARGET_COL, _SEQ_LEN)
    assert result.index[-1] == df.index[-1]
