import numpy as np
import pandas as pd
import torch

from src.features.dataset import VolatilityDataset

_RNG = np.random.default_rng(42)
_N = 100
_SEQ_LEN = 10
_FEATURE_COLS = ["f1", "f2", "f3"]
_TARGET_COL = "target"

_DF = pd.DataFrame({
    "f1": _RNG.normal(0, 1, _N).astype(np.float32),
    "f2": _RNG.normal(0, 1, _N).astype(np.float32),
    "f3": _RNG.normal(0, 1, _N).astype(np.float32),
    "target": _RNG.uniform(0.05, 0.5, _N).astype(np.float32),
})

_DATASET = VolatilityDataset(_DF, _FEATURE_COLS, _TARGET_COL, _SEQ_LEN)


def test_dataset_len():
    assert len(_DATASET) == _N - _SEQ_LEN + 1


def test_dataset_x_shape():
    X, _ = _DATASET[0]
    assert X.shape == (_SEQ_LEN, len(_FEATURE_COLS))


def test_dataset_y_is_scalar():
    _, y = _DATASET[0]
    assert y.ndim == 0


def test_dataset_tensors_are_float32():
    X, y = _DATASET[0]
    assert X.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataset_target_alignment():
    # target at index i must be targets[i + seq_len - 1]
    for i in [0, 5, len(_DATASET) - 1]:
        _, y = _DATASET[i]
        expected = _DF[_TARGET_COL].iloc[i + _SEQ_LEN - 1]
        assert np.isclose(y.item(), expected)


def test_dataset_last_index_does_not_raise():
    X, y = _DATASET[len(_DATASET) - 1]
    assert X.shape == (_SEQ_LEN, len(_FEATURE_COLS))
