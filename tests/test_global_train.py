import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.features.panel_dataset import (
    BalancedDateBatchSampler,
    PanelVolatilityDataset,
)
from src.models.global_lstm import GlobalVolatilityLSTM
from src.models.global_train import (
    fit_global_model_for_epochs,
    predict_global_model,
    train_global_model,
)
from src.models.train import set_seeds

FEATURE_COLS = ["feature_1", "feature_2"]


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=60, name="date")
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    rows = []
    for date_number in range(len(dates)):
        for ticker_number in range(2):
            feature = np.sin(date_number / 5.0) + ticker_number * 0.2
            rows.append(
                {
                    "feature_1": feature,
                    "feature_2": np.cos(date_number / 7.0),
                    "target": 0.15 + 0.02 * abs(feature),
                }
            )
    return pd.DataFrame(rows, index=index)


def _dataset(dates: pd.DatetimeIndex) -> PanelVolatilityDataset:
    return PanelVolatilityDataset(
        _panel(),
        FEATURE_COLS,
        "target",
        seq_len=5,
        anchor_dates=dates,
        ticker_to_id={"AAA": 0, "BBB": 1},
    )


def _model() -> GlobalVolatilityLSTM:
    return GlobalVolatilityLSTM(
        input_size=2,
        num_tickers=2,
        embedding_dim=2,
        hidden_size=8,
        dropout=0.0,
    )


def test_training_and_prediction_keep_positive_original_scale():
    dates = _panel().index.get_level_values("date").unique()
    train_dataset = _dataset(dates[:40])
    validation_dataset = _dataset(dates[45:55])
    sampler = BalancedDateBatchSampler(
        train_dataset,
        dates_per_batch=2,
        shuffle=True,
        seed=0,
    )
    set_seeds(0)

    result = train_global_model(
        _model(),
        DataLoader(train_dataset, batch_sampler=sampler),
        DataLoader(validation_dataset, batch_size=16),
        epochs=2,
        lr=1e-2,
        patience=2,
    )
    forecast = predict_global_model(
        result.model,
        validation_dataset,
        batch_size=16,
    )

    assert result.best_epoch in {1, 2}
    assert len(result.history) == 2
    assert forecast.index.equals(validation_dataset.sample_index)
    assert np.isfinite(forecast).all()
    assert (forecast > 0).all()


def test_fixed_epoch_refit_uses_nonempty_balanced_loader():
    dates = _panel().index.get_level_values("date").unique()
    dataset = _dataset(dates)
    sampler = BalancedDateBatchSampler(
        dataset,
        dates_per_batch=3,
        shuffle=True,
        seed=0,
    )

    model = fit_global_model_for_epochs(
        _model(),
        DataLoader(dataset, batch_sampler=sampler),
        epochs=1,
        lr=1e-2,
    )

    assert isinstance(model, GlobalVolatilityLSTM)
    assert not model.training
