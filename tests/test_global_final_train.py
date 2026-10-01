import numpy as np
import pandas as pd

from src.config import (
    BaselineConfig,
    DataConfig,
    GlobalConfig,
    GlobalEvalConfig,
    GlobalModelConfig,
    GlobalTrainConfig,
)
from src.data.panel import GLOBAL_FEATURE_COLS, TARGET_COL
from src.data.universe import ContextSeries, TargetAsset, Universe
from src.features.panel_dataset import PanelVolatilityDataset
from src.models.global_train import (
    predict_global_model,
    train_final_global_model,
)


def _universe() -> Universe:
    return Universe(
        schema_version=1,
        targets=(TargetAsset("AAA", "a"), TargetAsset("BBB", "b")),
        context=(ContextSeries("^VIX", "implied_volatility"),),
    )


def _config() -> GlobalConfig:
    return GlobalConfig(
        data=DataConfig(horizon=5),
        eval=GlobalEvalConfig(
            n_splits=2,
            min_train_size=20,
            val_frac=0.2,
            holdout_size=20,
        ),
        model=GlobalModelConfig(
            seq_len=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            embedding_dim=2,
            target_transform="log",
        ),
        train=GlobalTrainConfig(
            epochs=2,
            lr=1e-2,
            patience=1,
            batch_size=32,
            seed=3,
            dates_per_batch=2,
        ),
        baselines=BaselineConfig(ewma_span=12.0),
    )


def _panel(n_dates: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=n_dates, name="date")
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    rows = []
    for date_number in range(n_dates):
        for ticker_number in range(2):
            cycle = np.sin(date_number / 7 + ticker_number)
            rv = 0.14 + 0.03 * abs(cycle)
            rows.append(
                {
                    "log_returns": 0.01 * cycle,
                    "rv_5d": rv,
                    "rv_20d": rv + 0.01,
                    "rv_60d": rv + 0.02,
                    "vix_level": 0.18 + 0.01 * abs(cycle),
                    "vix_log_return": 0.004 * np.cos(date_number / 5),
                    "rv_target": rv + 0.005,
                }
            )
    return pd.DataFrame(rows, index=index)[[*GLOBAL_FEATURE_COLS, TARGET_COL]]


def test_final_global_training_fits_all_target_observable_dates():
    panel = _panel()
    config = _config()
    result = train_final_global_model(
        panel,
        _universe(),
        config,
        selected_epochs=1,
    )

    assert result.epochs == 1
    assert result.ticker_to_id == {"AAA": 0, "BBB": 1}
    assert len(result.fit_dates) == 80
    assert result.scaler.n_fit_rows_ == 160
    assert result.samples == 152
    assert len(result.dropped_anchor_dates) == 4

    scaled = result.scaler.transform(panel)
    dataset = PanelVolatilityDataset(
        scaled,
        GLOBAL_FEATURE_COLS,
        TARGET_COL,
        config.model.seq_len,
        result.fit_dates[-5:],
        result.ticker_to_id,
    )
    forecasts = predict_global_model(result.model, dataset, batch_size=16)
    assert len(forecasts) == 10
    assert np.isfinite(forecasts).all()
    assert (forecasts > 0).all()
