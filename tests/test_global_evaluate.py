import numpy as np
import pandas as pd
import pytest

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
from src.eval.global_evaluate import (
    evaluate_global_model,
    summarize_global_predictions,
)


def test_metric_summary_has_per_ticker_macro_and_micro_views():
    predictions = pd.DataFrame(
        [
            ("fold_1", "model", "2024-01-02", "AAA", 0.20, 0.21),
            ("fold_1", "model", "2024-01-03", "AAA", 0.22, 0.20),
            ("fold_1", "model", "2024-01-02", "BBB", 0.30, 0.32),
            ("fold_1", "model", "2024-01-03", "BBB", 0.31, 0.30),
            ("final_holdout", "model", "2025-01-02", "AAA", 0.25, 0.24),
            ("final_holdout", "model", "2025-01-02", "BBB", 0.35, 0.34),
        ],
        columns=["split", "model", "date", "ticker", "actual", "forecast"],
    )
    predictions["date"] = pd.to_datetime(predictions["date"])

    per_ticker, aggregate = summarize_global_predictions(predictions)

    assert set(per_ticker["split"]) == {
        "fold_1",
        "cv_combined",
        "final_holdout",
    }
    assert set(aggregate["scope"]) == {"macro", "micro"}
    assert set(aggregate["split"]) == {
        "fold_1",
        "cv_combined",
        "final_holdout",
    }


def test_metric_summary_can_expose_asset_group_failures():
    predictions = pd.DataFrame(
        [
            ("final_holdout", "model", "2025-01-02", "AAA", 0.20, 0.21),
            ("final_holdout", "model", "2025-01-02", "BBB", 0.30, 0.32),
        ],
        columns=["split", "model", "date", "ticker", "actual", "forecast"],
    )

    _, aggregate = summarize_global_predictions(
        predictions,
        {"AAA": "technology", "BBB": "etf"},
    )

    group_rows = aggregate[aggregate["scope"] == "group"]
    assert set(group_rows["group"]) == {"technology", "etf"}
    assert (group_rows["observations"] == 1).all()


def test_metric_summary_rejects_mismatched_model_coverage():
    predictions = pd.DataFrame(
        [
            ("fold_1", "model", "2025-01-02", "AAA", 0.20, 0.21),
            ("fold_1", "baseline", "2025-01-02", "AAA", 0.20, 0.19),
            ("fold_1", "baseline", "2025-01-02", "BBB", 0.30, 0.29),
        ],
        columns=["split", "model", "date", "ticker", "actual", "forecast"],
    )

    with pytest.raises(ValueError, match="identical date/ticker coverage"):
        summarize_global_predictions(predictions)


def _tiny_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=220, name="date")
    tickers = ["AAA", "BBB"]
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=["date", "ticker"],
    )
    rows = []
    for date_number in range(len(dates)):
        for ticker_number in range(len(tickers)):
            cycle = np.sin(date_number / 9.0 + ticker_number)
            log_return = 0.008 * cycle + 0.002 * np.cos(date_number / 4.0)
            rv_5d = 0.14 + 0.02 * abs(cycle) + ticker_number * 0.01
            rows.append(
                {
                    "log_returns": log_return,
                    "rv_5d": rv_5d,
                    "rv_20d": rv_5d + 0.01,
                    "rv_60d": rv_5d + 0.02,
                    "vix_level": 0.18 + 0.01 * abs(cycle),
                    "vix_log_return": 0.005 * np.cos(date_number / 5.0),
                    "rv_target": rv_5d + 0.005 * np.sin(date_number / 3.0),
                }
            )
    return pd.DataFrame(rows, index=index)[[*GLOBAL_FEATURE_COLS, TARGET_COL]]


def _tiny_config() -> GlobalConfig:
    return GlobalConfig(
        data=DataConfig(horizon=5),
        eval=GlobalEvalConfig(
            n_splits=2,
            min_train_size=30,
            val_frac=0.2,
            holdout_size=40,
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
            epochs=1,
            lr=1e-2,
            patience=1,
            batch_size=64,
            seed=3,
            dates_per_batch=2,
        ),
        baselines=BaselineConfig(ewma_span=12.0),
    )


def test_end_to_end_global_evaluation_smoke_without_garch():
    universe = Universe(
        schema_version=1,
        targets=(TargetAsset("AAA", "a"), TargetAsset("BBB", "b")),
        context=(ContextSeries("^VIX", "implied_volatility"),),
    )

    result = evaluate_global_model(
        _tiny_panel(),
        universe,
        _tiny_config(),
        include_garch=False,
    )

    assert set(result.predictions["model"]) == {
        "global_lstm",
        "naive",
        "ewma",
    }
    assert "final_holdout" in set(result.aggregate_metrics["split"])
    assert result.training_summary["selected_epochs"] == 1
    assert (result.predictions["forecast"] > 0).all()
