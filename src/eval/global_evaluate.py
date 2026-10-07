"""End-to-end walk-forward evaluation for the global volatility model."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.config import GlobalConfig
from src.data.panel import GLOBAL_FEATURE_COLS, TARGET_COL
from src.data.universe import Universe
from src.eval.metrics import mae, qlike, rmse
from src.eval.panel_splits import PanelHoldout, PanelSplit, build_panel_split_plan
from src.features.panel_dataset import (
    BalancedDateBatchSampler,
    PanelVolatilityDataset,
)
from src.features.panel_preprocessing import PanelFeatureScaler
from src.features.realized_vol import ANNUALIZED_FACTOR
from src.models.global_baselines import rolling_garch_forecast
from src.models.global_lstm import GlobalVolatilityLSTM
from src.models.global_train import (
    fit_global_model_for_epochs,
    predict_global_model,
    train_global_model,
)
from src.models.train import set_seeds

MODEL_NAME = "global_lstm"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalEvaluationResult:
    predictions: pd.DataFrame
    per_ticker_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    training_summary: dict[str, Any]
    split_plan: dict[str, Any]


def _new_model(config: GlobalConfig, num_tickers: int) -> GlobalVolatilityLSTM:
    return GlobalVolatilityLSTM(
        input_size=len(GLOBAL_FEATURE_COLS),
        num_tickers=num_tickers,
        embedding_dim=config.model.embedding_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    )


def _dataset(
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    ticker_to_id: dict[str, int],
    config: GlobalConfig,
) -> PanelVolatilityDataset:
    return PanelVolatilityDataset(
        panel,
        GLOBAL_FEATURE_COLS,
        TARGET_COL,
        config.model.seq_len,
        dates,
        ticker_to_id,
    )


def _prediction_frame(
    forecast: pd.Series,
    raw_panel: pd.DataFrame,
    *,
    split: str,
    model: str,
) -> pd.DataFrame:
    actual = raw_panel.loc[forecast.index, TARGET_COL].astype(float)
    frame = pd.DataFrame(
        {
            "actual": actual.to_numpy(),
            "forecast": forecast.to_numpy(),
        },
        index=forecast.index,
    ).reset_index()
    frame.insert(0, "model", model)
    frame.insert(0, "split", split)
    return frame


def _baseline_prediction_frames(
    panel: pd.DataFrame,
    universe: Universe,
    *,
    fit_dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    split_name: str,
    config: GlobalConfig,
    include_garch: bool,
) -> list[pd.DataFrame]:
    if config.data.horizon != 5:
        raise ValueError("The current naive panel feature requires horizon=5")

    by_model: dict[str, list[pd.Series]] = {
        "naive": [],
        "ewma": [],
    }
    if include_garch:
        by_model["garch"] = []

    for ticker_number, ticker in enumerate(universe.target_symbols, start=1):
        ticker_panel = panel.xs(ticker, level="ticker").sort_index()
        naive = ticker_panel.loc[forecast_dates, "rv_5d"].rename(ticker)
        ewma = (
            ticker_panel["log_returns"]
            .ewm(span=config.baselines.ewma_span)
            .std()
            .mul(ANNUALIZED_FACTOR)
            .loc[forecast_dates]
            .rename(ticker)
        )
        by_model["naive"].append(naive)
        by_model["ewma"].append(ewma)
        if include_garch:
            if (
                ticker_number == 1
                or ticker_number % 5 == 0
                or ticker_number == len(universe.target_symbols)
            ):
                LOGGER.info(
                    "%s GARCH progress: %d/%d tickers",
                    split_name,
                    ticker_number,
                    len(universe.target_symbols),
                )
            garch = rolling_garch_forecast(
                ticker_panel["log_returns"],
                fit_dates,
                forecast_dates,
                horizon=config.data.horizon,
            ).rename(ticker)
            by_model["garch"].append(garch)

    frames: list[pd.DataFrame] = []
    for model, forecasts in by_model.items():
        wide = pd.concat(forecasts, axis=1)
        wide = wide.loc[forecast_dates, list(universe.target_symbols)]
        stacked = wide.stack().rename("forecast")
        stacked.index.names = ["date", "ticker"]
        frames.append(
            _prediction_frame(
                stacked,
                panel,
                split=split_name,
                model=model,
            )
        )
    return frames


def _metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rmse": rmse(frame["actual"], frame["forecast"]),
        "mae": mae(frame["actual"], frame["forecast"]),
        "qlike": qlike(frame["actual"], frame["forecast"]),
        "observations": len(frame),
    }


def summarize_global_predictions(
    predictions: pd.DataFrame,
    ticker_groups: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate per-ticker, group, equal-ticker macro, and pooled metrics."""
    required = {"split", "model", "date", "ticker", "actual", "forecast"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Predictions cannot be empty")
    prediction_keys = ["split", "model", "date", "ticker"]
    if predictions.duplicated(prediction_keys).any():
        raise ValueError("Predictions contain duplicate split/model/date/ticker keys")
    values = predictions[["actual", "forecast"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Actuals and forecasts must be finite and strictly positive")
    prediction_tickers = set(predictions["ticker"])
    if ticker_groups is not None and set(ticker_groups) != prediction_tickers:
        raise ValueError(
            "ticker_groups must contain every prediction ticker exactly once"
        )
    for split, split_frame in predictions.groupby("split"):
        expected_coverage: set[tuple[pd.Timestamp, str]] | None = None
        for model, model_frame in split_frame.groupby("model"):
            coverage = set(zip(model_frame["date"], model_frame["ticker"]))
            if expected_coverage is None:
                expected_coverage = coverage
            elif coverage != expected_coverage:
                raise ValueError(
                    f"Models do not share identical date/ticker coverage in {split}: "
                    f"{model} differs"
                )

    evaluation_frames = [
        (split, frame) for split, frame in predictions.groupby("split")
    ]
    cv_predictions = predictions[predictions["split"].str.startswith("fold_")]
    if not cv_predictions.empty:
        evaluation_frames.append(("cv_combined", cv_predictions))

    per_ticker_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for split, split_frame in evaluation_frames:
        for model, model_frame in split_frame.groupby("model"):
            ticker_metrics = []
            for ticker, ticker_frame in model_frame.groupby("ticker"):
                values = _metric_values(ticker_frame)
                ticker_metrics.append(values)
                per_ticker_rows.append(
                    {
                        "split": split,
                        "model": model,
                        "ticker": ticker,
                        **values,
                    }
                )
            macro = pd.DataFrame(ticker_metrics)[["rmse", "mae", "qlike"]].mean()
            aggregate_rows.append(
                {
                    "split": split,
                    "model": model,
                    "scope": "macro",
                    "group": None,
                    "rmse": float(macro["rmse"]),
                    "mae": float(macro["mae"]),
                    "qlike": float(macro["qlike"]),
                    "observations": len(model_frame),
                }
            )
            aggregate_rows.append(
                {
                    "split": split,
                    "model": model,
                    "scope": "micro",
                    "group": None,
                    **_metric_values(model_frame),
                }
            )
            if ticker_groups is not None:
                grouped = model_frame.assign(
                    group=model_frame["ticker"].map(ticker_groups)
                )
                for group, group_frame in grouped.groupby("group"):
                    aggregate_rows.append(
                        {
                            "split": split,
                            "model": model,
                            "scope": "group",
                            "group": group,
                            **_metric_values(group_frame),
                        }
                    )
    return pd.DataFrame(per_ticker_rows), pd.DataFrame(aggregate_rows)


def _evaluate_fold(
    panel: pd.DataFrame,
    universe: Universe,
    split: PanelSplit,
    config: GlobalConfig,
    *,
    fold_number: int,
    include_garch: bool,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    ticker_to_id = {
        ticker: index for index, ticker in enumerate(universe.target_symbols)
    }
    scaler = PanelFeatureScaler(tuple(GLOBAL_FEATURE_COLS)).fit(
        panel,
        split.train_dates,
    )
    scaled = scaler.transform(panel)
    train_dataset = _dataset(scaled, split.train_dates, ticker_to_id, config)
    validation_dataset = _dataset(
        scaled,
        split.validation_dates,
        ticker_to_id,
        config,
    )
    test_dataset = _dataset(scaled, split.test_dates, ticker_to_id, config)
    train_sampler = BalancedDateBatchSampler(
        train_dataset,
        dates_per_batch=config.train.dates_per_batch,
        shuffle=True,
        seed=config.train.seed + fold_number,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
    )

    set_seeds(config.train.seed + fold_number)
    result = train_global_model(
        _new_model(config, len(ticker_to_id)),
        train_loader,
        validation_loader,
        epochs=config.train.epochs,
        lr=config.train.lr,
        patience=config.train.patience,
    )
    forecast = predict_global_model(
        result.model,
        test_dataset,
        batch_size=config.train.batch_size,
    )
    frames = [
        _prediction_frame(
            forecast,
            panel,
            split=split.name,
            model=MODEL_NAME,
        )
    ]
    frames.extend(
        _baseline_prediction_frames(
            panel,
            universe,
            fit_dates=split.train_dates,
            forecast_dates=split.test_dates,
            split_name=split.name,
            config=config,
            include_garch=include_garch,
        )
    )
    summary = {
        "split": split.name,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "scaler_fit_start": scaler.fit_dates_.min().date().isoformat(),
        "scaler_fit_end": scaler.fit_dates_.max().date().isoformat(),
        "scaler_fit_rows": scaler.n_fit_rows_,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "dropped_train_anchor_dates": len(train_dataset.dropped_anchor_dates),
        "history": list(result.history),
    }
    return frames, summary


def _evaluate_holdout(
    panel: pd.DataFrame,
    universe: Universe,
    holdout: PanelHoldout,
    config: GlobalConfig,
    *,
    selected_epochs: int,
    include_garch: bool,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    ticker_to_id = {
        ticker: index for index, ticker in enumerate(universe.target_symbols)
    }
    scaler = PanelFeatureScaler(tuple(GLOBAL_FEATURE_COLS)).fit(
        panel,
        holdout.fit_dates,
    )
    scaled = scaler.transform(panel)
    fit_dataset = _dataset(scaled, holdout.fit_dates, ticker_to_id, config)
    test_dataset = _dataset(scaled, holdout.test_dates, ticker_to_id, config)
    sampler = BalancedDateBatchSampler(
        fit_dataset,
        dates_per_batch=config.train.dates_per_batch,
        shuffle=True,
        seed=config.train.seed + 1_000,
    )
    set_seeds(config.train.seed + 1_000)
    model = fit_global_model_for_epochs(
        _new_model(config, len(ticker_to_id)),
        DataLoader(fit_dataset, batch_sampler=sampler),
        epochs=selected_epochs,
        lr=config.train.lr,
    )
    forecast = predict_global_model(
        model,
        test_dataset,
        batch_size=config.train.batch_size,
    )
    frames = [
        _prediction_frame(
            forecast,
            panel,
            split="final_holdout",
            model=MODEL_NAME,
        )
    ]
    frames.extend(
        _baseline_prediction_frames(
            panel,
            universe,
            fit_dates=holdout.fit_dates,
            forecast_dates=holdout.test_dates,
            split_name="final_holdout",
            config=config,
            include_garch=include_garch,
        )
    )
    summary = {
        "split": "final_holdout",
        "selected_epochs": selected_epochs,
        "scaler_fit_start": scaler.fit_dates_.min().date().isoformat(),
        "scaler_fit_end": scaler.fit_dates_.max().date().isoformat(),
        "scaler_fit_rows": scaler.n_fit_rows_,
        "fit_samples": len(fit_dataset),
        "test_samples": len(test_dataset),
        "dropped_fit_anchor_dates": len(fit_dataset.dropped_anchor_dates),
    }
    return frames, summary


def evaluate_global_model(
    panel: pd.DataFrame,
    universe: Universe,
    config: GlobalConfig,
    *,
    include_garch: bool = True,
) -> GlobalEvaluationResult:
    """Run purged CV, select epochs, refit, and score the untouched holdout."""
    plan = build_panel_split_plan(
        panel,
        n_splits=config.eval.n_splits,
        horizon=config.data.horizon,
        min_train_size=config.eval.min_train_size,
        val_frac=config.eval.val_frac,
        holdout_size=config.eval.holdout_size,
    )
    prediction_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_number, split in enumerate(plan.folds, start=1):
        LOGGER.info(
            "Starting %s (%d/%d)",
            split.name,
            fold_number,
            len(plan.folds),
        )
        frames, summary = _evaluate_fold(
            panel,
            universe,
            split,
            config,
            fold_number=fold_number,
            include_garch=include_garch,
        )
        prediction_frames.extend(frames)
        fold_summaries.append(summary)
        LOGGER.info(
            "Completed %s: selected epoch %d, validation loss %.6f",
            split.name,
            summary["best_epoch"],
            summary["best_validation_loss"],
        )

    selected_epochs = max(
        1,
        int(round(np.median([summary["best_epoch"] for summary in fold_summaries]))),
    )
    LOGGER.info("Refitting final holdout model for %d epochs", selected_epochs)
    holdout_frames, holdout_summary = _evaluate_holdout(
        panel,
        universe,
        plan.holdout,
        config,
        selected_epochs=selected_epochs,
        include_garch=include_garch,
    )
    prediction_frames.extend(holdout_frames)
    LOGGER.info("Completed final holdout evaluation")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ticker_groups = {asset.symbol: asset.group for asset in universe.targets}
    per_ticker, aggregate = summarize_global_predictions(
        predictions,
        ticker_groups,
    )
    return GlobalEvaluationResult(
        predictions=predictions,
        per_ticker_metrics=per_ticker,
        aggregate_metrics=aggregate,
        training_summary={
            "folds": fold_summaries,
            "selected_epochs": selected_epochs,
            "holdout": holdout_summary,
        },
        split_plan=plan.to_dict(),
    )
