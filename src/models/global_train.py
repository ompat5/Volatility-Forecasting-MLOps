"""Training and prediction helpers for the global volatility LSTM."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import GlobalConfig
from src.data.panel import GLOBAL_FEATURE_COLS, TARGET_COL, validate_global_panel
from src.data.universe import Universe
from src.features.panel_dataset import (
    BalancedDateBatchSampler,
    PanelVolatilityDataset,
)
from src.features.panel_preprocessing import PanelFeatureScaler
from src.models.global_lstm import GlobalVolatilityLSTM
from src.models.train import set_seeds


@dataclass(frozen=True)
class GlobalTrainingResult:
    model: GlobalVolatilityLSTM
    best_epoch: int
    best_validation_loss: float
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class FinalGlobalTrainingResult:
    """Components and provenance needed to package the deployable candidate."""

    model: GlobalVolatilityLSTM
    scaler: PanelFeatureScaler
    ticker_to_id: dict[str, int]
    fit_dates: pd.DatetimeIndex
    epochs: int
    samples: int
    dropped_anchor_dates: pd.DatetimeIndex


def _log_target_loss(
    log_prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if (target <= 0).any():
        raise ValueError("Log-volatility training requires positive targets")
    return nn.functional.mse_loss(log_prediction, torch.log(target))


def train_global_model(
    model: GlobalVolatilityLSTM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    patience: int,
) -> GlobalTrainingResult:
    """Train with early stopping on log-volatility validation loss."""
    if epochs <= 0 or lr <= 0 or patience <= 0:
        raise ValueError("epochs, lr, and patience must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_observations = 0
        for features, ticker_ids, target in train_loader:
            optimizer.zero_grad()
            loss = _log_target_loss(model(features, ticker_ids), target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(target)
            train_observations += len(target)

        model.eval()
        validation_loss_sum = 0.0
        validation_observations = 0
        with torch.no_grad():
            for features, ticker_ids, target in validation_loader:
                loss = _log_target_loss(model(features, ticker_ids), target)
                validation_loss_sum += loss.item() * len(target)
                validation_observations += len(target)
        if train_observations == 0 or validation_observations == 0:
            raise ValueError("Training and validation loaders must not be empty")

        train_loss = train_loss_sum / train_observations
        validation_loss = validation_loss_sum / validation_observations
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise ValueError(
                "Global training produced a non-finite train or validation loss"
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    return GlobalTrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        history=tuple(history),
    )


def fit_global_model_for_epochs(
    model: GlobalVolatilityLSTM,
    train_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
) -> GlobalVolatilityLSTM:
    """Fit on a fixed date set for the CV-selected epoch count."""
    if epochs <= 0 or lr <= 0:
        raise ValueError("epochs and lr must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        observations = 0
        for features, ticker_ids, target in train_loader:
            optimizer.zero_grad()
            loss = _log_target_loss(model(features, ticker_ids), target)
            loss.backward()
            optimizer.step()
            observations += len(target)
        if observations == 0:
            raise ValueError("Training loader must not be empty")
    model.eval()
    return model


def train_final_global_model(
    panel: pd.DataFrame,
    universe: Universe,
    config: GlobalConfig,
    *,
    selected_epochs: int,
) -> FinalGlobalTrainingResult:
    """Fit the post-evaluation candidate on every target-observable panel date."""
    if selected_epochs <= 0:
        raise ValueError("selected_epochs must be positive")
    validate_global_panel(panel, universe)
    fit_dates = panel.index.get_level_values("date").unique().sort_values()
    ticker_to_id = {
        ticker: index for index, ticker in enumerate(universe.target_symbols)
    }
    scaler = PanelFeatureScaler(tuple(GLOBAL_FEATURE_COLS)).fit(panel, fit_dates)
    scaled = scaler.transform(panel)
    dataset = PanelVolatilityDataset(
        scaled,
        GLOBAL_FEATURE_COLS,
        TARGET_COL,
        config.model.seq_len,
        fit_dates,
        ticker_to_id,
    )
    sampler = BalancedDateBatchSampler(
        dataset,
        dates_per_batch=config.train.dates_per_batch,
        shuffle=True,
        seed=config.train.seed,
    )

    set_seeds(config.train.seed)
    model = GlobalVolatilityLSTM(
        input_size=len(GLOBAL_FEATURE_COLS),
        num_tickers=len(ticker_to_id),
        embedding_dim=config.model.embedding_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    )
    model = fit_global_model_for_epochs(
        model,
        DataLoader(dataset, batch_sampler=sampler),
        epochs=selected_epochs,
        lr=config.train.lr,
    )
    return FinalGlobalTrainingResult(
        model=model,
        scaler=scaler,
        ticker_to_id=ticker_to_id,
        fit_dates=fit_dates,
        epochs=selected_epochs,
        samples=len(dataset),
        dropped_anchor_dates=dataset.dropped_anchor_dates,
    )


def predict_global_model(
    model: GlobalVolatilityLSTM,
    dataset: PanelVolatilityDataset,
    *,
    batch_size: int,
) -> pd.Series:
    """Predict positive volatility and retain exact date/ticker alignment."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, ticker_ids, _ in loader:
            output = model.predict_volatility(features, ticker_ids)
            predictions.append(output.cpu().numpy())
    if not predictions:
        return pd.Series(dtype=float, index=dataset.sample_index)
    values = np.concatenate(predictions)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Global model produced invalid volatility predictions")
    return pd.Series(values, index=dataset.sample_index, name="forecast")
