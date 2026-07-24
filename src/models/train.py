import torch
import torch.nn as nn
import copy
import pandas as pd
import numpy as np
import random

from src.models.lstm import VolatilityLSTM
from torch.utils.data import DataLoader
from src.features.dataset import VolatilityDataset
from src.features.realized_vol import build_features
from sklearn.preprocessing import StandardScaler
from src.config import Config


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model: VolatilityLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10
) -> VolatilityLSTM:
    """Train the LSTM model"""
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_train_pred = model(X_batch)
            loss = criterion(y_train_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                y_val_pred = model(X_val)
                val_loss += criterion(y_val_pred, y_val).item()
        val_loss /= len(val_loader)

        patience_counter += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        elif patience_counter >= patience:
            break
    
    model.load_state_dict(best_weights)
    return model


def predict(
    model: VolatilityLSTM,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int) -> pd.Series:
    """Make predictions using the trained LSTM model."""
    dataset = VolatilityDataset(df, feature_cols, target_col, seq_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in loader:
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    return pd.Series(predictions, index=df.index[seq_len - 1:])


def train_final_model(prices: pd.Series, config: Config) -> tuple[VolatilityLSTM, StandardScaler]:
    """Train one model on ALL available data, the deployable model."""
    FEATURE_COLS = ["log_returns", "rv_5d", "rv_20d", "rv_60d"]
    TARGET_COL = "rv_target"

    features = build_features(prices, config.data.horizon)
    
    n_val = int(len(features) * config.eval.val_frac)
    inner_train_df = features.iloc[:-n_val].copy()
    val_df = features.iloc[-n_val:].copy()

    scaler = StandardScaler()
    inner_train_df[FEATURE_COLS] = scaler.fit_transform(inner_train_df[FEATURE_COLS])
    val_df[FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS])

    train_loader = DataLoader(
        VolatilityDataset(inner_train_df, FEATURE_COLS, TARGET_COL, config.model.seq_len),
        batch_size=config.train.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        VolatilityDataset(val_df, FEATURE_COLS, TARGET_COL, config.model.seq_len),
        batch_size=config.train.batch_size,
        shuffle=False
    )

    model = VolatilityLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    model = train_model(
        model, train_loader, val_loader,
        epochs=config.train.epochs, lr=config.train.lr, patience=config.train.patience
    )
    return model, scaler