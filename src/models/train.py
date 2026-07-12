import torch
import torch.nn as nn
import copy
import pandas as pd
import numpy as np

from src.models.lstm import VolatilityLSTM
from torch.utils.data import DataLoader
from src.features.dataset import VolatilityDataset


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

