"""Shared LSTM with ticker identity for global volatility forecasting."""

from __future__ import annotations

import torch
from torch import nn


class GlobalVolatilityLSTM(nn.Module):
    """Encode market history jointly with a learned target-ticker embedding."""

    def __init__(
        self,
        *,
        input_size: int,
        num_tickers: int,
        embedding_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size <= 0 or num_tickers <= 0 or embedding_dim <= 0:
            raise ValueError("Model dimensions must be positive")
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
        head_size = max(8, hidden_size // 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + embedding_dim, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        ticker_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return unconstrained log realized volatility."""
        sequence, _ = self.lstm(features)
        history = sequence[:, -1, :]
        ticker = self.ticker_embedding(ticker_ids)
        return self.head(torch.cat([history, ticker], dim=1)).squeeze(-1)

    def predict_volatility(
        self,
        features: torch.Tensor,
        ticker_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return strictly positive realized volatility in the original scale."""
        return torch.exp(self(features, ticker_ids))
