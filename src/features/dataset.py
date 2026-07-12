import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VolatilityDataset(Dataset):
    """Sliding-window PyTorch Dataset over a volatility feature DataFrame."""
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], target_col: str, seq_len: int):
        self.seq_len = seq_len
        self.features = df[feature_cols].to_numpy(dtype=np.float32)
        self.targets = df[target_col].to_numpy(dtype=np.float32)
    
    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx: int):
        X = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return torch.from_numpy(X.copy()), torch.tensor(float(y))