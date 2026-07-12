import numpy as np
import pandas as pd

from collections import defaultdict
from src.eval.metrics import mae, qlike, rmse
from src.eval.walk_forward import walk_forward_splits
from src.features.realized_vol import realized_vol_target, build_features
from src.models.baselines import ewma_forecast, garch_forecast, naive_forecast
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from src.features.dataset import VolatilityDataset
from src.models.lstm import VolatilityLSTM
from src.models.train import predict, train_model


def evaluate_baselines(
    returns: pd.Series,
    horizon: int = 5,
    n_splits: int = 5,
    min_train_size: int = 252,
    ewma_span: float = 32.0,
    annualize: bool = True
) -> pd.DataFrame:
    """Evaluate the forecasts using various metrics."""
    metric_functions = {"rmse": rmse, "mae": mae, "qlike": qlike}
    model_names = ["naive", "ewma", "garch"]

    splits = walk_forward_splits(returns.index, n_splits, horizon, min_train_size)

    all_folds = []
    for train_dates, test_dates in splits:
        train_returns = returns.loc[train_dates]

        test_target = realized_vol_target(returns, horizon).loc[test_dates].dropna()
        valid_index = test_target.index

        naive_preds = naive_forecast(returns, horizon, annualize).loc[valid_index]
        ewma_preds = ewma_forecast(returns, ewma_span, annualize).loc[valid_index]
        garch_preds = garch_forecast(train_returns, valid_index, horizon, annualize)
        preds = {"naive": naive_preds, "ewma": ewma_preds, "garch": garch_preds}

        fold_metrics = {
            model: {name: metric(test_target, preds[model]) for name, metric in metric_functions.items()}
            for model in preds
        }

        all_folds.append(fold_metrics)
    
    results = defaultdict(dict)
    for model in model_names:
        for metric in metric_functions:
            results[model][metric] = np.mean([fold[model][metric] for fold in all_folds])
    
    return pd.DataFrame(results).T


def evaluate_lstm(
    prices: pd.Series,
    horizon: int = 5,
    n_splits: int = 5,
    min_train_size: int = 252,
    seq_len: int = 30,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.2,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    val_frac: float = 0.2
) -> pd.DataFrame:
    """Evaluate the LSTM model using various metrics."""
    FEATURE_COLS = ["log_returns", "rv_5d", "rv_20d", "rv_60d"]
    TARGET_COL = "rv_target"

    features = build_features(prices, horizon)
    splits = walk_forward_splits(features.index, n_splits, horizon, min_train_size)
    metric_functions = {"rmse": rmse, "mae": mae, "qlike": qlike}
    all_folds = []

    for train_dates, test_dates in splits:
        train_df = features.loc[train_dates]
    
        n_val = int(len(train_df) * val_frac)
        inner_train_df = train_df.iloc[:-n_val].copy()
        val_df = train_df.iloc[-n_val:].copy()
        test_df = features.loc[test_dates].copy()
        
        scaler = StandardScaler()
        inner_train_df[FEATURE_COLS] = scaler.fit_transform(inner_train_df[FEATURE_COLS])
        val_df[FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS])
        test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])

        train_loader = DataLoader(VolatilityDataset(inner_train_df, FEATURE_COLS, TARGET_COL, seq_len), batch_size=32, shuffle=True)
        val_loader = DataLoader(VolatilityDataset(val_df, FEATURE_COLS, TARGET_COL, seq_len), batch_size=32, shuffle=False)

        model = VolatilityLSTM(input_size=len(FEATURE_COLS), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr, patience=patience)

        preds = predict(model, test_df, FEATURE_COLS, TARGET_COL, seq_len)
        test_target = test_df[TARGET_COL].dropna()
        common_index = test_target.index.intersection(preds.index)
        
        fold_metrics = {name: metric(test_target.loc[common_index], preds.loc[common_index]) for name, metric in metric_functions.items()}
        all_folds.append(fold_metrics)

    results = {metric: np.mean([fold[metric] for fold in all_folds]) for metric in metric_functions}
    
    return pd.DataFrame(results, index=["lstm"])
