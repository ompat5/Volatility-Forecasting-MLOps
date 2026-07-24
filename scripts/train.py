"""Config-driven training entrypoint: walk-forward LSTM eval, logged to MLflow."""

import mlflow
import pandas as pd
import mlflow.pytorch
import mlflow.sklearn

from src.config import REPO_ROOT, load_config
from src.data.ingest import DEFAULT_RAW_DIR
from src.eval.evaluate import evaluate_lstm
from src.models.train import set_seeds, train_final_model

TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "aapl-lstm"


def main(ticker: str = "AAPL") -> None:
    cfg = load_config()
    set_seeds(cfg.train.seed)
    prices = pd.read_parquet(DEFAULT_RAW_DIR / f"{ticker}.parquet")["Adj Close"]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(cfg.flatten())
        results = evaluate_lstm(prices, cfg)
        metrics = results.loc["lstm"].to_dict()
        mlflow.log_metrics(metrics)
        model, scaler = train_final_model(prices, cfg)
        mlflow.pytorch.log_model(
            model, name="model", 
            serialization_format="pickle",
            registered_model_name="volatility-lstm"
        )
        mlflow.sklearn.log_model(scaler, name="scaler")
        print(results.round(4))


if __name__ == "__main__":
    main()
