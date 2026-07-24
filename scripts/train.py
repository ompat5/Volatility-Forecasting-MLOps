"""Config-driven training entrypoint: walk-forward LSTM eval, logged to MLflow."""

import mlflow
import pandas as pd

from src.config import REPO_ROOT, load_config
from src.data.ingest import DEFAULT_RAW_DIR
from src.eval.evaluate import evaluate_lstm
from src.models.train import set_seeds

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
        print(results.round(4))


if __name__ == "__main__":
    main()
