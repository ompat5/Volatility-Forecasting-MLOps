"""Fetch fresh prices, forecast, and write the daily monitoring report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from src.config import load_config
from src.data.ingest import fetch_ticker
from src.monitoring.pipeline import (
    load_monitoring_config,
    run_monitoring,
    write_outputs,
)


def _emit_alert(report: dict) -> None:
    status = report["status"]
    message = (
        f"{report['ticker']} monitoring status is {status}: "
        f"drift={report['feature_drift']['status']}, "
        f"regime={report['volatility_regime']['status']}, "
        f"error={report['forecast_error']['status']}"
    )
    print(message)
    if os.getenv("GITHUB_ACTIONS") == "true" and status != "ok":
        annotation = "error" if status == "critical" else "warning"
        print(f"::{annotation} title=Volatility monitoring::{message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/monitoring.yaml"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("monitoring/reference.json"),
    )
    parser.add_argument("--model-uri", default="model")
    parser.add_argument("--prices", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("monitoring-output"),
    )
    args = parser.parse_args()

    monitoring_config, model_config = load_monitoring_config(args.config)
    if args.prices:
        prices_frame = pd.read_parquet(args.prices)
    else:
        prices_frame = fetch_ticker(monitoring_config.ticker)
    prices = prices_frame["Adj Close"].dropna()

    reference = json.loads(args.reference.read_text())
    model = mlflow.pyfunc.load_model(args.model_uri)
    report, predictions = run_monitoring(
        model,
        prices,
        reference,
        monitoring_config,
        horizon=load_config().data.horizon,
        model_version=model_config["version"],
    )
    write_outputs(args.output_dir, report, predictions)
    prices_frame.to_parquet(args.output_dir / "latest_prices.parquet")
    _emit_alert(report)


if __name__ == "__main__":
    main()
