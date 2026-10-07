"""Run monitoring across all 34 deployed targets and shared VIX context."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from src.config import load_global_config
from src.data.global_prices import fetch_global_prices, load_cached_global_prices
from src.data.ingest import DEFAULT_RAW_DIR, DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe
from src.monitoring.global_pipeline import (
    load_global_monitoring_config,
    run_global_monitoring,
)
from src.monitoring.global_reporting import write_global_outputs
from src.serving.global_api import validate_global_forecaster


def _artifact_manifest(model) -> dict:
    try:
        runtime = model.unwrap_python_model()
    except (AttributeError, NotImplementedError) as exc:
        raise ValueError(
            "Global monitoring requires an inspectable MLflow pyfunc"
        ) from exc
    manifest = getattr(runtime, "manifest", None)
    if not isinstance(manifest, dict):
        raise ValueError("Global monitoring model lacks an artifact manifest")
    return manifest


def _emit_global_alert(report: dict) -> None:
    status = report["status"]
    components = report["component_status"]
    message = (
        f"Global monitoring status is {status}: "
        f"data={components['data_quality']}, drift={components['feature_drift']}, "
        f"regime={components['volatility_regime']}, "
        f"error={components['forecast_error']}"
    )
    print(message)
    if os.getenv("GITHUB_ACTIONS") == "true" and status != "ok":
        annotation = "error" if status == "critical" else "warning"
        print(f"::{annotation} title=Global volatility monitoring::{message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/global_monitoring.yaml"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("monitoring/global_reference.json"),
    )
    parser.add_argument("--universe", type=Path, default=DEFAULT_TICKERS_CONFIG)
    parser.add_argument("--model-uri", default="global_model")
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("global-monitoring-output"),
    )
    args = parser.parse_args()
    if args.prices is not None and args.refresh:
        parser.error("--prices and --refresh are mutually exclusive")

    universe = load_universe(args.universe)
    monitoring_config, model_config = load_global_monitoring_config(args.config)
    if model_config.get("role") != "global_candidate":
        raise ValueError("Global monitoring config must identify the pinned model")
    if args.prices is not None:
        model_input = pd.read_parquet(args.prices)
    elif args.refresh:
        model_input = fetch_global_prices(universe)
    else:
        model_input = load_cached_global_prices(universe, args.raw_dir)

    model = mlflow.pyfunc.load_model(args.model_uri)
    validate_global_forecaster(model)
    manifest = _artifact_manifest(model)
    reference = json.loads(args.reference.read_text())
    report, predictions = run_global_monitoring(
        model,
        model_input,
        reference,
        universe,
        monitoring_config,
        horizon=load_global_config().data.horizon,
        model_version=model_config["version"],
        artifact_manifest=manifest,
        run_date=(pd.Timestamp.now(tz="America/Toronto") if args.refresh else None),
    )
    write_global_outputs(args.output_dir, report, predictions)
    model_input.to_parquet(args.output_dir / "latest_prices.parquet", index=False)
    _emit_global_alert(report)


if __name__ == "__main__":
    main()
