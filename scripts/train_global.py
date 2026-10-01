"""Train, package, and register the evaluated 34-target global candidate.

Run with: `uv run python -m scripts.train_global`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_GLOBAL_MODEL_CONFIG,
    REPO_ROOT,
    load_global_config,
)
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe
from src.models.global_artifact import (
    GLOBAL_REGISTERED_MODEL_NAME,
    load_global_evaluation_evidence,
    save_global_artifact_components,
    sha256_file,
)
from src.models.global_train import train_final_global_model
from src.serving.global_model_wrapper import GlobalVolatilityForecaster

DEFAULT_PANEL_PATH = REPO_ROOT / "data" / "processed" / "global_panel.parquet"
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "global_model.json"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "global-volatility-lstm"


def _input_example(target_symbol: str, context_symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2025-09-01", periods=90)
    target_prices = 100.0 * np.exp(np.cumsum(0.001 + 0.004 * np.sin(np.arange(90) / 7)))
    context_prices = 18.0 + 2.0 * np.sin(np.arange(90) / 9)
    return pd.DataFrame(
        {
            "date": np.concatenate([dates.to_numpy(), dates.to_numpy()]),
            "ticker": [target_symbol] * 90 + [context_symbol] * 90,
            "adjusted_close": np.concatenate([target_prices, context_prices]),
        }
    )


def _model_signature(model_input: pd.DataFrame, horizon: int):
    model_output = pd.DataFrame(
        {
            "ticker": [model_input["ticker"].iloc[0]],
            "as_of_date": pd.to_datetime(["2026-01-02"]),
            "horizon_sessions": [horizon],
            "forecast": [0.2],
        }
    )
    return infer_signature(model_input, model_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_GLOBAL_MODEL_CONFIG)
    parser.add_argument("--universe", type=Path, default=DEFAULT_TICKERS_CONFIG)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    args = parser.parse_args()

    config = load_global_config(args.config)
    universe = load_universe(args.universe)
    evidence = load_global_evaluation_evidence(
        args.benchmark,
        panel_path=args.panel,
        config=config,
        universe=universe,
    )
    panel = pd.read_parquet(args.panel)
    selected_epochs = evidence["training"]["selected_epochs"]
    input_example = _input_example(
        universe.target_symbols[0],
        universe.context_symbol("implied_volatility"),
    )
    training = train_final_global_model(
        panel,
        universe,
        config,
        selected_epochs=selected_epochs,
    )

    final_micro = next(
        row
        for row in evidence["aggregate_metrics"]
        if row["split"] == "final_holdout"
        and row["scope"] == "micro"
        and row["model"] == "global_lstm"
    )
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="global-candidate-full-refit") as run:
        mlflow.log_params(
            {
                **config.flatten(),
                "artifact.selected_epochs": selected_epochs,
                "artifact.target_count": len(universe.target_symbols),
                "artifact.panel_sha256": sha256_file(args.panel),
                "artifact.production_default": False,
            }
        )
        mlflow.log_metrics(
            {
                "holdout_micro_rmse": final_micro["rmse"],
                "holdout_micro_mae": final_micro["mae"],
                "holdout_micro_qlike": final_micro["qlike"],
            }
        )
        mlflow.set_tags(
            {
                "model_scope": "global_34_target",
                "promotion_status": "candidate",
                "production_default": "false",
                "serving_enabled": "false",
                "evaluation_benchmark_sha256": sha256_file(args.benchmark),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = save_global_artifact_components(
                Path(tmpdir),
                training=training,
                config=config,
                universe=universe,
                evidence=evidence,
                benchmark_path=args.benchmark,
            )
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=GlobalVolatilityForecaster(),
                artifacts={
                    **{name: str(path) for name, path in artifacts.items()},
                    "benchmark": str(args.benchmark),
                },
                signature=_model_signature(input_example, config.data.horizon),
                input_example=input_example,
                code_paths=[str(REPO_ROOT / "src")],
                registered_model_name=GLOBAL_REGISTERED_MODEL_NAME,
                metadata={
                    "artifact_role": "global_candidate",
                    "target_count": len(universe.target_symbols),
                    "context_symbols": list(universe.context_symbols),
                },
            )

        client = MlflowClient()
        versions = client.search_model_versions(
            f"name = '{GLOBAL_REGISTERED_MODEL_NAME}' and run_id = '{run.info.run_id}'"
        )
        if len(versions) != 1:
            raise RuntimeError("Could not resolve the newly registered global version")
        registered_version = versions[0].version
        version_tags = {
            "promotion_status": "candidate",
            "production_default": "false",
            "serving_enabled": "false",
            "evaluation_benchmark_sha256": sha256_file(args.benchmark),
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(
                GLOBAL_REGISTERED_MODEL_NAME,
                registered_version,
                key,
                value,
            )

    print(
        f"Registered {GLOBAL_REGISTERED_MODEL_NAME} version {registered_version} "
        f"from run {run.info.run_id}"
    )
    print(f"Model URI: {model_info.model_uri}")
    print("Production default remains the separate AAPL volatility-lstm model")


if __name__ == "__main__":
    main()
