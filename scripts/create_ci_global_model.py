"""Create a deterministic 34-target global MLflow fixture for CI.

This is not a trained forecasting model. It exercises the complete global
artifact, universe, preprocessing, and serving contracts without publishing a
candidate or depending on a developer's MLflow registry.
"""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import tempfile

import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import torch

from src.config import REPO_ROOT, load_global_config
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.panel import GLOBAL_FEATURE_COLS
from src.data.universe import load_universe
from src.features.panel_preprocessing import PanelFeatureScaler
from src.models.global_artifact import save_global_artifact_components
from src.models.global_lstm import GlobalVolatilityLSTM
from src.models.global_train import FinalGlobalTrainingResult
from src.serving.global_model_wrapper import GlobalVolatilityForecaster

DEFAULT_OUTPUT = Path(".ci-global-model")
FIXTURE_FORECAST = 0.2
FIXTURE_DATES = 140


def _scaler_panel() -> pd.DataFrame:
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    dates = pd.bdate_range("2025-01-02", periods=FIXTURE_DATES, name="date")
    index = pd.MultiIndex.from_product(
        [dates, universe.target_symbols], names=["date", "ticker"]
    )
    date_number = np.repeat(np.arange(FIXTURE_DATES), len(universe.target_symbols))
    ticker_number = np.tile(
        np.arange(len(universe.target_symbols)), FIXTURE_DATES
    )
    cycle = np.sin(date_number / 8 + ticker_number / 5)
    return pd.DataFrame(
        {
            "log_returns": 0.01 * cycle,
            "rv_5d": 0.12 + 0.02 * np.abs(cycle),
            "rv_20d": 0.14 + 0.02 * np.abs(cycle),
            "rv_60d": 0.16 + 0.02 * np.abs(cycle),
            "vix_level": 0.18 + 0.01 * np.abs(cycle),
            "vix_log_return": 0.005 * np.cos(date_number / 6),
        },
        index=index,
    )


def build_ci_global_prices() -> pd.DataFrame:
    """Return deterministic raw histories for every target plus VIX."""
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    dates = pd.bdate_range("2025-09-01", periods=FIXTURE_DATES)
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for ticker_number, ticker in enumerate(universe.target_symbols):
        returns = 0.001 + 0.004 * np.sin(
            np.arange(FIXTURE_DATES) / 7 + ticker_number / 4
        )
        prices = (80.0 + ticker_number) * np.exp(np.cumsum(returns))
        rows.extend(zip(dates, [ticker] * FIXTURE_DATES, prices, strict=True))
    context = universe.context_symbol("implied_volatility")
    vix = 18.0 + 2.0 * np.sin(np.arange(FIXTURE_DATES) / 9)
    rows.extend(zip(dates, [context] * FIXTURE_DATES, vix, strict=True))
    return pd.DataFrame(rows, columns=["date", "ticker", "adjusted_close"])


def _benchmark(rows: int, dates: int) -> dict:
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    config = load_global_config()
    aggregate_metrics = [
        {
            "split": "final_holdout",
            "model": model,
            "scope": "micro",
            "rmse": 0.1,
            "mae": 0.08,
            "qlike": 0.5,
        }
        for model in ("global_lstm", "naive", "ewma", "garch")
    ]
    return {
        "schema_version": 1,
        "fixture_only": True,
        "panel": {"sha256": "ci-fixture", "rows": rows, "dates": dates},
        "universe": {
            "targets": list(universe.target_symbols),
            "context": list(universe.context_symbols),
        },
        "config": asdict(config),
        "includes_garch": True,
        "training": {"selected_epochs": 1},
        "aggregate_metrics": aggregate_metrics,
    }


def create_ci_global_model(output: Path = DEFAULT_OUTPUT) -> Path:
    """Write a deterministic full-universe pyfunc, refusing to overwrite."""
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing path: {output}. "
            "Remove it explicitly first."
        )

    config = load_global_config()
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    panel = _scaler_panel()
    fit_dates = panel.index.get_level_values("date").unique().sort_values()
    scaler = PanelFeatureScaler(tuple(GLOBAL_FEATURE_COLS)).fit(panel, fit_dates)
    model = GlobalVolatilityLSTM(
        input_size=len(GLOBAL_FEATURE_COLS),
        num_tickers=len(universe.target_symbols),
        embedding_dim=config.model.embedding_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head[-1].bias.fill_(float(np.log(FIXTURE_FORECAST)))
    model.eval()
    training = FinalGlobalTrainingResult(
        model=model,
        scaler=scaler,
        ticker_to_id={
            ticker: index for index, ticker in enumerate(universe.target_symbols)
        },
        fit_dates=fit_dates,
        epochs=1,
        samples=(len(fit_dates) - config.model.seq_len + 1)
        * len(universe.target_symbols),
        dropped_anchor_dates=fit_dates[: config.model.seq_len - 1],
    )
    evidence = _benchmark(len(panel), len(fit_dates))
    model_input = build_ci_global_prices()
    model_output = pd.DataFrame(
        {
            "ticker": list(universe.target_symbols),
            "as_of_date": [model_input["date"].max()] * len(universe.target_symbols),
            "horizon_sessions": [config.data.horizon]
            * len(universe.target_symbols),
            "forecast": [FIXTURE_FORECAST] * len(universe.target_symbols),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        benchmark_path = temp_root / "ci_global_benchmark.json"
        benchmark_path.write_text(json.dumps(evidence, indent=2) + "\n")
        components = save_global_artifact_components(
            temp_root / "components",
            training=training,
            config=config,
            universe=universe,
            evidence=evidence,
            benchmark_path=benchmark_path,
        )
        mlflow.pyfunc.save_model(
            path=str(output),
            python_model=GlobalVolatilityForecaster(),
            artifacts={
                **{name: str(path) for name, path in components.items()},
                "benchmark": str(benchmark_path),
            },
            signature=infer_signature(model_input, model_output),
            input_example=model_input,
            code_paths=[str(REPO_ROOT / "src")],
            metadata={
                "artifact_role": "global_candidate",
                "target_count": len(universe.target_symbols),
                "context_symbols": list(universe.context_symbols),
            },
        )

    (output / "CI_MODEL_NOTICE.txt").write_text(
        "CI fixture only. This artifact is not a trained volatility model.\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the generated MLflow model (default: .ci-global-model)",
    )
    args = parser.parse_args()
    output = create_ci_global_model(args.output)
    print(f"Created global CI model fixture at {output}")


if __name__ == "__main__":
    main()
