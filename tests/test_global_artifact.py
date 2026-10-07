from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import (
    BaselineConfig,
    DataConfig,
    GlobalConfig,
    GlobalEvalConfig,
    GlobalModelConfig,
    GlobalTrainConfig,
    REPO_ROOT,
)
from src.data.panel import GLOBAL_FEATURE_COLS, TARGET_COL
from src.data.universe import ContextSeries, TargetAsset, Universe
from src.features.panel_preprocessing import PanelFeatureScaler
from src.models.global_artifact import (
    load_global_evaluation_evidence,
    save_global_artifact_components,
    sha256_file,
)
from src.models.global_lstm import GlobalVolatilityLSTM
from src.models.global_train import FinalGlobalTrainingResult
from src.serving.global_model_wrapper import GlobalVolatilityForecaster


def _universe() -> Universe:
    return Universe(
        schema_version=1,
        targets=(TargetAsset("AAA", "group_a"), TargetAsset("BBB", "group_b")),
        context=(ContextSeries("^VIX", "implied_volatility"),),
    )


def _config() -> GlobalConfig:
    return GlobalConfig(
        data=DataConfig(horizon=5),
        eval=GlobalEvalConfig(
            n_splits=2,
            min_train_size=30,
            val_frac=0.2,
            holdout_size=20,
        ),
        model=GlobalModelConfig(
            seq_len=10,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            embedding_dim=2,
            target_transform="log",
        ),
        train=GlobalTrainConfig(
            epochs=2,
            lr=1e-3,
            patience=1,
            batch_size=32,
            seed=7,
            dates_per_batch=2,
        ),
        baselines=BaselineConfig(ewma_span=12.0),
    )


def _panel(n_dates: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=n_dates, name="date")
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    rows = []
    for date_number in range(n_dates):
        for ticker_number in range(2):
            cycle = np.sin(date_number / 8 + ticker_number)
            rv = 0.15 + 0.02 * abs(cycle)
            rows.append(
                {
                    "log_returns": 0.01 * cycle,
                    "rv_5d": rv,
                    "rv_20d": rv + 0.01,
                    "rv_60d": rv + 0.02,
                    "vix_level": 0.18 + 0.01 * abs(cycle),
                    "vix_log_return": 0.005 * np.cos(date_number / 6),
                    "rv_target": rv + 0.005,
                }
            )
    return pd.DataFrame(rows, index=index)[[*GLOBAL_FEATURE_COLS, TARGET_COL]]


def _evidence(panel_path: Path) -> dict:
    aggregate = [
        {
            "split": "final_holdout",
            "model": model,
            "scope": "micro",
            "rmse": 0.1,
            "mae": 0.08,
            "qlike": 0.5,
            "observations": 40,
        }
        for model in ["global_lstm", "naive", "ewma", "garch"]
    ]
    per_ticker = [
        {"split": "final_holdout", "model": model, "ticker": ticker}
        for model in ["global_lstm", "naive", "ewma", "garch"]
        for ticker in ["AAA", "BBB"]
    ]
    return {
        "schema_version": 1,
        "panel": {"sha256": sha256_file(panel_path), "rows": 200, "dates": 100},
        "universe": {"targets": ["AAA", "BBB"], "context": ["^VIX"]},
        "config": asdict(_config()),
        "includes_garch": True,
        "training": {"selected_epochs": 1},
        "aggregate_metrics": aggregate,
        "final_holdout_per_ticker_metrics": per_ticker,
    }


def _raw_prices(n_dates: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=n_dates)
    rows = []
    for ticker_number, ticker in enumerate(["AAA", "BBB"]):
        returns = 0.001 + 0.005 * np.sin(np.arange(n_dates) / 7 + ticker_number)
        prices = (100 + 20 * ticker_number) * np.exp(np.cumsum(returns))
        rows.extend(zip(dates, [ticker] * n_dates, prices))
    vix = 18 + 2 * np.sin(np.arange(n_dates) / 9)
    rows.extend(zip(dates, ["^VIX"] * n_dates, vix))
    return pd.DataFrame(rows, columns=["date", "ticker", "adjusted_close"])


def test_evaluation_evidence_must_match_panel_config_and_universe(tmp_path: Path):
    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"evaluated-panel")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(_evidence(panel_path)))

    evidence = load_global_evaluation_evidence(
        benchmark_path,
        panel_path=panel_path,
        config=_config(),
        universe=_universe(),
    )
    assert evidence["training"]["selected_epochs"] == 1

    panel_path.write_bytes(b"different-panel")
    with pytest.raises(ValueError, match="panel hash"):
        load_global_evaluation_evidence(
            benchmark_path,
            panel_path=panel_path,
            config=_config(),
            universe=_universe(),
        )


def test_saved_pyfunc_round_trip_preserves_global_contract(tmp_path: Path):
    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"evaluated-panel")
    benchmark_path = tmp_path / "benchmark.json"
    evidence = _evidence(panel_path)
    benchmark_path.write_text(json.dumps(evidence))

    panel = _panel()
    dates = panel.index.get_level_values("date").unique()
    scaler = PanelFeatureScaler(tuple(GLOBAL_FEATURE_COLS)).fit(panel, dates)
    model = GlobalVolatilityLSTM(
        input_size=len(GLOBAL_FEATURE_COLS),
        num_tickers=2,
        embedding_dim=2,
        hidden_size=8,
        dropout=0.0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head[-1].bias.fill_(float(np.log(0.2)))
    training = FinalGlobalTrainingResult(
        model=model,
        scaler=scaler,
        ticker_to_id={"AAA": 0, "BBB": 1},
        fit_dates=dates,
        epochs=1,
        samples=182,
        dropped_anchor_dates=dates[:9],
    )
    components = save_global_artifact_components(
        tmp_path / "components",
        training=training,
        config=_config(),
        universe=_universe(),
        evidence=evidence,
        benchmark_path=benchmark_path,
    )
    output = tmp_path / "mlflow-model"
    mlflow.pyfunc.save_model(
        path=str(output),
        python_model=GlobalVolatilityForecaster(),
        artifacts={
            **{name: str(path) for name, path in components.items()},
            "benchmark": str(benchmark_path),
        },
        code_paths=[str(REPO_ROOT / "src")],
    )

    loaded = mlflow.pyfunc.load_model(str(output))
    forecasts = loaded.predict(_raw_prices())

    assert forecasts["ticker"].tolist() == ["AAA", "BBB"]
    assert forecasts["horizon_sessions"].tolist() == [5, 5]
    assert forecasts["forecast"].tolist() == pytest.approx([0.2, 0.2])
    assert (output / "code" / "src" / "serving" / "global_model_wrapper.py").is_file()
    manifest = json.loads(components["manifest"].read_text())
    assert manifest["artifact_role"] == "global_candidate"
    assert manifest["training"]["scaler_fit_rows"] == 200

    components["model_state"].write_bytes(
        components["model_state"].read_bytes() + b"tampered"
    )
    context = SimpleNamespace(
        artifacts={
            **{name: str(path) for name, path in components.items()},
            "benchmark": str(benchmark_path),
        }
    )
    with pytest.raises(ValueError, match="model-state checksum"):
        GlobalVolatilityForecaster().load_context(context)
