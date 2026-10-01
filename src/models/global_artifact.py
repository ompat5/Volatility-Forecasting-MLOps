"""Evidence validation and component packaging for the global candidate."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import torch

from src.config import REPO_ROOT, GlobalConfig
from src.data.panel import GLOBAL_FEATURE_COLS, TARGET_COL
from src.data.universe import Universe
from src.features.global_inference import GLOBAL_INFERENCE_COLUMNS
from src.models.global_train import FinalGlobalTrainingResult

GLOBAL_REGISTERED_MODEL_NAME = "global-volatility-lstm"
REQUIRED_BENCHMARK_MODELS = {"global_lstm", "naive", "ewma", "garch"}


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def load_global_evaluation_evidence(
    benchmark_path: Path,
    *,
    panel_path: Path,
    config: GlobalConfig,
    universe: Universe,
) -> dict[str, Any]:
    """Require training inputs to match the accepted Phase 3 evaluation exactly."""
    with benchmark_path.open() as benchmark_file:
        evidence = json.load(benchmark_file)
    if evidence.get("schema_version") != 1:
        raise ValueError("Global benchmark schema_version must be 1")
    if evidence.get("includes_garch") is not True:
        raise ValueError("Global benchmark must include the GARCH comparator")
    if evidence.get("panel", {}).get("sha256") != sha256_file(panel_path):
        raise ValueError("Global panel hash does not match the accepted benchmark")
    if evidence.get("config") != asdict(config):
        raise ValueError("Global config does not match the accepted benchmark")
    if evidence.get("universe", {}).get("targets") != list(universe.target_symbols):
        raise ValueError("Global target universe does not match the benchmark")
    if evidence.get("universe", {}).get("context") != list(universe.context_symbols):
        raise ValueError("Global context universe does not match the benchmark")

    selected_epochs = evidence.get("training", {}).get("selected_epochs")
    if not isinstance(selected_epochs, int) or selected_epochs <= 0:
        raise ValueError("Global benchmark selected_epochs must be positive")
    final_micro = [
        row
        for row in evidence.get("aggregate_metrics", [])
        if row.get("split") == "final_holdout" and row.get("scope") == "micro"
    ]
    if (
        len(final_micro) != len(REQUIRED_BENCHMARK_MODELS)
        or {row.get("model") for row in final_micro} != REQUIRED_BENCHMARK_MODELS
    ):
        raise ValueError("Global benchmark lacks complete final-holdout model metrics")
    per_ticker = evidence.get("final_holdout_per_ticker_metrics", [])
    expected_rows = len(universe.target_symbols) * len(REQUIRED_BENCHMARK_MODELS)
    expected_pairs = {
        (model, ticker)
        for model in REQUIRED_BENCHMARK_MODELS
        for ticker in universe.target_symbols
    }
    actual_pairs = {
        (row.get("model"), row.get("ticker"))
        for row in per_ticker
        if row.get("split") == "final_holdout"
    }
    if len(per_ticker) != expected_rows or actual_pairs != expected_pairs:
        raise ValueError("Global benchmark lacks complete per-ticker holdout metrics")
    return evidence


def save_global_artifact_components(
    directory: Path,
    *,
    training: FinalGlobalTrainingResult,
    config: GlobalConfig,
    universe: Universe,
    evidence: dict[str, Any],
    benchmark_path: Path,
) -> dict[str, Path]:
    """Serialize weights, scaler, and the contract/provenance manifest."""
    if training.epochs != evidence["training"]["selected_epochs"]:
        raise ValueError("Artifact epoch count does not match evaluation evidence")
    expected_vocabulary = {
        ticker: index for index, ticker in enumerate(universe.target_symbols)
    }
    if training.ticker_to_id != expected_vocabulary:
        raise ValueError("Artifact ticker vocabulary does not match the universe")
    if tuple(training.scaler.feature_cols) != tuple(GLOBAL_FEATURE_COLS):
        raise ValueError("Artifact scaler feature order is incompatible")
    if training.scaler.n_fit_rows_ != evidence["panel"]["rows"]:
        raise ValueError("Artifact scaler was not fitted on the complete panel")
    if len(training.fit_dates) != evidence["panel"]["dates"]:
        raise ValueError("Artifact fit dates do not cover the complete panel")
    expected_samples = (len(training.fit_dates) - config.model.seq_len + 1) * len(
        universe.target_symbols
    )
    if training.samples != expected_samples:
        raise ValueError("Artifact training samples do not cover the complete panel")
    model = training.model
    if (
        model.lstm.input_size != len(GLOBAL_FEATURE_COLS)
        or model.lstm.hidden_size != config.model.hidden_size
        or model.lstm.num_layers != config.model.num_layers
        or model.ticker_embedding.num_embeddings != len(universe.target_symbols)
        or model.ticker_embedding.embedding_dim != config.model.embedding_dim
    ):
        raise ValueError("Artifact model architecture does not match configuration")

    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty artifact directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    model_state_path = directory / "model_state.pt"
    scaler_path = directory / "scaler.pkl"
    manifest_path = directory / "manifest.json"
    torch.save(training.model.state_dict(), model_state_path)
    joblib.dump(training.scaler.scaler, scaler_path)

    final_micro = {
        row["model"]: {metric: row[metric] for metric in ("rmse", "mae", "qlike")}
        for row in evidence["aggregate_metrics"]
        if row["split"] == "final_holdout" and row["scope"] == "micro"
    }
    manifest = {
        "schema_version": 1,
        "artifact_role": "global_candidate",
        "registered_model_name": GLOBAL_REGISTERED_MODEL_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "input_size": len(GLOBAL_FEATURE_COLS),
            "num_tickers": len(universe.target_symbols),
            "embedding_dim": config.model.embedding_dim,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
        },
        "preprocessing": {
            "feature_columns": GLOBAL_FEATURE_COLS,
            "seq_len": config.model.seq_len,
            "scaler": "StandardScaler",
            "scaler_fit_scope": "all_target_observable_panel_dates",
            "vix_level_divisor": 100.0,
        },
        "target": {
            "column": TARGET_COL,
            "horizon_sessions": config.data.horizon,
            "transform": config.model.target_transform,
            "output_constraint": "exp_log_volatility",
        },
        "universe": {
            "targets": [asdict(asset) for asset in universe.targets],
            "context": [asdict(series) for series in universe.context],
            "ticker_to_id": training.ticker_to_id,
        },
        "training": {
            "selected_epochs": training.epochs,
            "objective": "mean_squared_error_on_log_volatility",
            "seed": config.train.seed,
            "fit_start": training.fit_dates.min().date().isoformat(),
            "fit_end": training.fit_dates.max().date().isoformat(),
            "fit_dates": len(training.fit_dates),
            "scaler_fit_rows": training.scaler.n_fit_rows_,
            "samples": training.samples,
            "dropped_anchor_dates": [
                date.date().isoformat() for date in training.dropped_anchor_dates
            ],
            "panel": evidence["panel"],
            "config": asdict(config),
        },
        "evaluation": {
            "benchmark_path": _portable_path(benchmark_path),
            "benchmark_sha256": sha256_file(benchmark_path),
            "final_holdout_micro": final_micro,
        },
        "input_contract": {
            "format": "long_adjusted_close",
            "columns": list(GLOBAL_INFERENCE_COLUMNS),
            "context_required": universe.context_symbol("implied_volatility"),
            "target_subset_allowed": True,
        },
        "output_contract": {
            "columns": [
                "ticker",
                "as_of_date",
                "horizon_sessions",
                "forecast",
            ],
            "one_row_per_requested_target": True,
        },
        "components": {
            "model_state_sha256": sha256_file(model_state_path),
            "scaler_sha256": sha256_file(scaler_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "model_state": model_state_path,
        "scaler": scaler_path,
        "manifest": manifest_path,
    }
