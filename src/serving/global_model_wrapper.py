"""Self-contained MLflow wrapper for the global volatility model."""

from __future__ import annotations

import json

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from src.data.panel import GLOBAL_FEATURE_COLS, TARGET_COL
from src.features.global_inference import build_global_inference_batch
from src.models.global_artifact import (
    GLOBAL_REGISTERED_MODEL_NAME,
    sha256_file,
)
from src.models.global_lstm import GlobalVolatilityLSTM


def validate_global_artifact_manifest(manifest: dict) -> None:
    """Fail fast when an artifact cannot satisfy the inference contract."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Global artifact schema_version must be 1")
    if manifest.get("artifact_role") != "global_candidate":
        raise ValueError("Global artifact role must be 'global_candidate'")
    if manifest.get("registered_model_name") != GLOBAL_REGISTERED_MODEL_NAME:
        raise ValueError("Global artifact registered-model name is incompatible")

    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("feature_columns") != GLOBAL_FEATURE_COLS:
        raise ValueError("Global artifact feature schema is incompatible")
    seq_len = preprocessing.get("seq_len")
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError("Global artifact seq_len must be positive")

    target = manifest.get("target", {})
    if target.get("column") != TARGET_COL or target.get("transform") != "log":
        raise ValueError("Global artifact target contract is incompatible")
    if (
        not isinstance(target.get("horizon_sessions"), int)
        or target["horizon_sessions"] <= 0
    ):
        raise ValueError("Global artifact horizon must be positive")

    universe = manifest.get("universe", {})
    raw_targets = universe.get("targets")
    raw_context = universe.get("context")
    ticker_to_id = universe.get("ticker_to_id")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Global artifact must contain forecast targets")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("symbol"), str)
        or not item.get("symbol")
        or not isinstance(item.get("group"), str)
        or not item.get("group")
        for item in raw_targets
    ):
        raise ValueError("Global artifact target entries are invalid")
    if not isinstance(raw_context, list) or any(
        not isinstance(item, dict) for item in raw_context
    ):
        raise ValueError("Global artifact context entries are invalid")
    target_symbols = [item.get("symbol") for item in raw_targets]
    if len(set(target_symbols)) != len(target_symbols):
        raise ValueError("Global artifact target symbols must be unique")
    if not isinstance(ticker_to_id, dict):
        raise ValueError("Global artifact must contain a ticker vocabulary")
    if list(ticker_to_id) != target_symbols:
        raise ValueError("Global artifact ticker vocabulary order is incompatible")
    if list(ticker_to_id.values()) != list(range(len(target_symbols))):
        raise ValueError("Global artifact ticker IDs must be contiguous from zero")
    implied_volatility = [
        item.get("symbol")
        for item in raw_context or []
        if item.get("role") == "implied_volatility"
    ]
    if len(implied_volatility) != 1:
        raise ValueError("Global artifact needs one implied-volatility context series")
    input_contract = manifest.get("input_contract", {})
    if input_contract.get("context_required") != implied_volatility[0]:
        raise ValueError("Global artifact context input contract is incompatible")
    if input_contract.get("target_subset_allowed") is not True:
        raise ValueError("Global artifact must allow known target subsets")

    architecture = manifest.get("architecture", {})
    positive_dimensions = (
        "input_size",
        "embedding_dim",
        "hidden_size",
        "num_layers",
    )
    if any(
        not isinstance(architecture.get(field), int) or architecture[field] <= 0
        for field in positive_dimensions
    ):
        raise ValueError("Global artifact model dimensions must be positive")
    if architecture["input_size"] != len(GLOBAL_FEATURE_COLS):
        raise ValueError("Global artifact model input size is incompatible")
    if architecture.get("num_tickers") != len(target_symbols):
        raise ValueError("Global artifact model ticker count is incompatible")
    dropout = architecture.get("dropout")
    if not isinstance(dropout, (float, int)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("Global artifact dropout must be in [0, 1)")

    if input_contract.get("columns") != ["date", "ticker", "adjusted_close"]:
        raise ValueError("Global artifact input contract is incompatible")
    output_contract = manifest.get("output_contract", {})
    if output_contract.get("columns") != [
        "ticker",
        "as_of_date",
        "horizon_sessions",
        "forecast",
    ]:
        raise ValueError("Global artifact output contract is incompatible")


class GlobalVolatilityForecaster(mlflow.pyfunc.PythonModel):
    """Long-form adjusted closes in, one forecast row per requested target out."""

    def _initialize(self, model, scaler, manifest: dict) -> None:
        validate_global_artifact_manifest(manifest)
        if getattr(scaler, "n_features_in_", None) != len(GLOBAL_FEATURE_COLS):
            raise ValueError("Global artifact scaler feature count is incompatible")
        self.model = model
        self.model.eval()
        self.scaler = scaler
        self.manifest = manifest
        self.target_symbols = tuple(
            item["symbol"] for item in manifest["universe"]["targets"]
        )
        self.ticker_to_id = dict(manifest["universe"]["ticker_to_id"])
        self.context_symbol = next(
            item["symbol"]
            for item in manifest["universe"]["context"]
            if item["role"] == "implied_volatility"
        )
        self.seq_len = manifest["preprocessing"]["seq_len"]
        self.horizon = manifest["target"]["horizon_sessions"]

    def load_context(self, context) -> None:
        """Rehydrate model, scaler, and immutable contract from MLflow artifacts."""
        with open(context.artifacts["manifest"]) as manifest_file:
            manifest = json.load(manifest_file)
        components = manifest.get("components", {})
        if sha256_file(context.artifacts["model_state"]) != components.get(
            "model_state_sha256"
        ):
            raise ValueError("Global artifact model-state checksum mismatch")
        if sha256_file(context.artifacts["scaler"]) != components.get("scaler_sha256"):
            raise ValueError("Global artifact scaler checksum mismatch")
        if sha256_file(context.artifacts["benchmark"]) != manifest.get(
            "evaluation", {}
        ).get("benchmark_sha256"):
            raise ValueError("Global artifact evaluation-benchmark checksum mismatch")
        architecture = manifest["architecture"]
        model = GlobalVolatilityLSTM(
            input_size=architecture["input_size"],
            num_tickers=architecture["num_tickers"],
            embedding_dim=architecture["embedding_dim"],
            hidden_size=architecture["hidden_size"],
            num_layers=architecture["num_layers"],
            dropout=architecture["dropout"],
        )
        state_dict = torch.load(
            context.artifacts["model_state"],
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        scaler = joblib.load(context.artifacts["scaler"])
        self._initialize(model, scaler, manifest)

    def predict(self, context, model_input) -> pd.DataFrame:
        """Return positive five-session forecasts for requested known targets."""
        batch = build_global_inference_batch(
            model_input,
            target_symbols=self.target_symbols,
            context_symbol=self.context_symbol,
            ticker_to_id=self.ticker_to_id,
            seq_len=self.seq_len,
        )
        n_targets, seq_len, n_features = batch.features.shape
        scaled = self.scaler.transform(
            pd.DataFrame(
                batch.features.reshape(n_targets * seq_len, n_features),
                columns=GLOBAL_FEATURE_COLS,
            )
        ).reshape(n_targets, seq_len, n_features)
        features = torch.from_numpy(np.asarray(scaled, dtype=np.float32))
        ticker_ids = torch.from_numpy(batch.ticker_ids)
        self.model.eval()
        with torch.no_grad():
            forecasts = self.model.predict_volatility(features, ticker_ids).numpy()
        if not np.isfinite(forecasts).all() or (forecasts <= 0).any():
            raise ValueError("Global artifact produced invalid volatility forecasts")
        return pd.DataFrame(
            {
                "ticker": batch.tickers,
                "as_of_date": pd.DatetimeIndex(batch.as_of_dates),
                "horizon_sessions": self.horizon,
                "forecast": forecasts.astype(float),
            }
        )
