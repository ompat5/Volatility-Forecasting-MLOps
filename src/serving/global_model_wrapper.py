"""Self-contained MLflow wrapper for the global volatility model."""

from __future__ import annotations

import json

import joblib
import mlflow.pyfunc
import pandas as pd
import torch

from src.data.panel import GLOBAL_FEATURE_COLS
from src.features.global_inference import (
    build_global_forecast_frame,
    build_global_inference_batch,
    scale_global_inference_batch,
)
from src.models.global_artifact import (
    sha256_file,
    validate_global_artifact_manifest,
)
from src.models.global_lstm import GlobalVolatilityLSTM


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
        features = torch.from_numpy(
            scale_global_inference_batch(batch, self.scaler)
        )
        ticker_ids = torch.from_numpy(batch.ticker_ids)
        self.model.eval()
        with torch.no_grad():
            forecasts = self.model.predict_volatility(features, ticker_ids).numpy()
        return build_global_forecast_frame(
            batch,
            forecasts,
            horizon=self.horizon,
        )
