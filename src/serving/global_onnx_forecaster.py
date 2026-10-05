"""Raw-price global forecaster backed by the validated FP32 ONNX graph."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.panel import GLOBAL_FEATURE_COLS
from src.features.global_inference import (
    build_global_forecast_frame,
    build_global_inference_batch,
    scale_global_inference_batch,
)
from src.optimization.global_onnx import (
    FEATURES_INPUT_NAME,
    GLOBAL_OUTPUT_NAME,
    TICKER_IDS_INPUT_NAME,
    create_global_onnx_session,
    validate_global_onnx_bundle,
    validate_global_onnx_session,
)


class GlobalONNXVolatilityForecaster:
    """Keep global preprocessing in Python and execute only the core in ONNX."""

    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        artifact_manifest_path: Path,
        export_manifest_path: Path,
        *,
        threads: int = 1,
    ) -> None:
        artifact_manifest, _ = validate_global_onnx_bundle(
            model_path=model_path,
            scaler_path=scaler_path,
            artifact_manifest_path=artifact_manifest_path,
            export_manifest_path=export_manifest_path,
        )
        self.scaler = joblib.load(scaler_path)
        if getattr(self.scaler, "n_features_in_", None) != len(
            GLOBAL_FEATURE_COLS
        ):
            raise ValueError("Global ONNX scaler feature count is incompatible")
        self.manifest = artifact_manifest
        self.target_symbols = tuple(
            item["symbol"] for item in artifact_manifest["universe"]["targets"]
        )
        self.ticker_to_id = dict(
            artifact_manifest["universe"]["ticker_to_id"]
        )
        self.context_symbol = next(
            item["symbol"]
            for item in artifact_manifest["universe"]["context"]
            if item["role"] == "implied_volatility"
        )
        self.seq_len = artifact_manifest["preprocessing"]["seq_len"]
        self.horizon = artifact_manifest["target"]["horizon_sessions"]
        self.global_metadata = {
            "artifact_role": artifact_manifest["artifact_role"],
            "target_count": len(self.target_symbols),
            "context_symbols": [self.context_symbol],
        }
        self.session = create_global_onnx_session(model_path, threads=threads)
        validate_global_onnx_session(
            self.session,
            seq_len=self.seq_len,
            input_size=len(GLOBAL_FEATURE_COLS),
        )

    def predict_scaled(
        self,
        features: np.ndarray,
        ticker_ids: np.ndarray,
    ) -> np.ndarray:
        """Run a preprocessed batch while enforcing tensor and ID contracts."""
        feature_array = np.ascontiguousarray(features, dtype=np.float32)
        ticker_array = np.ascontiguousarray(ticker_ids, dtype=np.int64)
        expected_tail = (self.seq_len, len(GLOBAL_FEATURE_COLS))
        if feature_array.ndim != 3 or feature_array.shape[1:] != expected_tail:
            raise ValueError(
                f"Global ONNX features must have shape (batch, {expected_tail[0]}, "
                f"{expected_tail[1]})"
            )
        if ticker_array.ndim != 1 or len(ticker_array) != len(feature_array):
            raise ValueError("Global ONNX ticker IDs must match the feature batch")
        if not np.isfinite(feature_array).all():
            raise ValueError("Global ONNX features must be finite")
        if (
            (ticker_array < 0).any()
            or (ticker_array >= len(self.target_symbols)).any()
        ):
            raise ValueError("Global ONNX ticker IDs are outside the vocabulary")
        forecasts = self.session.run(
            [GLOBAL_OUTPUT_NAME],
            {
                FEATURES_INPUT_NAME: feature_array,
                TICKER_IDS_INPUT_NAME: ticker_array,
            },
        )[0]
        values = np.asarray(forecasts, dtype=float).reshape(-1)
        if values.shape != (len(feature_array),):
            raise ValueError("Global ONNX output batch size is incompatible")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Global ONNX produced invalid volatility forecasts")
        return values

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """Build shared features and return one forecast per requested target."""
        batch = build_global_inference_batch(
            model_input,
            target_symbols=self.target_symbols,
            context_symbol=self.context_symbol,
            ticker_to_id=self.ticker_to_id,
            seq_len=self.seq_len,
        )
        features = scale_global_inference_batch(batch, self.scaler)
        forecasts = self.predict_scaled(features, batch.ticker_ids)
        return build_global_forecast_frame(
            batch,
            forecasts,
            horizon=self.horizon,
        )
