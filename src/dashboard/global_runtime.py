"""Resolve the validated global candidate runtime used by its dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from src.config import REPO_ROOT
from src.optimization.global_onnx import export_global_lstm_to_onnx
from src.serving.global_api import validate_global_forecaster
from src.serving.global_onnx_forecaster import GlobalONNXVolatilityForecaster

GLOBAL_ONNX_NAME = "global_volatility_lstm_fp32.onnx"
GLOBAL_ONNX_MANIFEST_NAME = "global_volatility_lstm_fp32.json"


def global_dashboard_model_dir() -> Path:
    """Return an explicit candidate snapshot without falling back to AAPL."""
    configured = os.getenv("GLOBAL_DASHBOARD_MODEL_DIR")
    return Path(configured) if configured else REPO_ROOT / "global_model"


def load_global_dashboard_forecaster(
    model_dir: Path | None = None,
) -> GlobalONNXVolatilityForecaster:
    """Validate one candidate snapshot, export ONNX if absent, and load it."""
    model_dir = model_dir or global_dashboard_model_dir()
    artifacts_dir = model_dir / "artifacts"
    optimized_dir = model_dir / "optimized"
    state_path = artifacts_dir / "model_state.pt"
    scaler_path = artifacts_dir / "scaler.pkl"
    artifact_manifest_path = artifacts_dir / "manifest.json"
    model_path = optimized_dir / GLOBAL_ONNX_NAME
    export_manifest_path = optimized_dir / GLOBAL_ONNX_MANIFEST_NAME

    required = (state_path, scaler_path, artifact_manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete global candidate directory; missing: " + ", ".join(missing)
        )

    optimized_exists = (model_path.is_file(), export_manifest_path.is_file())
    if any(optimized_exists) and not all(optimized_exists):
        raise FileNotFoundError(
            "Incomplete global ONNX bundle; expected both the model and export "
            "manifest"
        )
    if not any(optimized_exists):
        export_global_lstm_to_onnx(
            model_state_path=state_path,
            artifact_manifest_path=artifact_manifest_path,
            scaler_path=scaler_path,
            output_path=model_path,
            export_manifest_path=export_manifest_path,
        )

    forecaster = GlobalONNXVolatilityForecaster(
        model_path,
        scaler_path,
        artifact_manifest_path,
        export_manifest_path,
        threads=1,
    )
    validate_global_forecaster(forecaster)
    return forecaster
