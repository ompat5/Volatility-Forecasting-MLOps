"""Resolve the validated global runtime used by the public dashboard."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from src.config import REPO_ROOT
from src.models.global_release import download_global_release
from src.optimization.global_onnx import export_global_lstm_to_onnx
from src.serving.global_api import validate_global_forecaster
from src.serving.global_onnx_forecaster import GlobalONNXVolatilityForecaster

GLOBAL_ONNX_NAME = "global_volatility_lstm_fp32.onnx"
GLOBAL_ONNX_MANIFEST_NAME = "global_volatility_lstm_fp32.json"
GLOBAL_RELEASE_CONFIG = REPO_ROOT / "configs" / "global_release.yaml"


def global_dashboard_model_dir() -> Path:
    """Prefer an explicit/local model, otherwise use verified temp storage."""
    configured = os.getenv("GLOBAL_DASHBOARD_MODEL_DIR")
    if configured:
        return Path(configured)
    local_model = REPO_ROOT / "global_model"
    if (local_model / "artifacts" / "manifest.json").is_file():
        return local_model
    return Path(tempfile.gettempdir()) / "global-volatility-dashboard-model"


def load_global_dashboard_forecaster(
    model_dir: Path | None = None,
) -> GlobalONNXVolatilityForecaster:
    """Validate one global snapshot, export ONNX if absent, and load it."""
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
        if model_dir.exists():
            raise FileNotFoundError(
                "Incomplete global model directory; missing: " + ", ".join(missing)
            )
        download_global_release(GLOBAL_RELEASE_CONFIG, model_dir)

    optimized_exists = (model_path.is_file(), export_manifest_path.is_file())
    if any(optimized_exists) and not all(optimized_exists):
        raise FileNotFoundError(
            "Incomplete global ONNX bundle; expected both the model and export manifest"
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
