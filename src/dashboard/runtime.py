"""Resolve or retrieve the model artifacts used by the dashboard."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from src.config import DEFAULT_MODEL_CONFIG, REPO_ROOT, load_config
from src.features.realized_vol import FEATURE_COLS
from src.optimization.onnx_export import export_lstm_to_onnx
from src.serving.artifacts import download_model
from src.serving.onnx_forecaster import ONNXVolatilityForecaster

MONITORING_CONFIG = REPO_ROOT / "configs" / "monitoring.yaml"


def dashboard_model_dir() -> Path:
    """Prefer an explicit/local model, otherwise use ephemeral cloud storage."""
    configured = os.getenv("DASHBOARD_MODEL_DIR")
    if configured:
        return Path(configured)
    local_model = REPO_ROOT / "model"
    if (local_model / "artifacts" / "model.pt").is_file():
        return local_model
    return Path(tempfile.gettempdir()) / "aapl-volatility-dashboard-model"


def load_dashboard_forecaster(
    model_dir: Path | None = None,
) -> ONNXVolatilityForecaster:
    """Ensure immutable artifacts exist, export FP32 ONNX, and load it once."""
    model_dir = model_dir or dashboard_model_dir()
    torch_path = model_dir / "artifacts" / "model.pt"
    scaler_path = model_dir / "artifacts" / "scaler.pkl"
    onnx_path = model_dir / "optimized" / "volatility_lstm_fp32.onnx"

    if not torch_path.is_file() or not scaler_path.is_file():
        if model_dir.exists():
            raise FileNotFoundError(
                f"Incomplete model directory at {model_dir}; expected "
                "artifacts/model.pt and artifacts/scaler.pkl"
            )
        download_model(MONITORING_CONFIG, model_dir)

    config = load_config(DEFAULT_MODEL_CONFIG)
    if not onnx_path.is_file():
        export_lstm_to_onnx(
            torch_path,
            onnx_path,
            seq_len=config.model.seq_len,
            input_size=len(FEATURE_COLS),
        )

    return ONNXVolatilityForecaster(
        onnx_path,
        scaler_path,
        seq_len=config.model.seq_len,
        threads=1,
    )
