"""Serve the complete 34-target global volatility model.

The public API has one deployment contract: long-form adjusted-close histories
for one or more known targets plus the shared VIX context. It intentionally
does not retain the former AAPL price-list endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
import mlflow

from src.config import REPO_ROOT
from src.serving.global_api import (
    GlobalPredictRequest,
    GlobalPredictResponse,
    global_request_frame,
    requested_global_tickers,
    validate_global_forecaster,
    validated_global_predictions,
)
from src.serving.global_onnx_forecaster import GlobalONNXVolatilityForecaster

_state: dict[str, object] = {}


def _load_forecaster():
    """Load the one configured global runtime and verify its full universe."""
    backend = os.getenv("MODEL_BACKEND", "mlflow").lower()
    if backend == "mlflow":
        model_uri = os.getenv("MODEL_URI", "/app/model").strip()
        if model_uri.startswith("models:/"):
            expected_registry_uri = r"models:/global-volatility-lstm/[1-9]\d*"
            if re.fullmatch(expected_registry_uri, model_uri) is None:
                raise ValueError(
                    "Registry-backed MODEL_URI must select an explicit numeric "
                    "global-volatility-lstm version"
                )
            mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
        model = mlflow.pyfunc.load_model(model_uri)
    elif backend == "onnx":
        artifact_dir = Path(os.getenv("MODEL_DIR", REPO_ROOT / "model"))
        model = GlobalONNXVolatilityForecaster(
            Path(
                os.getenv(
                    "ONNX_MODEL_PATH",
                    artifact_dir / "optimized" / "global_volatility_lstm_fp32.onnx",
                )
            ),
            Path(
                os.getenv("ONNX_SCALER_PATH", artifact_dir / "artifacts" / "scaler.pkl")
            ),
            Path(
                os.getenv(
                    "ONNX_ARTIFACT_MANIFEST_PATH",
                    artifact_dir / "artifacts" / "manifest.json",
                )
            ),
            Path(
                os.getenv(
                    "ONNX_EXPORT_MANIFEST_PATH",
                    artifact_dir / "optimized" / "global_volatility_lstm_fp32.json",
                )
            ),
            threads=int(os.getenv("ONNX_NUM_THREADS", "1")),
        )
    else:
        raise ValueError(
            f"Unsupported MODEL_BACKEND={backend!r}; expected 'mlflow' or 'onnx'"
        )
    validate_global_forecaster(model)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.clear()
    try:
        _state["model"] = _load_forecaster()
        yield
    finally:
        _state.clear()


app = FastAPI(title="Global Volatility Forecaster", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_scope": "global_34_target", "ready": True}


@app.post("/predict", response_model=GlobalPredictResponse)
def predict(request: GlobalPredictRequest):
    """Forecast every supplied target using its shared VIX context."""
    frame = global_request_frame(request)
    try:
        raw_predictions = _state["model"].predict(frame)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not produce global forecasts: {exc}",
        ) from exc

    requested_tickers = requested_global_tickers(frame)
    try:
        predictions = validated_global_predictions(
            raw_predictions,
            requested_tickers=requested_tickers,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Global model returned an invalid response: {exc}",
        ) from exc
    return GlobalPredictResponse(predictions=predictions)
