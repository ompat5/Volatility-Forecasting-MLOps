import os
import re
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field

from src.config import REPO_ROOT, load_config
from src.serving.global_api import (
    GlobalPredictRequest,
    GlobalPredictResponse,
    global_request_frame,
    requested_global_tickers,
    validate_global_forecaster,
    validated_global_predictions,
)
from src.serving.onnx_forecaster import ONNXVolatilityForecaster


class PredictRequest(BaseModel):
    prices: list[float] = Field(
        ...,
        min_length=90,
        description="List of adjusted close prices for at least the last 90 days",
    )


class PredictResponse(BaseModel):
    forecast: float
    horizon: int


_state = {}


def _load_forecaster():
    backend = os.getenv("MODEL_BACKEND", "mlflow").lower()
    if backend == "mlflow":
        model_uri = os.getenv("MODEL_URI", "models:/volatility-lstm/latest")
        if model_uri.startswith("models:/"):
            mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
        return mlflow.pyfunc.load_model(model_uri)
    if backend == "onnx":
        config = load_config()
        model_path = Path(
            os.getenv(
                "ONNX_MODEL_PATH",
                REPO_ROOT / "model" / "optimized" / "volatility_lstm_fp32.onnx",
            )
        )
        scaler_path = Path(
            os.getenv(
                "ONNX_SCALER_PATH",
                REPO_ROOT / "model" / "artifacts" / "scaler.pkl",
            )
        )
        threads = int(os.getenv("ONNX_NUM_THREADS", "1"))
        return ONNXVolatilityForecaster(
            model_path,
            scaler_path,
            seq_len=config.model.seq_len,
            threads=threads,
        )
    raise ValueError(
        f"Unsupported MODEL_BACKEND={backend!r}; expected 'mlflow' or 'onnx'"
    )


def _load_global_forecaster():
    """Load the opt-in global candidate without changing the AAPL default."""
    model_uri = os.getenv("GLOBAL_MODEL_URI")
    if model_uri is None or not model_uri.strip():
        return None
    model_uri = model_uri.strip()
    if model_uri.startswith("models:/"):
        expected_registry_uri = r"models:/global-volatility-lstm/[1-9]\d*"
        if re.fullmatch(expected_registry_uri, model_uri) is None:
            raise ValueError(
                "Registry-backed GLOBAL_MODEL_URI must select an explicit numeric "
                "global-volatility-lstm version"
            )
        mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    model = mlflow.pyfunc.load_model(model_uri)
    validate_global_forecaster(model)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.clear()
    try:
        _state["model"] = _load_forecaster()
        _state["global_model"] = _load_global_forecaster()
        yield
    finally:
        _state.clear()


app = FastAPI(title="Volatility Forecaster", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/global")
def global_health():
    ready = _state.get("global_model") is not None
    return {
        "status": "ok" if ready else "disabled",
        "configured": ready,
        "ready": ready,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    prices = pd.Series(request.prices)
    try:
        forecast = _state["model"].predict(prices)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not produce a forecast: {exc}"
        ) from exc
    horizon = load_config().data.horizon
    return PredictResponse(forecast=forecast, horizon=horizon)


@app.post("/predict/global", response_model=GlobalPredictResponse)
def predict_global(request: GlobalPredictRequest):
    model = _state.get("global_model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Global model is not configured; set GLOBAL_MODEL_URI to opt in",
        )
    frame = global_request_frame(request)
    try:
        raw_predictions = model.predict(frame)
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
