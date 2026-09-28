import os
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field

from src.config import REPO_ROOT, load_config
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = _load_forecaster()
    yield
    _state.clear()


app = FastAPI(title="Volatility Forecaster", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


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
