import mlflow
import pandas as pd

from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.config import REPO_ROOT, load_config

class PredictRequest(BaseModel):
    prices: list[float] = Field(..., min_length=90, description="List of adjusted close prices for at least the last 90 days")


class PredictResponse(BaseModel):
    forecast: float
    horizon: int


MODEL_URI = "models:/volatility-lstm/latest"
_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    _state["model"] = mlflow.pyfunc.load_model(MODEL_URI)
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
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not produce a forecast: {e}")
    horizon = load_config().data.horizon
    return PredictResponse(forecast=forecast, horizon=horizon)
