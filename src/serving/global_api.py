"""HTTP schemas and system-contract validation for global inference."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.config import load_global_config
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe


class GlobalPriceObservation(BaseModel):
    """One raw adjusted-close observation for a target or context series."""

    model_config = ConfigDict(str_strip_whitespace=True)

    date: date
    ticker: str = Field(min_length=1)
    adjusted_close: float = Field(gt=0, allow_inf_nan=False)


class GlobalPredictRequest(BaseModel):
    observations: list[GlobalPriceObservation] = Field(min_length=1)


class GlobalPrediction(BaseModel):
    ticker: str
    as_of_date: date
    horizon_sessions: int = Field(gt=0)
    forecast: float = Field(gt=0, allow_inf_nan=False)


class GlobalPredictResponse(BaseModel):
    predictions: list[GlobalPrediction]


def validate_global_forecaster(model) -> None:
    """Require the loaded artifact to match the complete repository contract."""
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    config = load_global_config()
    direct_metadata = getattr(model, "global_metadata", None)
    if direct_metadata is not None:
        model_metadata = direct_metadata
        runtime = model
    else:
        model_metadata = getattr(getattr(model, "metadata", None), "metadata", None)
    expected_metadata = {
        "artifact_role": "global_candidate",
        "target_count": len(universe.target_symbols),
        "context_symbols": list(universe.context_symbols),
    }
    if not isinstance(model_metadata, dict) or any(
        model_metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise ValueError("Configured runtime is not a compatible global model")
    if direct_metadata is None:
        try:
            runtime = model.unwrap_python_model()
        except (AttributeError, NotImplementedError) as exc:
            raise ValueError(
                "MODEL_URI must load an inspectable global MLflow pyfunc"
            ) from exc
    if tuple(getattr(runtime, "target_symbols", ())) != universe.target_symbols:
        raise ValueError("Global artifact target universe is incompatible")
    if getattr(runtime, "context_symbol", None) != universe.context_symbol(
        "implied_volatility"
    ):
        raise ValueError("Global artifact context universe is incompatible")
    if getattr(runtime, "horizon", None) != config.data.horizon:
        raise ValueError("Global artifact forecast horizon is incompatible")


def global_request_frame(request: GlobalPredictRequest) -> pd.DataFrame:
    """Convert validated HTTP observations to the artifact's DataFrame input."""
    return pd.DataFrame(
        [observation.model_dump(mode="json") for observation in request.observations]
    )


def requested_global_tickers(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return requested targets in the authoritative universe order."""
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    supplied_tickers = set(frame["ticker"])
    return tuple(
        ticker for ticker in universe.target_symbols if ticker in supplied_tickers
    )


def validated_global_predictions(
    raw_predictions: object,
    *,
    requested_tickers: tuple[str, ...],
) -> list[GlobalPrediction]:
    """Validate artifact output before exposing it as an HTTP response."""
    expected_columns = [
        "ticker",
        "as_of_date",
        "horizon_sessions",
        "forecast",
    ]
    if (
        not isinstance(raw_predictions, pd.DataFrame)
        or list(raw_predictions.columns) != expected_columns
    ):
        raise ValueError(f"Expected global model output columns {expected_columns}")
    if raw_predictions["ticker"].astype(str).tolist() != list(requested_tickers):
        raise ValueError(
            "Global model must return each requested target exactly once in "
            "universe order"
        )
    expected_horizon = load_global_config().data.horizon
    horizons = pd.to_numeric(raw_predictions["horizon_sessions"], errors="coerce")
    if horizons.isna().any() or not (horizons == expected_horizon).all():
        raise ValueError("Global model returned an incompatible forecast horizon")
    forecasts = pd.to_numeric(raw_predictions["forecast"], errors="coerce")
    if not np.isfinite(forecasts.to_numpy(dtype=float)).all() or (forecasts <= 0).any():
        raise ValueError("Global model returned invalid volatility forecasts")
    as_of_dates = pd.to_datetime(raw_predictions["as_of_date"], errors="raise")
    if as_of_dates.isna().any():
        raise ValueError("Global model returned missing as-of dates")
    return [
        GlobalPrediction(
            ticker=ticker,
            as_of_date=as_of.date(),
            horizon_sessions=int(horizon),
            forecast=float(forecast),
        )
        for ticker, as_of, horizon, forecast in zip(
            raw_predictions["ticker"],
            as_of_dates,
            horizons,
            forecasts,
            strict=True,
        )
    ]
