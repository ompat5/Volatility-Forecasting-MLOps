"""MLflow pyfunc wrapper: bundles the LSTM + its scaler + inference glue as one deployable unit."""

import joblib
import pandas as pd
import torch
import mlflow.pyfunc

from src.features.realized_vol import FEATURE_COLS, build_inference_features


class VolatilityForecaster(mlflow.pyfunc.PythonModel):
    """Raw prices in, one 5-day realized-vol forecast out. Scaling is inseparable from the model."""

    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def load_context(self, context):
        """Runs once when MLflow loads the model. Rehydrate model + scaler from logged artifacts."""
        self.model = torch.load(context.artifacts["model"], weights_only=False)
        self.model.eval()
        self.scaler = joblib.load(context.artifacts["scaler"])

    def predict(self, context, model_input) -> float:
        """One raw adjusted-close price series -> the latest 5-day realized-vol forecast."""
        prices = pd.Series(model_input)

        features = build_inference_features(prices)

        scaled = self.scaler.transform(features[FEATURE_COLS])

        window = scaled[-self.seq_len:]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            return self.model(x).item()
