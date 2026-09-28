"""Raw-price forecaster backed by an ONNX Runtime LSTM session."""

from __future__ import annotations

from pathlib import Path

import joblib
import onnxruntime as ort
import pandas as pd

from src.features.realized_vol import FEATURE_COLS
from src.optimization.inference import build_scaled_window
from src.optimization.onnx_export import INPUT_NAME, OUTPUT_NAME


class ONNXVolatilityForecaster:
    """Bundle FP32 ONNX inference with the fitted training scaler."""

    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        *,
        seq_len: int,
        threads: int = 1,
    ) -> None:
        if threads <= 0:
            raise ValueError("threads must be positive")

        self.seq_len = seq_len
        self.scaler = joblib.load(scaler_path)

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._validate_contract()

    def _validate_contract(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        expected_tail = [self.seq_len, len(FEATURE_COLS)]
        if len(inputs) != 1 or inputs[0].name != INPUT_NAME:
            raise ValueError(f"Expected one ONNX input named {INPUT_NAME!r}")
        if inputs[0].shape[1:] != expected_tail:
            raise ValueError(
                f"Expected ONNX input tail {expected_tail}, got {inputs[0].shape[1:]}"
            )
        if len(outputs) != 1 or outputs[0].name != OUTPUT_NAME:
            raise ValueError(f"Expected one ONNX output named {OUTPUT_NAME!r}")

    def predict(self, prices: pd.Series) -> float:
        """Build leakage-safe features and return one raw volatility forecast."""
        inputs = build_scaled_window(prices, self.scaler, self.seq_len)
        output = self.session.run([OUTPUT_NAME], {INPUT_NAME: inputs})[0]
        return float(output[0])
