"""Export the trained PyTorch LSTM core to ONNX."""

from __future__ import annotations

from pathlib import Path

import onnx
import torch

from src.models.lstm import VolatilityLSTM

INPUT_NAME = "features"
OUTPUT_NAME = "forecast"
DEFAULT_OPSET = 17


def load_lstm(model_path: Path) -> VolatilityLSTM:
    """Load a trusted, locally produced LSTM artifact onto the CPU."""
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(model, VolatilityLSTM):
        raise TypeError(
            f"Expected {VolatilityLSTM.__name__}, got {type(model).__name__}"
        )
    model.eval()
    return model


def export_lstm_to_onnx(
    model_path: Path,
    output_path: Path,
    *,
    seq_len: int,
    input_size: int,
    opset_version: int = DEFAULT_OPSET,
) -> Path:
    """Export and validate an FP32 ONNX graph for the LSTM core.

    Sequence length and feature count stay fixed to the trained model contract.
    Only the batch dimension is dynamic.
    """
    if seq_len <= 0 or input_size <= 0:
        raise ValueError("seq_len and input_size must be positive")

    model = load_lstm(model_path)
    if model.lstm.input_size != input_size:
        raise ValueError(
            f"Model expects {model.lstm.input_size} features, got {input_size}"
        )

    example = torch.zeros(1, seq_len, input_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAME: {0: "batch_size"},
                OUTPUT_NAME: {0: "batch_size"},
            },
        )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    return output_path
