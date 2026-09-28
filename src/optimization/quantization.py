"""Dynamic INT8 quantization for ONNX inference artifacts."""

from __future__ import annotations

from pathlib import Path
import tempfile

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def quantize_onnx_dynamic_int8(fp32_path: Path, int8_path: Path) -> Path:
    """Quantize eligible LSTM and linear weights, then validate the graph."""
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    # ONNX Runtime recommends shape inference and graph optimization before
    # quantization. Keep the intermediate graph temporary: the source FP32 and
    # final INT8 artifacts are the only meaningful outputs of this experiment.
    with tempfile.TemporaryDirectory() as temporary_dir:
        prepared_path = Path(temporary_dir) / "prepared.onnx"
        quant_pre_process(str(fp32_path), str(prepared_path))
        quantize_dynamic(
            model_input=str(prepared_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
    onnx.checker.check_model(onnx.load(int8_path))
    return int8_path
