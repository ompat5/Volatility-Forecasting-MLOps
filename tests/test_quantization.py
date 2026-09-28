from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.lstm import VolatilityLSTM
from src.optimization.onnx_export import INPUT_NAME, export_lstm_to_onnx
from src.optimization.quantization import quantize_onnx_dynamic_int8


def test_dynamic_int8_quantizes_lstm_and_executes(tmp_path: Path):
    torch.manual_seed(0)
    model = VolatilityLSTM(
        input_size=4,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    model_path = tmp_path / "model.pt"
    fp32_path = tmp_path / "model_fp32.onnx"
    int8_path = tmp_path / "model_int8.onnx"
    torch.save(model, model_path)
    export_lstm_to_onnx(
        model_path, fp32_path, seq_len=30, input_size=4
    )

    quantize_onnx_dynamic_int8(fp32_path, int8_path)

    graph = onnx.load(int8_path).graph
    assert any(node.op_type == "DynamicQuantizeLSTM" for node in graph.node)
    assert int8_path.stat().st_size < fp32_path.stat().st_size

    inputs = np.random.default_rng(0).normal(size=(3, 30, 4)).astype(np.float32)
    session = ort.InferenceSession(
        str(int8_path), providers=["CPUExecutionProvider"]
    )
    output = session.run(None, {INPUT_NAME: inputs})[0]
    assert output.shape == (3,)
    assert np.isfinite(output).all()
