from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.models.lstm import VolatilityLSTM
from src.optimization.onnx_export import INPUT_NAME, export_lstm_to_onnx

SEQ_LEN = 30
INPUT_SIZE = 4


def _saved_test_model(tmp_path: Path) -> tuple[VolatilityLSTM, Path]:
    torch.manual_seed(0)
    model = VolatilityLSTM(
        input_size=INPUT_SIZE,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    model_path = tmp_path / "model.pt"
    torch.save(model, model_path)
    return model, model_path


def test_export_creates_valid_onnx_graph(tmp_path: Path):
    _, model_path = _saved_test_model(tmp_path)
    output_path = tmp_path / "model.onnx"

    result = export_lstm_to_onnx(
        model_path,
        output_path,
        seq_len=SEQ_LEN,
        input_size=INPUT_SIZE,
    )

    assert result == output_path
    assert output_path.is_file()
    onnx.checker.check_model(onnx.load(output_path))


def test_onnx_matches_pytorch_for_dynamic_batch(tmp_path: Path):
    model, model_path = _saved_test_model(tmp_path)
    output_path = tmp_path / "model.onnx"
    export_lstm_to_onnx(
        model_path,
        output_path,
        seq_len=SEQ_LEN,
        input_size=INPUT_SIZE,
    )

    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(3, SEQ_LEN, INPUT_SIZE)).astype(np.float32)
    with torch.no_grad():
        pytorch_output = model(torch.from_numpy(inputs)).numpy()

    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    onnx_output = session.run(None, {INPUT_NAME: inputs})[0]

    np.testing.assert_allclose(onnx_output, pytorch_output, rtol=1e-4, atol=1e-6)
