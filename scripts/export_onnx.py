"""Export the production LSTM core to a validated FP32 ONNX graph.

Run with: `uv run python -m scripts.export_onnx`
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_MODEL_CONFIG, REPO_ROOT, load_config
from src.features.realized_vol import FEATURE_COLS
from src.optimization.onnx_export import DEFAULT_OPSET, export_lstm_to_onnx

DEFAULT_MODEL_PATH = REPO_ROOT / "model" / "artifacts" / "model.pt"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "model" / "optimized" / "volatility_lstm_fp32.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = export_lstm_to_onnx(
        args.model_path,
        args.output_path,
        seq_len=config.model.seq_len,
        input_size=len(FEATURE_COLS),
        opset_version=args.opset,
    )
    size_kib = output_path.stat().st_size / 1024
    print(f"Exported validated FP32 ONNX model -> {output_path} ({size_kib:.1f} KiB)")


if __name__ == "__main__":
    main()
