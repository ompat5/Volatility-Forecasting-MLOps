"""Export one global artifact snapshot to validated FP32 ONNX.

Run after an explicit registry snapshot has been exported:
`uv run python -m scripts.export_global_onnx`
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import REPO_ROOT
from src.optimization.global_onnx import export_global_lstm_to_onnx
from src.optimization.onnx_export import DEFAULT_OPSET

DEFAULT_ARTIFACT_DIR = REPO_ROOT / "global_model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--export-manifest-path", type=Path)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    output_path = args.output_path or (
        artifact_dir / "optimized" / "global_volatility_lstm_fp32.onnx"
    )
    export_manifest_path = args.export_manifest_path or (
        artifact_dir / "optimized" / "global_volatility_lstm_fp32.json"
    )
    result = export_global_lstm_to_onnx(
        model_state_path=artifact_dir / "artifacts" / "model_state.pt",
        artifact_manifest_path=artifact_dir / "artifacts" / "manifest.json",
        scaler_path=artifact_dir / "artifacts" / "scaler.pkl",
        output_path=output_path,
        export_manifest_path=export_manifest_path,
        opset_version=args.opset,
    )
    size_kib = result.model_path.stat().st_size / 1024
    print(
        f"Exported validated global FP32 ONNX model -> {result.model_path} "
        f"({size_kib:.1f} KiB)"
    )
    print(
        "All-ticker maximum absolute difference: "
        f"{result.parity['max_absolute_difference']:.3e}"
    )
    print(f"Wrote export manifest -> {result.manifest_path}")


if __name__ == "__main__":
    main()
