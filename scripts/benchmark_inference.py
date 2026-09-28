"""Benchmark production PyTorch and FP32 ONNX inference on AAPL data.

Run after exporting both artifacts:
`uv run python -m scripts.export_onnx`
`uv run python -m scripts.benchmark_inference`
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import onnxruntime as ort
import torch

from src.config import DEFAULT_MODEL_CONFIG, REPO_ROOT, load_config
from src.optimization.benchmark import benchmark_callables
from src.optimization.inference import build_scaled_window, load_adjusted_close
from src.optimization.onnx_export import INPUT_NAME, load_lstm

DEFAULT_MODEL_PATH = REPO_ROOT / "model" / "artifacts" / "model.pt"
DEFAULT_SCALER_PATH = REPO_ROOT / "model" / "artifacts" / "scaler.pkl"
DEFAULT_ONNX_PATH = (
    REPO_ROOT / "model" / "optimized" / "volatility_lstm_fp32.onnx"
)
DEFAULT_PRICES_PATH = REPO_ROOT / "data" / "raw" / "AAPL.parquet"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "benchmarks" / "onnx_fp32.json"


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def run_benchmark(
    *,
    model_path: Path,
    scaler_path: Path,
    onnx_path: Path,
    prices_path: Path,
    output_path: Path,
    config_path: Path,
    observations: int,
    warmup: int,
    iterations: int,
    threads: int,
) -> dict[str, object]:
    """Run core and end-to-end benchmarks and persist their JSON report."""
    if threads <= 0:
        raise ValueError("threads must be positive")

    config = load_config(config_path)
    model = load_lstm(model_path)
    scaler = joblib.load(scaler_path)
    prices = load_adjusted_close(prices_path, observations)

    torch.set_num_threads(threads)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = threads
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    core_array = build_scaled_window(prices, scaler, config.model.seq_len)
    core_tensor = torch.from_numpy(core_array)

    def pytorch_core() -> float:
        with torch.inference_mode():
            return float(model(core_tensor)[0])

    def onnx_core() -> float:
        return float(session.run(None, {INPUT_NAME: core_array})[0][0])

    def pytorch_end_to_end() -> float:
        array = build_scaled_window(prices, scaler, config.model.seq_len)
        with torch.inference_mode():
            return float(model(torch.from_numpy(array))[0])

    def onnx_end_to_end() -> float:
        array = build_scaled_window(prices, scaler, config.model.seq_len)
        return float(session.run(None, {INPUT_NAME: array})[0][0])

    pytorch_forecast = pytorch_end_to_end()
    onnx_forecast = onnx_end_to_end()
    latency = {
        "core": benchmark_callables(
            {"pytorch": pytorch_core, "onnx_fp32": onnx_core},
            warmup=warmup,
            iterations=iterations,
        ),
        "end_to_end": benchmark_callables(
            {
                "pytorch": pytorch_end_to_end,
                "onnx_fp32": onnx_end_to_end,
            },
            warmup=warmup,
            iterations=iterations,
        ),
    }

    result: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "cpu_threads_per_runtime": threads,
            "onnx_providers": session.get_providers(),
        },
        "config": {
            "warmup": warmup,
            "iterations": iterations,
            "input_shape": list(core_array.shape),
            "price_observations": len(prices),
            "price_start": prices.index.min().isoformat(),
            "price_end": prices.index.max().isoformat(),
        },
        "parity": {
            "pytorch_forecast": pytorch_forecast,
            "onnx_fp32_forecast": onnx_forecast,
            "absolute_difference": abs(pytorch_forecast - onnx_forecast),
        },
        "artifacts": {
            "pytorch": {
                "path": _relative_path(model_path),
                "size_bytes": model_path.stat().st_size,
            },
            "onnx_fp32": {
                "path": _relative_path(onnx_path),
                "size_bytes": onnx_path.stat().st_size,
            },
        },
        "latency_ms": latency,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _print_summary(result: dict[str, object], output_path: Path) -> None:
    latency = result["latency_ms"]
    artifacts = result["artifacts"]
    parity = result["parity"]
    print("Runtime      Core median/p95 (ms)    End-to-end median/p95 (ms)")
    for runtime in ("pytorch", "onnx_fp32"):
        core = latency["core"][runtime]
        end_to_end = latency["end_to_end"][runtime]
        print(
            f"{runtime:<12} "
            f"{core['median_ms']:.4f}/{core['p95_ms']:.4f}"
            f"{'':>10} {end_to_end['median_ms']:.4f}/{end_to_end['p95_ms']:.4f}"
        )
    print(
        "Artifact size: "
        f"PyTorch={artifacts['pytorch']['size_bytes']} bytes, "
        f"ONNX={artifacts['onnx_fp32']['size_bytes']} bytes"
    )
    print(f"Forecast absolute difference: {parity['absolute_difference']:.3e}")
    print(f"Wrote benchmark report -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--prices-path", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--observations", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    result = run_benchmark(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        onnx_path=args.onnx_path,
        prices_path=args.prices_path,
        output_path=args.output_path,
        config_path=args.config,
        observations=args.observations,
        warmup=args.warmup,
        iterations=args.iterations,
        threads=args.threads,
    )
    _print_summary(result, args.output_path)


if __name__ == "__main__":
    main()
