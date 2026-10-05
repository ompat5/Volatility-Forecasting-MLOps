"""Benchmark global PyTorch and FP32 ONNX inference across all 34 targets.

Run after `uv run python -m scripts.export_global_onnx`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src.config import REPO_ROOT
from src.data.ingest import DEFAULT_RAW_DIR, DEFAULT_TICKERS_CONFIG
from src.data.panel import load_adjusted_close
from src.data.universe import load_universe
from src.features.global_inference import (
    build_global_forecast_frame,
    build_global_inference_batch,
    scale_global_inference_batch,
)
from src.models.global_artifact import sha256_file
from src.optimization.benchmark import benchmark_callables
from src.optimization.global_onnx import (
    FEATURES_INPUT_NAME,
    GLOBAL_OUTPUT_NAME,
    TICKER_IDS_INPUT_NAME,
    load_global_lstm,
)
from src.serving.global_onnx_forecaster import GlobalONNXVolatilityForecaster

DEFAULT_ARTIFACT_DIR = REPO_ROOT / "global_model"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "benchmarks" / "global_onnx_fp32.json"


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def load_global_benchmark_prices(
    *,
    raw_dir: Path,
    observations: int,
) -> pd.DataFrame:
    """Load an equal recent raw-price window for every target plus VIX."""
    if observations <= 0:
        raise ValueError("observations must be positive")
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    frames: list[pd.DataFrame] = []
    for symbol in universe.all_symbols:
        prices = load_adjusted_close(symbol, raw_dir).tail(observations)
        if len(prices) < observations:
            raise ValueError(
                f"Need {observations} prices for {symbol}, found {len(prices)}"
            )
        frames.append(
            pd.DataFrame(
                {
                    "date": prices.index,
                    "ticker": symbol,
                    "adjusted_close": prices.to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def run_global_benchmark(
    *,
    artifact_dir: Path,
    model_input: pd.DataFrame,
    output_path: Path,
    core_warmup: int,
    core_iterations: int,
    end_to_end_warmup: int,
    end_to_end_iterations: int,
    threads: int,
) -> dict[str, object]:
    """Measure core and raw-history paths and persist reproducible evidence."""
    if threads <= 0:
        raise ValueError("threads must be positive")
    model_state_path = artifact_dir / "artifacts" / "model_state.pt"
    scaler_path = artifact_dir / "artifacts" / "scaler.pkl"
    artifact_manifest_path = artifact_dir / "artifacts" / "manifest.json"
    onnx_path = artifact_dir / "optimized" / "global_volatility_lstm_fp32.onnx"
    export_manifest_path = (
        artifact_dir / "optimized" / "global_volatility_lstm_fp32.json"
    )
    model, manifest = load_global_lstm(model_state_path, artifact_manifest_path)
    scaler = joblib.load(scaler_path)
    onnx_runtime = GlobalONNXVolatilityForecaster(
        onnx_path,
        scaler_path,
        artifact_manifest_path,
        export_manifest_path,
        threads=threads,
    )
    target_symbols = tuple(
        item["symbol"] for item in manifest["universe"]["targets"]
    )
    ticker_to_id = dict(manifest["universe"]["ticker_to_id"])
    context_symbol = manifest["input_contract"]["context_required"]
    seq_len = manifest["preprocessing"]["seq_len"]
    horizon = manifest["target"]["horizon_sessions"]
    batch = build_global_inference_batch(
        model_input,
        target_symbols=target_symbols,
        context_symbol=context_symbol,
        ticker_to_id=ticker_to_id,
        seq_len=seq_len,
    )
    if batch.tickers != target_symbols:
        raise ValueError("Global benchmark input must cover all target tickers")
    core_features = scale_global_inference_batch(batch, scaler)
    core_ticker_ids = np.ascontiguousarray(batch.ticker_ids, dtype=np.int64)
    core_tensor = torch.from_numpy(core_features)
    ticker_tensor = torch.from_numpy(core_ticker_ids)
    torch.set_num_threads(threads)

    def pytorch_core() -> np.ndarray:
        with torch.inference_mode():
            return model.predict_volatility(core_tensor, ticker_tensor).numpy()

    def onnx_core() -> np.ndarray:
        return onnx_runtime.session.run(
            [GLOBAL_OUTPUT_NAME],
            {
                FEATURES_INPUT_NAME: core_features,
                TICKER_IDS_INPUT_NAME: core_ticker_ids,
            },
        )[0]

    def pytorch_end_to_end() -> pd.DataFrame:
        runtime_batch = build_global_inference_batch(
            model_input,
            target_symbols=target_symbols,
            context_symbol=context_symbol,
            ticker_to_id=ticker_to_id,
            seq_len=seq_len,
        )
        features = scale_global_inference_batch(runtime_batch, scaler)
        with torch.inference_mode():
            forecasts = model.predict_volatility(
                torch.from_numpy(features),
                torch.from_numpy(runtime_batch.ticker_ids),
            ).numpy()
        return build_global_forecast_frame(
            runtime_batch,
            forecasts,
            horizon=horizon,
        )

    def onnx_end_to_end() -> pd.DataFrame:
        return onnx_runtime.predict(model_input)

    pytorch_output = pytorch_end_to_end()
    onnx_output = onnx_end_to_end()
    if onnx_output["ticker"].tolist() != list(target_symbols):
        raise ValueError("Global ONNX benchmark output lost target coverage")
    absolute = np.abs(
        onnx_output["forecast"].to_numpy()
        - pytorch_output["forecast"].to_numpy()
    )
    relative = absolute / np.maximum(
        np.abs(pytorch_output["forecast"].to_numpy()), 1e-12
    )
    latency = {
        "core": benchmark_callables(
            {"pytorch": pytorch_core, "onnx_fp32": onnx_core},
            warmup=core_warmup,
            iterations=core_iterations,
        ),
        "end_to_end": benchmark_callables(
            {
                "pytorch": pytorch_end_to_end,
                "onnx_fp32": onnx_end_to_end,
            },
            warmup=end_to_end_warmup,
            iterations=end_to_end_iterations,
        ),
    }
    per_ticker = [
        {
            "ticker": ticker,
            "pytorch_forecast": float(pytorch_forecast),
            "onnx_fp32_forecast": float(onnx_forecast),
            "absolute_difference": float(difference),
        }
        for ticker, pytorch_forecast, onnx_forecast, difference in zip(
            target_symbols,
            pytorch_output["forecast"],
            onnx_output["forecast"],
            absolute,
            strict=True,
        )
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_role": "global_candidate_onnx_fp32_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "cpu_threads_per_runtime": threads,
            "onnx_providers": onnx_runtime.session.get_providers(),
        },
        "config": {
            "core_warmup": core_warmup,
            "core_iterations": core_iterations,
            "end_to_end_warmup": end_to_end_warmup,
            "end_to_end_iterations": end_to_end_iterations,
            "core_features_shape": list(core_features.shape),
            "core_ticker_ids_shape": list(core_ticker_ids.shape),
            "input_rows": len(model_input),
            "target_count": len(target_symbols),
            "context_symbol": context_symbol,
            "as_of_start": min(batch.as_of_dates).isoformat(),
            "as_of_end": max(batch.as_of_dates).isoformat(),
        },
        "parity": {
            "all_target_embeddings_covered": True,
            "max_absolute_difference": float(absolute.max()),
            "mean_absolute_difference": float(absolute.mean()),
            "max_relative_difference": float(relative.max()),
            "per_ticker": per_ticker,
        },
        "artifacts": {
            "source_manifest": {
                "path": _relative_path(artifact_manifest_path),
                "sha256": sha256_file(artifact_manifest_path),
            },
            "pytorch_state": {
                "path": _relative_path(model_state_path),
                "sha256": sha256_file(model_state_path),
                "size_bytes": model_state_path.stat().st_size,
            },
            "onnx_fp32": {
                "path": _relative_path(onnx_path),
                "sha256": sha256_file(onnx_path),
                "size_bytes": onnx_path.stat().st_size,
            },
            "onnx_export_manifest": {
                "path": _relative_path(export_manifest_path),
                "sha256": sha256_file(export_manifest_path),
            },
        },
        "latency_ms": latency,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _print_summary(result: dict[str, object], output_path: Path) -> None:
    latency = result["latency_ms"]
    print("Runtime      Core median/p95 (ms)    End-to-end median/p95 (ms)")
    for runtime in ("pytorch", "onnx_fp32"):
        core = latency["core"][runtime]
        end_to_end = latency["end_to_end"][runtime]
        print(
            f"{runtime:<12} "
            f"{core['median_ms']:.4f}/{core['p95_ms']:.4f}"
            f"{'':>10} {end_to_end['median_ms']:.4f}/{end_to_end['p95_ms']:.4f}"
        )
    parity = result["parity"]
    print(
        "All-target maximum absolute difference: "
        f"{parity['max_absolute_difference']:.3e}"
    )
    print(f"Wrote benchmark report -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--observations", type=int, default=120)
    parser.add_argument("--core-warmup", type=int, default=100)
    parser.add_argument("--core-iterations", type=int, default=1_000)
    parser.add_argument("--end-to-end-warmup", type=int, default=5)
    parser.add_argument("--end-to-end-iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    model_input = load_global_benchmark_prices(
        raw_dir=args.raw_dir,
        observations=args.observations,
    )
    result = run_global_benchmark(
        artifact_dir=args.artifact_dir,
        model_input=model_input,
        output_path=args.output_path,
        core_warmup=args.core_warmup,
        core_iterations=args.core_iterations,
        end_to_end_warmup=args.end_to_end_warmup,
        end_to_end_iterations=args.end_to_end_iterations,
        threads=args.threads,
    )
    _print_summary(result, args.output_path)


if __name__ == "__main__":
    main()
