"""Quantize FP32 ONNX to INT8 and evaluate whether the trade-off is worthwhile.

Run with: `uv run python -m scripts.quantize_onnx`
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import onnxruntime as ort
import pandas as pd

from src.config import DEFAULT_MODEL_CONFIG, REPO_ROOT, load_config
from src.eval.metrics import mae, qlike, rmse
from src.features.realized_vol import TARGET_COL, build_features
from src.optimization.benchmark import benchmark_callables
from src.optimization.inference import build_scaled_window, load_adjusted_close
from src.optimization.onnx_export import INPUT_NAME
from src.optimization.quantization import quantize_onnx_dynamic_int8

DEFAULT_FP32_PATH = (
    REPO_ROOT / "model" / "optimized" / "volatility_lstm_fp32.onnx"
)
DEFAULT_INT8_PATH = (
    REPO_ROOT / "model" / "optimized" / "volatility_lstm_int8.onnx"
)
DEFAULT_SCALER_PATH = REPO_ROOT / "model" / "artifacts" / "scaler.pkl"
DEFAULT_PRICES_PATH = REPO_ROOT / "data" / "raw" / "AAPL.parquet"
DEFAULT_REPORT_PATH = REPO_ROOT / "benchmarks" / "onnx_int8.json"


def _session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _forecast(
    session: ort.InferenceSession,
    prices: pd.Series,
    scaler,
    seq_len: int,
) -> float:
    inputs = build_scaled_window(prices, scaler, seq_len)
    return float(session.run(None, {INPUT_NAME: inputs})[0][0])


def _metric_summary(actual: pd.Series, forecast: pd.Series) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "rmse": rmse(actual, forecast),
        "mae": mae(actual, forecast),
        "qlike": None,
    }
    if (forecast > 0).all():
        metrics["qlike"] = qlike(actual, forecast)
    return metrics


def run_quantization_experiment(
    *,
    fp32_path: Path,
    int8_path: Path,
    scaler_path: Path,
    prices_path: Path,
    config_path: Path,
    report_path: Path,
    evaluation_window: int,
    price_window: int,
    warmup: int,
    iterations: int,
    threads: int,
) -> dict[str, object]:
    """Quantize, evaluate a fixed slice, benchmark, and persist the evidence."""
    if evaluation_window <= 0 or price_window <= 0 or threads <= 0:
        raise ValueError("Window sizes and threads must be positive")

    quantize_onnx_dynamic_int8(fp32_path, int8_path)
    config = load_config(config_path)
    scaler = joblib.load(scaler_path)
    prices = load_adjusted_close(prices_path)
    features = build_features(prices, config.data.horizon)
    evaluation_dates = features.index[-evaluation_window:]
    if len(evaluation_dates) < evaluation_window:
        raise ValueError(
            f"Need {evaluation_window} evaluation rows, found {len(evaluation_dates)}"
        )

    sessions = {
        "onnx_fp32": _session(fp32_path, threads),
        "onnx_int8": _session(int8_path, threads),
    }
    predictions = {
        runtime: pd.Series(
            [
                _forecast(
                    session,
                    prices.loc[:as_of].tail(price_window),
                    scaler,
                    config.model.seq_len,
                )
                for as_of in evaluation_dates
            ],
            index=evaluation_dates,
        )
        for runtime, session in sessions.items()
    }
    actual = features.loc[evaluation_dates, TARGET_COL]
    metrics = {
        runtime: _metric_summary(actual, forecast)
        for runtime, forecast in predictions.items()
    }
    metric_delta = {
        metric: (
            None
            if metrics["onnx_fp32"][metric] is None
            or metrics["onnx_int8"][metric] is None
            else metrics["onnx_int8"][metric] - metrics["onnx_fp32"][metric]
        )
        for metric in ("rmse", "mae", "qlike")
    }
    prediction_delta = predictions["onnx_int8"] - predictions["onnx_fp32"]

    latest_prices = prices.tail(price_window)
    latest_input = build_scaled_window(
        latest_prices, scaler, config.model.seq_len
    )
    core_calls = {
        runtime: (
            lambda current_session=session: current_session.run(
                None, {INPUT_NAME: latest_input}
            )
        )
        for runtime, session in sessions.items()
    }
    end_to_end_calls = {
        runtime: (
            lambda current_session=session: _forecast(
                current_session,
                latest_prices,
                scaler,
                config.model.seq_len,
            )
        )
        for runtime, session in sessions.items()
    }
    latency = {
        "core": benchmark_callables(
            core_calls, warmup=warmup, iterations=iterations
        ),
        "end_to_end": benchmark_callables(
            end_to_end_calls, warmup=warmup, iterations=iterations
        ),
    }

    result: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "cpu_threads_per_runtime": threads,
        },
        "config": {
            "evaluation_observations": len(evaluation_dates),
            "evaluation_start": evaluation_dates.min().isoformat(),
            "evaluation_end": evaluation_dates.max().isoformat(),
            "price_window": price_window,
            "warmup": warmup,
            "iterations": iterations,
            "input_shape": list(latest_input.shape),
        },
        "artifacts": {
            "onnx_fp32_size_bytes": fp32_path.stat().st_size,
            "onnx_int8_size_bytes": int8_path.stat().st_size,
        },
        "prediction_delta": {
            "mean_absolute": float(prediction_delta.abs().mean()),
            "max_absolute": float(prediction_delta.abs().max()),
        },
        "metrics": metrics,
        "metric_delta_int8_minus_fp32": metric_delta,
        "non_positive_predictions": {
            runtime: int((forecast <= 0).sum())
            for runtime, forecast in predictions.items()
        },
        "latency_ms": latency,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _print_summary(result: dict[str, object], report_path: Path) -> None:
    print("Runtime      Core median/p95 (ms)    End-to-end median/p95 (ms)")
    for runtime in ("onnx_fp32", "onnx_int8"):
        core = result["latency_ms"]["core"][runtime]
        end_to_end = result["latency_ms"]["end_to_end"][runtime]
        print(
            f"{runtime:<12} "
            f"{core['median_ms']:.4f}/{core['p95_ms']:.4f}"
            f"{'':>10} {end_to_end['median_ms']:.4f}/{end_to_end['p95_ms']:.4f}"
        )
    print("Metrics:", json.dumps(result["metrics"], sort_keys=True))
    print("Metric delta:", json.dumps(result["metric_delta_int8_minus_fp32"]))
    print("Prediction delta:", json.dumps(result["prediction_delta"]))
    print("Artifact bytes:", json.dumps(result["artifacts"]))
    print(f"Wrote quantization report -> {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-path", type=Path, default=DEFAULT_FP32_PATH)
    parser.add_argument("--int8-path", type=Path, default=DEFAULT_INT8_PATH)
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_SCALER_PATH)
    parser.add_argument("--prices-path", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--evaluation-window", type=int, default=60)
    parser.add_argument("--price-window", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    result = run_quantization_experiment(
        fp32_path=args.fp32_path,
        int8_path=args.int8_path,
        scaler_path=args.scaler_path,
        prices_path=args.prices_path,
        config_path=args.config,
        report_path=args.report_path,
        evaluation_window=args.evaluation_window,
        price_window=args.price_window,
        warmup=args.warmup,
        iterations=args.iterations,
        threads=args.threads,
    )
    _print_summary(result, args.report_path)


if __name__ == "__main__":
    main()
