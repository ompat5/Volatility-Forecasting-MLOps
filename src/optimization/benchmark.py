"""Small, runtime-agnostic helpers for honest inference benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter_ns

import numpy as np

BenchmarkCallable = Callable[[], object]
LatencySummary = dict[str, float | int]


def summarize_latencies(samples_ns: list[int]) -> LatencySummary:
    """Summarize nanosecond timings in milliseconds."""
    if not samples_ns:
        raise ValueError("At least one latency sample is required")

    samples_ms = np.asarray(samples_ns, dtype=np.float64) / 1_000_000
    return {
        "iterations": len(samples_ns),
        "median_ms": float(np.median(samples_ms)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "min_ms": float(np.min(samples_ms)),
        "max_ms": float(np.max(samples_ms)),
    }


def benchmark_callables(
    callables: Mapping[str, BenchmarkCallable],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, LatencySummary]:
    """Warm up and time callables, rotating their order on every iteration.

    Rotation reduces systematic bias from always measuring one runtime first.
    Model/session construction must happen before calling this function so setup
    time is not mixed into steady-state inference latency.
    """
    if not callables:
        raise ValueError("At least one benchmark callable is required")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    names = list(callables)

    for iteration in range(warmup):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            callables[name]()

    samples: dict[str, list[int]] = {name: [] for name in names}
    for iteration in range(iterations):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            start = perf_counter_ns()
            callables[name]()
            samples[name].append(perf_counter_ns() - start)

    return {
        name: summarize_latencies(runtime_samples)
        for name, runtime_samples in samples.items()
    }
