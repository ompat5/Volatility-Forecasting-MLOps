from src.optimization.benchmark import benchmark_callables, summarize_latencies


def test_summarize_latencies_converts_to_milliseconds():
    summary = summarize_latencies([1_000_000, 2_000_000, 3_000_000])

    assert summary["iterations"] == 3
    assert summary["median_ms"] == 2.0
    assert summary["p95_ms"] == 2.9
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 3.0


def test_benchmark_callables_warms_up_and_collects_every_iteration():
    calls = {"first": 0, "second": 0}

    def first() -> None:
        calls["first"] += 1

    def second() -> None:
        calls["second"] += 1

    result = benchmark_callables(
        {"first": first, "second": second}, warmup=2, iterations=3
    )

    assert calls == {"first": 5, "second": 5}
    assert result["first"]["iterations"] == 3
    assert result["second"]["iterations"] == 3
    assert result["first"]["median_ms"] >= 0
    assert result["second"]["median_ms"] >= 0
