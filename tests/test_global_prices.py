import pandas as pd
import pytest

from src.data.global_prices import fetch_global_prices
from src.data.universe import ContextSeries, TargetAsset, Universe


@pytest.fixture
def small_universe() -> Universe:
    return Universe(
        schema_version=1,
        targets=(TargetAsset("AAA", "group"),),
        context=(ContextSeries("^VIX", "implied_volatility"),),
    )


def _history(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"Adj Close": [value, value + 1]},
        index=pd.date_range("2026-01-02", periods=2, tz="America/New_York"),
    )


def test_global_refresh_retries_transient_empty_history(
    monkeypatch: pytest.MonkeyPatch,
    small_universe: Universe,
):
    calls = {"AAA": 0, "^VIX": 0}

    def fake_fetch(symbol: str) -> pd.DataFrame:
        calls[symbol] += 1
        if symbol == "AAA" and calls[symbol] == 1:
            raise ValueError("temporary empty response")
        return _history(100.0 if symbol == "AAA" else 20.0)

    monkeypatch.setattr("src.data.global_prices.fetch_ticker", fake_fetch)

    result = fetch_global_prices(small_universe)

    assert calls == {"AAA": 2, "^VIX": 1}
    assert result["ticker"].drop_duplicates().tolist() == ["AAA", "^VIX"]


def test_global_refresh_fails_closed_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
    small_universe: Universe,
):
    calls = 0

    def fake_fetch(symbol: str) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        raise ValueError(f"no data for {symbol}")

    monkeypatch.setattr("src.data.global_prices.fetch_ticker", fake_fetch)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        fetch_global_prices(small_universe, attempts=3)

    assert calls == 3 * len(small_universe.all_symbols)
