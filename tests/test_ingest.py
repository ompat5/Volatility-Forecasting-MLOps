import pandas as pd
import pytest

from src.data import ingest


@pytest.fixture
def fake_history():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "Adj Close": 1.0,
            "Volume": 100,
        },
        index=dates,
    )


def test_load_ticker_list_reads_basket():
    tickers = ingest.load_ticker_list()
    assert "AAPL" in tickers
    assert "^VIX" in tickers


def test_cache_path_strips_caret(tmp_path):
    assert ingest._cache_path("^VIX", tmp_path) == tmp_path / "VIX.parquet"
    assert ingest._cache_path("AAPL", tmp_path) == tmp_path / "AAPL.parquet"


def test_fetch_ticker_raises_on_empty(monkeypatch):
    class FakeTicker:
        def __init__(self, _ticker):
            pass

        def history(self, **_kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(ingest.yf, "Ticker", FakeTicker)
    with pytest.raises(ValueError, match="no data"):
        ingest.fetch_ticker("BOGUS")


def test_ingest_ticker_writes_parquet(tmp_path, fake_history, monkeypatch):
    class FakeTicker:
        def __init__(self, _ticker):
            pass

        def history(self, **_kwargs):
            return fake_history

    monkeypatch.setattr(ingest.yf, "Ticker", FakeTicker)
    out_path = ingest.ingest_ticker("AAPL", raw_dir=tmp_path)

    assert out_path == tmp_path / "AAPL.parquet"
    result = pd.read_parquet(out_path)
    assert len(result) == 5
    assert result.index.name == "date"


def test_ingest_basket_collects_per_ticker_errors(tmp_path, fake_history, monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **_kwargs):
            if self.ticker == "BAD":
                return pd.DataFrame()
            return fake_history

    monkeypatch.setattr(ingest.yf, "Ticker", FakeTicker)
    results = ingest.ingest_basket(["AAPL", "BAD"], raw_dir=tmp_path)

    assert results["AAPL"] == tmp_path / "AAPL.parquet"
    assert isinstance(results["BAD"], ValueError)
