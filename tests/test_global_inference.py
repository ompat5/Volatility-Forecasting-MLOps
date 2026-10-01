import numpy as np
import pandas as pd
import pytest

from src.features.global_inference import build_global_inference_batch


def _raw_prices(n_dates: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=n_dates)
    rows = []
    for ticker_number, ticker in enumerate(["AAA", "BBB"]):
        returns = 0.001 + 0.005 * np.sin(np.arange(n_dates) / 7 + ticker_number)
        prices = (100 + 20 * ticker_number) * np.exp(np.cumsum(returns))
        rows.extend(zip(dates, [ticker] * n_dates, prices))
    vix = 18 + 2 * np.sin(np.arange(n_dates) / 9)
    rows.extend(zip(dates, ["^VIX"] * n_dates, vix))
    return pd.DataFrame(rows, columns=["date", "ticker", "adjusted_close"])


def _batch(frame: pd.DataFrame):
    return build_global_inference_batch(
        frame,
        target_symbols=("AAA", "BBB"),
        context_symbol="^VIX",
        ticker_to_id={"AAA": 0, "BBB": 1},
        seq_len=10,
    )


def test_global_inference_builds_one_isolated_window_per_target():
    batch = _batch(_raw_prices())

    assert batch.features.shape == (2, 10, 6)
    assert batch.tickers == ("AAA", "BBB")
    assert batch.ticker_ids.tolist() == [0, 1]
    assert batch.as_of_dates == (
        pd.Timestamp("2025-05-21"),
        pd.Timestamp("2025-05-21"),
    )
    assert np.isfinite(batch.features).all()


def test_global_inference_allows_a_known_target_subset():
    frame = _raw_prices()
    frame = frame[frame["ticker"].isin(["BBB", "^VIX"])]

    batch = _batch(frame)

    assert batch.tickers == ("BBB",)
    assert batch.ticker_ids.tolist() == [1]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame[frame["ticker"] != "^VIX"], "context ticker"),
        (
            lambda frame: frame.assign(
                ticker=frame["ticker"].replace({"AAA": "UNKNOWN"})
            ),
            "Unsupported inference tickers",
        ),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate date/ticker"),
    ],
)
def test_global_inference_rejects_broken_contracts(mutate, message):
    with pytest.raises(ValueError, match=message):
        _batch(mutate(_raw_prices()))


def test_global_inference_rejects_insufficient_complete_history():
    with pytest.raises(ValueError, match="complete feature rows"):
        _batch(_raw_prices(n_dates=65))
