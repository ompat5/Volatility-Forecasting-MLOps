import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.panel_preprocessing import PanelFeatureScaler

FEATURE_COLS = ("feature_1", "feature_2")


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=10, name="date")
    tickers = ["AAA", "BBB"]
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=["date", "ticker"],
    )
    rows = []
    for date_number in range(len(dates)):
        for ticker_number in range(len(tickers)):
            value = float(date_number + ticker_number)
            if date_number >= 6:
                value += 10_000.0
            rows.append(
                {
                    "feature_1": value,
                    "feature_2": value * 2.0 + 1.0,
                    "target": value + 0.25,
                }
            )
    return pd.DataFrame(rows, index=index)


def test_scaler_fits_only_explicit_training_dates():
    panel = _panel()
    dates = panel.index.get_level_values("date").unique()
    train_dates = dates[:6]
    training_rows = panel.loc[
        panel.index.get_level_values("date").isin(train_dates)
    ]

    scaler = PanelFeatureScaler(FEATURE_COLS).fit(panel, train_dates)

    assert np.allclose(
        scaler.scaler.mean_,
        training_rows[list(FEATURE_COLS)].mean().to_numpy(),
    )
    assert scaler.n_fit_rows_ == 12
    assert scaler.fit_dates_.equals(train_dates)
    assert scaler.scaler.mean_[0] < 100.0


def test_transform_preserves_target_and_index():
    panel = _panel()
    dates = panel.index.get_level_values("date").unique()
    scaler = PanelFeatureScaler(FEATURE_COLS).fit(panel, dates[:6])

    transformed = scaler.transform(panel)
    training_mask = transformed.index.get_level_values("date").isin(dates[:6])

    pdt.assert_index_equal(transformed.index, panel.index)
    pdt.assert_series_equal(transformed["target"], panel["target"])
    assert np.allclose(
        transformed.loc[training_mask, list(FEATURE_COLS)].mean(),
        0.0,
        atol=1e-7,
    )


def test_scaler_rejects_transform_before_fit():
    with pytest.raises(ValueError, match="fitted"):
        PanelFeatureScaler(FEATURE_COLS).transform(_panel())
