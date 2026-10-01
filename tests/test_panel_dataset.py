import numpy as np
import pandas as pd
import pytest
import torch

from src.features.panel_dataset import PanelVolatilityDataset

FEATURE_COLS = ["feature_1", "feature_2"]
TARGET_COL = "target"


def _panel(n_dates: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_dates, name="date")
    tickers = ["AAA", "BBB"]
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=["date", "ticker"],
    )
    rows = []
    for date_number in range(n_dates):
        for ticker in tickers:
            offset = 0.0 if ticker == "AAA" else 100.0
            rows.append(
                {
                    "feature_1": offset + date_number,
                    "feature_2": offset + date_number + 0.5,
                    "target": offset + date_number + 1.0,
                }
            )
    return pd.DataFrame(rows, index=index)


def _dataset(anchor_dates: pd.DatetimeIndex | None = None):
    panel = _panel()
    dates = panel.index.get_level_values("date").unique()
    return PanelVolatilityDataset(
        panel,
        FEATURE_COLS,
        TARGET_COL,
        seq_len=3,
        anchor_dates=dates if anchor_dates is None else anchor_dates,
        ticker_to_id={"AAA": 0, "BBB": 1},
    )


def test_sequences_never_cross_ticker_boundaries():
    dataset = _dataset()

    aaa_features, aaa_id, _ = dataset[0]
    bbb_features, bbb_id, _ = dataset[1]

    assert torch.equal(aaa_features[:, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(bbb_features[:, 0], torch.tensor([100.0, 101.0, 102.0]))
    assert aaa_id.item() == 0
    assert bbb_id.item() == 1


def test_dataset_drops_only_anchor_dates_without_enough_history():
    dataset = _dataset()

    assert len(dataset.dropped_anchor_dates) == 2
    assert len(dataset) == 6 * 2
    assert dataset.sample_index.get_level_values("ticker").value_counts().to_dict() == {
        "AAA": 6,
        "BBB": 6,
    }
    assert dataset.sample_index[0] == (
        pd.Timestamp("2024-01-04"),
        "AAA",
    )


def test_target_and_sample_index_align_with_window_end():
    dataset = _dataset()

    _, _, target = dataset[0]

    assert target.item() == pytest.approx(3.0)
    assert dataset.sample_index[0] == (
        pd.Timestamp("2024-01-04"),
        "AAA",
    )


def test_evaluation_anchor_can_use_earlier_feature_context():
    panel = _panel()
    dates = panel.index.get_level_values("date").unique()
    dataset = _dataset(anchor_dates=dates[4:])

    features, _, _ = dataset[0]

    assert dataset.dropped_anchor_dates.empty
    assert dataset.sample_index[0][0] == dates[4]
    assert np.array_equal(features[:, 0].numpy(), np.array([2.0, 3.0, 4.0]))


def test_dataset_requires_complete_ticker_vocabulary():
    panel = _panel()
    dates = panel.index.get_level_values("date").unique()

    with pytest.raises(ValueError, match="every panel ticker"):
        PanelVolatilityDataset(
            panel,
            FEATURE_COLS,
            TARGET_COL,
            seq_len=3,
            anchor_dates=dates,
            ticker_to_id={"AAA": 0},
        )


def test_dataset_rejects_unknown_anchor_date():
    panel = _panel()

    with pytest.raises(ValueError, match="absent from the panel"):
        PanelVolatilityDataset(
            panel,
            FEATURE_COLS,
            TARGET_COL,
            seq_len=3,
            anchor_dates=pd.DatetimeIndex(["2030-01-01"]),
            ticker_to_id={"AAA": 0, "BBB": 1},
        )
