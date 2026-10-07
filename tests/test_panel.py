from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.data.ingest import ticker_cache_path
from src.data.panel import (
    GLOBAL_FEATURE_COLS,
    VIX_LEVEL_COL,
    build_global_panel,
    build_panel_manifest,
    load_adjusted_close,
    normalize_session_index,
)
from src.data.universe import ContextSeries, TargetAsset, Universe
from src.features.realized_vol import TARGET_COL


@pytest.fixture
def small_universe() -> Universe:
    return Universe(
        schema_version=1,
        targets=(
            TargetAsset("AAA", "group_a"),
            TargetAsset("BBB", "group_b"),
        ),
        context=(ContextSeries("^VIX", "implied_volatility"),),
    )


def _write_prices(
    raw_dir: Path,
    symbol: str,
    dates: pd.DatetimeIndex,
    *,
    seed: int,
    initial_price: float,
) -> None:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, len(dates))
    prices = initial_price * np.exp(np.cumsum(returns))
    frame = pd.DataFrame({"Adj Close": prices}, index=dates)
    frame.index.name = "date"
    frame.to_parquet(ticker_cache_path(symbol, raw_dir))


@pytest.fixture
def raw_panel_data(tmp_path: Path, small_universe: Universe) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    equity_dates = pd.bdate_range(
        "2020-01-02", periods=150, tz="America/New_York"
    )
    vix_dates = pd.bdate_range(
        "2020-01-02", periods=150, tz="America/Chicago"
    )
    _write_prices(raw_dir, "AAA", equity_dates, seed=1, initial_price=100.0)
    _write_prices(raw_dir, "BBB", equity_dates, seed=2, initial_price=80.0)
    _write_prices(raw_dir, "^VIX", vix_dates, seed=3, initial_price=20.0)
    return raw_dir


def test_session_normalization_aligns_exchange_local_dates():
    new_york = pd.DatetimeIndex(
        ["2024-01-02 00:00:00"], tz="America/New_York"
    )
    chicago = pd.DatetimeIndex(
        ["2024-01-02 00:00:00"], tz="America/Chicago"
    )

    assert normalize_session_index(new_york).equals(
        normalize_session_index(chicago)
    )
    assert normalize_session_index(new_york).tz is None


def test_session_normalization_rejects_duplicate_local_dates():
    observations = pd.DatetimeIndex(
        ["2024-01-02 09:30:00", "2024-01-02 16:00:00"],
        tz="America/New_York",
    )

    with pytest.raises(ValueError, match="same trading session"):
        normalize_session_index(observations)


def test_build_global_panel_is_balanced_and_uses_all_targets(
    raw_panel_data: Path,
    small_universe: Universe,
):
    panel = build_global_panel(
        small_universe,
        raw_dir=raw_panel_data,
        horizon=5,
    )

    assert panel.index.names == ["date", "ticker"]
    assert list(panel.columns) == [*GLOBAL_FEATURE_COLS, TARGET_COL]
    assert set(panel.index.get_level_values("ticker")) == {"AAA", "BBB"}
    counts = panel.groupby(level="ticker").size()
    assert counts.nunique() == 1
    assert len(panel) == counts.iloc[0] * 2
    assert not panel.isna().any().any()
    assert (panel[TARGET_COL] > 0).all()


def test_vix_context_is_joined_on_the_same_session(
    raw_panel_data: Path,
    small_universe: Universe,
):
    panel = build_global_panel(small_universe, raw_dir=raw_panel_data)
    vix = load_adjusted_close("^VIX", raw_panel_data)
    date = panel.index.get_level_values("date")[10]

    expected = vix.loc[date] / 100.0
    actual = panel.loc[(date, "AAA"), VIX_LEVEL_COL]

    assert actual == pytest.approx(expected)
    assert panel.loc[(date, "BBB"), VIX_LEVEL_COL] == pytest.approx(expected)


def test_future_vix_changes_do_not_change_earlier_panel_rows(
    raw_panel_data: Path,
    small_universe: Universe,
):
    original = build_global_panel(small_universe, raw_dir=raw_panel_data)
    vix_path = ticker_cache_path("^VIX", raw_panel_data)
    vix_frame = pd.read_parquet(vix_path)
    cutoff = vix_frame.index[110].date()
    vix_frame.loc[vix_frame.index[110]:, "Adj Close"] *= 1.5
    vix_frame.to_parquet(vix_path)

    changed = build_global_panel(small_universe, raw_dir=raw_panel_data)
    earlier = original.index.get_level_values("date").date < cutoff

    pdt.assert_frame_equal(original.loc[earlier], changed.loc[earlier])


def test_manifest_records_exact_sources_and_balanced_coverage(
    raw_panel_data: Path,
    small_universe: Universe,
):
    panel = build_global_panel(small_universe, raw_dir=raw_panel_data)
    manifest = build_panel_manifest(
        panel,
        small_universe,
        raw_dir=raw_panel_data,
    )

    assert manifest["schema_version"] == 1
    assert len(manifest["universe"]["targets"]) == 2
    assert manifest["panel"]["rows"] == len(panel)
    assert manifest["panel"]["horizon"] == 5
    assert manifest["panel"]["balanced"] is True
    assert manifest["panel"]["features"] == GLOBAL_FEATURE_COLS
    assert len(set(manifest["panel"]["rows_per_ticker"].values())) == 1
    assert set(manifest["sources"]) == {"AAA", "BBB", "^VIX"}
    assert all(
        len(metadata["sha256"]) == 64
        for metadata in manifest["sources"].values()
    )
