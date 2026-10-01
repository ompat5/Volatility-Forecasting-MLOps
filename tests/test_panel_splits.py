import pandas as pd
import pytest

from src.eval.panel_splits import build_panel_split_plan


def _panel(n_dates: int = 800) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=n_dates, name="date")
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    return pd.DataFrame({"feature": 1.0, "target": 0.2}, index=index)


def _gap(
    all_dates: pd.DatetimeIndex,
    earlier: pd.DatetimeIndex,
    later: pd.DatetimeIndex,
) -> int:
    return int(((all_dates > earlier.max()) & (all_dates < later.min())).sum())


def test_split_plan_is_calendar_wide_purged_and_has_final_holdout():
    panel = _panel()
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    plan = build_panel_split_plan(
        panel,
        n_splits=3,
        horizon=5,
        min_train_size=100,
        val_frac=0.2,
        holdout_size=100,
    )

    assert len(plan.folds) == 3
    assert plan.holdout.test_dates.equals(all_dates[-100:])
    for split in plan.folds:
        assert split.train_dates.max() < split.validation_dates.min()
        assert split.validation_dates.max() < split.test_dates.min()
        assert _gap(all_dates, split.train_dates, split.validation_dates) == 5
        assert _gap(all_dates, split.validation_dates, split.test_dates) == 5
        assert split.train_dates.intersection(split.test_dates).empty
    assert plan.holdout.fit_dates.max() < plan.holdout.test_dates.min()
    assert _gap(all_dates, plan.holdout.fit_dates, plan.holdout.test_dates) == 5
    assert plan.holdout.fit_dates.intersection(plan.holdout.test_dates).empty


def test_final_holdout_refits_on_every_pre_embargo_date():
    panel = _panel()
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    plan = build_panel_split_plan(
        panel,
        n_splits=3,
        horizon=5,
        min_train_size=100,
        val_frac=0.2,
        holdout_size=100,
    )

    assert plan.holdout.fit_dates.equals(all_dates[: -100 - 5])


def test_fold_test_dates_never_touch_final_holdout():
    plan = build_panel_split_plan(
        _panel(),
        n_splits=3,
        horizon=5,
        min_train_size=100,
        val_frac=0.2,
        holdout_size=100,
    )

    for fold in plan.folds:
        assert fold.test_dates.intersection(plan.holdout.test_dates).empty
        assert fold.test_dates.max() < plan.holdout.test_dates.min()


def test_split_manifest_records_reproducible_boundaries():
    plan = build_panel_split_plan(
        _panel(),
        n_splits=3,
        horizon=5,
        min_train_size=100,
        val_frac=0.2,
        holdout_size=100,
    )

    manifest = plan.to_dict()

    assert manifest["horizon"] == 5
    assert manifest["holdout_size"] == 100
    assert len(manifest["folds"]) == 3
    assert manifest["holdout"]["test"]["count"] == 100


def test_split_plan_rejects_unbalanced_panel():
    panel = _panel().drop(index=(pd.Timestamp("2020-01-02"), "BBB"))

    with pytest.raises(ValueError, match="identical dates"):
        build_panel_split_plan(
            panel,
            n_splits=3,
            horizon=5,
            min_train_size=100,
            val_frac=0.2,
            holdout_size=100,
        )


def test_split_plan_fails_instead_of_silently_returning_too_few_folds():
    with pytest.raises(ValueError, match="training minimum"):
        build_panel_split_plan(
            _panel(n_dates=300),
            n_splits=3,
            horizon=5,
            min_train_size=200,
            val_frac=0.2,
            holdout_size=100,
        )
