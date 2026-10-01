"""Train-only feature scaling for the global panel."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PanelFeatureScaler:
    """A fitted scaler that records the exact dates used to estimate it."""

    feature_cols: tuple[str, ...]
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fit_dates_: pd.DatetimeIndex | None = field(default=None, init=False)
    n_fit_rows_: int | None = field(default=None, init=False)

    def fit(
        self,
        panel: pd.DataFrame,
        train_dates: pd.DatetimeIndex,
    ) -> PanelFeatureScaler:
        rows = _select_dates(panel, train_dates)
        missing = set(self.feature_cols) - set(rows.columns)
        if missing:
            raise ValueError(f"Panel is missing features: {sorted(missing)}")
        if rows.empty:
            raise ValueError("Cannot fit feature scaler on an empty training slice")
        self.scaler.fit(rows[list(self.feature_cols)])
        self.fit_dates_ = pd.DatetimeIndex(train_dates).unique().sort_values()
        self.n_fit_rows_ = len(rows)
        return self

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        if self.fit_dates_ is None:
            raise ValueError("PanelFeatureScaler must be fitted before transform")
        missing = set(self.feature_cols) - set(panel.columns)
        if missing:
            raise ValueError(f"Panel is missing features: {sorted(missing)}")
        transformed = panel.copy()
        transformed.loc[:, list(self.feature_cols)] = self.scaler.transform(
            panel[list(self.feature_cols)]
        )
        return transformed


def _select_dates(
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Panel must use a MultiIndex")
    if panel.index.names != ["date", "ticker"]:
        raise ValueError("Panel index must be named ['date', 'ticker']")
    requested = pd.DatetimeIndex(dates).unique().sort_values()
    panel_dates = panel.index.get_level_values("date")
    missing_dates = requested[~requested.isin(panel_dates)]
    if not missing_dates.empty:
        raise ValueError(
            f"Requested dates are absent from the panel: {missing_dates.tolist()}"
        )
    return panel.loc[panel_dates.isin(requested)]
