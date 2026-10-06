"""Candidate-only Streamlit dashboard for all 34 global forecast targets."""

# ruff: noqa: E402 - src imports must follow the Streamlit path bootstrap below.

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# Streamlit executes this file with dashboard/ as the import root. Add the
# repository root without requiring an editable package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import REPO_ROOT, load_global_config
from src.dashboard.data import status_label
from src.dashboard.global_data import (
    build_global_dashboard_snapshot,
    group_model_comparison,
    load_global_evaluation_tables,
    portfolio_model_comparison,
    select_global_ticker_view,
    ticker_model_comparison,
)
from src.dashboard.global_runtime import load_global_dashboard_forecaster
from src.data.global_prices import fetch_global_prices, load_cached_global_prices
from src.data.ingest import DEFAULT_RAW_DIR, DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe
from src.monitoring.global_pipeline import load_global_monitoring_config

REFERENCE_PATH = REPO_ROOT / "monitoring" / "global_reference.json"
MONITORING_CONFIG_PATH = REPO_ROOT / "configs" / "global_monitoring.yaml"
RESULTS_PATH = REPO_ROOT / "benchmarks" / "global_model.json"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _price_source() -> tuple[str, Path | None]:
    exact_snapshot = os.getenv("GLOBAL_DASHBOARD_PRICES_PATH")
    if exact_snapshot:
        return "snapshot", Path(exact_snapshot)
    if _truthy_env("GLOBAL_DASHBOARD_USE_CACHED_PRICES"):
        raw_dir = Path(os.getenv("GLOBAL_DASHBOARD_RAW_DIR", str(DEFAULT_RAW_DIR)))
        return "cache", raw_dir
    return "fresh", None


@st.cache_resource(show_spinner=False)
def _load_model():
    return load_global_dashboard_forecaster()


@st.cache_data(ttl=3_600, show_spinner=False)
def _load_prices() -> pd.DataFrame:
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    source, path = _price_source()
    if source == "snapshot":
        return pd.read_parquet(path)
    if source == "cache":
        return load_cached_global_prices(universe, path)
    return fetch_global_prices(universe)


@st.cache_data(ttl=3_600, show_spinner=False)
def _load_snapshot():
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    config = load_global_config()
    monitoring_config, model_metadata = load_global_monitoring_config(
        MONITORING_CONFIG_PATH
    )
    model = _load_model()
    prices = _load_prices()
    reference = json.loads(REFERENCE_PATH.read_text())
    source, _ = _price_source()
    run_date = (
        pd.Timestamp.now(tz="America/Toronto").tz_localize(None).normalize()
        if source == "fresh"
        else None
    )
    return build_global_dashboard_snapshot(
        model,
        prices,
        reference,
        universe,
        monitoring_config,
        horizon=config.data.horizon,
        model_version=model_metadata["version"],
        artifact_manifest=model.manifest,
        run_date=run_date,
    )


@st.cache_data(show_spinner=False)
def _load_evaluation():
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    return load_global_evaluation_tables(RESULTS_PATH, universe)


def _format_group(group: str) -> str:
    return group.replace("_", " ").title()


def _render_monitoring(view, snapshot) -> None:
    st.subheader("Candidate monitoring")
    st.caption(
        "Fleet status aggregates all 34 targets. Ticker and group cards are "
        "drill-downs, not separate production alerts."
    )
    fleet_col, drift_col, regime_col, error_col = st.columns(4)
    fleet_col.metric(
        "34-target fleet",
        status_label(view.monitoring["fleet_status"]),
    )

    drift = view.monitoring["feature_drift"]
    drift_col.metric("Ticker feature drift", status_label(drift["ticker"]["status"]))
    drift_col.caption(
        f"Max PSI {max(drift['ticker']['psi'].values()):.3f} · "
        f"group {status_label(drift['group']['status'])}"
    )

    regime = view.monitoring["volatility_regime"]
    regime_col.metric("Ticker volatility regime", status_label(regime["ticker"]["status"]))
    regime_col.caption(
        f"RV20 {regime['ticker']['latest_rv_20d']:.1%} · "
        f"training p95 {regime['ticker']['reference_p95']:.1%}"
    )

    error = view.monitoring["forecast_error"]
    error_col.metric("Ticker delayed error", status_label(error["ticker"]["status"]))
    error_col.caption(
        f"Recent/baseline RMSE {error['ticker']['recent_to_baseline_rmse_ratio']:.2f} · "
        f"group {status_label(error['group']['status'])}"
    )

    with st.expander("Complete monitoring context"):
        component_rows = pd.DataFrame(
            [
                {"component": name.replace("_", " ").title(), "status": status_label(status)}
                for name, status in snapshot.report["component_status"].items()
            ]
        )
        st.dataframe(component_rows, hide_index=True, width="stretch")
        context = snapshot.report["feature_drift"]["context"]
        quality = snapshot.report["data_quality"]
        st.markdown(
            f"**Shared VIX context:** {status_label(context['status'])}  \n"
            f"**Synchronized basket:** {quality['target_count']} targets + "
            f"{quality['context_count']} context series; maximum lag "
            f"{quality['max_stale_sessions']} market sessions.  \n"
            f"**Wall-clock freshness:** "
            f"{'enforced' if quality['wall_clock_freshness_enforced'] else 'offline replay'}"
        )


def _render_evaluation(view, evaluation) -> None:
    st.subheader("Untouched final-holdout comparison")
    st.caption(
        "Lower is better. These stored results use the final 252-session holdout; "
        "they are not recalculated from today’s dashboard data."
    )
    ticker_tab, group_tab, portfolio_tab = st.tabs(
        [view.ticker, _format_group(view.group), "Portfolio"]
    )
    with ticker_tab:
        st.dataframe(
            ticker_model_comparison(evaluation, view.ticker),
            width="stretch",
            column_config={
                metric: st.column_config.NumberColumn(format="%.4f")
                for metric in ("rmse", "mae", "qlike")
            },
        )
    with group_tab:
        st.dataframe(
            group_model_comparison(evaluation, view.group),
            width="stretch",
            column_config={
                metric: st.column_config.NumberColumn(format="%.4f")
                for metric in ("rmse", "mae", "qlike")
            },
        )
    with portfolio_tab:
        scope = st.radio(
            "Portfolio aggregation",
            ["micro", "macro"],
            horizontal=True,
            format_func=lambda value: (
                "Micro (all observations)"
                if value == "micro"
                else "Macro (equal ticker weight)"
            ),
        )
        st.dataframe(
            portfolio_model_comparison(evaluation, scope),
            width="stretch",
            column_config={
                metric: st.column_config.NumberColumn(format="%.4f")
                for metric in ("rmse", "mae", "qlike")
            },
        )
    st.caption(
        "The global LSTM passes the candidate evidence gate on pooled RMSE and "
        "MAE; GARCH retains the stronger pooled QLIKE result."
    )


def _render() -> None:
    st.set_page_config(
        page_title="Global Volatility Candidate",
        page_icon="🌐",
        layout="wide",
    )
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    groups = tuple(dict.fromkeys(asset.group for asset in universe.targets))
    tickers_by_group = {
        group: tuple(
            asset.symbol for asset in universe.targets if asset.group == group
        )
        for group in groups
    }

    st.title("Global Volatility Candidate")
    st.caption(
        "34 forecast targets · VIX context · Five-trading-day realized volatility · "
        "Annualized values"
    )
    st.warning(
        "Candidate evaluation only. The public production default and scheduled "
        "production monitoring remain AAPL-only; this page does not perform a "
        "global production cutover."
    )

    with st.sidebar:
        st.header("Universe")
        selected_group = st.selectbox(
            "Asset group",
            groups,
            format_func=_format_group,
        )
        selected_ticker = st.selectbox(
            "Forecast target",
            tickers_by_group[selected_group],
        )
        st.caption(
            f"{len(universe.target_symbols)} targets + "
            f"{universe.context_symbol('implied_volatility')} context"
        )
        st.write("Backend: **Global ONNX Runtime FP32**")
        source, path = _price_source()
        labels = {
            "fresh": "fresh all-universe download",
            "cache": "local ingestion cache (offline replay)",
            "snapshot": "exact Parquet snapshot (offline replay)",
        }
        st.caption(f"Data: {labels[source]}")
        if path is not None:
            st.caption(f"Source path: `{path}`")
        if st.button("Refresh complete universe"):
            _load_prices.clear()
            _load_snapshot.clear()
            st.rerun()
        st.divider()
        st.caption("Educational project—not investment advice.")

    try:
        with st.spinner(
            "Loading the verified candidate and one synchronized universe snapshot..."
        ):
            snapshot = _load_snapshot()
            evaluation = _load_evaluation()
            view = select_global_ticker_view(
                snapshot,
                universe,
                selected_ticker,
            )
    except Exception as exc:
        st.error(f"Global candidate dashboard could not be loaded: {exc}")
        st.stop()

    regime = view.monitoring["volatility_regime"]["ticker"]
    forecast_col, rv_col, group_col, as_of_col = st.columns(4)
    forecast_col.metric(
        f"{view.ticker} next 5-day volatility",
        f"{view.forecast:.1%}",
        help="Annualized realized-volatility forecast—not a price return.",
    )
    rv_col.metric("Current 20-day realized volatility", f"{regime['latest_rv_20d']:.1%}")
    group_col.metric("Asset group", _format_group(view.group))
    as_of_col.metric("Synchronized as of", view.as_of.date().isoformat())

    recent_tab, history_tab = st.tabs(
        ["Recent realized volatility", "Forecast vs realized"]
    )
    with recent_tab:
        recent = view.recent_volatility.rename(
            columns={
                "rv_5d": "5-day RV",
                "rv_20d": "20-day RV",
                "rv_60d": "60-day RV",
            }
        )
        st.line_chart(recent * 100, y_label="Annualized volatility (%)")
    with history_tab:
        history = view.history.loc[:, ["forecast", "actual"]].rename(
            columns={"forecast": "Forecast", "actual": "Realized"}
        )
        st.line_chart(history * 100, y_label="Annualized volatility (%)")

    _render_monitoring(view, snapshot)
    _render_evaluation(view, evaluation)
    st.caption(
        f"Candidate: {snapshot.report['model_version']} · Training reference ends "
        f"{pd.Timestamp(snapshot.report['reference_training_end']).date().isoformat()}"
    )


_render()
