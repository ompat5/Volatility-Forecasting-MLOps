"""AAPL-only Streamlit dashboard for the deployed volatility forecaster."""

# ruff: noqa: E402 - src imports must follow the Streamlit path bootstrap below.

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# Streamlit executes this file with dashboard/ as the import root. Add the
# repository root so the production src package resolves locally and in cloud
# deployments without requiring an editable package install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import REPO_ROOT, load_config
from src.dashboard.data import (
    build_dashboard_snapshot,
    load_model_results,
    status_label,
)
from src.dashboard.runtime import load_dashboard_forecaster
from src.data.ingest import fetch_ticker
from src.monitoring.pipeline import load_monitoring_config
from src.optimization.inference import load_adjusted_close

TICKER = "AAPL"
REFERENCE_PATH = REPO_ROOT / "monitoring" / "reference.json"
MONITORING_CONFIG_PATH = REPO_ROOT / "configs" / "monitoring.yaml"
RESULTS_PATH = REPO_ROOT / "dashboard" / "assets" / "model_results.csv"


@st.cache_resource(show_spinner=False)
def _load_model():
    return load_dashboard_forecaster()


@st.cache_data(ttl=3_600, show_spinner=False)
def _load_prices() -> pd.Series:
    local_path = os.getenv("DASHBOARD_PRICES_PATH")
    if local_path:
        return load_adjusted_close(Path(local_path))
    frame = fetch_ticker(TICKER)
    return frame["Adj Close"].dropna().sort_index()


def _render() -> None:
    st.set_page_config(
        page_title="AAPL Volatility Forecaster",
        page_icon="📈",
        layout="wide",
    )
    st.title("AAPL Volatility Forecaster")
    st.caption(
        "AAPL-only · Five-trading-day realized-volatility forecast · "
        "Annualized values"
    )
    st.info(
        "This deployment supports AAPL only. The wider ticker basket is cached "
        "for research but does not have deployed models."
    )

    with st.sidebar:
        st.header("Model")
        st.selectbox("Supported ticker", [TICKER], disabled=True)
        st.write("Backend: **ONNX Runtime FP32**")
        if st.button("Refresh market data"):
            _load_prices.clear()
            st.rerun()
        st.divider()
        st.caption("Educational project—not investment advice.")

    try:
        with st.spinner("Loading the verified model and current AAPL data..."):
            model = _load_model()
            prices = _load_prices()
            reference = json.loads(REFERENCE_PATH.read_text())
            monitoring_config, model_config = load_monitoring_config(
                MONITORING_CONFIG_PATH
            )
            config = load_config()
            snapshot = build_dashboard_snapshot(
                model,
                prices,
                reference,
                monitoring_config,
                horizon=config.data.horizon,
                model_version=model_config["version"],
            )
    except Exception as exc:
        st.error(f"Dashboard data could not be loaded: {exc}")
        st.stop()

    report = snapshot.report
    regime = report["volatility_regime"]
    error = report["forecast_error"]
    drift = report["feature_drift"]

    forecast_col, rv_col, as_of_col = st.columns(3)
    forecast_col.metric(
        "Next 5-day realized volatility",
        f"{report['forecast']:.1%}",
        help="Model forecast, annualized. This is volatility—not a price return.",
    )
    rv_col.metric("Current 20-day realized volatility", f"{regime['latest_rv_20d']:.1%}")
    as_of_col.metric("Market data as of", pd.Timestamp(report["as_of"]).date().isoformat())

    st.subheader("Recent realized volatility")
    recent = snapshot.recent_volatility.rename(
        columns={
            "rv_5d": "5-day RV",
            "rv_20d": "20-day RV",
            "rv_60d": "60-day RV",
        }
    )
    st.line_chart(recent * 100, y_label="Annualized volatility (%)")

    st.subheader("Forecast versus realized volatility")
    history = snapshot.predictions.query("record_type == 'backtest'")[
        ["forecast", "actual"]
    ].rename(columns={"forecast": "Forecast", "actual": "Realized"})
    st.line_chart(history * 100, y_label="Annualized volatility (%)")

    st.subheader("Monitoring")
    drift_col, regime_col, error_col = st.columns(3)
    drift_col.metric("Feature drift", status_label(drift["status"]))
    drift_col.caption(f"Maximum PSI: {max(drift['psi'].values()):.3f}")
    regime_col.metric("Volatility regime", status_label(regime["status"]))
    regime_col.caption(
        f"RV20 {regime['latest_rv_20d']:.1%} vs training p95 "
        f"{regime['reference_p95']:.1%}"
    )
    error_col.metric("Delayed forecast error", status_label(error["status"]))
    error_col.caption(
        "Recent/baseline RMSE ratio: "
        f"{error['recent_to_baseline_rmse_ratio']:.2f}"
    )

    with st.expander("How to read these monitoring signals"):
        st.markdown(
            "- **Feature drift:** PSI compares recent inputs with training data.\n"
            "- **Regime:** warns when RV20 exceeds the training 95th percentile.\n"
            "- **Error:** compares recent RMSE with the preceding monitoring window.\n\n"
            "Drift does not automatically mean a stressed volatility regime or a "
            "failed model; the signals answer different questions."
        )

    st.subheader("Walk-forward model comparison")
    st.caption(
        "Five expanding-window folds with a five-day gap. Lower is better. "
        "These are stored evaluation results, not live retraining metrics."
    )
    results = load_model_results(RESULTS_PATH)
    st.dataframe(
        results,
        hide_index=True,
        width="stretch",
        column_config={
            "rmse": st.column_config.NumberColumn(format="%.4f"),
            "mae": st.column_config.NumberColumn(format="%.4f"),
            "qlike": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "The LSTM leads RMSE and MAE; EWMA retains a slight QLIKE edge. "
        f"Model: {report['model_version']} · Training reference ends "
        f"{pd.Timestamp(report['reference_training_end']).date().isoformat()}"
    )


_render()
