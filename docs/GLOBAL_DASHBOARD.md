# Global dashboard

`dashboard/app.py` is the public 34-target Streamlit entrypoint. One cached
all-universe model/data/monitoring snapshot backs ticker and group selection, so
changing a control never creates a different as-of date or reruns the delayed
forecast history independently.

The page validates the complete target/VIX universe, model checksums, ONNX
sidecar, monitoring reference, and final-holdout evidence before it renders.
It shows target forecasts, realized-volatility history, delayed errors, fleet
and group monitoring, shared VIX context, and final-holdout comparisons.

```bash
uv run streamlit run dashboard/app.py
```

By default it refreshes all 35 series and enforces wall-clock freshness. Use
`GLOBAL_DASHBOARD_USE_CACHED_PRICES=1` for an explicitly labeled offline replay,
or `GLOBAL_DASHBOARD_PRICES_PATH=...` for an exact Parquet snapshot. The runtime
uses a local `global_model/` when present and otherwise downloads the immutable
global release pin.
