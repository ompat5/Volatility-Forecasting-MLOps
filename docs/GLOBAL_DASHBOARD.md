# Global candidate dashboard

Phase 8 adds a separate Streamlit entrypoint for the 34-target candidate. It
does not replace the deployed AAPL dashboard, publish a model release, enable
the candidate monitoring schedule, or change either API default.

## System contract

The dashboard treats the universe as one unit rather than 34 unrelated apps:

- all 34 forecast targets and the shared VIX context must be present;
- the candidate manifest must match the repository universe and five-session
  horizon;
- the FP32 ONNX graph and sidecar must be checksum-bound to that candidate;
- the final-holdout evidence must contain naive, EWMA, GARCH, and global-LSTM
  metrics for every target, every group, and both portfolio aggregations;
- VIX can be inspected as context but can never be selected as a forecast
  target.

One cached all-universe snapshot runs the same strict monitoring pipeline used
by Phase 7. Ticker selection only slices that result, so changing the UI control
does not reconstruct 60 × 34 historical forecasts. This also guarantees every
ticker card, group view, and fleet status comes from the same synchronized
as-of date.

## What the dashboard shows

- one positive five-session forecast for any of the 34 targets;
- recent 5-, 20-, and 60-session realized volatility;
- synchronized delayed forecast-versus-realized history;
- fleet, group, ticker, and shared-VIX monitoring context;
- untouched 252-session final-holdout comparisons at ticker, group, pooled
  micro, and equal-ticker macro levels.

The page carries a permanent candidate banner. A critical drift state is shown
as monitoring evidence, not translated into a model-failure or production
claim.

## Run locally

The default mode refreshes all 35 histories and enforces wall-clock freshness:

```bash
uv run streamlit run dashboard/global_app.py
```

For a reproducible offline replay from the ingestion cache:

```bash
GLOBAL_DASHBOARD_USE_CACHED_PRICES=1 \
  uv run streamlit run dashboard/global_app.py
```

An exact long-form Parquet snapshot can be supplied instead:

```bash
GLOBAL_DASHBOARD_PRICES_PATH=path/to/latest_prices.parquet \
  uv run streamlit run dashboard/global_app.py
```

`GLOBAL_DASHBOARD_MODEL_DIR` may point to an exported candidate snapshot. The
dashboard otherwise prefers a local `global_model/`; when neither exists, it
downloads and validates the checksum-pinned candidate in
`configs/global_release.yaml` into temporary storage. The directory must
contain the model state, scaler, and artifact manifest. If both optimized files
are absent, the dashboard exports and validates FP32 ONNX once. If only one
optimized file is present, startup fails rather than guessing whether the model
and sidecar belong together.

Cached and exact-snapshot modes are visibly labeled **offline replay** and skip
only the wall-clock age comparison. They retain universe completeness,
duplicate, finite/positive, synchronized-session, artifact, monitoring, and
evaluation-evidence validation.

## Acceptance evidence

The real exported candidate was loaded through the dashboard runtime and the
cached 34-target-plus-VIX basket produced:

- 34 synchronized live forecasts;
- 2,040 delayed backtest forecasts plus 34 live ledger rows;
- 120 recent volatility points for every target;
- the expected ticker, group, and portfolio evaluation tables.

The Streamlit page was also exercised end to end against that snapshot. The
existing `dashboard/app.py` AAPL entrypoint was not modified.

## Promotion boundary

Dashboard integration and immutable delivery complete candidate lifecycle
gates, not promotion. The downloaded archive is validated through the same
dashboard runtime before acceptance. Changing the production API or dashboard
default still requires a separate cutover review; see `docs/GLOBAL_RELEASE.md`.
