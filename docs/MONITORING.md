# Monitoring design

Phase 5 monitors the deployed **AAPL** LSTM. The project has a 35-ticker data
basket, but the current registered model was trained on AAPL only; the monitor
does not imply multi-ticker model coverage that does not yet exist.

## Daily pipeline

At 18:37 America/Toronto, Monday through Friday, GitHub Actions:

1. downloads the immutable `aapl-lstm-v1` MLflow pyfunc from GitHub Releases;
2. verifies its SHA-256 digest before extraction;
3. refreshes AAPL history from yfinance;
4. generates the latest five-trading-day realized-volatility forecast;
5. reconstructs the last 60 eligible as-of forecasts and joins targets that are
   now observable;
6. calculates feature drift, regime state, and rolling forecast error;
7. writes the report into the Actions job summary and uploads all outputs for 90
   days.

The workflow can be started manually with `workflow_dispatch`. GitHub schedules
run only from the default branch, so the schedule becomes active after this phase
branch is merged.

## What is monitored

### Feature drift

Population Stability Index (PSI) compares the most recent 60 feature rows with
fixed quantile bins captured from the model's training data:

- PSI below 0.10: `ok`
- PSI from 0.10 to 0.25: `warning`
- PSI at or above 0.25: `critical`

The reference in `monitoring/reference.json` is deliberately committed: it is
small, reviewable model metadata, not raw market data.

### Volatility regime

The latest 20-day realized volatility is compared with its training-reference
95th percentile. Crossing that level produces a warning and represents the
finance-specific regime-shift story: a calm-market model is now operating in a
stress regime.

### Delayed forecast error

The five-day target is unknowable at prediction time. Each run therefore
reconstructs recent forecasts using the same fixed model and only prices that
were available on each as-of date. It compares RMSE over the latest 20 eligible
predictions with the preceding 40:

- ratio below 1.50: `ok`
- ratio from 1.50 to 2.00: `warning`
- ratio at or above 2.00: `critical`

RMSE, MAE, and QLIKE are included in the report. A non-positive forecast makes
QLIKE undefined and is treated as critical.

## Outputs and alerting

Each run uploads:

- `predictions.csv` — 60 scored historical forecasts plus the current live one;
- `report.json` — machine-readable metrics and status;
- `report.md` — human-readable report shown in the workflow summary;
- `latest_prices.parquet` — the exact refreshed input snapshot.

Warnings and critical states emit GitHub workflow annotations. This is the MVP
alert channel; a Slack webhook is intentionally left as a stretch enhancement.

## Run locally

With an exported model already in `model/`:

```bash
uv run python -m scripts.run_monitoring \
  --prices data/raw/AAPL.parquet \
  --model-uri model
```

To reproduce the cloud path instead:

```bash
uv run python -m scripts.download_production_model --output downloaded-model
uv run python -m scripts.run_monitoring --model-uri downloaded-model
```
