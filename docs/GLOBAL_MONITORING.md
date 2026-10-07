# Global monitoring

The global monitoring workflow is the repository's only scheduled production
monitor. It refreshes all 34 forecast targets and the shared `^VIX` context,
downloads the checksum-pinned global artifact, and fails closed when the
complete universe is unavailable.

## Checks

- **Data quality:** every series is positive, finite, unique by `(date, ticker)`,
  sufficiently long, synchronized, and within the wall-clock freshness limit.
- **Feature drift:** fixed PSI bins compare ticker features and shared VIX
  features to the training panel.
- **Volatility regime:** each ticker's RV20 is compared with its own training
  p95 and p99 thresholds.
- **Delayed error:** 60 synchronized reconstructed forecasts are joined with
  realized targets only after the five-session horizon is observable. Reports
  include micro, macro, group, and ticker RMSE, MAE, QLIKE, and a recent versus
  preceding-window error ratio.

Ticker states aggregate into one fleet status, avoiding 34 independent alerts
for the same data event. In particular, a critical PSI signal describes a
distribution change; it does not by itself prove the model failed.

## Run locally

```bash
uv run python -m scripts.download_global_model
uv run python -m scripts.run_global_monitoring --refresh
```

The command writes `global-monitoring-output/` containing `report.json`, a
Markdown summary, every reconstructed prediction, and the exact input snapshot.

## Scheduled workflow

`.github/workflows/global-monitoring.yml` runs on weekdays at 19:07
America/Toronto and can also be dispatched manually. It uploads monitoring
artifacts for 90 days and emits one GitHub annotation when the fleet status is
not `ok`.

The accepted fresh release-backed run produced 34 live forecasts and 2,040
delayed rows. Data quality, regime, and delayed error were `ok`; feature drift
was `critical` relative to the long training reference and remains visible in
the dashboard and workflow output.
