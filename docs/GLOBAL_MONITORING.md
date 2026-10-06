# Global candidate monitoring

Phase 7 extends monitoring to the pooled 34-target candidate while preserving
the AAPL production monitor. VIX is the 35th ingested series and remains shared
context, not a forecast target.

## Why this is a separate path

The global candidate has different failure modes from a single-ticker model:
partial-universe data, stale symbols, one ticker dominating an aggregate, and
34 simultaneous alerts for one run. The global monitor therefore uses the same
authoritative universe as training and serving, fails closed on incomplete
coverage, and emits one fleet-level workflow annotation with drill-down data.

The existing `.github/workflows/monitoring.yml`, `configs/monitoring.yaml`, and
`monitoring/reference.json` remain the AAPL production path.

## Reference contract

`monitoring/global_reference.json` is generated from the exact accepted global
training panel:

```bash
uv run python -m scripts.build_global_monitoring_reference
```

The reference is checksum-bound to the panel recorded by the candidate
artifact. It contains:

- fixed PSI bins for the four asset features, separately for every ticker;
- fixed PSI bins for VIX level and VIX return, stored once rather than repeated
  34 times;
- per-ticker RV20 training p95 and p99 regime thresholds;
- the exact target order, groups, context role, training range, and panel hash.

Startup rejects a reference whose panel hash, features, universe, groups, or VIX
role differs from the loaded candidate.

## Daily checks

### Data quality

All 34 targets and VIX must be present, unique by date/ticker, positive, finite,
and have at least 90 observations. A series more than two observed market
sessions behind the freshest input fails the run. Forecast rows must cover all
targets exactly once, in universe order, on one synchronized as-of date.
Refreshed and scheduled runs also require the entire basket to be no more than
seven calendar days behind the run date. Offline cached replays skip only this
wall-clock comparison and record that fact in `data_quality`.

### Feature drift

PSI uses the same thresholds as the AAPL monitor:

- below `0.10`: `ok`;
- `0.10` to `0.25`: `warning`;
- at least `0.25`: `critical`.

Asset features are evaluated per ticker. Shared VIX drift is evaluated once so
it cannot be mistaken for 34 independent asset shifts.

### Volatility regime

Each ticker's current RV20 is compared with its own training distribution:

- at or below p95: `ok`;
- above p95: `warning`;
- above p99: `critical`.

### Delayed error

The monitor reconstructs 60 synchronized as-of forecasts. Every model call
receives only the 120 raw observations available through that as-of date. The
five-session target is joined only after prediction, once it is observable.

The latest 20 dates are compared with the preceding 40. Recent/baseline RMSE
ratios of `1.50` and `2.00` produce warning and critical states. Reports include
micro, equal-ticker macro, asset-group, and per-ticker RMSE, MAE, and QLIKE.

## Alert aggregation

Per-ticker states remain available in JSON, but the workflow emits one alert:

- fewer than 10% non-ok tickers: fleet `ok`, unless any ticker is critical;
- at least 10% non-ok tickers, or any critical ticker: fleet `warning`;
- at least 25% critical tickers: fleet `critical`.

This makes isolated failures visible without turning one outlier into a
portfolio-wide critical page. Data-contract failures still fail the run
immediately.

## Run locally

Use the exported candidate and cached data:

```bash
uv run python -m scripts.run_global_monitoring
```

Use an exact long-form Parquet snapshot instead:

```bash
uv run python -m scripts.run_global_monitoring \
  --prices path/to/global_prices.parquet \
  --model-uri global_model
```

Add `--refresh` to download all 35 histories from yfinance. The run writes:

- `report.json`: aggregate, group, and per-ticker diagnostics;
- `report.md`: concise workflow summary;
- `predictions.csv`: 2,040 delayed forecasts plus 34 live forecasts;
- `latest_prices.parquet`: the exact all-series input snapshot.

The first real cached-data run was as of 2026-06-26 and covered all 34 targets
plus VIX. Data quality was `ok`; drift was `critical` (34/34 asset distributions
and shared VIX differed from the full training reference); regime was `warning`
(7/34 elevated); delayed error was `warning` (14/34 affected) while pooled
micro recent/baseline RMSE remained `ok` at `1.267`. This is evidence that the
monitor detects broad distribution change, not evidence of model failure by
itself.

## Candidate workflow and promotion gate

`.github/workflows/global-monitoring.yml` supports manual candidate QA and a
weekday candidate schedule. Every run downloads the exact URL and SHA-256 in
`configs/global_release.yaml`; there are no floating versions or repository
variable overrides. GitHub runs schedules from the default branch, so the new
schedule becomes active only after the global branch is deliberately merged.
Manual dispatch on this branch provides the pre-merge acceptance run.

The AAPL schedule remains the production monitor. The global workflow is
explicitly candidate-only and does not change the production model, API, or
dashboard defaults. Release packaging and retrieval are documented in
`docs/GLOBAL_RELEASE.md`.
