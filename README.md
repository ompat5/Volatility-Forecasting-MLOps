# Volatility Forecasting MLOps

[![CI](https://github.com/ompat5/Volatility-Forecasting-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/ompat5/Volatility-Forecasting-MLOps/actions/workflows/ci.yml)

> An end-to-end MLOps system for five-day AAPL realized-volatility forecasting:
> leakage-safe evaluation, MLflow, FastAPI, Docker, CI, monitoring, optimized
> ONNX inference, and a live Streamlit demo.

## Result

The PyTorch LSTM reached **0.2061 RMSE** and **0.1408 MAE** on AAPL walk-forward
evaluation—about 6% and 7% better than EWMA respectively—while EWMA retained a
small QLIKE edge. The modest result is the point: the project emphasizes honest
time-series evaluation and the production lifecycle rather than an implausible
claim of crushing classical volatility models.

## Demo

**[Open the live AAPL volatility dashboard](https://volatility-forecaster.streamlit.app/)**

The Streamlit demo shows the latest forecast, recent realized volatility,
forecast versus realized history, stored model comparisons, and live monitoring
indicators. It is explicitly AAPL-only because the deployed model is AAPL-only.

## Architecture

```mermaid
flowchart TD
    A["Data ingestion<br/>(35-symbol cache + VIX)"] --> B["Leakage-safe features<br/>(returns + realized vol)"]
    B --> C["Baseline models<br/>(naive, EWMA, GARCH)"]
    B --> D["AAPL LSTM<br/>(PyTorch)"]
    C --> E["Walk-forward evaluation<br/>(tracked in MLflow)"]
    D --> E
    E --> F["MLflow registry +<br/>versioned model release"]
    F --> G["Inference API<br/>(FastAPI in Docker)"]
    F --> J["FP32 ONNX export<br/>+ measured optimization"]
    J --> G
    J --> H["Live AAPL-only<br/>Streamlit demo"]
    F --> I["Monitoring<br/>(drift + error alerts)"]
```

See [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) for the full evaluation protocol and week-by-week plan.

## Results

All models use the same five-fold expanding-window protocol with a five-day gap.

| Model | RMSE | MAE | QLIKE |
|---|---:|---:|---:|
| Naive | 0.2570 | 0.1698 | 1.6150 |
| EWMA | 0.2195 | 0.1508 | **0.5792** |
| GARCH(1,1) | 0.2441 | 0.1892 | 0.6962 |
| LSTM | **0.2061** | **0.1408** | 0.5822 |

The production default and scheduled production monitoring are **AAPL-only**.
The repository caches 34 forecast targets plus VIX context. A global-model
migration is in progress: its pooled LSTM candidate has been evaluated,
packaged, optimized, served behind an opt-in route, and given a separate
candidate monitor and candidate-only dashboard. It is not promoted, and its
schedule remains disabled until the immutable-release and cutover gates pass.

### Global candidate benchmark (not production)

The candidate is one shared LSTM for all 34 forecast targets, with a learned
ticker embedding and same-session VIX context. It trains on log volatility,
returns positive forecasts, balances every ticker in each training batch, and
uses one purged calendar for the whole basket. Five expanding folds selected a
four-epoch final refit, which was scored once on the untouched 252-session
holdout from 2025-06-18 through 2026-06-18 (8,568 observations per model).

| Model | Macro RMSE | Micro RMSE | Micro MAE | Micro QLIKE |
|---|---:|---:|---:|---:|
| Naive | 0.16156 | 0.18064 | 0.11894 | 1.62829 |
| EWMA | 0.12837 | 0.14536 | 0.09694 | 0.54039 |
| GARCH(1,1) | 0.12853 | 0.14436 | 0.09785 | **0.52160** |
| Global LSTM | **0.12581** | **0.14235** | **0.08837** | 0.72611 |

The global LSTM has the best holdout RMSE and MAE, but GARCH retains a clear
QLIKE advantage. At ticker level, the LSTM beats GARCH on RMSE for 21/34
targets, MAE for 28/34, and QLIKE for only 5/34. That mixed result is reported
as-is: it clears the implementation/evaluation gate, not the production
promotion gate. Reproducible configuration, split boundaries, aggregate/group
metrics, and per-ticker holdout metrics are in
[`benchmarks/global_model.json`](benchmarks/global_model.json).

The global artifact bundles its weights, scaler, ticker vocabulary, feature
schema, configuration, and benchmark evidence. Its long-form raw-price contract
and promotion safeguards are documented in
[`docs/GLOBAL_MODEL.md`](docs/GLOBAL_MODEL.md). `POST /predict/global` loads only
with an explicit global MLflow URI or ONNX backend; the existing AAPL
`POST /predict`, `/health`, model URI, and backend defaults remain unchanged.

### Global FP32 ONNX candidate (not production)

The global LSTM core now has a validated FP32 ONNX export with dynamic batching
across all 34 ticker embeddings. Raw-price features and scaling remain in shared
Python code. On the recorded one-thread run, ONNX reduced median 34-target core
latency from `1.8345 ms` to `0.7304 ms` (2.5×), but full request latency moved
only from `147.6555 ms` to `146.7871 ms` (~0.6%) because preprocessing dominates.
Maximum PyTorch/ONNX forecast difference was `5.96e-08`.

| Path | PyTorch median | ONNX FP32 median |
|---|---:|---:|
| Scaled `(34, 30, 6)` tensors + ticker IDs | 1.8345 ms | **0.7304 ms** |
| 4,200 raw-price rows → 34 forecasts | 147.6555 ms | **146.7871 ms** |

See [`docs/GLOBAL_ONNX.md`](docs/GLOBAL_ONNX.md) for the design and runbook and
[`benchmarks/global_onnx_fp32.json`](benchmarks/global_onnx_fp32.json) for the
complete hashes, environment, timings, and per-ticker parity evidence.

### Global candidate monitoring (not production)

Phase 7 monitors all 34 targets plus shared VIX context without changing the
AAPL production workflow. It rejects partial or stale universes, reconstructs
60 synchronized as-of forecasts without future inputs, and reports drift,
regime, and delayed error at portfolio, asset-group, and per-ticker levels.
One fleet alert replaces 34 independent annotations: 10% non-ok tickers produce
a warning, while critical requires at least 25% of the universe to be critical.

The first real cached-data run, as of 2026-06-26, had complete data quality,
broad critical feature drift, 7/34 elevated volatility regimes, and 14/34
error-affected tickers. Pooled recent/baseline RMSE remained `ok` at `1.267`.
See [`docs/GLOBAL_MONITORING.md`](docs/GLOBAL_MONITORING.md) for thresholds,
methodology, workflow activation, and interpretation.

### FP32 ONNX benchmark

The production LSTM core was exported to ONNX and measured on CPU with one
thread per runtime, 100 warm-up calls, and 1,000 timed calls. These are local
Darwin x86_64 measurements, not universal latency guarantees.

| Path | Runtime | Median | p95 |
|---|---|---:|---:|
| Scaled tensor → forecast | PyTorch | 0.3820 ms | 0.7280 ms |
| Scaled tensor → forecast | ONNX FP32 | **0.1036 ms** | **0.2153 ms** |
| Raw prices → forecast | PyTorch | 4.2884 ms | 6.3510 ms |
| Raw prices → forecast | ONNX FP32 | **3.9003 ms** | **6.1185 ms** |

ONNX reduced median core latency by about 73% (3.7× faster), but reduced
end-to-end median latency by about 9% because pandas feature construction and
scaling dominate the full path. The ONNX artifact is 74,079 bytes versus 75,764
bytes for PyTorch (about 2% smaller), and the production forecasts differed by
only `2.98e-08`. The machine-readable result is in
[`benchmarks/onnx_fp32.json`](benchmarks/onnx_fp32.json).

### INT8 quantization decision

Dynamic INT8 quantization was evaluated on the final 60 target-observable AAPL
dates in the cached dataset (2026-03-25 through 2026-06-18). FP32 and INT8 used
identical raw-price windows, features, scaler, and targets.

| Runtime | Size | Core median | End-to-end median | RMSE | MAE | QLIKE |
|---|---:|---:|---:|---:|---:|---:|
| ONNX FP32 | 74,079 B | 0.0755 ms | 3.4405 ms | 0.09472 | 0.08030 | 0.37972 |
| ONNX INT8 | 23,715 B | 0.0701 ms | 3.4142 ms | 0.09331 | 0.07932 | 0.35664 |

INT8 is 68% smaller, but improves median end-to-end latency by less than 1%.
Its forecasts differ from FP32 by `0.0054` on average and `0.0121` at maximum.
The slightly better metrics on this small slice are treated as quantization
noise, not a model improvement. **Decision: keep FP32 as the optimized serving
candidate** because the roughly 50 KiB saving and negligible application-level
speedup do not justify the larger numerical deviation. The full evidence is in
[`benchmarks/onnx_int8.json`](benchmarks/onnx_int8.json).

## How to run

```bash
uv sync
uv run pytest
uv run ruff check .
```

Build and evaluate the global candidate (the full run includes 204 per-fold and
holdout GARCH fits):

```bash
uv run python -m scripts.build_global_panel
uv run python -m scripts.build_global_splits
uv run python -m scripts.evaluate_global_model
uv run python -m scripts.train_global
uv run python -m scripts.export_global_model --version 3  # use the printed version
uv run python -m scripts.export_global_onnx
uv run python -m scripts.benchmark_global_inference
```

Train and register the current AAPL production model:

```bash
uv run python -m src.data.ingest
uv run python -m scripts.train
```

Run the API from the local MLflow registry:

```bash
uv run uvicorn src.serving.app:app --port 8000
curl http://127.0.0.1:8000/health
```

Opt into the global candidate with an explicit registry version:

```bash
GLOBAL_MODEL_URI=models:/global-volatility-lstm/3 \
  uv run uvicorn src.serving.app:app --port 8000
curl http://127.0.0.1:8000/health/global
```

`POST /predict/global` accepts `observations`, a long-form list of
`{date, ticker, adjusted_close}` records. Include `^VIX` and one or more known
targets; the response contains one positive five-session forecast per target.
The service validates the loaded artifact against the exact 34-target universe
at startup. Without a global URI or explicit ONNX backend, this endpoint returns
HTTP 503 while the AAPL service continues normally.

After exporting the global graph, explicitly select its ONNX runtime with
`GLOBAL_MODEL_URI` unset:

```bash
GLOBAL_MODEL_BACKEND=onnx \
  uv run uvicorn src.serving.app:app --port 8000
```

Or explicitly opt into the FP32 ONNX backend after exporting the local model:

```bash
uv run python -m scripts.export_model
uv run python -m scripts.export_onnx
MODEL_BACKEND=onnx uv run uvicorn src.serving.app:app --port 8000
```

MLflow remains the default backend. The ONNX path loads the graph and scaler
once at API startup and preserves the same raw-price request contract.

Build the production container from an exported registered model:

```bash
uv run python -m scripts.export_model
docker build -t volatility-forecaster .
docker run --rm -p 8000:8000 volatility-forecaster
```

The image still bakes in only the AAPL production artifact. To exercise the
separately exported global candidate without changing that image default, mount
the immutable snapshot explicitly:

```bash
docker run --rm \
  -e GLOBAL_MODEL_URI=/app/global_model \
  -v "$(pwd)/global_model:/app/global_model:ro" \
  -p 8000:8000 volatility-forecaster
```

Use the same mount with `-e GLOBAL_MODEL_BACKEND=onnx` instead of
`GLOBAL_MODEL_URI` to exercise the explicit global ONNX path.

To exercise the optimized backend in that image:

```bash
docker run --rm -e MODEL_BACKEND=onnx -p 8000:8000 volatility-forecaster
```

Export the trained LSTM core to a validated FP32 ONNX graph:

```bash
uv run python -m scripts.export_onnx
uv run python -m scripts.benchmark_inference
uv run python -m scripts.quantize_onnx
```

The ONNX graph accepts scaled tensors shaped `(batch, 30, 4)`. Raw-price
feature construction and the fitted `StandardScaler` intentionally remain in
Python so optimization does not change the model's preprocessing contract.

Run the AAPL dashboard with fresh market data:

```bash
uv run streamlit run dashboard/app.py
```

For a fully offline, reproducible view using the cached AAPL prices:

```bash
DASHBOARD_PRICES_PATH=data/raw/AAPL.parquet \
  uv run streamlit run dashboard/app.py
```

The dashboard reuses local model artifacts when available. If they are absent,
it downloads the checksum-pinned `aapl-lstm-v1` release into temporary storage
and exports the FP32 ONNX graph once per application process.

Run the separate 34-target candidate dashboard against the local ingestion
cache:

```bash
GLOBAL_DASHBOARD_USE_CACHED_PRICES=1 \
  uv run streamlit run dashboard/global_app.py
```

The candidate page loads one synchronized 34-target-plus-VIX snapshot, then
slices it instantly by ticker and group. It includes fleet/ticker monitoring
and ticker/group/portfolio final-holdout comparisons. It is intentionally not
the public default and requires an exported `global_model/` candidate. See the
[global dashboard runbook](docs/GLOBAL_DASHBOARD.md).

Run the global candidate monitor with the exported `global_model/` snapshot and
cached all-symbol histories:

```bash
uv run python -m scripts.run_global_monitoring
```

Add `--refresh` to refetch all 35 series. This remains candidate QA; it does not
replace the AAPL production schedule.

See the [AAPL monitoring runbook](docs/MONITORING.md) and
[global candidate runbook](docs/GLOBAL_MONITORING.md) for thresholds, artifacts,
and local commands.

## Status

Phases 1–5 are complete. Phase 6 optimization and demo work is merged to `main`:
FP32 ONNX export, parity validation, benchmarks, the measured INT8 decision, an
opt-in ONNX API backend, and the public AAPL-only Streamlit dashboard are all
implemented. Global migration Phases 1–8 now add the pooled-model path through
opt-in API, FP32 ONNX export, all-embedding parity, measured benchmarking,
explicit ONNX serving, and full-universe candidate monitoring without changing
that production default. Global migration Phase 8 adds the complete candidate
dashboard and group-level views as a separate entrypoint. Remaining global work
is the immutable release and eventual cutover; final portfolio polish follows
the migration. The current branch has **192 tests**.

### Continuous integration

Every push and pull request runs Ruff, the pytest suite, a Docker build, and
three end-to-end container smoke tests. The first uses the deterministic AAPL
fixture and proves global serving is disabled by default. The other two mount a
separate deterministic global fixture and require all 34 forecasts through the
MLflow and explicit ONNX runtimes. Neither fixture is trained or ever published.

Production images continue to use an explicitly exported registered model:

```bash
uv run python -m scripts.export_model
docker build -t volatility-forecaster .
```

### Scheduled monitoring

A weekday GitHub Actions pipeline refreshes AAPL data, downloads the immutable
checksum-pinned production model, generates a forecast, and reports feature
drift, volatility-regime state, and delayed rolling forecast error. Predictions,
the input snapshot, and Markdown/JSON reports are retained as workflow artifacts.

A separate global-candidate workflow implements the same lifecycle across all
34 targets with aggregate/group/ticker views and one fleet alert. Its weekday
schedule is deliberately dormant until an immutable global archive URL and
SHA-256 are configured; manual candidate runs are available now.

See [docs/MONITORING.md](docs/MONITORING.md) and
[docs/GLOBAL_MONITORING.md](docs/GLOBAL_MONITORING.md) for thresholds,
methodology, and local commands.
