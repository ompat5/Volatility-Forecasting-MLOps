# Global model candidate

The global candidate forecasts five-session realized volatility for 34 equities
and ETFs with one pooled LSTM. `^VIX` is a required context series, not a
forecast target. It now has an opt-in FastAPI path, while the AAPL-only model
remains the production API default, monitoring, ONNX, and dashboard model.

## Evidence-gated training

The packaging entrypoint refuses to train unless all of these match the
committed Phase 3 benchmark:

- balanced-panel SHA-256;
- complete global configuration;
- ordered 34-target vocabulary and VIX context;
- GARCH-inclusive final-holdout evidence; and
- the cross-validation-selected epoch count.

After that check, the candidate refits for four epochs on every
target-observable panel date. This is post-evaluation training: the frozen
holdout metrics remain the reported evidence, while the deployable candidate
uses all historical labels that are now known.

```bash
uv run python -m scripts.train_global
```

The script registers a separate `global-volatility-lstm` model. It never writes
to the AAPL `volatility-lstm` registry entry. New versions are tagged
`promotion_status=candidate`, `production_default=false`, and
`serving_enabled=false`.

## Artifact contents

One MLflow pyfunc contains:

- PyTorch `state_dict` weights;
- the fitted six-feature `StandardScaler`;
- ordered ticker-to-embedding IDs;
- target groups and VIX role;
- feature order, sequence length, horizon, and model architecture;
- full training configuration and panel coverage;
- Phase 3 benchmark metrics and benchmark SHA-256;
- model/scaler component checksums; and
- explicit input and output schemas.

The relevant `src` package is embedded through MLflow `code_paths`, so loading
the snapshot does not depend on running from this repository checkout.

The wrapper verifies the component and benchmark checksums before loading.
Export requires an immutable numeric version; `latest` is intentionally not an
option:

```bash
uv run python -m scripts.export_global_model --version 3
```

This writes the self-contained snapshot to the gitignored `global_model/`
directory and refuses to overwrite an existing snapshot.

## Inference contract

Input is a long-form DataFrame with exactly these columns:

| Column | Type | Meaning |
|---|---|---|
| `date` | datetime | Market session date |
| `ticker` | string | Known target symbol or `^VIX` |
| `adjusted_close` | positive float | Raw adjusted close; VIX remains in quoted points |

Every request must contain `^VIX` plus at least one known target. A target
subset is allowed. Each supplied series needs enough overlapping history to
produce 30 complete feature rows—normally at least 90 sessions because RV60 is
the longest rolling feature. Unknown symbols, missing VIX, duplicate
`(date, ticker)` keys, non-positive prices, and incomplete histories fail
closed.

Output contains one row per requested target:

| Column | Meaning |
|---|---|
| `ticker` | Forecast target |
| `as_of_date` | Latest target/VIX session used |
| `horizon_sessions` | Always 5 for this model version |
| `forecast` | Strictly positive annualized realized-volatility forecast |

The artifact performs feature construction and scaling internally; callers do
not send engineered features or embedding IDs.

## Opt-in FastAPI serving

The existing AAPL `POST /predict`, `GET /health`, `MODEL_URI`, and
`MODEL_BACKEND` behavior is unchanged. Global serving is enabled only when
`GLOBAL_MODEL_URI` is present:

```bash
GLOBAL_MODEL_URI=models:/global-volatility-lstm/3 \
  uv run uvicorn src.serving.app:app --port 8000
```

Registry URIs must select an explicit numeric version of the separate
`global-volatility-lstm` model; floating `latest` and model aliases are rejected.
A local exported snapshot such as `GLOBAL_MODEL_URI=global_model` is also
supported.

`GET /health/global` reports `disabled` when the optional artifact is absent and
`ok` only after it has loaded successfully. At startup the API unwraps the
pyfunc and verifies all of the following against repository configuration:

- global-candidate artifact metadata;
- the exact ordered set of 34 target tickers;
- the single `^VIX` context series; and
- the five-session forecast horizon.

This prevents an AAPL model or stale basket artifact from appearing ready under
the global route. A configured but incompatible artifact fails application
startup. With no global URI, the AAPL service starts normally and
`POST /predict/global` returns HTTP 503.

HTTP requests wrap the artifact's long-form input rows in `observations`:

```json
{
  "observations": [
    {"date": "2026-01-02", "ticker": "AAPL", "adjusted_close": 250.1},
    {"date": "2026-01-02", "ticker": "^VIX", "adjusted_close": 16.8}
  ]
}
```

The abbreviated example shows the shape only; real requests need enough history
to construct the complete 30-session feature window. The API checks the model's
output again and requires every requested target exactly once, in universe
order, with the expected horizon and a finite positive forecast.

CI generates a deterministic fixture with the production packaging contract
and the complete 34-target vocabulary. It starts the same Docker image twice:
once without a global URI to prove the AAPL-only default still works, and once
with the fixture mounted read-only to exercise all 34 forecasts end to end. The
fixture is explicitly marked as untrained and is never a release candidate.

## Phase 4 verification

The real-data registration run fitted 117,368 sequences across 3,481 dates;
the scaler fitted 118,354 panel rows. Loading the exact registered version and
passing the latest raw histories for all 34 targets plus VIX returned 34 finite,
positive forecasts dated 2026-06-26. Loading the explicit exported snapshot
also succeeded for an AAPL/SPY subset from `/tmp` with the repository absent
from `PYTHONPATH`.

The real candidate and the deterministic CI artifact both pass direct-pyfunc and
FastAPI parity checks.

## Phase 6: explicit FP32 ONNX runtime

The candidate's positive-output neural-network core can now be exported to a
two-input FP32 ONNX graph and selected with `GLOBAL_MODEL_BACKEND=onnx`.
Preprocessing and the fitted scaler remain shared Python code. Export verifies
every ticker embedding against PyTorch and writes a checksummed sidecar bound to
the exact source artifact.

On the recorded 34-target one-thread run, ONNX reduced median core latency from
1.8345 ms to 0.7304 ms (2.5×) but reduced full raw-history latency only from
147.6555 ms to 146.7871 ms (~0.6%). Maximum forecast difference was `5.96e-08`.
See [`docs/GLOBAL_ONNX.md`](GLOBAL_ONNX.md) for the export, benchmark, and serving
runbook and [`benchmarks/global_onnx_fp32.json`](../benchmarks/global_onnx_fp32.json)
for the complete evidence.

Candidate monitoring now covers all 34 targets at aggregate, asset-group, and
per-ticker levels; see [`docs/GLOBAL_MONITORING.md`](GLOBAL_MONITORING.md).
The AAPL scheduled monitor, dashboard, immutable release, and production default
remain authoritative until their later migration gates pass.
