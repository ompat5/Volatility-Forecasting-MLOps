# Global model candidate

The global candidate forecasts five-session realized volatility for 34 equities
and ETFs with one pooled LSTM. `^VIX` is a required context series, not a
forecast target. The AAPL-only model remains the production API, monitoring,
ONNX, and dashboard default.

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

## Phase 4 verification

The real-data registration run fitted 117,368 sequences across 3,481 dates;
the scaler fitted 118,354 panel rows. Loading the exact registered version and
passing the latest raw histories for all 34 targets plus VIX returned 34 finite,
positive forecasts dated 2026-06-26. Loading the explicit exported snapshot
also succeeded for an AAPL/SPY subset from `/tmp` with the repository absent
from `PYTHONPATH`.

This proves the artifact contract only. FastAPI, ONNX, monitoring, scheduled
jobs, and the dashboard remain AAPL-only until their later migration gates pass.
