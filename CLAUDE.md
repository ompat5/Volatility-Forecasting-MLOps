# CLAUDE.md

Project memory for Claude Code. Keep this file lean — it loads every session. Long reference material lives in `docs/BUILD_PLAN.md`.

## Project

Deep learning system that forecasts 5-day **realized volatility** for a basket of equities/ETFs, benchmarked against a GARCH baseline, wrapped in a full MLOps lifecycle (experiment tracking, deployed API, CI/CD, drift monitoring, inference optimization, demo).

Full week-by-week plan, architecture, and evaluation protocol: see `docs/BUILD_PLAN.md`. Read it before starting a new phase.

Goal: a portfolio-grade, end-to-end project for an ML / AI engineer resume. The MLOps wrapper is the point, not raw model accuracy.

## About me / how to help

- Recent data science grad. Strong on ML *theory*; new to most of the *tooling* (MLflow, FastAPI, Docker, CI/CD, ONNX, Streamlit). Explain the *why* behind tooling choices, not just the commands.
- Prefer incremental steps I can understand and run, over large dumps of code I can't follow.
- When introducing a new tool, give a one-line "what this is / why we use it here" before the code.
- Point out when I'm about to do something non-idiomatic or fragile.

## Environment & conventions

- Python version: 3.12 (pinned via uv — chosen over the system 3.14 for ML library compatibility: PyTorch/MLflow/ONNX/evidently wheels lag the newest CPython release)
- Env manager: uv
- Lint/format: ruff
- Tests: pytest
- Style: type hints on functions; small, testable modules in `src/`; notebooks in `notebooks/` are for exploration only — production logic lives in `src/`.

## Repo structure (target)

```
data/            # cached raw + processed data (gitignored)
src/
  features/      # returns, realized vol, feature pipeline
  models/        # baselines (GARCH/EWMA/naive) + DL (LSTM/TCN)
  eval/          # walk-forward splits + metrics (RMSE, MAE, QLIKE)
  serving/       # FastAPI app
notebooks/       # exploration only
tests/
configs/         # YAML/Hydra hyperparameters
docs/BUILD_PLAN.md
```

## Methodology guardrails (NON-NEGOTIABLE — finance time series)

These prevent data leakage, the single most common way this kind of project loses credibility. Never violate them, even if it simplifies code:

- **Predict volatility, never price or returns.**
- **Walk-forward / expanding-window validation only.** Never shuffle. Test dates must never precede train dates.
- **Fit scalers/normalizers on the training slice only**, then apply to validation/test. Never fit on the full dataset.
- **Verify feature lags.** A feature used to predict day *t* must use only information known at *t* or earlier; realized-vol features must not peek into the target window.
- **GARCH(1,1) is the baseline to beat.** A modest win or a tie is the honest, expected result. Be suspicious of any large improvement — it usually means leakage.

## Current status

<!-- Update this line as you progress so a fresh session orients instantly -->
Phase 0 — repo setup. Nothing built yet. Next: Phase 1 (data ingestion, realized-vol target, walk-forward harness, GARCH baseline).
