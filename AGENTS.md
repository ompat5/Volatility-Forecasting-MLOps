# AGENTS.md

## Project

This repository deploys one pooled model for five-session realized-volatility
forecasts across 34 equity and ETF targets. `^VIX` is required market context,
not a forecast target. FastAPI, Docker, Streamlit, CI, ONNX, and scheduled
monitoring all use this single global contract.

The immutable artifact is pinned in `configs/global_release.yaml`. Its release
tag contains `candidate` as historical provenance, but it is the active
repository artifact. See `docs/BUILD_PLAN.md` before changing architecture or
starting a new model iteration.

## Working style

- Python 3.12 via `uv`; lint and format with `ruff`; test with `pytest`.
- Keep reusable, typed production code under `src/`; notebooks are not part of
  the production contract.
- Explain new MLOps tooling briefly and make changes incrementally.

## Finance time-series guardrails

- Forecast volatility only, never price or returns.
- Use chronological, purged walk-forward evaluation only; never shuffle dates.
- Fit preprocessing on training dates only.
- Ensure target features use only information available at the prediction date.
- Preserve ticker isolation in sequence windows and full-universe alignment in
  panel splits, inference, serving, and monitoring.
- GARCH is a mandatory comparator. Large improvements require a leakage audit.
- The final holdout is frozen evidence, never a model-selection surface.

## Current status

The global system is fully implemented and deployed in the repository:

- Balanced 34-target + VIX panel, purged splits, train-only scaling, pooled
  LSTM with ticker embeddings, and globally fair batches.
- Per-ticker, group, macro, and micro evaluation against naive, EWMA, and GARCH.
- Evidence-gated MLflow artifact, full-universe FastAPI and Docker contract,
  parity-validated FP32 ONNX runtime, Streamlit dashboard, and scheduled
  monitoring.
- CI validates the complete universe through both MLflow and ONNX containers.

The published final-holdout result is mixed: global LSTM improves pooled RMSE
and MAE versus GARCH, while GARCH retains better QLIKE (`0.52160` versus
`0.72611`). Improve successors using validation-only selection and require new
time-forward evidence before promotion.
