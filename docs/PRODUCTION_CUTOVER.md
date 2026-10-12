# Global deployment

**Decision date:** 2026-10-07 (America/Toronto)
**Decision:** the pooled 34-target model is the repository's sole serving and
monitoring contract.

## Active contract

- `POST /predict` accepts long-form adjusted-close observations for one or more
  targets plus mandatory `^VIX` context and returns a positive five-session
  volatility forecast per requested target.
- Docker supports the checksum-bound MLflow pyfunc and its parity-validated
  FP32 ONNX core through `MODEL_BACKEND`.
- `dashboard/app.py` is the 34-target Streamlit dashboard.
- One weekday workflow refreshes the 34 targets plus VIX and records data
  quality, drift, regime, and delayed-error evidence.

## Evidence boundary

The final 252-session holdout is frozen evidence, not a selection surface. The
global model's pooled RMSE/MAE are `0.14235`/`0.08837` versus GARCH's
`0.14436`/`0.09785`; GARCH retains better QLIKE (`0.52160` versus `0.72611`).
Any successor must be selected only on validation folds and then evaluated on
new time-forward data.

## Rollback

Rollback is a Git/release deployment decision. Do not mutate, relabel, or
partially replace the checksum-pinned production artifact.
