# Global deployment cutover

**Decision date:** 2026-10-07 (America/Toronto)
**Decision:** deploy the global model as the repository's only active serving
and monitoring path.

This is an explicit product decision. It replaces the AAPL-only deployment
surface with the pooled 34-target model while retaining the prior implementation
in Git history and its immutable historical release.

## Cutover performed

- `POST /predict` now accepts the global long-form price contract and returns
  one forecast per supplied target; the old AAPL price-list endpoint is removed.
- The Docker image bakes one global artifact and supports its MLflow and FP32
  ONNX backends through the same `MODEL_BACKEND` setting.
- `dashboard/app.py` is the global 34-target dashboard.
- The AAPL scheduled workflow, AAPL deployment configuration, AAPL dashboard
  runtime, and AAPL serving modules are removed from active deployment code.
- One weekday global monitoring workflow refreshes all 34 targets plus VIX.

## Evidence retained

The cutover does not change the published model evidence. On the final
252-session holdout, the global model improves pooled RMSE (`0.14235` versus
GARCH `0.14436`) and MAE (`0.08837` versus `0.09785`), while GARCH has stronger
QLIKE (`0.52160` versus `0.72611`). The dashboard and README report that
trade-off. The immutable release has SHA-256
`c40b48f186a9255f6afcfffde1a5ca02f0ff2379ab9d68da87d0b024d353ee0a`.

## Rollback

Rollback is a Git/release deployment decision, not a model mutation: deploy the
last AAPL-only revision and its checksum-pinned `aapl-lstm-v1` release from Git
history. Do not overwrite or relabel either immutable artifact.
