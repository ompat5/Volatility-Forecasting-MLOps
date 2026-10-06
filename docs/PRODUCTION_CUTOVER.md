# Global-model production cutover review

**Review date:** 2026-10-06 (America/Toronto)  
**Decision:** **No-go for direct production cutover.**

The global model is an operationally sound, immutable **candidate**, but it
must not replace the AAPL production default yet. This is a model-quality
decision, not a deployment-quality failure. The AAPL API, dashboard, release,
and scheduled monitor remain the production path.

## Scope and evidence reviewed

The review covered the complete system rather than treating a successful model
export as a promotion:

| Gate | Evidence | Result |
|---|---|---|
| Leakage-safe evaluation | Five expanding, purged calendar folds; a 252-session final holdout with 8,568 pooled observations; train-only preprocessing | Pass |
| Predictive quality | Global LSTM micro RMSE `0.14235` and MAE `0.08837` versus GARCH `0.14436` and `0.09785` | Partial pass |
| Volatility loss | Global LSTM micro QLIKE `0.72611` versus GARCH `0.52160`; the LSTM is worse by 39.2% and wins QLIKE for only 5/34 targets | **Fail** |
| Immutable artifact | `global-volatility-lstm` version 3, GitHub prerelease archive, SHA-256 `c40b48f186a9255f6afcfffde1a5ca02f0ff2379ab9d68da87d0b024d353ee0a` | Pass |
| Runtime integrity | MLflow and checksum-bound FP32 ONNX paths validated for every ticker embedding; maximum ONNX/PyTorch difference `5.96e-08` | Pass |
| API and container | Long-form global API validates the exact 34-target-plus-VIX contract; CI exercises all 34 targets through both runtimes | Pass as a candidate path |
| Dashboard | Separate global dashboard renders all target, group, fleet, and holdout views from one synchronized snapshot | Pass as a candidate path |
| Fresh monitoring | Release-backed 35-series run: data quality, regime, and delayed-error checks `ok`; 34 live forecasts and 2,040 delayed rows | Pass with an alert |
| Drift interpretation | PSI is `critical` for all 34 targets and VIX relative to the long training reference; it is not a model-failure signal, but it must not be silently ignored | Open operational risk |
| Rollback | AAPL model/release/API/dashboard/workflow are unchanged and independent of the global candidate | Pass |

The immutable release and the latest successful CI run are available at
[the candidate release](https://github.com/ompat5/Volatility-Forecasting-MLOps/releases/tag/global-volatility-lstm-v3-candidate)
and [CI run 37399472232](https://github.com/ompat5/Volatility-Forecasting-MLOps/actions/runs/37399472232).

## Why direct cutover is rejected

The project’s evaluation guardrail is to benchmark against GARCH, not only to
produce a deployable neural-network artifact. RMSE and MAE improve modestly,
but QLIKE is the loss most directly aligned with variance forecasting and the
global model materially regresses on it. Replacing the default with this model
would overstate the evidence.

The current package is also deliberately and consistently a candidate:

- its artifact manifest has role `global_candidate`;
- MLflow version 3 has `promotion_status=candidate`,
  `production_default=false`, and `serving_enabled=false`;
- global serving is `POST /predict/global`, not a replacement for the legacy
  AAPL contract at `POST /predict`;
- the global dashboard and monitoring workflow are labeled candidate-only; and
- its GitHub asset is a prerelease.

Changing only one of those labels or defaults would create an incoherent
system. A future promotion must create a new production identity and change
all dependent contracts together.

## What remains safe to integrate

Merging the additive candidate implementation into `main` is **not** itself a
production cutover: the legacy AAPL defaults remain intact and the global
workflow would begin candidate monitoring from the default branch. That merge
should be reviewed as a normal pull request after its checks pass; it must not
be described as enabling global production inference.

The candidate schedule may run alongside the existing AAPL schedule. Its
alerts are candidate evidence and must remain visibly separate from AAPL
production alerts.

## Required path before a future promotion

1. Improve the global model using only the existing walk-forward validation
   folds for design and hyperparameter choices. Plausible hypotheses include a
   variance-aware training objective or calibrated forecast transformation; do
   not tune against the published final-holdout result.
2. Preserve the published 2025-06-18 to 2026-06-18 final holdout as historical
   evidence. Because its result informed this decision, a revised model needs
   a new time-forward promotion holdout (or a pre-registered rolling shadow
   evaluation) before it can claim a fresh final result.
3. Require a preregistered promotion rule that includes at least pooled QLIKE
   non-inferiority to GARCH, alongside complete coverage, finite positive
   forecasts, and no material pooled RMSE/MAE regression. Report per-ticker
   and group trade-offs rather than hiding them in one average.
4. Investigate the broad PSI alert without merely loosening thresholds. Any
   changed reference window, bins, or alert policy must be versioned,
   justified, and validated independently; the current alert remains visible
   until then.
5. Publish a new, checksum-pinned **production** artifact only after those
   gates pass. Its manifest role, MLflow tags, release status, API readiness,
   dashboard wording, and monitoring workflow must all identify the same
   immutable version.
6. Keep `POST /predict` backward compatible during the first production
   rollout. Promote the long-form global endpoint explicitly rather than
   silently changing an AAPL price-list request into a different schema.
7. Run the production candidate and AAPL path in parallel for a fixed,
   documented observation window. Compare delayed global forecasts with a
   concurrently computed GARCH baseline, not merely with the global model’s
   own prior error window.

## Rollback plan for a later promotion

Rollback is a configuration and deployment reversal, never a retraining step:

1. Point the serving image back to the checksum-pinned `aapl-lstm-v1` artifact
   with the current `MODEL_URI`/`MODEL_BACKEND` settings and remove the global
   production runtime setting.
2. Restore the AAPL Streamlit entrypoint as the public dashboard; retain the
   global dashboard as a clearly labeled candidate or investigation view.
3. Keep the AAPL weekday monitoring workflow authoritative. Disable the global
   production schedule or demote it to candidate observation, while preserving
   its artifacts for incident analysis.
4. Record the immutable global artifact version, input snapshot, monitoring
   report, and reason for rollback. Do not overwrite the release or mutate its
   checksum.

The current AAPL system already satisfies this rollback state, so no production
configuration was changed by this review.
