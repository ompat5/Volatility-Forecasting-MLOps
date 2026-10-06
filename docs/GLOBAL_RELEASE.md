# Immutable global candidate release

The selected registry version is published as the GitHub **prerelease**
`global-volatility-lstm-v3-candidate`. This is immutable candidate delivery,
not a production promotion: the AAPL API, dashboard, artifact, and production
monitor remain the defaults.

## Pinned identity

| Field | Value |
|---|---|
| Registered model | `global-volatility-lstm` version `3` |
| Release tag | `global-volatility-lstm-v3-candidate` |
| Asset | `global-volatility-lstm-v3-candidate.tar.gz` |
| Archive size | 334,241 bytes |
| Files | 80 |
| SHA-256 | `c40b48f186a9255f6afcfffde1a5ca02f0ff2379ab9d68da87d0b024d353ee0a` |

The URL and digest are checked into `configs/global_release.yaml`. Consumers do
not use a floating release, registry alias, repository variable, or a manually
entered checksum.

[Open the immutable candidate prerelease](https://github.com/ompat5/Volatility-Forecasting-MLOps/releases/tag/global-volatility-lstm-v3-candidate).

## Reproducible packaging

Build the archive from the explicit exported registry snapshot:

```bash
uv run python -m scripts.package_global_candidate_release \
  --source global_model \
  --output /tmp/global-volatility-lstm-v3-candidate.tar.gz \
  --registered-model-version 3
```

The packager refuses to overwrite its output and rejects CI fixtures. Before it
writes anything it requires:

- registry metadata naming `global-volatility-lstm` version `3`;
- the global-candidate artifact role and exact repository universe/horizon;
- valid model-state and scaler checksums;
- a checksum-bound FP32 ONNX graph and sidecar;
- loadable MLflow and ONNX runtimes.

Archive entries are sorted and their timestamps, owners, permissions, and gzip
header are normalized. Two independent real-candidate packaging runs produced
the same 334,241 bytes and SHA-256 shown above.

## Verified download

```bash
uv run python -m scripts.download_global_candidate_model \
  --output /tmp/global-release-check
```

The downloader verifies the archive digest before extraction, uses safe tar
extraction, and then repeats the registry, artifact, MLflow, and ONNX runtime
validation on the extracted directory. A post-extraction failure removes the
destination rather than leaving an ambiguous model cache.

The separate global dashboard uses this pinned release automatically when no
explicit or local candidate directory exists. The candidate monitoring
workflow downloads the same configuration before every run.

## Monitoring schedule boundary

The release removes the repository-variable gate from
`.github/workflows/global-monitoring.yml`: the workflow owns a weekday
candidate schedule and reads only the checked-in release pin. GitHub executes
scheduled workflows from the default branch, so this schedule becomes active
only when the global branch is deliberately merged. Manual workflow dispatch
on the candidate branch is the pre-merge acceptance test.

The candidate workflow remains separate from the AAPL production workflow and
emits candidate artifacts and fleet alerts. Enabling it does not change an API
route, dashboard URL, production model, or promotion label.

## Release acceptance

The public asset was downloaded through the checked-in URL into a clean
directory. Its SHA-256 matched both independent local builds and GitHub's asset
digest, after which MLflow and ONNX validation both passed.

Acceptance then used only that downloaded directory:

- the dashboard ONNX path built 34 live forecasts, 2,074 monitoring-ledger
  rows, and 4,080 chart rows from the cached replay;
- the MLflow monitoring path produced the same complete 34-target contract;
- a fresh-data run downloaded all 34 targets plus VIX, enforced wall-clock
  freshness, and produced 34 synchronized forecasts plus 2,040 delayed rows as
  of 2026-10-02;
- fresh-run data quality, regime, and delayed error were `ok`; feature drift
  remained `critical`, which is a distribution-change signal rather than a
  failed release.

The first fresh attempt received an empty transient Yahoo response for `GE`.
The shared global loader now retries each symbol once, then still fails the
entire run if either attempt is unsuccessful. The acceptance rerun passed with
complete coverage; partial-universe fallback was never introduced.

## Remaining gate

Release download, dashboard-runtime, cached-monitoring, and fresh-monitoring
acceptance have passed. The next step is an explicit production-cutover review.
That review must decide whether and how to change defaults; this release does
not make that decision implicitly.
