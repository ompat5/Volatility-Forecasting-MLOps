# Global FP32 ONNX candidate

ONNX is a portable model graph executed here by ONNX Runtime. This phase uses it
to replace only the global PyTorch neural-network core; raw-price validation,
feature construction, the fitted `StandardScaler`, ticker ordering, and response
construction remain shared Python code.

The global production default has not changed. MLflow remains the opt-in global
backend unless `GLOBAL_MODEL_BACKEND=onnx` is explicitly selected, and the
separate AAPL service remains the production default.

## Export contract

Export starts from an explicit `global_model/` snapshot:

```bash
uv run python -m scripts.export_global_onnx
```

The graph contains the shared LSTM encoder, ticker embedding, prediction head,
and exponential log-volatility transform. Its inputs and output are:

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `features` | FP32 | `(batch, 30, 6)` | Scaled leakage-safe feature windows |
| `ticker_ids` | INT64 | `(batch,)` | Stable IDs from the artifact vocabulary |
| `forecast` | FP32 | `(batch,)` | Strictly positive realized-volatility forecasts |

Only the batch dimension is dynamic. Sequence length, feature order, and ticker
IDs remain fixed to the source artifact contract.

The export writes:

- `global_model/optimized/global_volatility_lstm_fp32.onnx`; and
- `global_model/optimized/global_volatility_lstm_fp32.json`.

The sidecar binds the ONNX checksum to the source artifact-manifest hash,
model-state hash, scaler hash, target count, horizon, tensor contract, and parity
evidence for every ticker ID. The serving runtime verifies those values before
creating an ONNX session. A graph, scaler, or source-manifest mismatch fails
startup.

## Parity and benchmark

The exporter checks every one of the 34 ticker embeddings against PyTorch using
a deterministic batch. The real version-3 candidate's maximum absolute
difference was `5.96e-08`. The raw-price benchmark also covered all 34 targets
and recorded the same maximum difference.

Run the reproducible benchmark with:

```bash
uv run python -m scripts.benchmark_global_inference
```

The recorded run used one CPU thread per runtime, 34 targets, 120 observations
per target and VIX, 100 warm-ups plus 1,000 measurements for the core, and 5
warm-ups plus 100 measurements for the full request.

| Path | Runtime | Median | p95 |
|---|---|---:|---:|
| Scaled `(34, 30, 6)` tensors + IDs → forecasts | PyTorch | 1.8345 ms | 2.2085 ms |
| Scaled `(34, 30, 6)` tensors + IDs → forecasts | ONNX FP32 | **0.7304 ms** | **0.9156 ms** |
| 4,200 long-form raw-price rows → 34 forecasts | PyTorch | 147.6555 ms | 164.0970 ms |
| 4,200 long-form raw-price rows → 34 forecasts | ONNX FP32 | **146.7871 ms** | **162.6878 ms** |

ONNX is about 2.5× faster for the neural-network core, but only about 0.6%
faster end to end. The graph is 87,579 bytes versus 87,856 bytes for the
PyTorch state, so there is no meaningful size reduction. The honest conclusion
is that pandas feature construction and scaling—not neural-network execution—are
the optimization target for whole-request latency.

The full environment, hashes, timings, and per-ticker parity values are in
`benchmarks/global_onnx_fp32.json`. These are local Darwin x86_64 measurements,
not universal latency guarantees.

## Explicit serving mode

After exporting, opt in locally with:

```bash
GLOBAL_MODEL_BACKEND=onnx \
  uv run uvicorn src.serving.app:app --port 8000
```

`GLOBAL_MODEL_URI` must be unset in ONNX mode so there is no ambiguous runtime.
Optional path overrides are `GLOBAL_ONNX_MODEL_PATH`,
`GLOBAL_ONNX_SCALER_PATH`, `GLOBAL_ONNX_ARTIFACT_MANIFEST_PATH`, and
`GLOBAL_ONNX_EXPORT_MANIFEST_PATH`. `GLOBAL_ONNX_NUM_THREADS` defaults to one.

For Docker, mount the separately exported snapshot exactly as in the MLflow
candidate path and select the backend:

```bash
docker run --rm \
  -e GLOBAL_MODEL_BACKEND=onnx \
  -v "$(pwd)/global_model:/app/global_model:ro" \
  -p 8000:8000 volatility-forecaster
```

CI exercises three modes with the same image: AAPL-only default, opt-in global
MLflow, and explicit global ONNX. The ONNX smoke test requires all 34 finite,
positive forecasts. Global candidate monitoring is the next completed gate;
dashboard, immutable release, and production cutover remain later gates.
