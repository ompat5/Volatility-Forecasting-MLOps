# Global FP32 ONNX runtime

The deployed global model supports `MODEL_BACKEND=onnx`. Only the pooled LSTM
core runs in ONNX; the raw-price feature contract and fitted scaler stay in
Python and are shared with the MLflow pyfunc.

The exported graph receives dynamic batches of `(batch, 30, 6)` features plus
one ticker ID per row, and emits strictly positive volatility forecasts. Its
sidecar binds the graph to the model-state checksum, scaler checksum, artifact
manifest checksum, vocabulary, and all 34 embedding IDs.

```bash
uv run python -m scripts.download_global_model
docker build -t volatility-forecaster .
docker run --rm -e MODEL_BACKEND=onnx -p 8000:8000 volatility-forecaster
```

All-embedding parity against PyTorch has maximum absolute difference
`5.96e-08`. On the recorded one-thread run, ONNX reduced median core latency
from `1.8345 ms` to `0.7304 ms` (2.5×), while end-to-end latency changed only
from `147.6555 ms` to `146.7871 ms` because preprocessing dominates.

Export and benchmark evidence is in
[benchmarks/global_onnx_fp32.json](../benchmarks/global_onnx_fp32.json).
