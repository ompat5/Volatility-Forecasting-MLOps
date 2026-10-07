# Global model deployment contract

The repository deploys one pooled LSTM for all 34 target tickers. `^VIX` is
mandatory context and is never a forecast target. The legacy AAPL serving path
has been removed from active deployment code.

## Artifact

The exact model is pinned in `configs/global_release.yaml` and contains:

- PyTorch weights and the fitted six-feature `StandardScaler`;
- ordered ticker embedding IDs, groups, VIX context role, and model schema;
- the source panel/configuration and final-holdout evidence;
- model and scaler checksums; and
- an FP32 ONNX graph plus checksum-bound sidecar.

The release tag contains `candidate` because the immutable artifact was
published during candidate acceptance. That provenance does not create a
second runtime: it is the only artifact the API, dashboard, and monitor load.

## API

Start the model from the pinned snapshot:

```bash
uv run python -m scripts.download_global_model
MODEL_URI=global_model uv run uvicorn src.serving.app:app --port 8000
```

`POST /predict` takes:

| Column | Meaning |
|---|---|
| `date` | Market-session date |
| `ticker` | One target symbol or `^VIX` |
| `adjusted_close` | Positive adjusted close; VIX uses quoted points |

The body wraps records in `{"observations": [...]}`. The request must include
VIX plus at least one known target and enough synchronized history to construct
30 complete feature rows. Unknown symbols, duplicate keys, non-positive prices,
missing VIX, or incomplete histories fail closed.

The result has one row per requested target in universe order with `ticker`,
`as_of_date`, `horizon_sessions`, and strictly positive annualized `forecast`.

## Backends

`MODEL_BACKEND=mlflow` is the default and reads `MODEL_URI`. Registry URIs must
pin an explicit numeric `global-volatility-lstm` version. `MODEL_BACKEND=onnx`
uses the ONNX graph and sidecar beneath `MODEL_DIR` (default `/app/model` in a
container). Both backends validate the same complete target/VIX contract at
startup.

The Docker image always bakes one global artifact into `/app/model`:

```bash
docker build -t volatility-forecaster .
docker run --rm -p 8000:8000 volatility-forecaster
docker run --rm -e MODEL_BACKEND=onnx -p 8000:8000 volatility-forecaster
```

CI validates the MLflow and ONNX variants against a deterministic all-34-target
fixture. It has no AAPL-only container mode.
