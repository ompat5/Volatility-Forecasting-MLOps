# Immutable global deployment release

The global deployment uses one checksum-pinned archive. It was originally
published under the immutable prerelease tag
`global-volatility-lstm-v3-candidate`; the suffix records its training and
evaluation provenance, not a second serving path.

| Field | Value |
|---|---|
| Registered model | `global-volatility-lstm` version `3` |
| Release tag | `global-volatility-lstm-v3-candidate` |
| Asset | `global-volatility-lstm-v3-candidate.tar.gz` |
| SHA-256 | `c40b48f186a9255f6afcfffde1a5ca02f0ff2379ab9d68da87d0b024d353ee0a` |

The URL and digest are fixed in `configs/global_release.yaml`. No floating
registry alias, GitHub variable, or manually entered checksum is accepted.

```bash
uv run python -m scripts.download_global_model --output global_model
```

The downloader verifies the archive before extraction, uses safe tar handling,
then validates the registry metadata, artifact manifest, model/scaler checksums,
MLflow runtime, and checksum-bound ONNX runtime. It removes the destination on
post-extraction failure.

The release is reused unchanged by Docker preparation, the Streamlit dashboard,
and the weekday global monitor. This prevents preprocessing, embedding
vocabulary, or model-version drift across those surfaces.
