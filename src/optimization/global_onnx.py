"""Export and validate the positive global LSTM core as FP32 ONNX."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from src.models.global_artifact import (
    sha256_file,
    validate_global_artifact_manifest,
)
from src.models.global_lstm import GlobalVolatilityLSTM
from src.optimization.onnx_export import DEFAULT_OPSET

FEATURES_INPUT_NAME = "features"
TICKER_IDS_INPUT_NAME = "ticker_ids"
GLOBAL_OUTPUT_NAME = "forecast"
GLOBAL_ONNX_EXPORT_SCHEMA_VERSION = 1
PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-6


@dataclass(frozen=True)
class GlobalONNXExportResult:
    model_path: Path
    manifest_path: Path
    parity: dict[str, object]


class _PositiveGlobalONNXModule(nn.Module):
    """Export adapter that preserves the model's exponential output contract."""

    def __init__(self, model: GlobalVolatilityLSTM) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        features: torch.Tensor,
        ticker_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.predict_volatility(features, ticker_ids)


def load_global_artifact_manifest(path: Path) -> dict:
    """Read and validate the source MLflow artifact manifest."""
    with path.open() as manifest_file:
        manifest = json.load(manifest_file)
    validate_global_artifact_manifest(manifest)
    return manifest


def load_global_lstm(
    model_state_path: Path,
    artifact_manifest_path: Path,
) -> tuple[GlobalVolatilityLSTM, dict]:
    """Rehydrate the trusted global candidate and verify its state checksum."""
    manifest = load_global_artifact_manifest(artifact_manifest_path)
    expected_hash = manifest["components"]["model_state_sha256"]
    if sha256_file(model_state_path) != expected_hash:
        raise ValueError("Global artifact model-state checksum mismatch")
    architecture = manifest["architecture"]
    model = GlobalVolatilityLSTM(
        input_size=architecture["input_size"],
        num_tickers=architecture["num_tickers"],
        embedding_dim=architecture["embedding_dim"],
        hidden_size=architecture["hidden_size"],
        num_layers=architecture["num_layers"],
        dropout=architecture["dropout"],
    )
    state_dict = torch.load(
        model_state_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, manifest


def create_global_onnx_session(
    model_path: Path,
    *,
    threads: int = 1,
) -> ort.InferenceSession:
    """Create a deterministic CPU session with an explicit thread budget."""
    if threads <= 0:
        raise ValueError("threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def validate_global_onnx_session(
    session: ort.InferenceSession,
    *,
    seq_len: int,
    input_size: int,
) -> None:
    """Require the two-input dynamic-batch contract used by global serving."""
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {FEATURES_INPUT_NAME, TICKER_IDS_INPUT_NAME}:
        raise ValueError(
            "Global ONNX inputs must be exactly 'features' and 'ticker_ids'"
        )
    features = inputs[FEATURES_INPUT_NAME]
    ticker_ids = inputs[TICKER_IDS_INPUT_NAME]
    if features.type != "tensor(float)" or features.shape[1:] != [
        seq_len,
        input_size,
    ]:
        raise ValueError("Global ONNX feature input contract is incompatible")
    if ticker_ids.type != "tensor(int64)" or len(ticker_ids.shape) != 1:
        raise ValueError("Global ONNX ticker-ID input contract is incompatible")
    outputs = session.get_outputs()
    if (
        len(outputs) != 1
        or outputs[0].name != GLOBAL_OUTPUT_NAME
        or outputs[0].type != "tensor(float)"
        or len(outputs[0].shape) != 1
    ):
        raise ValueError("Global ONNX output contract is incompatible")


def _parity_inputs(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    architecture = manifest["architecture"]
    seq_len = manifest["preprocessing"]["seq_len"]
    num_tickers = architecture["num_tickers"]
    rng = np.random.default_rng(42)
    features = rng.normal(
        size=(num_tickers, seq_len, architecture["input_size"])
    ).astype(np.float32)
    ticker_ids = np.arange(num_tickers, dtype=np.int64)
    return features, ticker_ids


def validate_global_onnx_parity(
    model: GlobalVolatilityLSTM,
    session: ort.InferenceSession,
    manifest: dict,
) -> dict[str, object]:
    """Compare every ticker embedding between PyTorch and ONNX."""
    features, ticker_ids = _parity_inputs(manifest)
    with torch.inference_mode():
        pytorch_output = model.predict_volatility(
            torch.from_numpy(features),
            torch.from_numpy(ticker_ids),
        ).numpy()
    onnx_output = session.run(
        [GLOBAL_OUTPUT_NAME],
        {
            FEATURES_INPUT_NAME: features,
            TICKER_IDS_INPUT_NAME: ticker_ids,
        },
    )[0]
    if not np.isfinite(onnx_output).all() or (onnx_output <= 0).any():
        raise ValueError("Global ONNX graph produced invalid volatility forecasts")
    try:
        np.testing.assert_allclose(
            onnx_output,
            pytorch_output,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )
    except AssertionError as exc:
        raise ValueError("Global ONNX graph failed PyTorch parity") from exc
    absolute = np.abs(onnx_output - pytorch_output)
    relative = absolute / np.maximum(np.abs(pytorch_output), 1e-12)
    return {
        "validated_ticker_ids": ticker_ids.tolist(),
        "max_absolute_difference": float(absolute.max()),
        "max_relative_difference": float(relative.max()),
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
    }


def export_global_lstm_to_onnx(
    *,
    model_state_path: Path,
    artifact_manifest_path: Path,
    scaler_path: Path,
    output_path: Path,
    export_manifest_path: Path,
    opset_version: int = DEFAULT_OPSET,
) -> GlobalONNXExportResult:
    """Export, validate, and bind FP32 ONNX to one global artifact snapshot."""
    if opset_version <= 0:
        raise ValueError("opset_version must be positive")
    model, manifest = load_global_lstm(
        model_state_path,
        artifact_manifest_path,
    )
    if sha256_file(scaler_path) != manifest["components"]["scaler_sha256"]:
        raise ValueError("Global artifact scaler checksum mismatch")

    architecture = manifest["architecture"]
    seq_len = manifest["preprocessing"]["seq_len"]
    adapter = _PositiveGlobalONNXModule(model).eval()
    example_features = torch.zeros(
        1,
        seq_len,
        architecture["input_size"],
        dtype=torch.float32,
    )
    example_ticker_ids = torch.zeros(1, dtype=torch.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            adapter,
            (example_features, example_ticker_ids),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[FEATURES_INPUT_NAME, TICKER_IDS_INPUT_NAME],
            output_names=[GLOBAL_OUTPUT_NAME],
            dynamic_axes={
                FEATURES_INPUT_NAME: {0: "batch_size"},
                TICKER_IDS_INPUT_NAME: {0: "batch_size"},
                GLOBAL_OUTPUT_NAME: {0: "batch_size"},
            },
        )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    session = create_global_onnx_session(output_path)
    validate_global_onnx_session(
        session,
        seq_len=seq_len,
        input_size=architecture["input_size"],
    )
    parity = validate_global_onnx_parity(model, session, manifest)
    export_manifest = {
        "schema_version": GLOBAL_ONNX_EXPORT_SCHEMA_VERSION,
        "artifact_role": "global_candidate_onnx_fp32",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "registered_model_name": manifest["registered_model_name"],
            "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
            "model_state_sha256": manifest["components"]["model_state_sha256"],
            "scaler_sha256": manifest["components"]["scaler_sha256"],
        },
        "contract": {
            "features_input": {
                "name": FEATURES_INPUT_NAME,
                "dtype": "float32",
                "shape": ["batch", seq_len, architecture["input_size"]],
            },
            "ticker_ids_input": {
                "name": TICKER_IDS_INPUT_NAME,
                "dtype": "int64",
                "shape": ["batch"],
            },
            "output": {
                "name": GLOBAL_OUTPUT_NAME,
                "dtype": "float32",
                "shape": ["batch"],
                "constraint": manifest["target"]["output_constraint"],
            },
            "target_count": architecture["num_tickers"],
            "horizon_sessions": manifest["target"]["horizon_sessions"],
        },
        "validation": parity,
        "onnx": {
            "opset_version": opset_version,
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
    }
    export_manifest_path.write_text(
        json.dumps(export_manifest, indent=2, sort_keys=True) + "\n"
    )
    return GlobalONNXExportResult(
        model_path=output_path,
        manifest_path=export_manifest_path,
        parity=parity,
    )


def load_global_onnx_export_manifest(path: Path) -> dict:
    """Read the ONNX sidecar used to bind serving files to their source."""
    with path.open() as manifest_file:
        export_manifest = json.load(manifest_file)
    if export_manifest.get("schema_version") != GLOBAL_ONNX_EXPORT_SCHEMA_VERSION:
        raise ValueError("Global ONNX export schema_version must be 1")
    if export_manifest.get("artifact_role") != "global_candidate_onnx_fp32":
        raise ValueError("Global ONNX export artifact role is incompatible")
    return export_manifest


def validate_global_onnx_bundle(
    *,
    model_path: Path,
    scaler_path: Path,
    artifact_manifest_path: Path,
    export_manifest_path: Path,
) -> tuple[dict, dict]:
    """Verify all optimized serving files belong to the same source artifact."""
    artifact_manifest = load_global_artifact_manifest(artifact_manifest_path)
    export_manifest = load_global_onnx_export_manifest(export_manifest_path)
    source = export_manifest.get("source", {})
    if source.get("artifact_manifest_sha256") != sha256_file(
        artifact_manifest_path
    ):
        raise ValueError("Global ONNX source-manifest checksum mismatch")
    if source.get("model_state_sha256") != artifact_manifest["components"].get(
        "model_state_sha256"
    ):
        raise ValueError("Global ONNX source model-state checksum mismatch")
    if (
        source.get("scaler_sha256")
        != artifact_manifest["components"].get("scaler_sha256")
        or sha256_file(scaler_path) != source.get("scaler_sha256")
    ):
        raise ValueError("Global ONNX scaler checksum mismatch")
    if sha256_file(model_path) != export_manifest.get("onnx", {}).get("sha256"):
        raise ValueError("Global ONNX graph checksum mismatch")
    architecture = artifact_manifest["architecture"]
    contract = export_manifest.get("contract", {})
    if (
        contract.get("target_count") != architecture["num_tickers"]
        or contract.get("horizon_sessions")
        != artifact_manifest["target"]["horizon_sessions"]
    ):
        raise ValueError("Global ONNX export contract is incompatible")
    if export_manifest.get("validation", {}).get(
        "validated_ticker_ids"
    ) != list(range(architecture["num_tickers"])):
        raise ValueError("Global ONNX export lacks all-ticker parity evidence")
    return artifact_manifest, export_manifest
