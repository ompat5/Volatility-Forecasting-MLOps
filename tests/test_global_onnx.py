import json
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd
import pytest
import torch

import src.serving.app as app_module
from src.models.global_artifact import sha256_file
from src.models.global_lstm import GlobalVolatilityLSTM
from src.optimization.global_onnx import (
    FEATURES_INPUT_NAME,
    GLOBAL_OUTPUT_NAME,
    TICKER_IDS_INPUT_NAME,
    export_global_lstm_to_onnx,
)
from scripts.benchmark_global_inference import run_global_benchmark
from src.serving.global_model_wrapper import GlobalVolatilityForecaster
from src.serving.global_onnx_forecaster import GlobalONNXVolatilityForecaster


@pytest.fixture(scope="module")
def global_onnx_bundle(tmp_path_factory, global_ci_fixture):
    source, _, prices = global_ci_fixture
    artifact_dir = tmp_path_factory.mktemp("global_onnx") / "artifact"
    shutil.copytree(source, artifact_dir)
    manifest_path = artifact_dir / "artifacts" / "manifest.json"
    state_path = artifact_dir / "artifacts" / "model_state.pt"
    scaler_path = artifact_dir / "artifacts" / "scaler.pkl"
    manifest = json.loads(manifest_path.read_text())
    architecture = manifest["architecture"]

    # Use non-constant parameters so parity genuinely exercises every embedding.
    torch.manual_seed(123)
    model = GlobalVolatilityLSTM(
        input_size=architecture["input_size"],
        num_tickers=architecture["num_tickers"],
        embedding_dim=architecture["embedding_dim"],
        hidden_size=architecture["hidden_size"],
        num_layers=architecture["num_layers"],
        dropout=architecture["dropout"],
    ).eval()
    torch.save(model.state_dict(), state_path)
    manifest["components"]["model_state_sha256"] = sha256_file(state_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    onnx_path = artifact_dir / "optimized" / "global_volatility_lstm_fp32.onnx"
    export_manifest_path = (
        artifact_dir / "optimized" / "global_volatility_lstm_fp32.json"
    )
    export_result = export_global_lstm_to_onnx(
        model_state_path=state_path,
        artifact_manifest_path=manifest_path,
        scaler_path=scaler_path,
        output_path=onnx_path,
        export_manifest_path=export_manifest_path,
    )
    scaler = joblib.load(scaler_path)
    pytorch = GlobalVolatilityForecaster()
    pytorch._initialize(model, scaler, manifest)
    onnx = GlobalONNXVolatilityForecaster(
        onnx_path,
        scaler_path,
        manifest_path,
        export_manifest_path,
    )
    return {
        "artifact_dir": artifact_dir,
        "manifest_path": manifest_path,
        "scaler_path": scaler_path,
        "onnx_path": onnx_path,
        "export_manifest_path": export_manifest_path,
        "export_result": export_result,
        "pytorch": pytorch,
        "onnx": onnx,
        "prices": prices,
    }


def test_export_validates_every_ticker_embedding(global_onnx_bundle):
    result = global_onnx_bundle["export_result"]
    evidence = result.parity

    assert evidence["validated_ticker_ids"] == list(range(34))
    assert evidence["max_absolute_difference"] < 1e-5
    assert result.model_path.is_file()
    assert result.manifest_path.is_file()


def test_global_onnx_dynamic_batch_matches_pytorch(global_onnx_bundle):
    pytorch = global_onnx_bundle["pytorch"].model
    onnx = global_onnx_bundle["onnx"]
    rng = np.random.default_rng(7)
    features = rng.normal(size=(3, 30, 6)).astype(np.float32)
    ticker_ids = np.asarray([0, 17, 33], dtype=np.int64)

    with torch.inference_mode():
        expected = pytorch.predict_volatility(
            torch.from_numpy(features), torch.from_numpy(ticker_ids)
        ).numpy()
    actual = onnx.session.run(
        [GLOBAL_OUTPUT_NAME],
        {
            FEATURES_INPUT_NAME: features,
            TICKER_IDS_INPUT_NAME: ticker_ids,
        },
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
    assert (actual > 0).all()


def test_global_onnx_raw_price_path_matches_pytorch(global_onnx_bundle):
    prices = global_onnx_bundle["prices"]
    pytorch = global_onnx_bundle["pytorch"].predict(None, prices)
    onnx = global_onnx_bundle["onnx"].predict(prices)

    assert onnx["ticker"].tolist() == pytorch["ticker"].tolist()
    assert len(onnx) == 34
    np.testing.assert_allclose(
        onnx["forecast"], pytorch["forecast"], rtol=1e-4, atol=1e-6
    )

    subset = prices[prices["ticker"].isin(["AAPL", "SPY", "^VIX"])]
    subset_output = global_onnx_bundle["onnx"].predict(subset)
    assert subset_output["ticker"].tolist() == ["AAPL", "SPY"]


def test_global_onnx_rejects_tampered_graph(global_onnx_bundle, tmp_path: Path):
    tampered = tmp_path / "tampered.onnx"
    shutil.copyfile(global_onnx_bundle["onnx_path"], tampered)
    tampered.write_bytes(tampered.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="graph checksum"):
        GlobalONNXVolatilityForecaster(
            tampered,
            global_onnx_bundle["scaler_path"],
            global_onnx_bundle["manifest_path"],
            global_onnx_bundle["export_manifest_path"],
        )


def test_fastapi_serves_global_onnx_backend(
    global_onnx_bundle,
    monkeypatch,
):
    prices = global_onnx_bundle["prices"].copy()
    expected = global_onnx_bundle["onnx"].predict(prices)
    monkeypatch.setenv("MODEL_BACKEND", "onnx")
    monkeypatch.setenv("ONNX_MODEL_PATH", str(global_onnx_bundle["onnx_path"]))
    monkeypatch.setenv("ONNX_SCALER_PATH", str(global_onnx_bundle["scaler_path"]))
    monkeypatch.setenv(
        "ONNX_ARTIFACT_MANIFEST_PATH",
        str(global_onnx_bundle["manifest_path"]),
    )
    monkeypatch.setenv(
        "ONNX_EXPORT_MANIFEST_PATH",
        str(global_onnx_bundle["export_manifest_path"]),
    )
    prices["date"] = prices["date"].dt.date.astype(str)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/predict",
            json={"observations": prices.to_dict(orient="records")},
        )

    assert response.status_code == 200
    actual = pd.DataFrame(response.json()["predictions"])
    assert actual["ticker"].tolist() == expected["ticker"].tolist()
    np.testing.assert_allclose(
        actual["forecast"], expected["forecast"], rtol=1e-4, atol=1e-6
    )


def test_global_benchmark_records_all_target_core_and_end_to_end_paths(
    global_onnx_bundle,
    tmp_path: Path,
):
    output_path = tmp_path / "benchmark.json"

    result = run_global_benchmark(
        artifact_dir=global_onnx_bundle["artifact_dir"],
        model_input=global_onnx_bundle["prices"],
        output_path=output_path,
        core_warmup=0,
        core_iterations=2,
        end_to_end_warmup=0,
        end_to_end_iterations=2,
        threads=1,
    )

    assert result["config"]["target_count"] == 34
    assert result["config"]["core_features_shape"] == [34, 30, 6]
    assert result["parity"]["all_target_embeddings_covered"] is True
    assert len(result["parity"]["per_ticker"]) == 34
    assert result["parity"]["max_absolute_difference"] < 1e-5
    assert result["latency_ms"]["core"]["pytorch"]["iterations"] == 2
    assert result["latency_ms"]["end_to_end"]["onnx_fp32"]["iterations"] == 2
    assert output_path.is_file()
