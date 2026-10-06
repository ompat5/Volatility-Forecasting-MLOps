import hashlib
from pathlib import Path
import shutil

import pytest
import yaml

from src.config import REPO_ROOT
from src.models.global_release import (
    GlobalReleaseConfig,
    create_global_release_archive,
    download_global_release,
    load_global_release_config,
    validate_global_release_directory,
)
from src.optimization.global_onnx import export_global_lstm_to_onnx


@pytest.fixture(scope="module")
def global_release_bundle(tmp_path_factory, global_ci_fixture) -> dict:
    source, _, _ = global_ci_fixture
    artifact_dir = tmp_path_factory.mktemp("global_release") / "artifact"
    shutil.copytree(source, artifact_dir)
    export_global_lstm_to_onnx(
        model_state_path=artifact_dir / "artifacts" / "model_state.pt",
        artifact_manifest_path=artifact_dir / "artifacts" / "manifest.json",
        scaler_path=artifact_dir / "artifacts" / "scaler.pkl",
        output_path=(
            artifact_dir / "optimized" / "global_volatility_lstm_fp32.onnx"
        ),
        export_manifest_path=(
            artifact_dir / "optimized" / "global_volatility_lstm_fp32.json"
        ),
    )
    return {"artifact_dir": artifact_dir}


def _release_source(tmp_path: Path, global_release_bundle: dict) -> Path:
    source = tmp_path / "global_model"
    shutil.copytree(global_release_bundle["artifact_dir"], source)
    (source / "CI_MODEL_NOTICE.txt").unlink()
    (source / "registered_model_meta").write_text(
        yaml.safe_dump(
            {
                "model_name": "global-volatility-lstm",
                "model_version": "3",
                "artifact_path": "registry-version-3",
            }
        )
    )
    return source


def _write_release_config(path: Path, archive: Path, digest: str) -> None:
    tag = "global-volatility-lstm-v3-candidate"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "release": {
                    "version": tag,
                    "role": "global_candidate",
                    "tag": tag,
                    "asset_name": f"{tag}.tar.gz",
                    "url": (
                        "https://github.com/owner/repository/releases/download/"
                        f"{tag}/{tag}.tar.gz"
                    ),
                    "sha256": digest,
                    "registered_model_name": "global-volatility-lstm",
                    "registered_model_version": 3,
                },
            },
            sort_keys=False,
        )
    )


def test_global_release_archive_is_deterministic(
    tmp_path: Path,
    global_release_bundle: dict,
):
    source = _release_source(tmp_path, global_release_bundle)
    first = create_global_release_archive(
        source,
        tmp_path / "first.tar.gz",
        registered_model_version=3,
    )
    second = create_global_release_archive(
        source,
        tmp_path / "second.tar.gz",
        registered_model_version=3,
    )

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.files > 10


def test_global_release_rejects_ci_fixture(global_release_bundle: dict):
    with pytest.raises(ValueError, match="CI fixtures cannot be published"):
        validate_global_release_directory(
            global_release_bundle["artifact_dir"],
            registered_model_version=3,
        )


def test_global_release_rejects_wrong_registry_version(
    tmp_path: Path,
    global_release_bundle: dict,
):
    source = _release_source(tmp_path, global_release_bundle)

    with pytest.raises(ValueError, match="registry model version"):
        validate_global_release_directory(
            source,
            registered_model_version=4,
        )


def test_global_release_config_rejects_mutable_or_invalid_identity(
    tmp_path: Path,
):
    archive = tmp_path / "candidate.tar.gz"
    archive.write_bytes(b"candidate")
    config_path = tmp_path / "release.yaml"
    _write_release_config(
        config_path,
        archive,
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )

    config = load_global_release_config(config_path)

    assert config.registered_model_version == 3
    assert config.tag == "global-volatility-lstm-v3-candidate"

    payload = yaml.safe_load(config_path.read_text())
    payload["release"]["url"] = "https://github.com/owner/repository/releases/latest"
    config_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="immutable GitHub asset"):
        load_global_release_config(config_path)


def test_committed_global_release_pin_is_complete():
    config = load_global_release_config(
        REPO_ROOT / "configs" / "global_release.yaml"
    )

    assert config.tag == "global-volatility-lstm-v3-candidate"
    assert config.registered_model_name == "global-volatility-lstm"
    assert config.registered_model_version == 3
    assert config.url.endswith(f"/{config.tag}/{config.asset_name}")


def test_downloaded_global_release_is_validated_again(
    tmp_path: Path,
    global_release_bundle: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _release_source(tmp_path, global_release_bundle)
    archive = create_global_release_archive(
        source,
        tmp_path / "global-volatility-lstm-v3-candidate.tar.gz",
        registered_model_version=3,
    )
    config_path = tmp_path / "release.yaml"
    config_path.touch()
    monkeypatch.setattr(
        "src.models.global_release.load_global_release_config",
        lambda _path: GlobalReleaseConfig(
            version="global-volatility-lstm-v3-candidate",
            tag="global-volatility-lstm-v3-candidate",
            asset_name="global-volatility-lstm-v3-candidate.tar.gz",
            url=archive.path.as_uri(),
            sha256=archive.sha256,
            registered_model_name="global-volatility-lstm",
            registered_model_version=3,
        ),
    )

    downloaded = download_global_release(config_path, tmp_path / "downloaded")

    assert (downloaded / "MLmodel").is_file()
    assert (downloaded / "optimized" / "global_volatility_lstm_fp32.onnx").is_file()
