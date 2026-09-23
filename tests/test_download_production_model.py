import hashlib
import tarfile
from pathlib import Path

import pytest
import yaml

from scripts.download_production_model import download_model


def _write_config(path: Path, archive: Path, digest: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "monitoring": {
                    "ticker": "AAPL",
                    "current_window": 20,
                    "backtest_window": 10,
                    "recent_error_window": 3,
                    "psi_warning": 0.1,
                    "psi_critical": 0.25,
                    "error_ratio_warning": 1.5,
                    "error_ratio_critical": 2.0,
                },
                "model": {
                    "version": "test-v1",
                    "url": archive.as_uri(),
                    "sha256": digest,
                },
            }
        )
    )


def test_download_model_verifies_and_extracts_archive(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "MLmodel").write_text("fixture")
    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source / "MLmodel", arcname="MLmodel")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    config = tmp_path / "monitoring.yaml"
    _write_config(config, archive, digest)

    output = download_model(config, tmp_path / "model")

    assert (output / "MLmodel").read_text() == "fixture"


def test_download_model_rejects_checksum_mismatch(tmp_path):
    archive = tmp_path / "model.tar.gz"
    archive.write_bytes(b"not the expected model")
    config = tmp_path / "monitoring.yaml"
    _write_config(config, archive, "0" * 64)

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_model(config, tmp_path / "model")
