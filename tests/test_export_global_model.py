from pathlib import Path

import pytest

import scripts.export_global_model as export_module
from scripts.export_global_model import export_global_model


def test_global_export_requires_explicit_positive_version(tmp_path: Path):
    with pytest.raises(ValueError, match="positive"):
        export_global_model(0, tmp_path / "global-model")


def test_global_export_refuses_to_overwrite(tmp_path: Path):
    destination = tmp_path / "global-model"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_global_model(1, destination)


def test_global_export_uses_explicit_registry_version(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "MLmodel").write_text("candidate")
    requested = []

    def fake_download(uri):
        requested.append(uri)
        return str(source)

    monkeypatch.setattr(
        export_module.mlflow.artifacts,
        "download_artifacts",
        fake_download,
    )
    destination = tmp_path / "snapshot"

    result = export_global_model(7, destination)

    assert requested == ["models:/global-volatility-lstm/7"]
    assert result == destination
    assert (destination / "MLmodel").read_text() == "candidate"
