"""Retrieve immutable production model artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

from src.monitoring.pipeline import load_monitoring_config


def download_model(config_path: Path, output: Path) -> Path:
    """Fetch, checksum, and safely extract the configured model archive."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output}")

    _, model_config = load_monitoring_config(config_path)
    url = model_config["url"]
    expected_digest = model_config["sha256"]
    if "PLACEHOLDER" in url or "PLACEHOLDER" in expected_digest:
        raise ValueError("Production model release metadata has not been configured")

    with tempfile.TemporaryDirectory() as temporary_dir:
        archive = Path(temporary_dir) / "model.tar.gz"
        with urlopen(url, timeout=60) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)

        actual_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"Model checksum mismatch: expected {expected_digest}, "
                f"got {actual_digest}"
            )

        output.mkdir(parents=True)
        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(output, filter="data")
        except Exception:
            shutil.rmtree(output)
            raise
    return output
