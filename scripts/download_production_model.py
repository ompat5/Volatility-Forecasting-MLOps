"""Download and verify the immutable production model release asset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path
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

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "model.tar.gz"
        with urlopen(url, timeout=60) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)

        actual_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"Model checksum mismatch: expected {expected_digest}, got {actual_digest}"
            )

        output.mkdir(parents=True)
        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(output, filter="data")
        except Exception:
            shutil.rmtree(output)
            raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/monitoring.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("model"))
    args = parser.parse_args()
    output = download_model(args.config, args.output)
    print(f"Downloaded verified production model to {output}")


if __name__ == "__main__":
    main()
