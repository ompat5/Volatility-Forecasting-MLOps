"""Export one explicit registered global-model version to a local snapshot.

The version is mandatory: candidate builds must never float on ``latest``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import mlflow

from src.config import REPO_ROOT
from src.models.global_artifact import GLOBAL_REGISTERED_MODEL_NAME

DEFAULT_DESTINATION = REPO_ROOT / "global_model"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"


def export_global_model(version: int, destination: Path) -> Path:
    """Snapshot an immutable registry version without overwriting any path."""
    if version <= 0:
        raise ValueError("Global model version must be positive")
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing path: {destination}. "
            "Remove it explicitly or choose another destination."
        )
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{GLOBAL_REGISTERED_MODEL_NAME}/{version}"
    source = mlflow.artifacts.download_artifacts(model_uri)
    shutil.copytree(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
    )
    args = parser.parse_args()
    destination = export_global_model(args.version, args.destination)
    print(
        f"Exported models:/{GLOBAL_REGISTERED_MODEL_NAME}/{args.version} "
        f"to {destination}"
    )


if __name__ == "__main__":
    main()
