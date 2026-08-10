"""Export the registered pyfunc model to a local ./model dir for the Docker build context.

The container loads the model by path (no mlflow.db), so we snapshot the registered
version's self-contained artifact folder into REPO_ROOT/model before building the image.
Run after training: `uv run python -m scripts.export_model`.
"""

import shutil
from pathlib import Path

import mlflow

from src.config import REPO_ROOT

MODEL_URI = "models:/volatility-lstm/latest"
DEST = REPO_ROOT / "model"


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    # Resolves the registered version to a local, self-contained MLflow model directory.
    src = mlflow.artifacts.download_artifacts(MODEL_URI)

    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(src, DEST)
    print(f"Exported {MODEL_URI} -> {DEST}")


if __name__ == "__main__":
    main()
