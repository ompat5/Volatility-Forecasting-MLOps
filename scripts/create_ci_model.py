"""Create a deterministic MLflow model fixture for CI container builds.

This is not a trained forecasting model. It has the same packaging and inference
contract as the production artifact, which lets CI prove that the Docker image
loads a model and serves predictions without needing the local MLflow registry.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import joblib
import mlflow.pyfunc
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.features.realized_vol import FEATURE_COLS
from src.models.lstm import VolatilityLSTM
from src.serving.model_wrapper import VolatilityForecaster

DEFAULT_OUTPUT = Path(".ci-model")
FIXTURE_FORECAST = 0.2


def create_ci_model(output: Path = DEFAULT_OUTPUT) -> Path:
    """Write a small deterministic pyfunc artifact, refusing to overwrite files."""
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing path: {output}. Remove it explicitly first."
        )

    scaler_data = pd.DataFrame(
        [
            [-0.02, 0.10, 0.15, 0.20],
            [-0.01, 0.12, 0.16, 0.21],
            [0.01, 0.14, 0.17, 0.22],
            [0.02, 0.16, 0.18, 0.23],
        ],
        columns=FEATURE_COLS,
    )
    scaler = StandardScaler().fit(scaler_data)

    model = VolatilityLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
    )
    # A constant positive output keeps the fixture deterministic and makes it
    # unmistakably a packaging test rather than a trained market model.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc.bias.fill_(FIXTURE_FORECAST)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)
        model_path = artifact_dir / "model.pt"
        scaler_path = artifact_dir / "scaler.pkl"
        torch.save(model, model_path)
        joblib.dump(scaler, scaler_path)

        mlflow.pyfunc.save_model(
            path=str(output),
            python_model=VolatilityForecaster(seq_len=30),
            artifacts={"model": str(model_path), "scaler": str(scaler_path)},
        )

    (output / "CI_MODEL_NOTICE.txt").write_text(
        "CI fixture only. This artifact is not a trained volatility model.\n"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the generated MLflow model (default: .ci-model)",
    )
    args = parser.parse_args()
    output = create_ci_model(args.output)
    print(f"Created CI model fixture at {output}")


if __name__ == "__main__":
    main()
