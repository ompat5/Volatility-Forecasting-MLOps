from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import pytest

from scripts.create_ci_model import FIXTURE_FORECAST, create_ci_model


def test_ci_model_has_production_inference_contract(tmp_path: Path):
    output = create_ci_model(tmp_path / "ci-model")

    loaded = mlflow.pyfunc.load_model(str(output))
    prices = pd.Series([100.0 + index * 0.1 for index in range(120)])

    assert loaded.predict(prices) == pytest.approx(FIXTURE_FORECAST)
    assert (output / "CI_MODEL_NOTICE.txt").is_file()


def test_ci_model_refuses_to_overwrite_existing_path(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_ci_model(output)
