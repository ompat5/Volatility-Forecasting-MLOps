from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import pytest

from scripts.create_ci_global_model import (
    build_ci_global_prices,
    create_ci_global_model,
)


@pytest.fixture(scope="session")
def global_ci_fixture(tmp_path_factory) -> tuple[Path, object, pd.DataFrame]:
    """Build the full-universe artifact once for global integration tests."""
    output = create_ci_global_model(
        tmp_path_factory.mktemp("ci_global") / "model"
    )
    return output, mlflow.pyfunc.load_model(str(output)), build_ci_global_prices()
