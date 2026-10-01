from pathlib import Path

from fastapi.testclient import TestClient
import mlflow.pyfunc
import pandas as pd
import pytest

from scripts.create_ci_global_model import (
    FIXTURE_FORECAST,
    build_ci_global_prices,
    create_ci_global_model,
)
import src.serving.app as app_module
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe


class _LegacyStub:
    def predict(self, _prices):
        return 0.25


@pytest.fixture(scope="module")
def global_fixture(tmp_path_factory) -> tuple[Path, object, pd.DataFrame]:
    output = create_ci_global_model(
        tmp_path_factory.mktemp("ci_global") / "model"
    )
    return output, mlflow.pyfunc.load_model(str(output)), build_ci_global_prices()


def test_ci_global_model_covers_the_complete_universe(global_fixture):
    output, loaded, prices = global_fixture
    universe = load_universe(DEFAULT_TICKERS_CONFIG)

    forecasts = loaded.predict(prices)

    assert forecasts["ticker"].tolist() == list(universe.target_symbols)
    assert len(forecasts) == 34
    assert forecasts["horizon_sessions"].tolist() == [5] * 34
    assert forecasts["forecast"].tolist() == pytest.approx(
        [FIXTURE_FORECAST] * 34
    )
    assert loaded.metadata.metadata == {
        "artifact_role": "global_candidate",
        "target_count": 34,
        "context_symbols": ["^VIX"],
    }
    assert (output / "CI_MODEL_NOTICE.txt").is_file()


def test_global_api_matches_direct_pyfunc_predictions(
    global_fixture,
    monkeypatch,
):
    output, loaded, prices = global_fixture
    direct = loaded.predict(prices)

    def fake_load_model(uri):
        return loaded if uri == str(output) else _LegacyStub()

    monkeypatch.setattr("mlflow.pyfunc.load_model", fake_load_model)
    monkeypatch.setenv("MODEL_URI", "/models/aapl")
    monkeypatch.setenv("GLOBAL_MODEL_URI", str(output))
    payload = prices.copy()
    payload["date"] = payload["date"].dt.date.astype(str)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/predict/global",
            json={"observations": payload.to_dict(orient="records")},
        )

    assert response.status_code == 200
    api = pd.DataFrame(response.json()["predictions"])
    assert api["ticker"].tolist() == direct["ticker"].tolist()
    assert api["as_of_date"].tolist() == [
        value.date().isoformat() for value in direct["as_of_date"]
    ]
    assert api["horizon_sessions"].tolist() == direct[
        "horizon_sessions"
    ].tolist()
    assert api["forecast"].tolist() == pytest.approx(
        direct["forecast"].tolist()
    )


def test_global_api_maps_missing_vix_to_validation_error(
    global_fixture,
    monkeypatch,
):
    output, loaded, prices = global_fixture

    def fake_load_model(uri):
        return loaded if uri == str(output) else _LegacyStub()

    monkeypatch.setattr("mlflow.pyfunc.load_model", fake_load_model)
    monkeypatch.setenv("MODEL_URI", "/models/aapl")
    monkeypatch.setenv("GLOBAL_MODEL_URI", str(output))
    prices = prices[prices["ticker"] != "^VIX"].copy()
    prices["date"] = prices["date"].dt.date.astype(str)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/predict/global",
            json={"observations": prices.to_dict(orient="records")},
        )

    assert response.status_code == 422
    assert "context ticker ^VIX" in response.json()["detail"]


def test_ci_global_model_refuses_to_overwrite_existing_path(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_ci_global_model(output)
