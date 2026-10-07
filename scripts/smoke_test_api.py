"""Wait for the global API container and require all 34 forecasts."""

from __future__ import annotations

import argparse
import json
import math
import time
from urllib.request import Request, urlopen

from scripts.create_ci_global_model import build_ci_global_prices
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:  # noqa: S310 - local API only
        return json.load(response)


def wait_for_health(base_url: str, timeout: float = 90.0) -> None:
    """Poll until the global API reports ready or startup times out."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            body = _get_json(f"{base_url}/health")
            if body == {
                "status": "ok",
                "model_scope": "global_34_target",
                "ready": True,
            }:
                return
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"API did not become healthy within {timeout}s: {last_error}")


def check_prediction(base_url: str) -> None:
    """Require one finite positive forecast for every configured target."""
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    prices = build_ci_global_prices()
    prices["date"] = prices["date"].dt.date.astype(str)
    payload = json.dumps({"observations": prices.to_dict(orient="records")}).encode()
    request = Request(
        f"{base_url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - local API only
        body = json.load(response)

    predictions = body.get("predictions")
    if not isinstance(predictions, list):
        raise RuntimeError(f"Invalid global prediction response: {body}")
    if [row.get("ticker") for row in predictions] != list(universe.target_symbols):
        raise RuntimeError("Prediction response does not cover the complete universe")
    if any(
        row.get("horizon_sessions") != 5
        or not isinstance(row.get("forecast"), (int, float))
        or not math.isfinite(row["forecast"])
        or row["forecast"] <= 0
        for row in predictions
    ):
        raise RuntimeError(f"Invalid global forecast values: {body}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    wait_for_health(base_url, args.timeout)
    check_prediction(base_url)
    print("Global container smoke test passed")


if __name__ == "__main__":
    main()
