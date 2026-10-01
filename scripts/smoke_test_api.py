"""Wait for a running API container, then exercise health and prediction paths."""

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
    with urlopen(url, timeout=5) as response:  # noqa: S310 - caller supplies local API URL
        return json.load(response)


def wait_for_health(base_url: str, timeout: float = 90.0) -> None:
    """Poll until the API reports healthy or the startup deadline expires."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if _get_json(f"{base_url}/health") == {"status": "ok"}:
                return
        # Container startup can briefly refuse or reset the connection before
        # Uvicorn begins accepting requests. Both are transient OS-level errors.
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"API did not become healthy within {timeout}s: {last_error}")


def check_prediction(base_url: str) -> None:
    """Verify that the packaged model can serve one valid end-to-end request."""
    payload = json.dumps(
        {"prices": [100.0 + index * 0.1 for index in range(120)]}
    ).encode()
    request = Request(
        f"{base_url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - local API URL
        body = json.load(response)

    if body.get("horizon") != 5:
        raise RuntimeError(f"Unexpected prediction horizon: {body}")
    forecast = body.get("forecast")
    if not isinstance(forecast, (int, float)) or not math.isfinite(forecast):
        raise RuntimeError(f"Invalid forecast response: {body}")


def check_global_readiness(base_url: str, *, expected: bool) -> None:
    """Verify that global readiness is explicit and separate from AAPL health."""
    body = _get_json(f"{base_url}/health/global")
    expected_body = {
        "status": "ok" if expected else "disabled",
        "configured": expected,
        "ready": expected,
    }
    if body != expected_body:
        raise RuntimeError(f"Unexpected global readiness response: {body}")


def check_global_prediction(base_url: str) -> None:
    """Require one valid forecast for every configured target ticker."""
    universe = load_universe(DEFAULT_TICKERS_CONFIG)
    prices = build_ci_global_prices()
    prices["date"] = prices["date"].dt.date.astype(str)
    payload = json.dumps({"observations": prices.to_dict(orient="records")}).encode()
    request = Request(
        f"{base_url}/predict/global",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - local API URL
        body = json.load(response)

    predictions = body.get("predictions")
    if not isinstance(predictions, list):
        raise RuntimeError(f"Invalid global prediction response: {body}")
    if [row.get("ticker") for row in predictions] != list(
        universe.target_symbols
    ):
        raise RuntimeError(
            "Global prediction response does not cover the complete target universe"
        )
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
    parser.add_argument(
        "--expect-global",
        action="store_true",
        help="Require the optional global model and exercise all 34 targets",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_for_health(base_url, args.timeout)
    check_prediction(base_url)
    check_global_readiness(base_url, expected=args.expect_global)
    if args.expect_global:
        check_global_prediction(base_url)
    print("Container smoke test passed")


if __name__ == "__main__":
    main()
