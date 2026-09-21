"""Wait for a running API container, then exercise health and prediction paths."""

from __future__ import annotations

import argparse
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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
        except (HTTPError, URLError, TimeoutError) as exc:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    wait_for_health(args.base_url.rstrip("/"), args.timeout)
    check_prediction(args.base_url.rstrip("/"))
    print("Container smoke test passed")


if __name__ == "__main__":
    main()
