"""Download and verify the immutable production model release asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.serving.artifacts import download_model


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
