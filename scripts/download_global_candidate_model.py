"""Download one explicitly selected, checksum-pinned global candidate archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.serving.artifacts import download_verified_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("GLOBAL_MODEL_URL"))
    parser.add_argument("--sha256", default=os.getenv("GLOBAL_MODEL_SHA256"))
    parser.add_argument("--output", type=Path, default=Path("global_model"))
    args = parser.parse_args()
    if not args.url or not args.sha256:
        parser.error("--url and --sha256 (or matching environment variables) are required")
    output = download_verified_model(args.url, args.sha256, args.output)
    print(f"Downloaded verified global candidate to {output}")


if __name__ == "__main__":
    main()
