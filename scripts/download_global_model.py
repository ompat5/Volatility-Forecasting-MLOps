"""Download the checksum-pinned global model used by every deployment path."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import REPO_ROOT
from src.models.global_release import download_global_release

DEFAULT_RELEASE_CONFIG = REPO_ROOT / "configs" / "global_release.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("global_model"))
    args = parser.parse_args()
    output = download_global_release(args.config, args.output)
    print(f"Downloaded verified global model to {output}")


if __name__ == "__main__":
    main()
