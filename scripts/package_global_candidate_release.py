"""Build one validated, deterministic global candidate release archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.global_release import create_global_release_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("global_model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registered-model-version", type=int, required=True)
    args = parser.parse_args()
    result = create_global_release_archive(
        args.source,
        args.output,
        registered_model_version=args.registered_model_version,
    )
    print(f"Validated {result.files} files from registry version {result.registered_model_version}")
    print(f"Archive: {result.path} ({result.size_bytes} bytes)")
    print(f"SHA-256: {result.sha256}")


if __name__ == "__main__":
    main()
