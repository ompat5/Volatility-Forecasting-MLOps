"""Build fixed all-target drift and regime references from the training panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import REPO_ROOT
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe
from src.models.global_artifact import sha256_file
from src.monitoring.drift import build_global_reference

DEFAULT_PANEL = REPO_ROOT / "data" / "processed" / "global_panel.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "monitoring" / "global_reference.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--universe", type=Path, default=DEFAULT_TICKERS_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    universe = load_universe(args.universe)
    reference = build_global_reference(
        panel,
        universe,
        panel_sha256=sha256_file(args.panel),
        n_bins=args.bins,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(universe.target_symbols)}-target monitoring reference "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
