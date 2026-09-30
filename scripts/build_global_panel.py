"""Build the balanced global-model panel and its reproducibility manifest.

Run with: `uv run python -m scripts.build_global_panel`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DEFAULT_MODEL_CONFIG, load_config
from src.data.ingest import DEFAULT_RAW_DIR, DEFAULT_TICKERS_CONFIG
from src.data.panel import build_global_panel, build_panel_manifest
from src.data.universe import load_universe

DEFAULT_PANEL_PATH = Path("data/processed/global_panel.parquet")
DEFAULT_MANIFEST_PATH = Path("data/processed/global_panel_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_TICKERS_CONFIG)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Model config supplying the default forecast horizon",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional override; defaults to data.horizon in model config",
    )
    parser.add_argument("--output-panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    args = parser.parse_args()

    universe = load_universe(args.config)
    horizon = args.horizon
    if horizon is None:
        horizon = load_config(args.model_config).data.horizon
    panel = build_global_panel(
        universe,
        raw_dir=args.raw_dir,
        horizon=horizon,
    )
    manifest = build_panel_manifest(
        panel,
        universe,
        raw_dir=args.raw_dir,
        horizon=horizon,
    )

    args.output_panel.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output_panel)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Wrote {len(panel):,} rows for {len(universe.target_symbols)} targets "
        f"to {args.output_panel}"
    )
    print(f"Wrote reproducibility manifest to {args.output_manifest}")


if __name__ == "__main__":
    main()
