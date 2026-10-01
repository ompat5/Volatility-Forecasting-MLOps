"""Build and record the purged global-model evaluation calendar.

Run after `scripts.build_global_panel`:
`uv run python -m scripts.build_global_splits`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_GLOBAL_MODEL_CONFIG, load_global_config
from src.eval.panel_splits import build_panel_split_plan

DEFAULT_PANEL_PATH = Path("data/processed/global_panel.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/global_split_plan.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_GLOBAL_MODEL_CONFIG,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    config = load_global_config(args.config)
    plan = build_panel_split_plan(
        panel,
        n_splits=config.eval.n_splits,
        horizon=config.data.horizon,
        min_train_size=config.eval.min_train_size,
        val_frac=config.eval.val_frac,
        holdout_size=config.eval.holdout_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
    print(
        f"Wrote {len(plan.folds)} purged folds plus a "
        f"{len(plan.holdout.test_dates)}-date holdout to {args.output}"
    )


if __name__ == "__main__":
    main()
