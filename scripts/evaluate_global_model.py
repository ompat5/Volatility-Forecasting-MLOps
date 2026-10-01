"""Evaluate the global LSTM and same-date baselines on all 34 targets.

Run with: `uv run python -m scripts.evaluate_global_model`
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_GLOBAL_MODEL_CONFIG, load_global_config
from src.data.ingest import DEFAULT_TICKERS_CONFIG
from src.data.universe import load_universe
from src.eval.global_evaluate import evaluate_global_model

DEFAULT_PANEL_PATH = Path("data/processed/global_panel.parquet")
DEFAULT_OUTPUT_DIR = Path("data/processed/global_evaluation")
DEFAULT_BENCHMARK_PATH = Path("benchmarks/global_model.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_GLOBAL_MODEL_CONFIG)
    parser.add_argument("--universe", type=Path, default=DEFAULT_TICKERS_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=None,
        help=(
            "Concise summary path. The canonical default-input run writes "
            "benchmarks/global_model.json automatically; custom runs must opt in."
        ),
    )
    parser.add_argument(
        "--skip-garch",
        action="store_true",
        help="Development smoke test only; full reported runs must include GARCH",
    )
    args = parser.parse_args()
    if args.skip_garch and args.benchmark_output is not None:
        parser.error("--benchmark-output cannot be used with --skip-garch")
    if args.skip_garch and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_OUTPUT_DIR.with_name("global_evaluation_smoke")
    canonical_run = (
        args.panel == DEFAULT_PANEL_PATH
        and args.config == DEFAULT_GLOBAL_MODEL_CONFIG
        and args.universe == DEFAULT_TICKERS_CONFIG
    )
    benchmark_output = args.benchmark_output
    if benchmark_output is None and canonical_run and not args.skip_garch:
        benchmark_output = DEFAULT_BENCHMARK_PATH

    panel = pd.read_parquet(args.panel)
    config = load_global_config(args.config)
    universe = load_universe(args.universe)
    result = evaluate_global_model(
        panel,
        universe,
        config,
        include_garch=not args.skip_garch,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    result.per_ticker_metrics.to_csv(
        args.output_dir / "per_ticker_metrics.csv",
        index=False,
    )
    result.aggregate_metrics.to_csv(
        args.output_dir / "aggregate_metrics.csv",
        index=False,
    )
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(result.training_summary, indent=2) + "\n"
    )
    (args.output_dir / "split_plan.json").write_text(
        json.dumps(result.split_plan, indent=2) + "\n"
    )
    evaluation_manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel": {
            "path": str(args.panel),
            "sha256": _sha256(args.panel),
            "rows": len(panel),
            "dates": panel.index.get_level_values("date").nunique(),
        },
        "universe": {
            "targets": list(universe.target_symbols),
            "context": list(universe.context_symbols),
        },
        "config": asdict(config),
        "includes_garch": not args.skip_garch,
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2) + "\n"
    )
    if benchmark_output is not None:
        aggregate_metrics = result.aggregate_metrics[
            result.aggregate_metrics["split"].isin(["cv_combined", "final_holdout"])
        ]
        holdout_per_ticker = result.per_ticker_metrics[
            result.per_ticker_metrics["split"] == "final_holdout"
        ]
        benchmark = {
            **evaluation_manifest,
            "split_plan": result.split_plan,
            "training": {
                "folds": [
                    {
                        "split": fold["split"],
                        "best_epoch": fold["best_epoch"],
                        "best_validation_loss": fold["best_validation_loss"],
                    }
                    for fold in result.training_summary["folds"]
                ],
                "selected_epochs": result.training_summary["selected_epochs"],
            },
            "aggregate_metrics": json.loads(
                aggregate_metrics.to_json(orient="records")
            ),
            "final_holdout_per_ticker_metrics": json.loads(
                holdout_per_ticker.to_json(orient="records")
            ),
        }
        benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        benchmark_output.write_text(json.dumps(benchmark, indent=2) + "\n")

    holdout = result.aggregate_metrics.query(
        "split == 'final_holdout' and scope in ['macro', 'micro']"
    )
    print(holdout.to_string(index=False))
    print(f"Wrote global evaluation artifacts to {args.output_dir}")
    if benchmark_output is not None:
        print(f"Wrote concise benchmark to {benchmark_output}")


if __name__ == "__main__":
    main()
