"""Snapshot training-feature distributions used by the drift monitor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.ingest import DEFAULT_RAW_DIR
from src.features.realized_vol import FEATURE_COLS, build_inference_features
from src.monitoring.drift import build_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument(
        "--prices",
        type=Path,
        default=None,
        help="Optional parquet input; defaults to data/raw/{ticker}.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("monitoring/reference.json"),
    )
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    prices_path = args.prices or DEFAULT_RAW_DIR / f"{args.ticker}.parquet"
    prices = pd.read_parquet(prices_path)["Adj Close"]
    features = build_inference_features(prices)[FEATURE_COLS]
    reference = build_reference(features, ticker=args.ticker, n_bins=args.bins)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
    print(f"Wrote monitoring reference to {args.output}")


if __name__ == "__main__":
    main()
