"""Download and cache raw OHLCV data for the ticker basket via yfinance.

Each ticker is cached independently to data/raw/{ticker}.parquet so later
phases (features, baselines, DL) never need network access. Re-running this
script re-downloads full history and overwrites the cache.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TICKERS_CONFIG = REPO_ROOT / "configs" / "tickers.yaml"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"


def load_ticker_list(config_path: Path = DEFAULT_TICKERS_CONFIG) -> list[str]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["tickers"]


def _cache_path(ticker: str, raw_dir: Path) -> Path:
    safe_name = ticker.replace("^", "")
    return raw_dir / f"{safe_name}.parquet"


def fetch_ticker(ticker: str) -> pd.DataFrame:
    """Download full available daily OHLCV history for a single ticker."""
    df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r}")
    df.index.name = "date"
    return df


def ingest_ticker(ticker: str, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = fetch_ticker(ticker)
    out_path = _cache_path(ticker, raw_dir)
    df.to_parquet(out_path)
    logger.info("Cached %s rows for %s -> %s", len(df), ticker, out_path)
    return out_path


def ingest_basket(
    tickers: list[str] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> dict[str, Path | Exception]:
    """Ingest every ticker in the basket; failures are caught per-ticker so one
    bad symbol doesn't abort the whole run."""
    tickers = tickers or load_ticker_list()
    results: dict[str, Path | Exception] = {}
    for ticker in tickers:
        try:
            results[ticker] = ingest_ticker(ticker, raw_dir)
        except Exception as exc:  # noqa: BLE001 - report per-ticker, keep going
            logger.error("Failed to ingest %s: %s", ticker, exc)
            results[ticker] = exc
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TICKERS_CONFIG,
        help="Path to tickers.yaml",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to cache raw parquet files",
    )
    args = parser.parse_args()

    tickers = load_ticker_list(args.config)
    results = ingest_basket(tickers, args.raw_dir)

    failures = {t: e for t, e in results.items() if isinstance(e, Exception)}
    logger.info("Ingested %d/%d tickers", len(results) - len(failures), len(results))
    if failures:
        logger.warning("Failed tickers: %s", list(failures))


if __name__ == "__main__":
    main()
