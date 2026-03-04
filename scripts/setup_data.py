#!/usr/bin/env python3
"""
scripts/setup_data.py
=====================
Initial data download: backfill 2 years of OHLCV for the entire universe.
Run once on first setup, then use update_universe() for incremental updates.

Usage:
    python scripts/setup_data.py
    python scripts/setup_data.py --small    # Test with 10 tickers
"""
import sys
import logging
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.universe import get_universe
from src.data.stock_fetcher import download_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    small = "--small" in sys.argv
    period = "2y"

    tickers = get_universe()
    if small:
        tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA",
                   "TSLA", "AMD", "JPM", "XLK", "XLF"]
        logger.info(f"Small mode: {len(tickers)} tickers")
    else:
        logger.info(f"Full universe: {len(tickers)} tickers")

    logger.info(f"Downloading {period} of daily data…")
    data = download_universe(tickers, period=period, save=True)

    logger.info(f"\nComplete: {len(data)}/{len(tickers)} tickers downloaded")
    failed = [t for t in tickers if t not in data]
    if failed:
        logger.warning(f"Failed: {failed}")

    # Quick summary
    for ticker in list(data.keys())[:5]:
        df = data[ticker]
        logger.info(
            f"  {ticker}: {len(df)} rows | "
            f"{df.index[0].date()} → {df.index[-1].date()} | "
            f"Latest close: ${df['close'].iloc[-1]:.2f}"
        )


if __name__ == "__main__":
    main()
