"""
scripts/backfill_iv_from_flatfiles.py
=====================================
Build a 2-year real IV history for every SP100 ticker by:
  1. Loading each day's options flat file
  2. Finding ATM contracts (~30 DTE) for each ticker
  3. Inverting Black-Scholes to extract implied volatility from the mid-price
  4. Saving to iv_snapshots/{TICKER}_iv_history.parquet

This populates the exact files that volatility.py's _load_iv_snapshot_history()
reads, immediately replacing the HV×1.15 proxy with real market IV.

Usage:
    python scripts/backfill_iv_from_flatfiles.py
    python scripts/backfill_iv_from_flatfiles.py --start 2024-03-01 --end 2025-03-01
    python scripts/backfill_iv_from_flatfiles.py --tickers AAPL,MSFT,NVDA
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from config.settings import IV_SNAPSHOT_DIR, RAW_DIR, RISK_FREE_RATE
from config.universe import get_universe
from src.data.massive_s3 import (
    download_options_day,
    list_available_dates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Black-Scholes IV Solver ─────────────────────────────────────────────────

def _bs_price(S, K, T, r, sigma, opt_type="call"):
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(market_price, S, K, T, r=RISK_FREE_RATE, opt_type="call"):
    """
    Invert Black-Scholes to find implied volatility from a market price.
    Returns IV as a decimal (e.g., 0.25 for 25%) or None if it can't solve.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    # Quick intrinsic value check
    if opt_type == "call":
        intrinsic = max(S - K * math.exp(-r * T), 0)
    else:
        intrinsic = max(K * math.exp(-r * T) - S, 0)

    if market_price < intrinsic * 0.95:
        return None  # price below intrinsic — bad data

    try:
        iv = brentq(
            lambda sigma: _bs_price(S, K, T, r, sigma, opt_type) - market_price,
            0.01,   # 1% vol floor
            5.0,    # 500% vol ceiling
            xtol=1e-6,
            maxiter=100,
        )
        return round(iv, 6)
    except (ValueError, RuntimeError):
        return None


# ─── Main Backfill Logic ─────────────────────────────────────────────────────

def compute_atm_iv_for_day(
    trade_date: date,
    tickers: list[str],
    stock_prices: dict[str, float],
) -> dict[str, float]:
    """
    For a single trading day, download the options flat file,
    find ATM calls ~30 DTE for each ticker, solve for IV.

    Returns:
        {ticker: atm_iv_pct} e.g. {"AAPL": 28.5, "MSFT": 22.1}
    """
    opts_df = download_options_day(trade_date, underlyings=tickers)
    if opts_df.empty:
        return {}

    results = {}

    for ticker in tickers:
        price = stock_prices.get(ticker)
        if not price or price <= 0:
            continue

        # Filter to this ticker's calls
        tk_opts = opts_df[
            (opts_df["underlying"] == ticker) &
            (opts_df["type"] == "call")
        ].copy()
        if tk_opts.empty:
            continue

        # Compute DTE for each contract
        tk_opts["exp_date"] = pd.to_datetime(tk_opts["expiration"])
        tk_opts["dte"] = (tk_opts["exp_date"] - pd.Timestamp(trade_date)).dt.days

        # Target: 20-40 DTE range (centered on 30)
        near_term = tk_opts[(tk_opts["dte"] >= 20) & (tk_opts["dte"] <= 45)]
        if near_term.empty:
            near_term = tk_opts[(tk_opts["dte"] >= 10) & (tk_opts["dte"] <= 60)]
        if near_term.empty:
            continue

        # Find the ATM strike (closest to current price)
        near_term = near_term.copy()
        near_term["strike_dist"] = (near_term["strike"] - price).abs()
        atm_candidates = near_term.nsmallest(4, "strike_dist")

        # Compute IV for each candidate, take the median
        ivs = []
        for _, row in atm_candidates.iterrows():
            mid = (row["open"] + row["close"]) / 2  # day-agg mid approximation
            if mid <= 0:
                mid = row["close"]
            if mid <= 0:
                continue

            T = row["dte"] / 365.0
            iv = implied_vol(mid, price, row["strike"], T, RISK_FREE_RATE, "call")
            if iv and 0.02 < iv < 3.0:  # sanity: 2% to 300%
                ivs.append(iv)

        if ivs:
            atm_iv = float(np.median(ivs)) * 100  # convert to percentage
            results[ticker] = round(atm_iv, 2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Backfill IV history from Massive options flat files")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: full universe)")
    args = parser.parse_args()

    end_date = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start_date = date.fromisoformat(args.start) if args.start else end_date - timedelta(days=730)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = get_universe()

    logger.info(f"Backfilling IV for {len(tickers)} tickers from {start_date} to {end_date}")

    # Get available trading days (use stocks list — same trading calendar)
    logger.info("Listing available trading days...")
    available = list_available_dates("options")
    target_dates = [d for d in available if start_date <= d <= end_date]
    logger.info(f"Found {len(target_dates)} trading days with options data")

    if not target_dates:
        logger.error("No trading days found. Check S3 credentials and date range.")
        return

    # Load stock prices for each day (needed for ATM determination)
    # Pre-load from parquet cache if available, otherwise download
    logger.info("Loading stock price data...")
    stock_price_cache: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        parquet_path = RAW_DIR / f"{ticker.lower()}_daily.parquet"
        if parquet_path.exists():
            try:
                stock_price_cache[ticker] = pd.read_parquet(parquet_path)
            except Exception:
                pass

    # Accumulate IV per ticker across all days
    iv_history: dict[str, list[dict]] = {t: [] for t in tickers}
    days_processed = 0
    days_with_data = 0

    for i, trade_date in enumerate(target_dates):
        if i % 20 == 0:
            logger.info(
                f"Day {i+1}/{len(target_dates)}: {trade_date} "
                f"({days_with_data} days with IV data so far)"
            )

        # Get stock prices for this date
        day_prices: dict[str, float] = {}
        for ticker in tickers:
            df = stock_price_cache.get(ticker)
            if df is not None and not df.empty:
                ts = pd.Timestamp(trade_date)
                if ts in df.index:
                    day_prices[ticker] = float(df.loc[ts, "close"])

        # Skip stock flat-file fallback (Options Starter plan — no stock files)
        # Rely solely on yfinance parquet cache for stock prices

        if not day_prices:
            continue

        # Compute ATM IV from options flat file
        day_ivs = compute_atm_iv_for_day(trade_date, tickers, day_prices)

        if day_ivs:
            days_with_data += 1
            for ticker, atm_iv in day_ivs.items():
                iv_history[ticker].append({
                    "date": trade_date,
                    "atm_iv": atm_iv,
                })

        days_processed += 1

    # Save IV history parquets
    logger.info(f"\nProcessed {days_processed} days, {days_with_data} had IV data")
    saved = 0
    for ticker, rows in iv_history.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        path = IV_SNAPSHOT_DIR / f"{ticker}_iv_history.parquet"

        # Merge with existing snapshot data if present
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            except Exception:
                pass

        df.to_parquet(path)
        saved += 1
        if saved <= 5:
            logger.info(f"  {ticker}: {len(df)} days of IV history → {path.name}")

    logger.info(f"\n✅ Saved IV history for {saved} tickers to {IV_SNAPSHOT_DIR}")
    logger.info(f"   Your algo will now use REAL IV rank instead of HV×1.15 proxy")

    # Summary stats
    lengths = [len(v) for v in iv_history.values() if v]
    if lengths:
        logger.info(f"   Median days per ticker: {sorted(lengths)[len(lengths)//2]}")
        logger.info(f"   Min: {min(lengths)}, Max: {max(lengths)}")


if __name__ == "__main__":
    main()
