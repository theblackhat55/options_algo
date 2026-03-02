"""
src/data/stock_fetcher.py
=========================
Download and cache daily OHLCV data for the stock universe.
Primary: yfinance (free). Backup: Polygon REST API.

Usage:
    from src.data.stock_fetcher import download_universe, update_universe, load_ticker

    data = download_universe(["AAPL", "MSFT", "SPY"], period="2y")
    data = update_universe(tickers)        # incremental update
    df   = load_ticker("AAPL")             # load from cache
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import RAW_DIR, POLYGON_API_KEY

logger = logging.getLogger(__name__)

_RATE_LIMIT_SLEEP = 0.5   # seconds between individual downloads


# ─── Download ─────────────────────────────────────────────────────────────────

def download_universe(
    tickers: list[str],
    period: str = "2y",
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for all tickers in the universe.

    Args:
        tickers: List of ticker symbols
        period: yfinance period string (1y, 2y, 5y, max)
        save: Whether to persist data to parquet files

    Returns:
        {ticker: OHLCV DataFrame} — failed tickers are omitted.
    """
    logger.info(f"Downloading {len(tickers)} tickers (period={period})")
    data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    # ── Attempt batch download ────────────────────────────────────────────────
    try:
        raw = yf.download(
            tickers,
            period=period,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].dropna(how="all")

                df = _clean_ohlcv(df)
                if df is None or len(df) < 50:
                    failed.append(ticker)
                    continue

                data[ticker] = df
                if save:
                    _save_parquet(ticker, df)

            except Exception as exc:
                logger.debug(f"Batch: post-process failed for {ticker}: {exc}")
                failed.append(ticker)

    except Exception as exc:
        logger.warning(f"Batch download failed ({exc}); falling back to individual")
        failed = list(tickers)

    # ── Individual fallback for failed tickers ────────────────────────────────
    if failed:
        logger.info(f"Retrying {len(failed)} tickers individually")
        for ticker in failed:
            df = _download_single(ticker, period=period)
            if df is not None:
                data[ticker] = df
                if save:
                    _save_parquet(ticker, df)
            time.sleep(_RATE_LIMIT_SLEEP)

    still_failed = [t for t in tickers if t not in data]
    logger.info(
        f"Downloaded {len(data)}/{len(tickers)} tickers. "
        f"Failed: {still_failed[:10]}{'...' if len(still_failed) > 10 else ''}"
    )
    return data


def _download_single(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Download a single ticker with error handling."""
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        return _clean_ohlcv(raw)
    except Exception as exc:
        logger.warning(f"Individual download failed for {ticker}: {exc}")
        return None


# ─── Incremental Update ───────────────────────────────────────────────────────

def update_universe(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Incremental update: append new candles to existing cached data.
    Falls back to full 2-year download if no cache exists.

    Returns:
        {ticker: up-to-date OHLCV DataFrame}
    """
    data: dict[str, pd.DataFrame] = {}
    needs_full: list[str] = []

    today = datetime.now().date()

    for ticker in tickers:
        cached = load_ticker(ticker)
        if cached is None or len(cached) == 0:
            needs_full.append(ticker)
            continue

        last_date = cached.index[-1].date()
        days_stale = (today - last_date).days

        if days_stale <= 1:
            # Already up to date (or today's data not yet available)
            data[ticker] = cached
            continue

        # Download only missing period
        start = (last_date + timedelta(days=1)).isoformat()
        try:
            new = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if not new.empty:
                new = _clean_ohlcv(new)
                if new is not None:
                    combined = pd.concat([cached, new])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    data[ticker] = combined
                    _save_parquet(ticker, combined)
                    continue
        except Exception as exc:
            logger.debug(f"Incremental update failed for {ticker}: {exc}")

        # Fall through: use cached
        data[ticker] = cached

    if needs_full:
        logger.info(f"Full download needed for {len(needs_full)} tickers")
        full = download_universe(needs_full, period="2y", save=True)
        data.update(full)

    return data


# ─── Cache I/O ────────────────────────────────────────────────────────────────

def load_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """Load cached OHLCV from parquet. Returns None if not found."""
    path = _parquet_path(ticker)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        except Exception as exc:
            logger.warning(f"Failed to load parquet for {ticker}: {exc}")
    return None


def load_universe(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load cached data for multiple tickers."""
    return {t: df for t in tickers if (df := load_ticker(t)) is not None}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalise column names, drop empties, sort by date."""
    if df is None or df.empty:
        return None
    df = df.copy()
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(subset=["close"], inplace=True)
    return df if len(df) >= 10 else None


def _parquet_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker.lower()}_daily.parquet"


def _save_parquet(ticker: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_parquet_path(ticker))
    except Exception as exc:
        logger.warning(f"Failed to save parquet for {ticker}: {exc}")


def get_current_price(ticker: str, data: dict[str, pd.DataFrame] = None) -> Optional[float]:
    """Get most recent closing price for a ticker."""
    if data and ticker in data:
        return float(data[ticker]["close"].iloc[-1])
    df = load_ticker(ticker)
    if df is not None and len(df) > 0:
        return float(df["close"].iloc[-1])
    return None
