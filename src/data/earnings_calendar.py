"""
src/data/earnings_calendar.py
==============================
Fetch and cache earnings dates and other catalysts (ex-div, FOMC).
Primary: Finnhub. Backup: yfinance.

Usage:
    from src.data.earnings_calendar import get_earnings_dates, build_calendar_cache
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from config.settings import FINNHUB_API_KEY, CALENDAR_DIR

logger = logging.getLogger(__name__)

_CACHE_FILE = CALENDAR_DIR / "earnings_cache.json"
_CACHE_TTL_HOURS = 12     # Refresh cache if older than 12 hours


# ─── Earnings ─────────────────────────────────────────────────────────────────

def get_earnings_dates(ticker: str, days_ahead: int = 60) -> list[date]:
    """
    Return list of upcoming earnings dates for a ticker.
    Uses Finnhub with local JSON cache.
    """
    cache = _load_cache()
    today_str = date.today().isoformat()
    cache_key = f"{ticker}_{today_str}"

    # P3 FIX #12: Prune stale entries (keys not matching today's date) to prevent
    # unbounded cache growth when the scanner runs daily.
    stale_keys = [k for k in list(cache.keys()) if not k.endswith(f"_{today_str}")]
    if stale_keys:
        for k in stale_keys:
            del cache[k]
        logger.debug(f"Pruned {len(stale_keys)} stale earnings cache entries")
        _save_cache(cache)

    if cache_key in cache:
        return [datetime.strptime(d, "%Y-%m-%d").date() for d in cache[cache_key]]

    dates = _fetch_finnhub_earnings(ticker, days_ahead)
    if not dates:
        dates = _fetch_yfinance_earnings(ticker)

    # Cache for today
    cache[cache_key] = [d.isoformat() for d in dates]
    _save_cache(cache)
    return dates


def _fetch_finnhub_earnings(ticker: str, days_ahead: int = 60) -> list[date]:
    """Fetch earnings calendar from Finnhub."""
    if not FINNHUB_API_KEY:
        return []
    try:
        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {
            "symbol": ticker,
            "from": date.today().isoformat(),
            "to": (date.today() + timedelta(days=days_ahead)).isoformat(),
            "token": FINNHUB_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        dates = []
        for item in data.get("earningsCalendar", []):
            d_str = item.get("date")
            if d_str:
                dates.append(datetime.strptime(d_str, "%Y-%m-%d").date())
        return sorted(dates)

    except Exception as exc:
        logger.warning(f"Finnhub earnings fetch failed for {ticker}: {exc}")
        return []


def _fetch_yfinance_earnings(ticker: str) -> list[date]:
    """Fallback: get next earnings from yfinance calendar."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).calendar
        if info is None:
            return []
        # yfinance returns a dict with 'Earnings Date' key
        if isinstance(info, dict):
            earn = info.get("Earnings Date")
            if earn is None:
                return []
            if hasattr(earn, "__iter__") and not isinstance(earn, str):
                return [e.date() if hasattr(e, "date") else e for e in earn]
            if hasattr(earn, "date"):
                return [earn.date()]
        return []
    except Exception as exc:
        logger.debug(f"yfinance earnings failed for {ticker}: {exc}")
        return []


def build_calendar_cache(tickers: list[str]) -> dict[str, list[str]]:
    """
    Pre-build earnings cache for the whole universe.
    Call this once during nightly scan to warm the cache.
    """
    cache = _load_cache()
    today = date.today().isoformat()

    for ticker in tickers:
        cache_key = f"{ticker}_{today}"
        if cache_key not in cache:
            dates = _fetch_finnhub_earnings(ticker, days_ahead=60)
            cache[cache_key] = [d.isoformat() for d in dates]

    _save_cache(cache)
    return cache


# ─── Cache Helpers ────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as exc:
        logger.warning(f"Failed to save earnings cache: {exc}")
