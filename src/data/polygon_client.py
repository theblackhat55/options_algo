"""
src/data/polygon_client.py
==========================
Rate-limited Polygon.io client using requests (not SDK pagination).
Free tier: 5 calls/min → 1 call every 13 seconds.
"""
import os
import time
import logging
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except Exception:
    pass

logger = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"
_api_key = None
_last_call_time = 0.0
_delay = 13  # free tier default


def _get_key():
    global _api_key
    if _api_key is None:
        _api_key = os.getenv("POLYGON_API_KEY", "")
        if not _api_key:
            raise ValueError("POLYGON_API_KEY not set")
    return _api_key


def set_tier(tier: str = "free"):
    global _delay
    if tier.lower() in ("starter", "paid", "premium"):
        _delay = 0.2
        logger.info("Polygon rate limit: PAID (unlimited)")
    else:
        _delay = 13
        logger.info(f"Polygon rate limit: FREE (13s between calls)")


def _throttle():
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < _delay:
        wait = _delay - elapsed
        logger.debug(f"Polygon throttle: waiting {wait:.1f}s")
        time.sleep(wait)
    _last_call_time = time.time()


def _request(endpoint: str, params: dict = None) -> dict:
    """Make a single throttled GET request to Polygon."""
    _throttle()
    key = _get_key()
    if params is None:
        params = {}
    params["apiKey"] = key
    url = f"{BASE_URL}{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 429:
        logger.warning("Rate limited — waiting 60s and retrying")
        time.sleep(60)
        _throttle()
        resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_options_contracts(underlying: str, limit: int = 50, **kwargs) -> list:
    """List options contracts — single page only (no auto-pagination)."""
    params = {"underlying_ticker": underlying, "limit": limit}
    params.update(kwargs)
    data = _request("/v3/reference/options/contracts", params)
    return data.get("results", [])


def get_options_chain(underlying: str, expiration_date: str = None,
                      strike_price_gte: float = None, strike_price_lte: float = None,
                      contract_type: str = None, limit: int = 250) -> list:
    """Get options chain snapshot for a ticker."""
    params = {"limit": limit}
    if expiration_date:
        params["expiration_date"] = expiration_date
    if strike_price_gte:
        params["strike_price.gte"] = strike_price_gte
    if strike_price_lte:
        params["strike_price.lte"] = strike_price_lte
    if contract_type:
        params["contract_type"] = contract_type
    data = _request(f"/v3/snapshot/options/{underlying}", params)
    return data.get("results", [])


def get_stock_snapshot(ticker: str) -> dict:
    """Get current stock snapshot (price, volume, etc)."""
    data = _request(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
    return data.get("ticker", {})


def get_aggs(ticker: str, multiplier: int, timespan: str,
             from_date: str, to_date: str, limit: int = 5000) -> list:
    """Get OHLCV aggregates."""
    params = {"adjusted": "true", "sort": "asc", "limit": limit}
    data = _request(
        f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
        params
    )
    return data.get("results", [])


def get_last_quote(ticker: str) -> dict:
    """Get last quote for a ticker."""
    data = _request(f"/v2/last/nbbo/{ticker}")
    return data.get("results", {})
