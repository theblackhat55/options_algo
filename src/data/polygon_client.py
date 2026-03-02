"""
src/data/polygon_client.py
==========================
Rate-limited Polygon.io REST client wrapper.
Free tier: 5 calls/min → 1 call every 13 seconds to stay safe.
Starter tier: unlimited → no delay needed.
"""
import os
import time
import logging
from functools import wraps
from polygon import RESTClient

logger = logging.getLogger(__name__)

# Rate limit config
FREE_TIER_DELAY = 13  # seconds between calls (5/min = 12s, add 1s buffer)
STARTER_TIER_DELAY = 0.1  # minimal delay for paid tier

_last_call_time = 0.0
_client = None
_delay = FREE_TIER_DELAY  # default to free tier


def get_client() -> RESTClient:
    """Get or create the singleton Polygon REST client."""
    global _client
    if _client is None:
        api_key = os.getenv("POLYGON_API_KEY", "")
        if not api_key:
            raise ValueError("POLYGON_API_KEY not set in environment or .env")
        _client = RESTClient(api_key)
        logger.info("Polygon client initialized")
    return _client


def set_tier(tier: str = "free"):
    """Set rate limit tier: 'free' (5/min) or 'starter' (unlimited)."""
    global _delay
    if tier.lower() == "starter":
        _delay = STARTER_TIER_DELAY
        logger.info("Polygon rate limit: STARTER (unlimited)")
    else:
        _delay = FREE_TIER_DELAY
        logger.info(f"Polygon rate limit: FREE ({FREE_TIER_DELAY}s between calls)")


def throttle():
    """Wait if needed to respect rate limits."""
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < _delay:
        wait = _delay - elapsed
        logger.debug(f"Polygon throttle: waiting {wait:.1f}s")
        time.sleep(wait)
    _last_call_time = time.time()


def rate_limited(func):
    """Decorator to add rate limiting to any Polygon API call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        throttle()
        return func(*args, **kwargs)
    return wrapper


@rate_limited
def list_options_contracts(underlying: str, **kwargs):
    """Rate-limited wrapper for listing options contracts."""
    client = get_client()
    return list(client.list_options_contracts(underlying, **kwargs))


@rate_limited
def get_snapshot_option(underlying: str, option_contract: str):
    """Rate-limited wrapper for option snapshot."""
    client = get_client()
    return client.get_snapshot_option(underlying, option_contract)


@rate_limited
def list_snapshot_options_chain(underlying: str, **kwargs):
    """Rate-limited wrapper for full options chain snapshot."""
    client = get_client()
    return list(client.list_snapshot_options_chain(underlying, **kwargs))


@rate_limited
def get_aggs(ticker: str, multiplier: int, timespan: str, from_: str, to: str, **kwargs):
    """Rate-limited wrapper for aggregates (OHLCV bars)."""
    client = get_client()
    return list(client.get_aggs(ticker, multiplier, timespan, from_, to, **kwargs))
