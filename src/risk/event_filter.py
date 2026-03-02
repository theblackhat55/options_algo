"""
src/risk/event_filter.py
========================
Filter out stocks with upcoming earnings, ex-dividend dates, or other
binary events within the trade's expiration window.

Rule: Never enter an options trade that spans an earnings event.
      Exception: intentional earnings plays (not in this system's scope).

Usage:
    from src.risk.event_filter import is_safe_to_trade, EventFilterResult
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import requests

from config.settings import FINNHUB_API_KEY
from src.data.earnings_calendar import get_earnings_dates

logger = logging.getLogger(__name__)


@dataclass
class EventFilterResult:
    safe: bool
    ticker: str
    reason: Optional[str]
    earnings_in_days: Optional[int]
    events: list[dict] = field(default_factory=list)


def is_safe_to_trade(
    ticker: str,
    target_dte: int,
    buffer_days: int = 3,   # Extra days of clearance after earnings
) -> EventFilterResult:
    """
    Check if a stock is safe to enter an options trade.

    Unsafe if:
    1. Earnings fall within [today, expiry + buffer_days]
    2. Other binary events (not yet implemented — placeholder)

    Args:
        ticker: Stock symbol
        target_dte: Days to option expiration
        buffer_days: Additional days required after event

    Returns:
        EventFilterResult
    """
    today = date.today()
    expiry = today + timedelta(days=target_dte + buffer_days)
    events = []

    # ── Earnings Check ────────────────────────────────────────────────────────
    earnings = get_earnings_dates(ticker, days_ahead=target_dte + buffer_days + 5)

    for earn_date in earnings:
        if today <= earn_date <= expiry:
            days_until = (earn_date - today).days
            events.append({
                "type": "earnings",
                "date": earn_date.isoformat(),
                "days_until": days_until,
            })
            return EventFilterResult(
                safe=False,
                ticker=ticker,
                reason=(
                    f"Earnings on {earn_date} ({days_until}d) "
                    f"within {target_dte}DTE expiry window"
                ),
                earnings_in_days=days_until,
                events=events,
            )

    return EventFilterResult(
        safe=True,
        ticker=ticker,
        reason=None,
        earnings_in_days=None,
        events=events,
    )


def filter_safe_tickers(
    tickers: list[str],
    dte_map: dict[str, int],
    max_per_batch: int = 50,
) -> dict[str, EventFilterResult]:
    """
    Batch-check a list of tickers for event safety.

    Args:
        tickers: List of ticker symbols to check
        dte_map: {ticker: target_dte} mapping
        max_per_batch: Cap to avoid API rate limits

    Returns:
        {ticker: EventFilterResult}
    """
    import time
    results = {}
    checked = 0

    for ticker in tickers[:max_per_batch]:
        dte = dte_map.get(ticker, 45)
        result = is_safe_to_trade(ticker, dte)
        results[ticker] = result
        checked += 1

        if not result.safe:
            logger.info(f"  FILTER {ticker}: {result.reason}")

        if checked % 10 == 0:
            time.sleep(0.5)   # Finnhub rate limiting

    safe_count = sum(1 for r in results.values() if r.safe)
    logger.info(f"Event filter: {safe_count}/{len(results)} tickers are safe")
    return results
