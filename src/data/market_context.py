"""
src/data/market_context.py
==========================
Fetch market-wide context: VIX, sector ETF performance, market breadth.
Used to gate overall signal generation (avoid trading in crash conditions).

Usage:
    from src.data.market_context import get_market_context

    ctx = get_market_context(data)
    if ctx.vix_level < 30 and ctx.market_regime != "CRASH":
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config.universe import SECTOR_ETF_MAP

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Current market-wide conditions."""
    vix_level: float            # Current VIX (from ^VIX or VXX proxy)
    vix_regime: str             # "LOW" (<15), "NORMAL" (15-25), "HIGH" (25-35), "EXTREME" (>35)
    spy_trend: str              # "UPTREND", "RANGE", "DOWNTREND"
    spy_return_5d: float        # SPY 5-day return (%)
    spy_return_20d: float       # SPY 20-day return (%)
    advance_decline: float      # Approximate advance/decline ratio
    market_regime: str          # "BULL", "SIDEWAYS", "BEAR", "CRASH"
    sector_leaders: list[str]   # Top 3 performing sectors (30d)
    sector_laggards: list[str]  # Bottom 3 performing sectors (30d)
    breadth_score: float        # 0-1 (% of universe above 50 EMA)
    notes: str = ""


def get_market_context(
    data: dict[str, pd.DataFrame],
    vix_level: Optional[float] = None,
) -> MarketContext:
    """
    Compute current market context from universe price data.

    Args:
        data: {ticker: OHLCV DataFrame} for the full universe
        vix_level: Override VIX level (if fetched separately)

    Returns:
        MarketContext object
    """
    # ── SPY Trend ─────────────────────────────────────────────────────────────
    spy_df = data.get("SPY")
    spy_trend = "RANGE"
    spy_ret_5d = 0.0
    spy_ret_20d = 0.0

    if spy_df is not None and len(spy_df) >= 50:
        close = spy_df["close"]
        spy_ret_5d = round((close.iloc[-1] / close.iloc[-5] - 1) * 100, 2)
        spy_ret_20d = round((close.iloc[-1] / close.iloc[-21] - 1) * 100, 2)

        import pandas_ta as ta
        ema20 = ta.ema(close, length=20).iloc[-1]
        ema50 = ta.ema(close, length=50).iloc[-1]
        price = close.iloc[-1]

        if price > ema20 > ema50:
            spy_trend = "UPTREND"
        elif price < ema20 < ema50:
            spy_trend = "DOWNTREND"
        else:
            spy_trend = "RANGE"

    # ── VIX Proxy ─────────────────────────────────────────────────────────────
    if vix_level is None:
        vix_level = _estimate_vix(data)

    if vix_level < 15:
        vix_regime = "LOW"
    elif vix_level < 25:
        vix_regime = "NORMAL"
    elif vix_level < 35:
        vix_regime = "HIGH"
    else:
        vix_regime = "EXTREME"

    # ── Market Regime ─────────────────────────────────────────────────────────
    if vix_level >= 35 and spy_ret_20d < -10:
        market_regime = "CRASH"
    elif spy_trend == "DOWNTREND" and vix_level > 25:
        market_regime = "BEAR"
    elif spy_trend == "UPTREND" and vix_level < 25:
        market_regime = "BULL"
    else:
        market_regime = "SIDEWAYS"

    # ── Sector Performance ────────────────────────────────────────────────────
    sector_perf: dict[str, float] = {}
    for sector, etf in SECTOR_ETF_MAP.items():
        etf_df = data.get(etf)
        if etf_df is not None and len(etf_df) >= 31:
            ret = (etf_df["close"].iloc[-1] / etf_df["close"].iloc[-21] - 1) * 100
            sector_perf[sector] = round(ret, 2)

    sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
    sector_leaders = [s for s, _ in sorted_sectors[:3]]
    sector_laggards = [s for s, _ in sorted_sectors[-3:]]

    # ── Market Breadth (% of universe above 50 EMA) ───────────────────────────
    above_ema50 = 0
    total_counted = 0
    try:
        import pandas_ta as ta
        for ticker, df in data.items():
            if df is None or len(df) < 55:
                continue
            close = df["close"]
            ema50 = ta.ema(close, length=50).iloc[-1]
            if not np.isnan(ema50):
                total_counted += 1
                if close.iloc[-1] > ema50:
                    above_ema50 += 1
    except Exception:
        pass

    breadth = round(above_ema50 / max(total_counted, 1), 3)
    adv_decline = round(breadth / (1 - breadth + 0.001), 2) if breadth < 1 else 10.0

    notes = []
    if vix_level > 30:
        notes.append(f"Elevated VIX ({vix_level:.0f}) — reduce size")
    if market_regime == "CRASH":
        notes.append("CRASH conditions — suspend all new trades")
    if breadth < 0.3:
        notes.append(f"Weak breadth ({breadth:.0%}) — avoid longs")
    elif breadth > 0.7:
        notes.append(f"Strong breadth ({breadth:.0%}) — favours bulls")

    return MarketContext(
        vix_level=round(vix_level, 1),
        vix_regime=vix_regime,
        spy_trend=spy_trend,
        spy_return_5d=spy_ret_5d,
        spy_return_20d=spy_ret_20d,
        advance_decline=adv_decline,
        market_regime=market_regime,
        sector_leaders=sector_leaders,
        sector_laggards=sector_laggards,
        breadth_score=breadth,
        notes=" | ".join(notes),
    )


def _estimate_vix(data: dict[str, pd.DataFrame]) -> float:
    """
    Estimate VIX from SPY realized volatility (poor man's VIX proxy).
    Returns 20 as default if SPY data unavailable.
    """
    spy_df = data.get("SPY")
    if spy_df is None or len(spy_df) < 22:
        return 20.0

    close = spy_df["close"]
    log_ret = np.log(close / close.shift(1)).dropna()
    hv_20 = log_ret.tail(20).std() * np.sqrt(252) * 100  # Annualised %
    return round(float(hv_20) * 1.15, 1)   # VIX typically ~15% above realized vol
