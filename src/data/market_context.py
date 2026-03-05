"""
src/data/market_context.py
==========================
Fetch market-wide context: VIX (real + proxy), sector ETF performance, market breadth.
Used to gate overall signal generation (avoid trading in crash/high-volatility conditions).

V2 additions:
  - _fetch_real_vix(): pulls actual VIX close from yfinance with fallback to HV proxy
  - MarketContext.vix_5d_avg: rolling 5-day VIX average
  - MarketContext.vix_spike: True when VIX jumped > VIX_SPIKE_THRESHOLD_PCT above avg
  - MarketContext.vix_tier: NORMAL / CAUTION / DEFENSIVE / LIQUIDATION
  - MarketContext.spy_return_5d already present; now also used in vix_tier gating

Usage:
    from src.data.market_context import get_market_context

    ctx = get_market_context(data)
    if ctx.vix_tier == "LIQUIDATION":
        return  # halt all new trades
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    VIX_CAUTION_LEVEL,
    VIX_DEFENSIVE_LEVEL,
    VIX_LIQUIDATION_LEVEL,
    VIX_SPIKE_WINDOW,
    VIX_SPIKE_THRESHOLD_PCT,
)
from config.universe import SECTOR_ETF_MAP

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Current market-wide conditions."""
    vix_level: float            # Current VIX (real or HV proxy)
    vix_regime: str             # "LOW" (<15), "NORMAL" (15-25), "HIGH" (25-35), "EXTREME" (>35)
    spy_trend: str              # "UPTREND", "RANGE", "DOWNTREND"
    spy_return_5d: float        # SPY 5-day return (%)
    spy_return_20d: float       # SPY 20-day return (%)
    advance_decline: float      # Approximate advance/decline ratio
    market_regime: str          # "BULL", "SIDEWAYS", "BEAR", "CRASH"
    sector_leaders: list[str]   # Top 3 performing sectors (30d)
    sector_laggards: list[str]  # Bottom 3 performing sectors (30d)
    breadth_score: float        # 0-1 (% of universe above 50 EMA)
    notes: str = ""             # Human-readable summary of conditions

    # ── V2: VIX Circuit-Breaker Fields (defaulted for backward compatibility) ───
    vix_5d_avg: float = 0.0     # 5-day rolling VIX average (spike baseline)
    vix_spike: bool = False     # True if VIX rose > VIX_SPIKE_THRESHOLD_PCT above avg
    vix_tier: str = "NORMAL"    # NORMAL / CAUTION / DEFENSIVE / LIQUIDATION


def _fetch_real_vix() -> Optional[tuple[float, float]]:
    """
    Pull the latest actual VIX and VIX_SPIKE_WINDOW-day average.

    Attempt order:
        1. IBKR live VIX (real-time, most accurate during market hours).
        2. yfinance ^VIX historical close (free, always available).

    For the 5-day average we always use yfinance historical data because
    IBKR only provides a point-in-time live value, not history.

    Returns:
        (vix_close, vix_5d_avg) tuple, both rounded to 1 decimal.
        Returns None on any failure so the caller can fall back to the HV proxy.
    """
    vix_live: Optional[float] = None

    # ── 1. Try IBKR for the current live VIX ─────────────────────────────────
    try:
        from src.data.ibkr_client import connect_ibkr, disconnect_ibkr, fetch_vix_ibkr
        ib = connect_ibkr()
        if ib is not None:
            vix_live = fetch_vix_ibkr(ib)
            disconnect_ibkr(ib)
            if vix_live:
                logger.info(f"VIX from IBKR: {vix_live}")
    except Exception as exc:
        logger.debug(f"IBKR VIX attempt failed (non-fatal): {exc}")

    # ── 2. yfinance for historical closes (always needed for 5d avg) ──────────
    try:
        import yfinance as yf
        hist = yf.Ticker("^VIX").history(period=f"{VIX_SPIKE_WINDOW + 10}d")
        if hist.empty:
            if vix_live:
                return round(vix_live, 1), round(vix_live, 1)
            return None
        closes = hist["Close"].dropna()
        if len(closes) < 1:
            if vix_live:
                return round(vix_live, 1), round(vix_live, 1)
            return None

        # Use IBKR live if available, else fall back to yfinance last close
        vix_close = round(float(vix_live if vix_live else closes.iloc[-1]), 1)
        vix_avg = round(float(closes.tail(VIX_SPIKE_WINDOW).mean()), 1)
        return vix_close, vix_avg
    except Exception as exc:
        logger.warning(f"Real VIX fetch failed: {exc}")
        if vix_live:
            return round(vix_live, 1), round(vix_live, 1)
        return None


def _classify_vix_tier(vix_level: float) -> str:
    """Map a raw VIX level to the circuit-breaker tier string."""
    # Import at call-time to pick up any env-override reloads in tests.
    from config.settings import (
        VIX_LIQUIDATION_LEVEL as _LIQ,
        VIX_DEFENSIVE_LEVEL as _DEF,
        VIX_CAUTION_LEVEL as _CAU,
    )
    if vix_level >= _LIQ:
        return "LIQUIDATION"
    if vix_level >= _DEF:
        return "DEFENSIVE"
    if vix_level >= _CAU:
        return "CAUTION"
    return "NORMAL"



def get_market_context(
    data: dict[str, pd.DataFrame],
    vix_level: Optional[float] = None,
) -> MarketContext:
    """
    Compute current market context from universe price data.

    V2: tries real VIX first (yfinance ^VIX), falls back to HV × 1.15 proxy.
    Adds vix_5d_avg, vix_spike, and vix_tier fields.

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

        import pandas_ta_classic as ta
        ema20 = ta.ema(close, length=20).iloc[-1]
        ema50 = ta.ema(close, length=50).iloc[-1]
        price = close.iloc[-1]

        if price > ema20 > ema50:
            spy_trend = "UPTREND"
        elif price < ema20 < ema50:
            spy_trend = "DOWNTREND"
        else:
            spy_trend = "RANGE"

    # ── VIX: real first, HV proxy as fallback ─────────────────────────────────
    vix_is_proxy = False
    vix_5d_avg_val: float = 0.0
    if vix_level is None:
        vix_result = _fetch_real_vix()
        if vix_result is not None:
            vix_level, vix_5d_avg_val = vix_result
            logger.info(f"  Real VIX: {vix_level} (5d avg: {vix_5d_avg_val})")
        else:
            vix_level = _estimate_vix(data)
            vix_5d_avg_val = vix_level
            vix_is_proxy = True
            logger.warning("Using HV proxy for VIX — real VIX fetch failed")
    else:
        # vix_level was explicitly provided — no avg available
        vix_5d_avg_val = vix_level

    # ── VIX Spike Detection ───────────────────────────────────────────────────
    vix_5d_avg = vix_5d_avg_val
    vix_spike = False
    if vix_5d_avg > 0 and not vix_is_proxy:
        spike_ratio = (vix_level - vix_5d_avg) / vix_5d_avg * 100
        vix_spike = spike_ratio > VIX_SPIKE_THRESHOLD_PCT
        if vix_spike:
            logger.warning(
                f"  VIX SPIKE detected: {vix_level} vs {vix_5d_avg} avg "
                f"(+{spike_ratio:.0f}%)"
            )

    # ── VIX Tier (use extracted helper for testability) ─────────────────────
    vix_tier = _classify_vix_tier(vix_level)

    # ── VIX Regime (legacy field kept for backward compat) ────────────────────
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
        import pandas_ta_classic as ta
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

    # ── Notes ─────────────────────────────────────────────────────────────────
    notes = []
    if vix_is_proxy:
        notes.append("VIX is HV proxy (no real data)")
    if vix_tier == "LIQUIDATION":
        notes.append(f"VIX LIQUIDATION ({vix_level:.0f}) — all new trades halted")
    elif vix_tier == "DEFENSIVE":
        notes.append(f"VIX DEFENSIVE ({vix_level:.0f}) — neutral strategies only")
    elif vix_tier == "CAUTION":
        notes.append(f"VIX CAUTION ({vix_level:.0f}) — credit/neutral only")
    if vix_spike:
        notes.append(f"VIX SPIKE: {vix_level} vs {vix_5d_avg} 5d-avg")
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
        vix_5d_avg=vix_5d_avg,
        vix_spike=vix_spike,
        vix_tier=vix_tier,
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
