"""
src/analysis/technical.py
=========================
Classify each stock into a trading regime using technical indicators.

Regimes drive strategy selection:
  STRONG_UPTREND / UPTREND      → Bull call spread or Bull put spread (credit)
  RANGE_BOUND                   → Iron condor or Long butterfly
  DOWNTREND / STRONG_DOWNTREND  → Bear put spread or Bear call spread (credit)
  SQUEEZE                       → Long butterfly (breakout anticipated)
  REVERSAL_UP / REVERSAL_DOWN   → Directional spreads on oversold/overbought
  OVERSOLD_BOUNCE               → NEW: down >2 ATR in 5d + bouncing → no directional short
  OVERBOUGHT_DROP               → NEW: up >2 ATR in 5d + fading → no directional long

V2 changes:
  - Added short-term momentum (3d ROC) to direction_score
  - Added OVERSOLD_BOUNCE and OVERBOUGHT_DROP regimes to prevent fighting snap-backs
  - Mean-reversion guard: if stock dropped >2 ATR in 5 days and RSI<35 and
    3d ROC positive, classify as OVERSOLD_BOUNCE instead of DOWNTREND/STRONG_DOWNTREND
  - If stock rallied >2 ATR in 5 days and RSI>65 and 3d ROC negative,
    classify as OVERBOUGHT_DROP instead of UPTREND/STRONG_UPTREND
  - Snap-back guard inside trending regimes: even if ADX is trending, a
    positive 3d ROC >1% inside a bearish setup (or vice versa) gets
    reclassified to a snap-back regime so selector skips the directional trade

Usage:
    from src.analysis.technical import classify_regime, classify_universe

    regime = classify_regime("AAPL", df)
    regimes = classify_universe(data_dict)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from config.settings import (
    EMA_FAST, EMA_MEDIUM, EMA_SLOW,
    ADX_PERIOD, ADX_TRENDING_THRESHOLD,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    BB_PERIOD, BB_STD, ATR_PERIOD,
    SNAPBACK_ATR_THRESHOLD, SNAPBACK_ROC_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─── Regime Enum ──────────────────────────────────────────────────────────────

class Regime(str, Enum):
    STRONG_UPTREND   = "STRONG_UPTREND"
    UPTREND          = "UPTREND"
    RANGE_BOUND      = "RANGE_BOUND"
    DOWNTREND        = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    SQUEEZE          = "SQUEEZE"           # BB squeeze → breakout imminent
    REVERSAL_UP      = "REVERSAL_UP"       # Oversold bounce candidate
    REVERSAL_DOWN    = "REVERSAL_DOWN"     # Overbought reversal candidate
    OVERSOLD_BOUNCE  = "OVERSOLD_BOUNCE"   # NEW: dropped >2 ATR but bouncing — don't short
    OVERBOUGHT_DROP  = "OVERBOUGHT_DROP"   # NEW: rallied >2 ATR but fading — don't buy


# ─── StockRegime Dataclass ────────────────────────────────────────────────────

@dataclass
class StockRegime:
    ticker: str
    regime: Regime
    direction_score: float     # -1.0 (strong bear) … +1.0 (strong bull)
    trend_strength: float      # 0.0 (no trend) … 1.0 (very strong trend)
    volatility_state: str      # "expanding" | "contracting" | "normal"
    rsi: float                 # 14-period RSI
    adx: float                 # 14-period ADX
    bb_squeeze: bool           # True if Bollinger bandwidth in bottom 15%ile
    ema_alignment: str         # "bullish" | "bearish" | "mixed"
    support: float             # 20-day low (soft support level)
    resistance: float          # 20-day high (soft resistance level)
    atr: float                 # 14-day ATR (dollar)
    atr_pct: float             # ATR as % of price
    price: float               # Latest close price
    volume_trend: str          # "rising" | "falling" | "neutral" (10d vs 30d avg)
    roc_3d: float = 0.0        # NEW: 3-day rate of change (%)
    atr_move_5d: float = 0.0   # NEW: 5-day price move expressed in ATR units


# ─── Main Classification Function ────────────────────────────────────────────

def classify_regime(ticker: str, df: pd.DataFrame) -> Optional[StockRegime]:
    """
    Classify a stock's current regime from its OHLCV DataFrame.

    Requires at least 60 rows of daily data.
    Returns None if data is insufficient or an error occurs.
    """
    if df is None or len(df) < 60:
        logger.debug(f"{ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
        return None

    try:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # ── EMAs ─────────────────────────────────────────────────────────────
        ema_fast = ta.ema(close, length=EMA_FAST)
        ema_med  = ta.ema(close, length=EMA_MEDIUM)
        ema_slow = ta.ema(close, length=EMA_SLOW)

        ema_f = float(ema_fast.iloc[-1])
        ema_m = float(ema_med.iloc[-1])
        ema_s = float(ema_slow.iloc[-1])
        price = float(close.iloc[-1])

        if any(np.isnan([ema_f, ema_m, ema_s, price])):
            return None

        # ── ADX ───────────────────────────────────────────────────────────────
        adx_df   = ta.adx(high, low, close, length=ADX_PERIOD)
        adx_val  = float(adx_df.iloc[-1][f"ADX_{ADX_PERIOD}"])
        plus_di  = float(adx_df.iloc[-1][f"DMP_{ADX_PERIOD}"])
        minus_di = float(adx_df.iloc[-1][f"DMN_{ADX_PERIOD}"])

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi_series = ta.rsi(close, length=RSI_PERIOD)
        rsi = float(rsi_series.iloc[-1])

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb = ta.bbands(close, length=BB_PERIOD, std=BB_STD)
        # Column names vary by pandas_ta version; find them dynamically
        bb_upper_col = next((c for c in bb.columns if c.startswith("BBU_")), None)
        bb_lower_col = next((c for c in bb.columns if c.startswith("BBL_")), None)
        bb_mid_col   = next((c for c in bb.columns if c.startswith("BBM_")), None)
        bb_bw_col    = next((c for c in bb.columns if c.startswith("BBB_")), None)

        if bb_upper_col is None or bb_lower_col is None or bb_mid_col is None:
            raise ValueError(f"Unexpected BB column names: {bb.columns.tolist()}")

        bb_squeeze = False
        if bb_bw_col and bb_bw_col in bb.columns:
            bw_pctile = float(bb[bb_bw_col].rank(pct=True).iloc[-1])
            bb_squeeze = bw_pctile < 0.15

        # ── ATR ───────────────────────────────────────────────────────────────
        atr_series = ta.atr(high, low, close, length=ATR_PERIOD)
        atr = float(atr_series.iloc[-1])
        atr_pct = round(atr / price * 100, 2) if price > 0 else 0.0

        atr_ma = float(atr_series.rolling(10).mean().iloc[-1])
        if atr > atr_ma * 1.10:
            vol_state = "expanding"
        elif atr < atr_ma * 0.90:
            vol_state = "contracting"
        else:
            vol_state = "normal"

        # ── Support / Resistance ──────────────────────────────────────────────
        support    = float(low.tail(20).min())
        resistance = float(high.tail(20).max())

        # ── EMA Alignment ─────────────────────────────────────────────────────
        if ema_f > ema_m > ema_s:
            ema_align = "bullish"
        elif ema_f < ema_m < ema_s:
            ema_align = "bearish"
        else:
            ema_align = "mixed"

        # ── Direction Score ───────────────────────────────────────────────────
        direction_score = 0.0

        if ema_align == "bullish":
            direction_score += 0.30
        elif ema_align == "bearish":
            direction_score -= 0.30

        direction_score += 0.20 if price > ema_s else -0.20
        direction_score += 0.20 if plus_di > minus_di else -0.20
        direction_score += 0.15 if rsi > 50 else -0.15

        # Short-term momentum (5d return)
        ret_5d = (price / float(close.iloc[-5]) - 1) if len(close) >= 5 else 0
        direction_score += min(max(ret_5d * 5, -0.15), 0.15)

        direction_score = round(max(-1.0, min(1.0, direction_score)), 3)

        # ── NEW: Short-term momentum indicators ───────────────────────────────
        # 3-day rate of change (%) — detects snap-backs and fades
        roc_3d = round((price / float(close.iloc[-3]) - 1) * 100, 2) if len(close) >= 3 else 0.0

        # 5-day move expressed in ATR units — detects extreme short-term moves
        if len(close) >= 5 and atr > 0:
            move_5d = price - float(close.iloc[-5])
            atr_move_5d = round(move_5d / atr, 2)
        else:
            atr_move_5d = 0.0

        # ── Trend Strength ────────────────────────────────────────────────────
        trend_strength = round(min(adx_val / 50, 1.0), 3)

        # ── Volume Trend ─────────────────────────────────────────────────────
        vol_10d = float(volume.tail(10).mean())
        vol_30d = float(volume.tail(30).mean())
        if vol_10d > vol_30d * 1.15:
            volume_trend = "rising"
        elif vol_10d < vol_30d * 0.85:
            volume_trend = "falling"
        else:
            volume_trend = "neutral"

        # ── Regime Classification ─────────────────────────────────────────────
        #
        # Priority order (highest to lowest):
        #   P1. Mean-reversion detection (OVERSOLD_BOUNCE / OVERBOUGHT_DROP)
        #       — stock moved >SNAPBACK_ATR_THRESHOLD ATRs in 5d with
        #         confirming RSI extreme AND short-term momentum reversal
        #   P2. Bollinger Squeeze
        #   P3. Classic RSI extremes (REVERSAL_UP / REVERSAL_DOWN)
        #   P4. ADX-trend regimes — with snap-back guard inside trending block
        #   P5. Range-bound default

        # P1 — Mean-reversion (snap-back guard)
        # Dropped >threshold ATRs AND RSI oversold AND bouncing on 3d ROC
        if atr_move_5d < -SNAPBACK_ATR_THRESHOLD and rsi < 35 and roc_3d > SNAPBACK_ROC_THRESHOLD:
            regime = Regime.OVERSOLD_BOUNCE

        # Rallied >threshold ATRs AND RSI overbought AND fading on 3d ROC
        elif atr_move_5d > SNAPBACK_ATR_THRESHOLD and rsi > 65 and roc_3d < -SNAPBACK_ROC_THRESHOLD:
            regime = Regime.OVERBOUGHT_DROP

        # P2 — Bollinger Squeeze
        elif bb_squeeze:
            regime = Regime.SQUEEZE

        # P3 — Classic reversal
        elif rsi <= RSI_OVERSOLD and direction_score <= -0.2:
            regime = Regime.REVERSAL_UP
        elif rsi >= RSI_OVERBOUGHT and direction_score >= 0.2:
            regime = Regime.REVERSAL_DOWN

        # P4 — ADX-trending regimes
        elif adx_val >= ADX_TRENDING_THRESHOLD:
            # Snap-back guard inside trending block:
            # If trending bearish but 3d ROC is positive >1% → don't short, reclassify
            if direction_score <= -0.4 and roc_3d > 1.0:
                regime = Regime.OVERSOLD_BOUNCE
            # If trending bullish but 3d ROC is negative <-1% → don't buy, reclassify
            elif direction_score >= 0.4 and roc_3d < -1.0:
                regime = Regime.OVERBOUGHT_DROP
            elif direction_score >= 0.4:
                regime = Regime.STRONG_UPTREND
            elif direction_score >= 0.1:
                regime = Regime.UPTREND
            elif direction_score <= -0.4:
                regime = Regime.STRONG_DOWNTREND
            elif direction_score <= -0.1:
                regime = Regime.DOWNTREND
            else:
                regime = Regime.RANGE_BOUND

        # P5 — Default: range-bound
        else:
            regime = Regime.RANGE_BOUND

        return StockRegime(
            ticker=ticker,
            regime=regime,
            direction_score=direction_score,
            trend_strength=trend_strength,
            volatility_state=vol_state,
            rsi=round(rsi, 1),
            adx=round(adx_val, 1),
            bb_squeeze=bb_squeeze,
            ema_alignment=ema_align,
            support=round(support, 2),
            resistance=round(resistance, 2),
            atr=round(atr, 2),
            atr_pct=atr_pct,
            price=round(price, 2),
            volume_trend=volume_trend,
            roc_3d=roc_3d,
            atr_move_5d=atr_move_5d,
        )

    except Exception as exc:
        logger.error(f"Regime classification failed for {ticker}: {exc}")
        return None


def classify_universe(
    data: dict[str, pd.DataFrame],
) -> list[StockRegime]:
    """Classify all stocks in the universe. Skips ETFs/indexes if desired."""
    results = []
    for ticker, df in data.items():
        r = classify_regime(ticker, df)
        if r is not None:
            results.append(r)

    logger.info(f"Classified {len(results)}/{len(data)} stocks")
    return results


def get_regime_summary(regimes: list[StockRegime]) -> dict[str, int]:
    """Return count of each regime in the list."""
    from collections import Counter
    return dict(Counter(r.regime.value for r in regimes))
