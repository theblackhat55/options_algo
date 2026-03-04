"""
src/analysis/technical.py
=========================
Classify each stock into a trading regime using technical indicators.

V2 changes:
  - Added short-term momentum (3d ROC) to direction_score
  - Added OVERSOLD_BOUNCE and OVERBOUGHT_DROP regimes
  - Mean-reversion guard: if stock dropped >2 ATR in 5 days and RSI<35,
    classify as OVERSOLD_BOUNCE instead of DOWNTREND
  - If stock rallied >2 ATR in 5 days and RSI>65,
    classify as OVERBOUGHT_DROP instead of UPTREND
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from config.settings import (
    EMA_FAST, EMA_MEDIUM, EMA_SLOW,
    ADX_PERIOD, ADX_TRENDING_THRESHOLD,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    BB_PERIOD, BB_STD, ATR_PERIOD,
)

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    STRONG_UPTREND   = "STRONG_UPTREND"
    UPTREND          = "UPTREND"
    RANGE_BOUND      = "RANGE_BOUND"
    DOWNTREND        = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    SQUEEZE          = "SQUEEZE"
    REVERSAL_UP      = "REVERSAL_UP"
    REVERSAL_DOWN    = "REVERSAL_DOWN"
    OVERSOLD_BOUNCE  = "OVERSOLD_BOUNCE"    # NEW: down big but bouncing
    OVERBOUGHT_DROP  = "OVERBOUGHT_DROP"    # NEW: up big but fading


@dataclass
class StockRegime:
    ticker: str
    regime: Regime
    direction_score: float
    trend_strength: float
    volatility_state: str
    rsi: float
    adx: float
    bb_squeeze: bool
    ema_alignment: str
    support: float
    resistance: float
    atr: float
    atr_pct: float
    price: float
    volume_trend: str
    roc_3d: float = 0.0          # NEW: 3-day rate of change (%)
    atr_move_5d: float = 0.0     # NEW: 5-day move in ATR units


def classify_regime(ticker: str, df: pd.DataFrame) -> Optional[StockRegime]:
    if df is None or len(df) < 60:
        logger.debug(f"{ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
        return None

    try:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)

        ema_fast = ta.ema(close, length=EMA_FAST)
        ema_med  = ta.ema(close, length=EMA_MEDIUM)
        ema_slow = ta.ema(close, length=EMA_SLOW)

        ema_f = float(ema_fast.iloc[-1])
        ema_m = float(ema_med.iloc[-1])
        ema_s = float(ema_slow.iloc[-1])
        price = float(close.iloc[-1])

        if any(np.isnan([ema_f, ema_m, ema_s, price])):
            return None

        adx_df   = ta.adx(high, low, close, length=ADX_PERIOD)
        adx_val  = float(adx_df.iloc[-1][f"ADX_{ADX_PERIOD}"])
        plus_di  = float(adx_df.iloc[-1][f"DMP_{ADX_PERIOD}"])
        minus_di = float(adx_df.iloc[-1][f"DMN_{ADX_PERIOD}"])

        rsi_series = ta.rsi(close, length=RSI_PERIOD)
        rsi = float(rsi_series.iloc[-1])

        bb = ta.bbands(close, length=BB_PERIOD, std=BB_STD)
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

        support    = float(low.tail(20).min())
        resistance = float(high.tail(20).max())

        if ema_f > ema_m > ema_s:
            ema_align = "bullish"
        elif ema_f < ema_m < ema_s:
            ema_align = "bearish"
        else:
            ema_align = "mixed"

        # === Direction Score (enhanced) ===
        direction_score = 0.0
        if ema_align == "bullish":
            direction_score += 0.30
        elif ema_align == "bearish":
            direction_score -= 0.30

        direction_score += 0.20 if price > ema_s else -0.20
        direction_score += 0.20 if plus_di > minus_di else -0.20
        direction_score += 0.15 if rsi > 50 else -0.15

        ret_5d = (price / float(close.iloc[-5]) - 1) if len(close) >= 5 else 0
        direction_score += min(max(ret_5d * 5, -0.15), 0.15)

        direction_score = round(max(-1.0, min(1.0, direction_score)), 3)

        # === NEW: Short-term momentum ===
        roc_3d = round((price / float(close.iloc[-3]) - 1) * 100, 2) if len(close) >= 3 else 0.0

        # 5-day move in ATR units
        if len(close) >= 5 and atr > 0:
            move_5d = price - float(close.iloc[-5])
            atr_move_5d = round(move_5d / atr, 2)
        else:
            atr_move_5d = 0.0

        trend_strength = round(min(adx_val / 50, 1.0), 3)

        vol_10d = float(volume.tail(10).mean())
        vol_30d = float(volume.tail(30).mean())
        if vol_10d > vol_30d * 1.15:
            volume_trend = "rising"
        elif vol_10d < vol_30d * 0.85:
            volume_trend = "falling"
        else:
            volume_trend = "neutral"

        # === Regime Classification (enhanced) ===

        # PRIORITY 1: Mean-reversion detection
        # Stock dropped >2 ATR in 5 days but short-term bouncing → don't short
        if atr_move_5d < -2.0 and rsi < 35 and roc_3d > 0.5:
            regime = Regime.OVERSOLD_BOUNCE
        elif atr_move_5d > 2.0 and rsi > 65 and roc_3d < -0.5:
            regime = Regime.OVERBOUGHT_DROP

        # PRIORITY 2: Bollinger squeeze
        elif bb_squeeze:
            regime = Regime.SQUEEZE

        # PRIORITY 3: Classic reversal detection
        elif rsi <= RSI_OVERSOLD and direction_score <= -0.2:
            regime = Regime.REVERSAL_UP
        elif rsi >= RSI_OVERBOUGHT and direction_score >= 0.2:
            regime = Regime.REVERSAL_DOWN

        # PRIORITY 4: Trend regimes (with snap-back guard)
        elif adx_val >= ADX_TRENDING_THRESHOLD:
            # NEW: If trending down but 3d ROC is positive and >1%, downgrade
            if direction_score <= -0.4 and roc_3d > 1.0:
                regime = Regime.OVERSOLD_BOUNCE
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


def classify_universe(data: dict[str, pd.DataFrame]) -> list[StockRegime]:
    results = []
    for ticker, df in data.items():
        r = classify_regime(ticker, df)
        if r is not None:
            results.append(r)
    logger.info(f"Classified {len(results)}/{len(data)} stocks")
    return results


def get_regime_summary(regimes: list[StockRegime]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(r.regime.value for r in regimes))
