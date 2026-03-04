"""
src/strategy/selector.py
========================
V2: Added OVERSOLD_BOUNCE and OVERBOUGHT_DROP regimes.
These map to SKIP (no trade) or neutral strategies instead of
directional trades that fight the snap-back.

Also added market_context awareness to reduce directional
exposure when SPY is reversing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.analysis.technical import Regime, StockRegime
from src.analysis.volatility import IVAnalysis

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    BULL_CALL_SPREAD  = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD   = "BEAR_PUT_SPREAD"
    BULL_PUT_SPREAD   = "BULL_PUT_SPREAD"
    BEAR_CALL_SPREAD  = "BEAR_CALL_SPREAD"
    IRON_CONDOR       = "IRON_CONDOR"
    LONG_BUTTERFLY    = "LONG_BUTTERFLY"
    SKIP              = "SKIP"


@dataclass
class StrategyRecommendation:
    ticker: str
    strategy: StrategyType
    direction: str
    regime: str
    iv_regime: str
    confidence: float
    rationale: str
    target_dte: int
    risk_reward: str
    priority: int
    ev_estimate: float = 0.0


_MATRIX: dict[tuple, tuple] = {
    # Strong Uptrend
    (Regime.STRONG_UPTREND, "HIGH"):    (StrategyType.BULL_PUT_SPREAD,   "BULLISH", "CREDIT"),
    (Regime.STRONG_UPTREND, "NORMAL"):  (StrategyType.BULL_CALL_SPREAD,  "BULLISH", "DEBIT"),
    (Regime.STRONG_UPTREND, "LOW"):     (StrategyType.BULL_CALL_SPREAD,  "BULLISH", "DEBIT"),

    # Uptrend
    (Regime.UPTREND, "HIGH"):           (StrategyType.BULL_PUT_SPREAD,   "BULLISH", "CREDIT"),
    (Regime.UPTREND, "NORMAL"):         (StrategyType.BULL_CALL_SPREAD,  "BULLISH", "DEBIT"),
    (Regime.UPTREND, "LOW"):            (StrategyType.BULL_CALL_SPREAD,  "BULLISH", "DEBIT"),

    # Range Bound
    (Regime.RANGE_BOUND, "HIGH"):       (StrategyType.IRON_CONDOR,       "NEUTRAL", "CREDIT"),
    (Regime.RANGE_BOUND, "NORMAL"):     (StrategyType.IRON_CONDOR,       "NEUTRAL", "CREDIT"),
    (Regime.RANGE_BOUND, "LOW"):        (StrategyType.LONG_BUTTERFLY,    "NEUTRAL", "DEBIT"),

    # Downtrend
    (Regime.DOWNTREND, "HIGH"):         (StrategyType.BEAR_CALL_SPREAD,  "BEARISH", "CREDIT"),
    (Regime.DOWNTREND, "NORMAL"):       (StrategyType.BEAR_PUT_SPREAD,   "BEARISH", "DEBIT"),
    (Regime.DOWNTREND, "LOW"):          (StrategyType.BEAR_PUT_SPREAD,   "BEARISH", "DEBIT"),

    # Strong Downtrend
    (Regime.STRONG_DOWNTREND, "HIGH"):  (StrategyType.BEAR_CALL_SPREAD,  "BEARISH", "CREDIT"),
    (Regime.STRONG_DOWNTREND, "NORMAL"):(StrategyType.BEAR_PUT_SPREAD,   "BEARISH", "DEBIT"),
    (Regime.STRONG_DOWNTREND, "LOW"):   (StrategyType.BEAR_PUT_SPREAD,   "BEARISH", "DEBIT"),

    # Squeeze
    (Regime.SQUEEZE, "HIGH"):           (StrategyType.IRON_CONDOR,       "NEUTRAL", "CREDIT"),
    (Regime.SQUEEZE, "NORMAL"):         (StrategyType.LONG_BUTTERFLY,    "NEUTRAL", "DEBIT"),
    (Regime.SQUEEZE, "LOW"):            (StrategyType.LONG_BUTTERFLY,    "NEUTRAL", "DEBIT"),

    # Classic reversals
    (Regime.REVERSAL_UP, "HIGH"):       (StrategyType.BULL_PUT_SPREAD,   "BULLISH", "CREDIT"),
    (Regime.REVERSAL_UP, "NORMAL"):     (StrategyType.BULL_CALL_SPREAD,  "BULLISH", "DEBIT"),
    (Regime.REVERSAL_UP, "LOW"):        (StrategyType.BULL_CALL_SPREAD,  "BULLISH", "DEBIT"),
    (Regime.REVERSAL_DOWN, "HIGH"):     (StrategyType.BEAR_CALL_SPREAD,  "BEARISH", "CREDIT"),
    (Regime.REVERSAL_DOWN, "NORMAL"):   (StrategyType.BEAR_PUT_SPREAD,   "BEARISH", "DEBIT"),
    (Regime.REVERSAL_DOWN, "LOW"):      (StrategyType.BEAR_PUT_SPREAD,   "BEARISH", "DEBIT"),

    # === NEW: Snap-back regimes → neutral or skip ===
    # Oversold bounce: stock dropped hard but is bouncing. Don't short it.
    # If IV is high, sell an iron condor (neutral). Otherwise skip.
    (Regime.OVERSOLD_BOUNCE, "HIGH"):   (StrategyType.IRON_CONDOR,       "NEUTRAL", "CREDIT"),
    (Regime.OVERSOLD_BOUNCE, "NORMAL"): (StrategyType.SKIP,              "NEUTRAL", "NONE"),
    (Regime.OVERSOLD_BOUNCE, "LOW"):    (StrategyType.SKIP,              "NEUTRAL", "NONE"),

    # Overbought drop: stock rallied hard but is fading. Don't go long.
    (Regime.OVERBOUGHT_DROP, "HIGH"):   (StrategyType.IRON_CONDOR,       "NEUTRAL", "CREDIT"),
    (Regime.OVERBOUGHT_DROP, "NORMAL"): (StrategyType.SKIP,              "NEUTRAL", "NONE"),
    (Regime.OVERBOUGHT_DROP, "LOW"):    (StrategyType.SKIP,              "NEUTRAL", "NONE"),
}

_DTE_MAP = {"CREDIT": 45, "DEBIT": 21}
_DTE_BUTTERFLY = 30


def select_strategy(
    regime: StockRegime,
    iv: IVAnalysis,
    spy_return_5d: float = 0.0,     # NEW: pass market context
) -> StrategyRecommendation:
    key = (regime.regime, iv.iv_regime)
    result = _MATRIX.get(key)

    if result is None:
        return StrategyRecommendation(
            ticker=regime.ticker, strategy=StrategyType.SKIP,
            direction="NEUTRAL", regime=regime.regime.value,
            iv_regime=iv.iv_regime, confidence=0.0,
            rationale="No strategy match", target_dte=0,
            risk_reward="NONE", priority=99,
        )

    strategy_type, direction, risk_reward = result

    # === NEW: Market context override ===
    # If SPY bounced >1% in 5 days and we're about to enter a bear spread,
    # downgrade to SKIP — don't fight the broad tape
    if direction == "BEARISH" and spy_return_5d > 1.0:
        logger.info(f"{regime.ticker}: SKIP — bearish signal but SPY bounced {spy_return_5d:.1f}%")
        return StrategyRecommendation(
            ticker=regime.ticker, strategy=StrategyType.SKIP,
            direction="NEUTRAL", regime=regime.regime.value,
            iv_regime=iv.iv_regime, confidence=0.0,
            rationale=f"Skipped: SPY +{spy_return_5d:.1f}% vs bearish signal",
            target_dte=0, risk_reward="NONE", priority=99,
        )

    # If SPY dropped >1% in 5 days and we're about to enter a bull spread,
    # downgrade to SKIP
    if direction == "BULLISH" and spy_return_5d < -1.0:
        logger.info(f"{regime.ticker}: SKIP — bullish signal but SPY dropped {spy_return_5d:.1f}%")
        return StrategyRecommendation(
            ticker=regime.ticker, strategy=StrategyType.SKIP,
            direction="NEUTRAL", regime=regime.regime.value,
            iv_regime=iv.iv_regime, confidence=0.0,
            rationale=f"Skipped: SPY {spy_return_5d:.1f}% vs bullish signal",
            target_dte=0, risk_reward="NONE", priority=99,
        )

    if strategy_type == StrategyType.LONG_BUTTERFLY:
        target_dte = _DTE_BUTTERFLY
    else:
        target_dte = _DTE_MAP.get(risk_reward, 30)

    confidence = _compute_confidence(regime, iv, strategy_type)
    rationale = _build_rationale(regime, iv, strategy_type)

    return StrategyRecommendation(
        ticker=regime.ticker, strategy=strategy_type,
        direction=direction, regime=regime.regime.value,
        iv_regime=iv.iv_regime, confidence=round(confidence, 3),
        rationale=rationale, target_dte=target_dte,
        risk_reward=risk_reward, priority=0,
    )


_CREDIT_STRATEGIES = {
    StrategyType.IRON_CONDOR,
    StrategyType.BULL_PUT_SPREAD,
    StrategyType.BEAR_CALL_SPREAD,
}
_DEBIT_STRATEGIES = {
    StrategyType.BULL_CALL_SPREAD,
    StrategyType.BEAR_PUT_SPREAD,
    StrategyType.LONG_BUTTERFLY,
}


def _compute_confidence(regime, iv, strategy):
    score = 0.50

    if regime.trend_strength > 0.60:
        score += 0.15
    elif regime.trend_strength > 0.35:
        score += 0.08

    if iv.iv_rank >= 80 and strategy in _CREDIT_STRATEGIES:
        score += 0.15
    elif iv.iv_rank >= 60 and strategy in _CREDIT_STRATEGIES:
        score += 0.08
    elif iv.iv_rank <= 20 and strategy in _DEBIT_STRATEGIES:
        score += 0.10
    elif iv.iv_rank <= 40 and strategy in _DEBIT_STRATEGIES:
        score += 0.05

    if iv.iv_hv_ratio > 1.4 and strategy in _CREDIT_STRATEGIES:
        score += 0.10
    elif iv.iv_hv_ratio < 0.9 and strategy in _DEBIT_STRATEGIES:
        score += 0.08

    if iv.iv_trend == "FALLING" and strategy in _CREDIT_STRATEGIES:
        score += 0.05
    elif iv.iv_trend == "RISING" and strategy in _DEBIT_STRATEGIES:
        score -= 0.05

    if regime.regime == Regime.REVERSAL_UP and regime.rsi < 25:
        score += 0.10
    if regime.regime == Regime.REVERSAL_DOWN and regime.rsi > 75:
        score += 0.10

    if regime.bb_squeeze and strategy == StrategyType.LONG_BUTTERFLY:
        score += 0.10

    if regime.volume_trend == "rising" and strategy in (
        StrategyType.BULL_CALL_SPREAD, StrategyType.BULL_PUT_SPREAD
    ) and regime.direction_score > 0:
        score += 0.05
    elif regime.volume_trend == "rising" and strategy in (
        StrategyType.BEAR_PUT_SPREAD, StrategyType.BEAR_CALL_SPREAD
    ) and regime.direction_score < 0:
        score += 0.05

    # NEW: Penalise confidence if short-term momentum opposes direction
    if hasattr(regime, 'roc_3d'):
        if strategy in (StrategyType.BEAR_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD) and regime.roc_3d > 1.5:
            score -= 0.15  # Stock bouncing, bearish confidence drops
        elif strategy in (StrategyType.BULL_CALL_SPREAD, StrategyType.BULL_PUT_SPREAD) and regime.roc_3d < -1.5:
            score -= 0.15  # Stock dropping, bullish confidence drops

    return min(max(score, 0.0), 1.0)


def _build_rationale(regime, iv, strategy):
    parts = [
        f"{regime.regime.value} (ADX={regime.adx}, RSI={regime.rsi})",
        f"IV rank {iv.iv_rank}% [{iv.iv_regime}] | IV/HV={iv.iv_hv_ratio}",
    ]
    if regime.bb_squeeze:
        parts.append("BB squeeze")
    if iv.iv_trend != "FLAT":
        parts.append(f"IV {iv.iv_trend.lower()}")
    if regime.volume_trend == "rising":
        parts.append("vol rising")
    if hasattr(regime, 'roc_3d') and abs(regime.roc_3d) > 1.0:
        parts.append(f"3d ROC={regime.roc_3d:+.1f}%")
    if hasattr(regime, 'atr_move_5d') and abs(regime.atr_move_5d) > 1.5:
        parts.append(f"5d ATR move={regime.atr_move_5d:+.1f}")
    return " | ".join(parts)
