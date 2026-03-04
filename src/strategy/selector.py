"""
src/strategy/selector.py
========================
Maps stock regime + IV analysis → recommended options strategy.

Strategy Matrix V2:
────────────────────────────────────────────────────────────────────────────────
                        IV HIGH (Sell)    IV NORMAL         IV LOW (Buy)
────────────────────────────────────────────────────────────────────────────────
STRONG UPTREND          Bull Put Spread   Bull Call Spread  Bull Call Spread
UPTREND                 Bull Put Spread   Bull Call Spread  Bull Call Spread
RANGE BOUND             Iron Condor       Iron Condor       Long Butterfly
DOWNTREND               Bear Call Spread  Bear Put Spread   Bear Put Spread
STRONG DOWNTREND        Bear Call Spread  Bear Put Spread   Bear Put Spread
SQUEEZE                 Iron Condor       Long Butterfly    Long Butterfly
REVERSAL UP             Bull Put Spread   Bull Call Spread  Bull Call Spread
REVERSAL DOWN           Bear Call Spread  Bear Put Spread   Bear Put Spread
OVERSOLD_BOUNCE (NEW)   Iron Condor       SKIP              SKIP
OVERBOUGHT_DROP (NEW)   Iron Condor       SKIP              SKIP
────────────────────────────────────────────────────────────────────────────────

V2 changes:
  - Added OVERSOLD_BOUNCE and OVERBOUGHT_DROP regime rows to the matrix.
    These map to IRON_CONDOR when IV is HIGH (neutral premium sale),
    or SKIP otherwise — preventing directional trades against snap-backs.
  - Added spy_return_5d market-context gate to select_strategy():
    * If SPY bounced >SPY_DIRECTIONAL_GATE_PCT% in 5d AND the signal is
      bearish → SKIP (don't fight a rising tape with bear spreads)
    * If SPY dropped >SPY_DIRECTIONAL_GATE_PCT% in 5d AND the signal is
      bullish → SKIP (don't fade a falling market with bull spreads)
  - _compute_confidence() now penalises confidence when 3d ROC opposes
    the strategy direction (short-term momentum counter-signal).
  - _build_rationale() now includes roc_3d and atr_move_5d when material.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.analysis.technical import Regime, StockRegime
from src.analysis.volatility import IVAnalysis
from config.settings import SPY_DIRECTIONAL_GATE_PCT

logger = logging.getLogger(__name__)


# ─── Strategy Enum ────────────────────────────────────────────────────────────

class StrategyType(str, Enum):
    BULL_CALL_SPREAD  = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD   = "BEAR_PUT_SPREAD"
    BULL_PUT_SPREAD   = "BULL_PUT_SPREAD"    # Credit spread — bullish
    BEAR_CALL_SPREAD  = "BEAR_CALL_SPREAD"   # Credit spread — bearish
    IRON_CONDOR       = "IRON_CONDOR"
    LONG_BUTTERFLY    = "LONG_BUTTERFLY"
    SKIP              = "SKIP"


# ─── StrategyRecommendation Dataclass ────────────────────────────────────────

@dataclass
class StrategyRecommendation:
    ticker: str
    strategy: StrategyType
    direction: str              # "BULLISH" | "BEARISH" | "NEUTRAL"
    regime: str                 # Regime.value
    iv_regime: str              # "HIGH" | "NORMAL" | "LOW"
    confidence: float           # 0.0–1.0
    rationale: str
    target_dte: int
    risk_reward: str            # "CREDIT" | "DEBIT"
    priority: int               # 1 = highest (set by composite ranker)
    ev_estimate: float = 0.0    # Expected value estimate


# ─── Strategy Matrix ──────────────────────────────────────────────────────────
# Key: (Regime, iv_regime_str)  →  (StrategyType, direction, risk_reward)

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

    # Bollinger Squeeze
    (Regime.SQUEEZE, "HIGH"):           (StrategyType.IRON_CONDOR,       "NEUTRAL", "CREDIT"),
    (Regime.SQUEEZE, "NORMAL"):         (StrategyType.LONG_BUTTERFLY,    "NEUTRAL", "DEBIT"),
    (Regime.SQUEEZE, "LOW"):            (StrategyType.LONG_BUTTERFLY,    "NEUTRAL", "DEBIT"),

    # Reversal candidates
    (Regime.REVERSAL_UP, "HIGH"):       (StrategyType.BULL_PUT_SPREAD,   "BULLISH", "CREDIT"),
    (Regime.REVERSAL_UP, "NORMAL"):     (StrategyType.BULL_CALL_SPREAD,  "BULLISH", "DEBIT"),
    (Regime.REVERSAL_UP, "LOW"):        (StrategyType.BULL_CALL_SPREAD,  "BULLISH", "DEBIT"),

    (Regime.REVERSAL_DOWN, "HIGH"):     (StrategyType.BEAR_CALL_SPREAD,  "BEARISH", "CREDIT"),
    (Regime.REVERSAL_DOWN, "NORMAL"):   (StrategyType.BEAR_PUT_SPREAD,   "BEARISH", "DEBIT"),
    (Regime.REVERSAL_DOWN, "LOW"):      (StrategyType.BEAR_PUT_SPREAD,   "BEARISH", "DEBIT"),

    # ── NEW: Snap-back regimes ──────────────────────────────────────────────
    # OVERSOLD_BOUNCE: stock dropped hard but is bouncing — don't go short.
    #   HIGH IV → Iron Condor (neutral premium sale, wide strikes capture bounce range)
    #   NORMAL / LOW → SKIP (no edge without premium richness)
    (Regime.OVERSOLD_BOUNCE, "HIGH"):   (StrategyType.IRON_CONDOR,       "NEUTRAL", "CREDIT"),
    (Regime.OVERSOLD_BOUNCE, "NORMAL"): (StrategyType.SKIP,              "NEUTRAL", "NONE"),
    (Regime.OVERSOLD_BOUNCE, "LOW"):    (StrategyType.SKIP,              "NEUTRAL", "NONE"),

    # OVERBOUGHT_DROP: stock rallied hard but is fading — don't go long.
    #   HIGH IV → Iron Condor (neutral premium sale)
    #   NORMAL / LOW → SKIP
    (Regime.OVERBOUGHT_DROP, "HIGH"):   (StrategyType.IRON_CONDOR,       "NEUTRAL", "CREDIT"),
    (Regime.OVERBOUGHT_DROP, "NORMAL"): (StrategyType.SKIP,              "NEUTRAL", "NONE"),
    (Regime.OVERBOUGHT_DROP, "LOW"):    (StrategyType.SKIP,              "NEUTRAL", "NONE"),
}

# Target DTE by risk_reward type
_DTE_MAP = {
    "CREDIT": 45,   # Premium selling: optimal theta decay window
    "DEBIT":  21,   # Directional: shorter to keep cost down
}
_DTE_BUTTERFLY = 30


# ─── Main Selection Function ──────────────────────────────────────────────────

def select_strategy(
    regime: StockRegime,
    iv: IVAnalysis,
    spy_return_5d: float = 0.0,   # NEW: current 5-day SPY return (%)
) -> StrategyRecommendation:
    """
    Select the optimal options strategy for a stock given its
    technical regime and IV analysis.

    Args:
        regime: StockRegime from classify_regime()
        iv: IVAnalysis from analyze_iv()
        spy_return_5d: 5-day SPY return (%). Used to block directional trades
            when the broad market is moving against the signal direction.

    Returns:
        StrategyRecommendation (may be SKIP)
    """
    key = (regime.regime, iv.iv_regime)
    result = _MATRIX.get(key)

    if result is None:
        return StrategyRecommendation(
            ticker=regime.ticker,
            strategy=StrategyType.SKIP,
            direction="NEUTRAL",
            regime=regime.regime.value,
            iv_regime=iv.iv_regime,
            confidence=0.0,
            rationale="No strategy match",
            target_dte=0,
            risk_reward="NONE",
            priority=99,
        )

    strategy_type, direction, risk_reward = result

    # ── NEW: SPY market-context gate ─────────────────────────────────────────
    # If SPY bounced strongly in the last 5 days, bearish individual-stock
    # signals are fighting an upward tape — skip them.
    if direction == "BEARISH" and spy_return_5d > SPY_DIRECTIONAL_GATE_PCT:
        logger.info(
            f"{regime.ticker}: SKIP — bearish signal but SPY +{spy_return_5d:.1f}% "
            f"(gate={SPY_DIRECTIONAL_GATE_PCT}%)"
        )
        return StrategyRecommendation(
            ticker=regime.ticker,
            strategy=StrategyType.SKIP,
            direction="NEUTRAL",
            regime=regime.regime.value,
            iv_regime=iv.iv_regime,
            confidence=0.0,
            rationale=f"Skipped: bearish signal vs SPY +{spy_return_5d:.1f}% (5d tape gate)",
            target_dte=0,
            risk_reward="NONE",
            priority=99,
        )

    # If SPY dropped strongly, bullish signals are fighting a falling tape — skip.
    if direction == "BULLISH" and spy_return_5d < -SPY_DIRECTIONAL_GATE_PCT:
        logger.info(
            f"{regime.ticker}: SKIP — bullish signal but SPY {spy_return_5d:.1f}% "
            f"(gate={SPY_DIRECTIONAL_GATE_PCT}%)"
        )
        return StrategyRecommendation(
            ticker=regime.ticker,
            strategy=StrategyType.SKIP,
            direction="NEUTRAL",
            regime=regime.regime.value,
            iv_regime=iv.iv_regime,
            confidence=0.0,
            rationale=f"Skipped: bullish signal vs SPY {spy_return_5d:.1f}% (5d tape gate)",
            target_dte=0,
            risk_reward="NONE",
            priority=99,
        )

    # ── Matrix-level SKIP (e.g. OVERSOLD_BOUNCE/NORMAL) ────────────────────
    # Return immediately with zero confidence — no scoring needed.
    if strategy_type == StrategyType.SKIP:
        return StrategyRecommendation(
            ticker=regime.ticker,
            strategy=StrategyType.SKIP,
            direction="NEUTRAL",
            regime=regime.regime.value,
            iv_regime=iv.iv_regime,
            confidence=0.0,
            rationale=f"Skipped: {regime.regime.value} + IV={iv.iv_regime} has no edge",
            target_dte=0,
            risk_reward="NONE",
            priority=99,
        )

    if strategy_type == StrategyType.LONG_BUTTERFLY:
        target_dte = _DTE_BUTTERFLY
    else:
        target_dte = _DTE_MAP.get(risk_reward, 30)

    confidence = _compute_confidence(regime, iv, strategy_type)
    rationale = _build_rationale(regime, iv, strategy_type)

    return StrategyRecommendation(
        ticker=regime.ticker,
        strategy=strategy_type,
        direction=direction,
        regime=regime.regime.value,
        iv_regime=iv.iv_regime,
        confidence=round(confidence, 3),
        rationale=rationale,
        target_dte=target_dte,
        risk_reward=risk_reward,
        priority=0,   # Set later by composite ranker
    )


# ─── Confidence Scoring ───────────────────────────────────────────────────────

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


def _compute_confidence(
    regime: StockRegime,
    iv: IVAnalysis,
    strategy: StrategyType,
) -> float:
    """
    Score from 0.0–1.0 indicating setup quality.
    Higher = more confluent signals.

    V2 additions:
    - Penalises confidence when 3d ROC opposes strategy direction
    - Rewards confidence when IV-RV spread confirms premium richness for credit trades
    """
    score = 0.50  # Base

    # 1. Trend strength adds conviction to directional plays
    if regime.trend_strength > 0.60:
        score += 0.15
    elif regime.trend_strength > 0.35:
        score += 0.08

    # 2. IV extreme vs strategy type
    if iv.iv_rank >= 80 and strategy in _CREDIT_STRATEGIES:
        score += 0.15
    elif iv.iv_rank >= 60 and strategy in _CREDIT_STRATEGIES:
        score += 0.08
    elif iv.iv_rank <= 20 and strategy in _DEBIT_STRATEGIES:
        score += 0.10
    elif iv.iv_rank <= 40 and strategy in _DEBIT_STRATEGIES:
        score += 0.05

    # 3. IV/HV ratio confirms premium richness
    if iv.iv_hv_ratio > 1.4 and strategy in _CREDIT_STRATEGIES:
        score += 0.10
    elif iv.iv_hv_ratio < 0.9 and strategy in _DEBIT_STRATEGIES:
        score += 0.08

    # 4. IV trend alignment
    if iv.iv_trend == "FALLING" and strategy in _CREDIT_STRATEGIES:
        score += 0.05   # IV falling after selling = gains on vega
    elif iv.iv_trend == "RISING" and strategy in _DEBIT_STRATEGIES:
        score -= 0.05   # Rising IV hurts debit buyers

    # 5. RSI extremes for reversal plays
    if regime.regime == Regime.REVERSAL_UP and regime.rsi < 25:
        score += 0.10
    if regime.regime == Regime.REVERSAL_DOWN and regime.rsi > 75:
        score += 0.10

    # 6. Bollinger squeeze confirms butterfly entry
    if regime.bb_squeeze and strategy == StrategyType.LONG_BUTTERFLY:
        score += 0.10

    # 7. Volume trend confirms direction
    if regime.volume_trend == "rising" and strategy in (
        StrategyType.BULL_CALL_SPREAD, StrategyType.BULL_PUT_SPREAD
    ) and regime.direction_score > 0:
        score += 0.05
    elif regime.volume_trend == "rising" and strategy in (
        StrategyType.BEAR_PUT_SPREAD, StrategyType.BEAR_CALL_SPREAD
    ) and regime.direction_score < 0:
        score += 0.05

    # 8. NEW: 3d ROC counter-signal penalty
    # If the 3-day short-term momentum opposes the strategy direction,
    # reduce confidence — the snap-back risk is real even if longer signals hold.
    roc_3d = getattr(regime, "roc_3d", 0.0)
    if strategy in (StrategyType.BEAR_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD) and roc_3d > 1.5:
        score -= 0.15   # Stock bouncing; bearish confidence reduced
    elif strategy in (StrategyType.BULL_CALL_SPREAD, StrategyType.BULL_PUT_SPREAD) and roc_3d < -1.5:
        score -= 0.15   # Stock dropping; bullish confidence reduced

    # 9. NEW: IV-RV spread bonus for credit strategies
    # Genuine premium richness (IV > HV by a meaningful margin) adds edge
    if strategy in _CREDIT_STRATEGIES and getattr(iv, "premium_rich", False):
        score += 0.05

    return min(max(score, 0.0), 1.0)


def _build_rationale(
    regime: StockRegime,
    iv: IVAnalysis,
    strategy: StrategyType,
) -> str:
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

    # NEW: include snap-back indicators when material
    roc_3d = getattr(regime, "roc_3d", 0.0)
    if abs(roc_3d) > 1.0:
        parts.append(f"3d ROC={roc_3d:+.1f}%")

    atr_move_5d = getattr(regime, "atr_move_5d", 0.0)
    if abs(atr_move_5d) > 1.5:
        parts.append(f"5d ATR move={atr_move_5d:+.1f}")

    # NEW: flag premium richness
    if getattr(iv, "premium_rich", False) and strategy in _CREDIT_STRATEGIES:
        parts.append(f"IV-RV spread={iv.iv_rv_spread:+.1f}pts ✓")

    return " | ".join(parts)
