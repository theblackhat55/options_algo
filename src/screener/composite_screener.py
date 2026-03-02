"""
src/screener/composite_screener.py
===================================
Combine all filters (regime, IV, liquidity, event) into a single ranked output.
This is the unified interface used by the nightly pipeline.

Usage:
    from src.screener.composite_screener import run_screener

    results = run_screener(data, options_chains=None)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.analysis.technical import StockRegime
from src.analysis.volatility import IVAnalysis
from src.analysis.relative_strength import RSAnalysis
from src.strategy.selector import StrategyRecommendation, StrategyType

logger = logging.getLogger(__name__)


@dataclass
class ScreenerResult:
    """Single stock screening result with all metrics."""
    ticker: str
    rank: int
    score: float
    regime: StockRegime
    iv: IVAnalysis
    rs: Optional[RSAnalysis]
    recommendation: StrategyRecommendation
    passes_all_filters: bool
    filter_notes: str


def run_screener(
    regime_map: dict[str, StockRegime],
    iv_map: dict[str, IVAnalysis],
    rs_map: dict[str, RSAnalysis] = None,
    min_confidence: float = 0.55,
    min_iv_rank: float = 0,
    max_iv_rank: float = 100,
    required_direction: str = None,      # "BULLISH" | "BEARISH" | "NEUTRAL" | None
    strategies_allowed: list[str] = None,
) -> list[ScreenerResult]:
    """
    Run the full composite screener and return ranked results.

    Args:
        regime_map: {ticker: StockRegime}
        iv_map: {ticker: IVAnalysis}
        rs_map: {ticker: RSAnalysis} (optional)
        min_confidence: Minimum confidence score to include
        min_iv_rank / max_iv_rank: IV rank filter range
        required_direction: Only show BULLISH/BEARISH/NEUTRAL if specified
        strategies_allowed: Filter to specific strategy types

    Returns:
        List of ScreenerResult sorted by score descending.
    """
    from src.strategy.selector import select_strategy

    results = []

    for ticker in regime_map:
        if ticker not in iv_map:
            continue

        regime = regime_map[ticker]
        iv = iv_map[ticker]
        rs = rs_map.get(ticker) if rs_map else None

        rec = select_strategy(regime, iv)

        if rec.strategy == StrategyType.SKIP:
            continue

        # Apply filters
        passes = True
        notes = []

        if rec.confidence < min_confidence:
            passes = False
            notes.append(f"conf {rec.confidence:.2f} < {min_confidence}")

        if not (min_iv_rank <= iv.iv_rank <= max_iv_rank):
            passes = False
            notes.append(f"IV rank {iv.iv_rank:.0f} outside [{min_iv_rank:.0f}, {max_iv_rank:.0f}]")

        if required_direction and rec.direction != required_direction:
            passes = False
            notes.append(f"direction {rec.direction} ≠ {required_direction}")

        if strategies_allowed and rec.strategy.value not in strategies_allowed:
            passes = False
            notes.append(f"strategy {rec.strategy.value} not in allowed list")

        # Composite score
        score = _compute_screener_score(rec, regime, iv, rs)

        results.append(ScreenerResult(
            ticker=ticker,
            rank=0,
            score=score,
            regime=regime,
            iv=iv,
            rs=rs,
            recommendation=rec,
            passes_all_filters=passes,
            filter_notes=" | ".join(notes),
        ))

    # Sort by score
    results.sort(key=lambda r: (r.passes_all_filters, r.score), reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    logger.info(
        f"Screener: {len(results)} results, "
        f"{sum(1 for r in results if r.passes_all_filters)} pass all filters"
    )
    return results


def _compute_screener_score(
    rec: StrategyRecommendation,
    regime: StockRegime,
    iv: IVAnalysis,
    rs: Optional[RSAnalysis],
) -> float:
    """Normalised score 0-100 for ranking."""
    score = rec.confidence * 50   # Base: 0-50 from confidence

    # IV alignment bonus
    if iv.iv_rank >= 70 and rec.risk_reward == "CREDIT":
        score += 15
    elif iv.iv_rank <= 30 and rec.risk_reward == "DEBIT":
        score += 10

    # Trend strength
    score += regime.trend_strength * 15

    # RS bonus
    if rs:
        if rec.direction == "BULLISH" and rs.outperforming_spy:
            score += 10
        elif rec.direction == "BEARISH" and not rs.outperforming_spy:
            score += 10

    # Volatility state
    if regime.volatility_state == "expanding" and rec.risk_reward == "DEBIT":
        score += 5
    elif regime.volatility_state == "contracting" and rec.risk_reward == "CREDIT":
        score += 5

    return round(score, 1)
