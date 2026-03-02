"""
src/analysis/relative_strength.py
==================================
Mansfield Relative Strength (RS) vs SPY and sector ETFs.
Used to prefer stocks that are outperforming the market (longs)
or underperforming (shorts/bearish spreads).

Usage:
    from src.analysis.relative_strength import compute_rs, rank_universe_rs
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from config.universe import get_sector, SECTOR_ETF_MAP

logger = logging.getLogger(__name__)


@dataclass
class RSAnalysis:
    ticker: str
    rs_vs_spy: float        # Raw RS vs SPY (current price ratio normalised)
    rs_rank: float          # Percentile rank within the universe (0-100)
    rs_trend: str           # "IMPROVING" | "STEADY" | "WEAKENING"
    sector: str
    sector_rs: float        # Sector ETF RS vs SPY
    outperforming_spy: bool
    outperforming_sector: bool


def compute_rs_raw(
    ticker_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback: int = 52,     # weeks ≈ 252 trading days / 5
) -> float:
    """
    Mansfield-style RS: (ticker 52wk % change) / (benchmark 52wk % change).
    Values > 1 = outperforming. Values < 1 = underperforming.
    """
    # Use weekly closes (every 5th day) for Mansfield RS
    n_days = lookback * 5
    try:
        t_close = ticker_df["close"].tail(n_days)
        b_close = benchmark_df["close"].tail(n_days)

        if len(t_close) < 20 or len(b_close) < 20:
            return 1.0

        t_ret = float(t_close.iloc[-1]) / float(t_close.iloc[0]) - 1
        b_ret = float(b_close.iloc[-1]) / float(b_close.iloc[0]) - 1

        # Normalise so RS = 0 when equal to benchmark
        if b_ret == -1:
            return 0.0

        rs = (1 + t_ret) / (1 + b_ret) - 1
        return round(rs, 4)

    except Exception as exc:
        logger.debug(f"RS computation failed: {exc}")
        return 0.0


def compute_rs(
    ticker: str,
    data: dict[str, pd.DataFrame],
    lookback: int = 52,
) -> Optional[RSAnalysis]:
    """
    Compute relative strength for a single ticker vs SPY and its sector ETF.
    """
    ticker_df = data.get(ticker)
    spy_df = data.get("SPY")

    if ticker_df is None or spy_df is None:
        return None

    sector = get_sector(ticker)
    sector_etf = SECTOR_ETF_MAP.get(sector)
    sector_df = data.get(sector_etf) if sector_etf else None

    try:
        rs_vs_spy = compute_rs_raw(ticker_df, spy_df, lookback)

        sector_rs = 0.0
        if sector_df is not None:
            sector_rs = compute_rs_raw(sector_df, spy_df, lookback)

        outperforming_spy = rs_vs_spy > 0
        outperforming_sector = (
            (rs_vs_spy > sector_rs) if sector_df is not None else outperforming_spy
        )

        # RS Trend: compare current 26wk RS to 13wk-ago RS
        rs_trend = _compute_rs_trend(ticker_df, spy_df)

        return RSAnalysis(
            ticker=ticker,
            rs_vs_spy=rs_vs_spy,
            rs_rank=50.0,   # Filled in by rank_universe_rs()
            rs_trend=rs_trend,
            sector=sector,
            sector_rs=sector_rs,
            outperforming_spy=outperforming_spy,
            outperforming_sector=outperforming_sector,
        )

    except Exception as exc:
        logger.debug(f"RS analysis failed for {ticker}: {exc}")
        return None


def rank_universe_rs(
    data: dict[str, pd.DataFrame],
) -> dict[str, RSAnalysis]:
    """
    Compute and rank RS for all tickers.
    Fills in the percentile rank within the universe.
    """
    results: dict[str, RSAnalysis] = {}
    for ticker in data:
        rs = compute_rs(ticker, data)
        if rs is not None:
            results[ticker] = rs

    # Assign percentile ranks
    raw_scores = {t: r.rs_vs_spy for t, r in results.items()}
    sorted_tickers = sorted(raw_scores, key=raw_scores.get)
    n = len(sorted_tickers)
    for i, ticker in enumerate(sorted_tickers):
        results[ticker].rs_rank = round(i / max(n - 1, 1) * 100, 1)

    logger.info(f"RS computed for {len(results)} tickers")
    return results


def _compute_rs_trend(
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> str:
    """Compare RS now vs RS 13 weeks ago to detect improvement/weakening."""
    try:
        rs_now = compute_rs_raw(ticker_df, spy_df, lookback=26)
        # Slice data to 13 weeks ago
        days_13wk = 65
        if len(ticker_df) < days_13wk + 20:
            return "STEADY"
        t_old = ticker_df.iloc[:-days_13wk] if len(ticker_df) > days_13wk else ticker_df
        b_old = spy_df.iloc[:-days_13wk] if len(spy_df) > days_13wk else spy_df
        rs_13wk_ago = compute_rs_raw(t_old, b_old, lookback=26)

        delta = rs_now - rs_13wk_ago
        if delta > 0.05:
            return "IMPROVING"
        elif delta < -0.05:
            return "WEAKENING"
        return "STEADY"
    except Exception:
        return "STEADY"
