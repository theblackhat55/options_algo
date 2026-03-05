"""
src/analysis/levels.py
======================
Support/Resistance detection using volume profile analysis.
Identifies key price levels where volume concentrated historically,
then classifies the current price relationship to those levels.

Used by: selector.py (confidence scoring), nightly_scan.py (trade context)

Usage:
    from src.analysis.levels import analyze_levels, LevelAnalysis
    levels = analyze_levels("AAPL", df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import SR_LOOKBACK_DAYS, SR_VOLUME_BINS, SR_PROXIMITY_PCT

logger = logging.getLogger(__name__)


@dataclass
class LevelAnalysis:
    """Result of support/resistance analysis for a single ticker."""
    ticker: str
    price: float

    # Volume-profile derived levels (sorted ascending)
    support_levels: list = field(default_factory=list)
    resistance_levels: list = field(default_factory=list)

    # Nearest levels to current price
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    support_distance_pct: float = 0.0     # % distance from price to nearest support
    resistance_distance_pct: float = 0.0  # % distance from price to nearest resistance

    # Classification
    near_support: bool = False             # price within SR_PROXIMITY_PCT of support
    near_resistance: bool = False          # price within SR_PROXIMITY_PCT of resistance
    breakout_above: bool = False           # price broke above resistance with volume
    breakdown_below: bool = False          # price broke below support with volume

    # Volume profile metrics
    high_volume_node: float = 0.0          # Price level with highest volume (POC)
    volume_profile_skew: float = 0.0       # >0 = more volume above price, <0 = below


def analyze_levels(
    ticker: str,
    df: pd.DataFrame,
    lookback: int = SR_LOOKBACK_DAYS,
    n_bins: int = SR_VOLUME_BINS,
    proximity_pct: float = SR_PROXIMITY_PCT,
) -> Optional[LevelAnalysis]:
    """
    Analyze support and resistance levels using volume profile.

    The volume profile divides the lookback price range into bins,
    sums volume in each bin, and identifies high-volume nodes (HVN)
    as support/resistance. Price levels where volume clustered act
    as magnets — they attract price and provide support/resistance.

    Args:
        ticker: Symbol
        df: OHLCV DataFrame with columns: close, high, low, volume
        lookback: Number of trading days to analyze
        n_bins: Number of price bins for volume profile
        proximity_pct: % threshold for "near" a level

    Returns:
        LevelAnalysis or None if insufficient data
    """
    if df is None or len(df) < lookback:
        return None

    try:
        recent = df.tail(lookback).copy()
        close = recent["close"].astype(float)
        high = recent["high"].astype(float)
        low = recent["low"].astype(float)
        volume = recent["volume"].astype(float)
        price = float(close.iloc[-1])

        if price <= 0:
            return None

        # ── Volume Profile ────────────────────────────────────────────────────
        price_min = float(low.min())
        price_max = float(high.max())

        if price_max <= price_min:
            return None

        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_profile = np.zeros(n_bins)

        for i in range(len(recent)):
            bar_low = float(low.iloc[i])
            bar_high = float(high.iloc[i])
            bar_vol = float(volume.iloc[i])

            if bar_vol <= 0 or bar_high <= bar_low:
                continue

            low_bin = np.searchsorted(bin_edges, bar_low, side="right") - 1
            high_bin = np.searchsorted(bin_edges, bar_high, side="left")

            low_bin = max(0, min(low_bin, n_bins - 1))
            high_bin = max(0, min(high_bin, n_bins - 1))

            n_covered = high_bin - low_bin + 1
            if n_covered > 0:
                vol_per_bin = bar_vol / n_covered
                vol_profile[low_bin:high_bin + 1] += vol_per_bin

        if vol_profile.sum() <= 0:
            return None

        # ── Identify High Volume Nodes (HVN) ─────────────────────────────────
        threshold = np.percentile(vol_profile, 75)
        hvn_mask = vol_profile >= threshold
        hvn_prices = bin_centers[hvn_mask]

        # Point of Control (POC)
        poc_idx = np.argmax(vol_profile)
        poc = float(bin_centers[poc_idx])

        # ── Classify as Support or Resistance ────────────────────────────────
        support_levels = sorted([float(p) for p in hvn_prices if p < price])
        resistance_levels = sorted([float(p) for p in hvn_prices if p > price])

        # Add rolling lows/highs as classic S/R
        rolling_support = float(low.tail(20).min())
        rolling_resistance = float(high.tail(20).max())

        if rolling_support < price and rolling_support not in support_levels:
            support_levels.append(rolling_support)
            support_levels.sort()

        if rolling_resistance > price and rolling_resistance not in resistance_levels:
            resistance_levels.append(rolling_resistance)
            resistance_levels.sort()

        # ── Nearest Levels ────────────────────────────────────────────────────
        nearest_support = max(support_levels) if support_levels else price * 0.95
        nearest_resistance = min(resistance_levels) if resistance_levels else price * 1.05

        support_dist = abs(price - nearest_support) / price * 100
        resistance_dist = abs(nearest_resistance - price) / price * 100

        near_support = support_dist <= proximity_pct
        near_resistance = resistance_dist <= proximity_pct

        # ── Breakout Detection ────────────────────────────────────────────────
        prev_resistance = float(high.iloc[:-1].tail(20).max())
        prev_support = float(low.iloc[:-1].tail(20).min())
        vol_today = float(volume.iloc[-1])
        vol_20d_avg = float(volume.tail(20).mean())

        breakout_above = (
            price > prev_resistance and
            vol_20d_avg > 0 and
            vol_today > vol_20d_avg * 1.5
        )

        breakdown_below = (
            price < prev_support and
            vol_20d_avg > 0 and
            vol_today > vol_20d_avg * 1.5
        )

        # ── Volume Profile Skew ───────────────────────────────────────────────
        price_bin = int(np.searchsorted(bin_edges, price, side="right")) - 1
        price_bin = max(0, min(price_bin, n_bins - 1))
        vol_above = vol_profile[price_bin + 1:].sum() if price_bin < n_bins - 1 else 0.0
        vol_below = vol_profile[:price_bin].sum() if price_bin > 0 else 0.0
        total_vol = vol_above + vol_below
        vol_skew = round((vol_above - vol_below) / total_vol, 3) if total_vol > 0 else 0.0

        return LevelAnalysis(
            ticker=ticker,
            price=round(price, 2),
            support_levels=[round(s, 2) for s in support_levels[-3:]],
            resistance_levels=[round(r, 2) for r in resistance_levels[:3]],
            nearest_support=round(nearest_support, 2),
            nearest_resistance=round(nearest_resistance, 2),
            support_distance_pct=round(support_dist, 2),
            resistance_distance_pct=round(resistance_dist, 2),
            near_support=near_support,
            near_resistance=near_resistance,
            breakout_above=breakout_above,
            breakdown_below=breakdown_below,
            high_volume_node=round(poc, 2),
            volume_profile_skew=vol_skew,
        )

    except Exception as exc:
        logger.error(f"Level analysis failed for {ticker}: {exc}")
        return None


def analyze_universe_levels(
    data: dict,
) -> dict:
    """Run level analysis on all tickers in the universe."""
    results = {}
    for ticker, df in data.items():
        la = analyze_levels(ticker, df)
        if la is not None:
            results[ticker] = la
    logger.info(f"Level analysis: {len(results)}/{len(data)} tickers")
    return results
