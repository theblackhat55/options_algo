"""
src/analysis/patterns.py
========================
Pattern detection for entry timing signals.
Detects: RSI divergence, inside bars, volume climax, squeeze breakouts, anchored VWAP.

Used by: technical.py (wired into StockRegime.ta_signals)

Usage:
    from src.analysis.patterns import detect_patterns, PatternSignals
    patterns = detect_patterns("AAPL", df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from config.settings import (
    RSI_PERIOD, BB_PERIOD, BB_STD,
    DIVERGENCE_LOOKBACK, VOLUME_CLIMAX_MULTIPLIER,
    BREAKOUT_VOLUME_MULTIPLIER,
)

logger = logging.getLogger(__name__)


@dataclass
class PatternSignals:
    """Collection of pattern-based timing signals for a ticker."""
    ticker: str

    # RSI Divergence
    bullish_divergence: bool = False   # Price lower low, RSI higher low
    bearish_divergence: bool = False   # Price higher high, RSI lower high
    divergence_strength: float = 0.0   # 0-1 magnitude of the divergence

    # Inside Bar (consolidation → breakout imminent)
    inside_bar: bool = False           # Today's range contained within yesterday's

    # Volume Climax (exhaustion or initiation)
    volume_climax: bool = False        # Volume > VOLUME_CLIMAX_MULTIPLIER × 20d avg
    climax_direction: str = "NEUTRAL"  # "UP" if close > open, "DOWN" if close < open

    # Squeeze Breakout (BB squeeze released)
    squeeze_fired: bool = False        # Was in squeeze yesterday, not today
    squeeze_direction: str = "NEUTRAL" # "UP" or "DOWN" based on close vs BB mid

    # VWAP (anchored from 20-day low/high)
    above_anchored_vwap: bool = False  # Price above VWAP anchored from recent swing low
    below_anchored_vwap: bool = False  # Price below VWAP anchored from recent swing high

    # Composite pattern score (-1.0 bearish to +1.0 bullish)
    pattern_score: float = 0.0


def detect_patterns(
    ticker: str,
    df: pd.DataFrame,
    divergence_lookback: int = DIVERGENCE_LOOKBACK,
) -> Optional[PatternSignals]:
    """
    Detect all pattern signals for a ticker.

    Requires at least 60 rows of daily OHLCV data.
    Returns None if data is insufficient.
    """
    if df is None or len(df) < 60:
        return None

    try:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        open_ = df["open"].astype(float)
        price = float(close.iloc[-1])

        signals = PatternSignals(ticker=ticker)
        score = 0.0

        # ── RSI Divergence ────────────────────────────────────────────────────
        rsi = ta.rsi(close, length=RSI_PERIOD)
        if rsi is not None and len(rsi.dropna()) >= divergence_lookback:
            _detect_divergence(signals, close, rsi, divergence_lookback)
            if signals.bullish_divergence:
                score += 0.25 * (1 + signals.divergence_strength)
            if signals.bearish_divergence:
                score -= 0.25 * (1 + signals.divergence_strength)

        # ── Inside Bar ────────────────────────────────────────────────────────
        if len(high) >= 2 and len(low) >= 2:
            signals.inside_bar = (
                float(high.iloc[-1]) <= float(high.iloc[-2]) and
                float(low.iloc[-1]) >= float(low.iloc[-2]) and
                (float(high.iloc[-2]) - float(low.iloc[-2])) > 0
            )

        # ── Volume Climax ─────────────────────────────────────────────────────
        vol_20d = float(volume.tail(20).mean())
        vol_today = float(volume.iloc[-1])
        if vol_20d > 0 and vol_today > vol_20d * VOLUME_CLIMAX_MULTIPLIER:
            signals.volume_climax = True
            bar_close = float(close.iloc[-1])
            bar_open = float(open_.iloc[-1])
            if bar_close > bar_open:
                signals.climax_direction = "UP"
                score += 0.15
            elif bar_close < bar_open:
                signals.climax_direction = "DOWN"
                score -= 0.15

        # ── Squeeze Breakout ──────────────────────────────────────────────────
        bb = ta.bbands(close, length=BB_PERIOD, std=BB_STD)
        if bb is not None and len(bb) >= 2:
            bb_bw_col = next((c for c in bb.columns if c.startswith("BBB_")), None)
            bb_mid_col = next((c for c in bb.columns if c.startswith("BBM_")), None)

            if bb_bw_col and bb_mid_col:
                bw = bb[bb_bw_col].dropna()
                if len(bw) >= 20:
                    bw_ranks = bw.rank(pct=True)
                    bw_pctile_today = float(bw_ranks.iloc[-1])
                    bw_pctile_yesterday = float(bw_ranks.iloc[-2])

                    was_squeeze = bw_pctile_yesterday < 0.15
                    not_squeeze_now = bw_pctile_today >= 0.15

                    if was_squeeze and not_squeeze_now:
                        signals.squeeze_fired = True
                        bb_mid = float(bb[bb_mid_col].iloc[-1])
                        if price > bb_mid:
                            signals.squeeze_direction = "UP"
                            score += 0.20
                        else:
                            signals.squeeze_direction = "DOWN"
                            score -= 0.20

        # ── Anchored VWAP ─────────────────────────────────────────────────────
        try:
            swing_low_idx = low.tail(20).idxmin()
            swing_high_idx = high.tail(20).idxmax()

            vwap_from_low = _compute_anchored_vwap(df, swing_low_idx)
            vwap_from_high = _compute_anchored_vwap(df, swing_high_idx)

            if vwap_from_low is not None and price > vwap_from_low:
                signals.above_anchored_vwap = True
                score += 0.10

            if vwap_from_high is not None and price < vwap_from_high:
                signals.below_anchored_vwap = True
                score -= 0.10
        except Exception:
            pass  # VWAP anchoring is optional

        # ── Volume Confirmation for Breakout ──────────────────────────────────
        if vol_20d > 0 and vol_today > vol_20d * BREAKOUT_VOLUME_MULTIPLIER and len(close) >= 2:
            daily_return = float(close.iloc[-1] / close.iloc[-2] - 1)
            if daily_return > 0.005:
                score += 0.10
            elif daily_return < -0.005:
                score -= 0.10

        signals.pattern_score = round(max(-1.0, min(1.0, score)), 3)
        return signals

    except Exception as exc:
        logger.error(f"Pattern detection failed for {ticker}: {exc}")
        return None


def _detect_divergence(
    signals: PatternSignals,
    close: pd.Series,
    rsi: pd.Series,
    lookback: int,
) -> None:
    """
    Detect bullish and bearish RSI divergences.

    Bullish: price makes lower low, RSI makes higher low.
    Bearish: price makes higher high, RSI makes lower high.
    """
    recent_close = close.tail(lookback).values
    recent_rsi = rsi.dropna().tail(lookback).values

    if len(recent_close) < 10 or len(recent_rsi) < 10:
        return

    min_len = min(len(recent_close), len(recent_rsi))
    recent_close = recent_close[-min_len:]
    recent_rsi = recent_rsi[-min_len:]

    half = min_len // 2

    price_first_low = np.min(recent_close[:half])
    price_second_low = np.min(recent_close[half:])
    rsi_first_low = np.min(recent_rsi[:half])
    rsi_second_low = np.min(recent_rsi[half:])

    price_first_high = np.max(recent_close[:half])
    price_second_high = np.max(recent_close[half:])
    rsi_first_high = np.max(recent_rsi[:half])
    rsi_second_high = np.max(recent_rsi[half:])

    rsi_range = max(recent_rsi) - min(recent_rsi)

    # Bullish: price lower low + RSI higher low
    if price_second_low < price_first_low and rsi_second_low > rsi_first_low:
        signals.bullish_divergence = True
        if rsi_range > 0:
            signals.divergence_strength = round(
                abs(rsi_second_low - rsi_first_low) / rsi_range, 3
            )

    # Bearish: price higher high + RSI lower high
    if price_second_high > price_first_high and rsi_second_high < rsi_first_high:
        signals.bearish_divergence = True
        if rsi_range > 0:
            signals.divergence_strength = round(
                abs(rsi_first_high - rsi_second_high) / rsi_range, 3
            )


def _compute_anchored_vwap(
    df: pd.DataFrame,
    anchor_idx,
) -> Optional[float]:
    """
    Compute VWAP anchored from a specific index (date) to the present.
    VWAP = cumulative(typical_price × volume) / cumulative(volume)
    """
    try:
        loc = df.index.get_loc(anchor_idx)
        subset = df.iloc[loc:].copy()
        if len(subset) < 2:
            return None

        typical_price = (
            subset["high"].astype(float) +
            subset["low"].astype(float) +
            subset["close"].astype(float)
        ) / 3

        vol = subset["volume"].astype(float)
        cum_vol = vol.cumsum()
        cum_tpv = (typical_price * vol).cumsum()

        if float(cum_vol.iloc[-1]) <= 0:
            return None

        return round(float(cum_tpv.iloc[-1] / cum_vol.iloc[-1]), 2)

    except Exception:
        return None


def detect_universe_patterns(
    data: dict,
) -> dict:
    """Detect patterns for all tickers in the universe."""
    results = {}
    for ticker, df in data.items():
        ps = detect_patterns(ticker, df)
        if ps is not None:
            results[ticker] = ps
    logger.info(f"Pattern detection: {len(results)}/{len(data)} tickers")
    return results
