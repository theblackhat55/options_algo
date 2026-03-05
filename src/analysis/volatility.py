"""
src/analysis/volatility.py
==========================
Implied volatility analysis: IV rank, IV percentile, IV/HV ratio.
Determines whether to buy premium (low IV) or sell premium (high IV).

V2 changes:
  - Added iv_rv_spread field: current_iv minus hv_20 (vol points)
  - Added premium_rich flag: True when IV-RV spread > MIN_IV_RV_SPREAD_CREDIT vol points
    This prevents selling premium when IV looks "high" purely because HV spiked
    (i.e., a selloff pushed realized vol up but options haven't re-priced yet).

Usage:
    from src.analysis.volatility import analyze_iv, analyze_universe_iv

    iv = analyze_iv("AAPL", price_df, options_chain)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    IV_LOOKBACK_DAYS, HV_WINDOW, IV_HIGH_THRESHOLD, IV_LOW_THRESHOLD,
    MIN_IV_RV_SPREAD_CREDIT, IV_SNAPSHOT_DIR, IV_SNAPSHOT_MIN_HISTORY,
)
from src.data.options_fetcher import get_atm_iv

logger = logging.getLogger(__name__)


# ─── IVAnalysis Dataclass ─────────────────────────────────────────────────────

@dataclass
class IVAnalysis:
    ticker: str
    current_iv: float         # Current 30-day implied vol (as %, e.g. 32.5)
    iv_rank: float            # IV rank 0-100: where current IV sits in 1-yr range
    iv_percentile: float      # % of past year days where IV was lower
    hv_20: float              # 20-day historical (realized) vol (%)
    hv_60: float              # 60-day historical vol (%)
    iv_hv_ratio: float        # current_iv / hv_20 (>1 = premium; <1 = discount)
    iv_regime: str            # "HIGH" | "NORMAL" | "LOW"
    premium_action: str       # "SELL" | "NEUTRAL" | "BUY"
    iv_trend: str             # "RISING" | "FLAT" | "FALLING"
    iv_30d_avg: float         # 30-day moving average of IV (%)
    skew: float               # Put/call IV skew proxy (positive = puts expensive)
    iv_rv_spread: float = 0.0  # NEW: current_iv - hv_20 (vol points); positive = premium rich
    premium_rich: bool = False  # NEW: True when iv_rv_spread > MIN_IV_RV_SPREAD_CREDIT


# ─── Core Computations ────────────────────────────────────────────────────────

def compute_historical_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Annualised close-to-close historical volatility.
    Returns a Series aligned with the input index.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)


def compute_iv_rank(iv_series: pd.Series, lookback: int = 252) -> float:
    """
    IV Rank = (Current IV − 52wk Low) / (52wk High − 52wk Low) × 100

    A rank of 70+ means IV is elevated vs recent history → sell premium.
    A rank of 30− means IV is depressed → buy premium.
    """
    recent = iv_series.tail(lookback).dropna()
    if len(recent) < 2:
        return 50.0

    current = float(recent.iloc[-1])
    lo = float(recent.min())
    hi = float(recent.max())

    if hi == lo:
        return 50.0

    return round((current - lo) / (hi - lo) * 100, 1)


def compute_iv_percentile(iv_series: pd.Series, lookback: int = 252) -> float:
    """
    IV Percentile = % of days in the past year where IV was LOWER than today.
    More robust than IV Rank because it is not distorted by single outlier spikes.
    """
    recent = iv_series.tail(lookback).dropna()
    if len(recent) < 2:
        return 50.0

    current = float(recent.iloc[-1])
    pct = float((recent < current).sum()) / len(recent) * 100
    return round(pct, 1)


# ─── IV Snapshot History Loader ───────────────────────────────────────────────

def _load_iv_snapshot_history(
    ticker: str,
    close_series: pd.Series,
) -> tuple[pd.Series, bool]:
    """
    Try to load real IV history from Parquet snapshots captured by
    scripts/capture_iv_snapshot.py. Falls back to HV×1.15 proxy if
    fewer than IV_SNAPSHOT_MIN_HISTORY days are available.

    Returns:
        (iv_series, is_proxy) — series aligned to close_series index
    """
    import pathlib

    snapshot_path = pathlib.Path(IV_SNAPSHOT_DIR) / f"{ticker}_iv_history.parquet"
    try:
        if snapshot_path.exists():
            snap_df = pd.read_parquet(snapshot_path)
            # Expect columns: date (index), atm_iv (float, already in %)
            if "atm_iv" in snap_df.columns and len(snap_df) >= IV_SNAPSHOT_MIN_HISTORY:
                snap_df.index = pd.to_datetime(snap_df.index)
                iv_series = snap_df["atm_iv"].dropna()
                if len(iv_series) >= IV_SNAPSHOT_MIN_HISTORY:
                    logger.debug(
                        f"{ticker}: using real IV snapshot ({len(iv_series)} days)"
                    )
                    return iv_series, False
    except Exception as exc:
        logger.debug(f"{ticker}: IV snapshot load failed — {exc}")

    # Fallback: HV × 1.15 proxy
    iv_proxy = compute_historical_volatility(close_series, window=30) * 100 * 1.15
    logger.debug(
        f"{ticker}: IV proxy (HV×1.15) — "
        f"snapshot has fewer than {IV_SNAPSHOT_MIN_HISTORY} days of real data"
    )
    return iv_proxy, True


# ─── Main Analysis Function ───────────────────────────────────────────────────

def analyze_iv(
    ticker: str,
    price_df: pd.DataFrame,
    options_chain: Optional[pd.DataFrame] = None,
) -> Optional[IVAnalysis]:
    """
    Full IV analysis for a single stock.

    If an options chain is provided, ATM IV is extracted from it.
    Otherwise, 30-day HV × 1.15 is used as an IV proxy.

    V2: also computes iv_rv_spread and premium_rich flag to distinguish
    "IV is high because a selloff spiked realized vol" (proxy inflation)
    from "IV is genuinely elevated vs realized vol" (true premium richness).

    Returns:
        IVAnalysis or None if insufficient data.
    """
    if price_df is None or len(price_df) < 60:
        return None

    try:
        close = price_df["close"].astype(float)
        current_price = float(close.iloc[-1])

        # ── Historical Volatility ─────────────────────────────────────────────
        hv20_series = compute_historical_volatility(close, window=20)
        hv60_series = compute_historical_volatility(close, window=60)

        hv_20 = float(hv20_series.iloc[-1]) * 100   # Convert to %
        hv_60 = float(hv60_series.iloc[-1]) * 100

        if np.isnan(hv_20) or hv_20 <= 0:
            hv_20 = 20.0   # Sensible default

        # ── Implied Volatility ────────────────────────────────────────────────
        iv_is_proxy = False  # True when using HV*1.15 fallback (no real chain data)
        if options_chain is not None and not options_chain.empty:
            current_iv_raw = get_atm_iv(options_chain, current_price)
            if current_iv_raw <= 0 or np.isnan(current_iv_raw):
                current_iv_raw = hv_20 / 100 * 1.15
                iv_is_proxy = True   # real chain present but ATM IV invalid
        else:
            current_iv_raw = (hv_20 / 100) * 1.15    # IV proxy — no options chain
            iv_is_proxy = True
            logger.debug(f"{ticker}: spot IV proxy — no options chain provided")

        current_iv = current_iv_raw * 100   # Already % if from get_atm_iv

        # If get_atm_iv returned a fraction (0-1), scale up
        if current_iv < 1.0:
            current_iv *= 100

        # ── IV History: prefer real snapshot, fallback to HV×1.15 proxy ───────
        iv_history_series, iv_is_proxy = _load_iv_snapshot_history(
            ticker, close
        )
        # Recheck proxy flag: if we got real snapshot data, iv_is_proxy stays False
        # only if the snapshot was used AND the spot current_iv was from options chain
        if iv_is_proxy and not iv_is_proxy:
            iv_is_proxy = True   # keep existing proxy flag from spot IV logic above

        # ── IV Rank / Percentile ──────────────────────────────────────────────
        iv_rank = compute_iv_rank(iv_history_series, IV_LOOKBACK_DAYS)
        iv_pctile = compute_iv_percentile(iv_history_series, IV_LOOKBACK_DAYS)

        # ── IV/HV Ratio ───────────────────────────────────────────────────────
        iv_hv_ratio = round(current_iv / hv_20, 2) if hv_20 > 0 else 1.0

        # ── 30-Day IV Moving Average ──────────────────────────────────────────
        iv_30d_avg = round(float(iv_history_series.tail(30).mean()), 1)

        # ── IV Regime ─────────────────────────────────────────────────────────
        if iv_rank >= IV_HIGH_THRESHOLD:
            iv_regime = "HIGH"
            premium_action = "SELL"
        elif iv_rank <= IV_LOW_THRESHOLD:
            iv_regime = "LOW"
            premium_action = "BUY"
        else:
            iv_regime = "NORMAL"
            premium_action = "NEUTRAL"

        # ── IV Trend ──────────────────────────────────────────────────────────
        recent_iv = iv_history_series.tail(10).dropna()
        if len(recent_iv) >= 5:
            iv_5d_ago = float(recent_iv.iloc[-5])
            iv_now = float(recent_iv.iloc[-1])
            if iv_now > iv_5d_ago * 1.05:
                iv_trend = "RISING"
            elif iv_now < iv_5d_ago * 0.95:
                iv_trend = "FALLING"
            else:
                iv_trend = "FLAT"
        else:
            iv_trend = "FLAT"

        # ── NEW: IV-RV Spread ─────────────────────────────────────────────────
        # Positive value = implied vol is trading above realized vol (genuine premium)
        # If spread is thin or negative, "HIGH IV" is mostly just high realized vol
        # and there isn't much edge in selling that premium.
        # IMPORTANT: when IV is a proxy (HV * 1.15) the spread is mechanical
        # (always ~15% of HV) and carries no signal — suppress premium_rich.
        iv_rv_spread = round(current_iv - hv_20, 1)
        premium_rich = (iv_rv_spread > MIN_IV_RV_SPREAD_CREDIT) and not iv_is_proxy

        # ── Put/Call Skew Proxy ───────────────────────────────────────────────
        # Positive skew = puts more expensive (market hedging, bearish fear)
        skew = 0.0
        if options_chain is not None and not options_chain.empty:
            skew = _compute_skew_proxy(options_chain, current_price)

        return IVAnalysis(
            ticker=ticker,
            current_iv=round(current_iv, 1),
            iv_rank=iv_rank,
            iv_percentile=iv_pctile,
            hv_20=round(hv_20, 1),
            hv_60=round(hv_60, 1),
            iv_hv_ratio=iv_hv_ratio,
            iv_regime=iv_regime,
            premium_action=premium_action,
            iv_trend=iv_trend,
            iv_30d_avg=iv_30d_avg,
            skew=round(skew, 2),
            iv_rv_spread=iv_rv_spread,
            premium_rich=premium_rich,
        )

    except Exception as exc:
        logger.error(f"IV analysis failed for {ticker}: {exc}")
        return None


def analyze_universe_iv(
    data: dict[str, pd.DataFrame],
    chains: Optional[dict[str, pd.DataFrame]] = None,
) -> dict[str, IVAnalysis]:
    """Analyze IV for all tickers in the universe."""
    results = {}
    for ticker, df in data.items():
        chain = chains.get(ticker) if chains else None
        iv = analyze_iv(ticker, df, chain)
        if iv is not None:
            results[ticker] = iv
    logger.info(f"IV analyzed for {len(results)}/{len(data)} tickers")
    return results


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compute_skew_proxy(chain: pd.DataFrame, current_price: float) -> float:
    """
    Compute a simple put/call skew proxy.
    Compare IV of 25-delta puts vs 25-delta calls.
    Positive = puts more expensive (bearish skew, normal for equities).
    """
    try:
        # OTM puts: strike ~5% below ATM
        put_target = current_price * 0.95
        call_target = current_price * 1.05

        puts = chain[chain["type"] == "put"]
        calls = chain[chain["type"] == "call"]

        if puts.empty or calls.empty:
            return 0.0

        put_row = puts.iloc[(puts["strike"] - put_target).abs().argsort()[:1]]
        call_row = calls.iloc[(calls["strike"] - call_target).abs().argsort()[:1]]

        put_iv = float(put_row["implied_volatility"].iloc[0])
        call_iv = float(call_row["implied_volatility"].iloc[0])

        if put_iv <= 0 or call_iv <= 0:
            return 0.0

        return (put_iv - call_iv) / ((put_iv + call_iv) / 2)

    except Exception:
        return 0.0
