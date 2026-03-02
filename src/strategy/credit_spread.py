"""
src/strategy/credit_spread.py
=============================
Construct bull put spreads and bear call spreads (credit spreads).

Bull Put Spread (bullish):   Sell OTM put + Buy further OTM put
Bear Call Spread (bearish):  Sell OTM call + Buy further OTM call

Usage:
    from src.strategy.credit_spread import construct_bull_put_spread, construct_bear_call_spread

    spread = construct_bull_put_spread("AAPL", current_price, chain, target_dte=45)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.options_analytics import (
    prob_otm, credit_spread_ev, annualised_return_on_risk,
    RISK_FREE_RATE,
)

logger = logging.getLogger(__name__)


# ─── CreditSpread Dataclass ───────────────────────────────────────────────────

@dataclass
class CreditSpread:
    ticker: str
    spread_type: str           # "BULL_PUT" or "BEAR_CALL"
    expiration: str
    dte: int
    short_strike: float
    long_strike: float
    short_premium: float
    long_premium: float
    net_credit: float          # Maximum reward
    max_risk: float            # Width − credit
    max_reward: float          # = net_credit
    risk_reward_ratio: float   # max_risk / net_credit (lower = better)
    breakeven: float
    prob_profit: float         # % probability stock stays on correct side
    prob_touch: float          # % probability of touching short strike
    short_delta: float
    width: float
    annual_ror: float          # Annualised return on risk (%)
    ev: float                  # Expected value per contract ($)
    notes: str = ""


# ─── Bull Put Spread ──────────────────────────────────────────────────────────

def construct_bull_put_spread(
    ticker: str,
    current_price: float,
    chain: pd.DataFrame,
    target_dte: int = 45,
    target_delta: float = 0.25,
    spread_width: float = 5.0,
) -> Optional[CreditSpread]:
    """
    Bull Put Spread: Sell OTM put, buy lower-strike OTM put.
    Profit if stock stays above the short put strike at expiry.

    Args:
        ticker: Symbol
        current_price: Latest close price
        chain: Filtered options chain DataFrame
        target_dte: Target days to expiration (will find nearest)
        target_delta: Target short put delta (0.20–0.30 typical)
        spread_width: Strike width in dollars (will snap to available strikes)

    Returns:
        CreditSpread or None if no suitable strikes found.
    """
    try:
        puts = chain[chain["type"] == "put"].copy()
        if puts.empty:
            logger.debug(f"{ticker}: no puts in chain")
            return None

        puts = _add_dte(puts)

        # Find expiration closest to target_dte
        exp, actual_dte = _pick_expiration(puts, target_dte)
        if exp is None:
            return None

        exp_puts = puts[puts["expiration"] == exp].sort_values("strike")
        if len(exp_puts) < 2:
            return None

        # ── Short Put Selection ───────────────────────────────────────────────
        short_row = _pick_short_strike_put(exp_puts, current_price, target_delta)
        if short_row is None:
            return None

        short_strike = float(short_row["strike"])
        long_strike = _snap_strike(short_strike - spread_width, exp_puts["strike"].values, direction=-1)

        long_row = _get_row_by_strike(exp_puts, long_strike)
        if long_row is None:
            # Try next-lowest available strike
            lower = exp_puts[exp_puts["strike"] < short_strike]
            if lower.empty:
                return None
            long_row = lower.iloc[-1]  # Highest strike below short

        actual_long_strike = float(long_row["strike"])
        width = round(short_strike - actual_long_strike, 2)

        if width <= 0:
            return None

        # ── Pricing ───────────────────────────────────────────────────────────
        short_premium = float(short_row.get("mid", short_row.get("last", 0)))
        long_premium  = float(long_row.get("mid", long_row.get("last", 0)))

        # Fall back to ask/2 if mid not available
        if short_premium <= 0:
            short_premium = float(short_row.get("ask", 0)) * 0.9
        if long_premium <= 0:
            long_premium = float(long_row.get("ask", 0)) * 0.9

        net_credit = round(short_premium - long_premium, 2)
        if net_credit <= 0.10:   # Minimum $0.10 credit
            logger.debug(f"{ticker}: bull put credit too small ({net_credit})")
            return None

        max_risk = round(width - net_credit, 2)
        if max_risk <= 0:
            return None

        breakeven = round(short_strike - net_credit, 2)
        rr_ratio = round(max_risk / net_credit, 2)

        # ── Probability Estimates ─────────────────────────────────────────────
        short_delta_val = abs(float(short_row.get("delta", 0) or 0))
        iv = float(short_row.get("implied_volatility", 0) or 0)

        if iv > 0 and actual_dte > 0:
            T = actual_dte / 365
            if iv > 1:
                iv = iv / 100   # Convert % to decimal if needed
            pop = prob_otm(current_price, short_strike, T, iv, RISK_FREE_RATE, "put") * 100
        elif short_delta_val > 0:
            pop = round((1 - short_delta_val) * 100, 1)
        else:
            pop = round((1 - target_delta) * 100, 1)

        # Probability of touching short strike (barrier)
        from src.analysis.options_analytics import prob_touch
        pot = 0.0
        if iv > 0 and actual_dte > 0:
            T = actual_dte / 365
            if iv > 1:
                iv_dec = iv / 100
            else:
                iv_dec = iv
            pot = prob_touch(current_price, short_strike, T, iv_dec, RISK_FREE_RATE) * 100

        # Expected value
        ev = credit_spread_ev(net_credit, max_risk, pop / 100)
        ror = annualised_return_on_risk(net_credit, max_risk, actual_dte)

        return CreditSpread(
            ticker=ticker,
            spread_type="BULL_PUT",
            expiration=exp,
            dte=actual_dte,
            short_strike=short_strike,
            long_strike=actual_long_strike,
            short_premium=round(short_premium, 2),
            long_premium=round(long_premium, 2),
            net_credit=net_credit,
            max_risk=max_risk,
            max_reward=net_credit,
            risk_reward_ratio=rr_ratio,
            breakeven=breakeven,
            prob_profit=round(pop, 1),
            prob_touch=round(pot, 1),
            short_delta=round(short_delta_val, 3),
            width=width,
            annual_ror=ror,
            ev=ev,
        )

    except Exception as exc:
        logger.error(f"Bull put spread construction failed for {ticker}: {exc}")
        return None


# ─── Bear Call Spread ─────────────────────────────────────────────────────────

def construct_bear_call_spread(
    ticker: str,
    current_price: float,
    chain: pd.DataFrame,
    target_dte: int = 45,
    target_delta: float = 0.25,
    spread_width: float = 5.0,
) -> Optional[CreditSpread]:
    """
    Bear Call Spread: Sell OTM call, buy higher-strike OTM call.
    Profit if stock stays below the short call strike at expiry.
    """
    try:
        calls = chain[chain["type"] == "call"].copy()
        if calls.empty:
            return None

        calls = _add_dte(calls)
        exp, actual_dte = _pick_expiration(calls, target_dte)
        if exp is None:
            return None

        exp_calls = calls[calls["expiration"] == exp].sort_values("strike")
        if len(exp_calls) < 2:
            return None

        # ── Short Call Selection ──────────────────────────────────────────────
        short_row = _pick_short_strike_call(exp_calls, current_price, target_delta)
        if short_row is None:
            return None

        short_strike = float(short_row["strike"])
        long_strike = _snap_strike(short_strike + spread_width, exp_calls["strike"].values, direction=1)

        long_row = _get_row_by_strike(exp_calls, long_strike)
        if long_row is None:
            higher = exp_calls[exp_calls["strike"] > short_strike]
            if higher.empty:
                return None
            long_row = higher.iloc[0]

        actual_long_strike = float(long_row["strike"])
        width = round(actual_long_strike - short_strike, 2)

        if width <= 0:
            return None

        # ── Pricing ───────────────────────────────────────────────────────────
        short_premium = float(short_row.get("mid", short_row.get("last", 0)))
        long_premium  = float(long_row.get("mid", long_row.get("last", 0)))

        if short_premium <= 0:
            short_premium = float(short_row.get("ask", 0)) * 0.9
        if long_premium <= 0:
            long_premium = float(long_row.get("ask", 0)) * 0.9

        net_credit = round(short_premium - long_premium, 2)
        if net_credit <= 0.10:
            return None

        max_risk = round(width - net_credit, 2)
        if max_risk <= 0:
            return None

        breakeven = round(short_strike + net_credit, 2)
        rr_ratio = round(max_risk / net_credit, 2)

        # ── Probability Estimates ─────────────────────────────────────────────
        short_delta_val = abs(float(short_row.get("delta", 0) or 0))
        iv = float(short_row.get("implied_volatility", 0) or 0)

        if iv > 0 and actual_dte > 0:
            T = actual_dte / 365
            if iv > 1:
                iv = iv / 100
            pop = prob_otm(current_price, short_strike, T, iv, RISK_FREE_RATE, "call") * 100
        elif short_delta_val > 0:
            pop = round((1 - short_delta_val) * 100, 1)
        else:
            pop = round((1 - target_delta) * 100, 1)

        from src.analysis.options_analytics import prob_touch
        pot = 0.0
        if iv > 0 and actual_dte > 0:
            T = actual_dte / 365
            iv_dec = iv / 100 if iv > 1 else iv
            pot = prob_touch(current_price, short_strike, T, iv_dec, RISK_FREE_RATE) * 100

        ev = credit_spread_ev(net_credit, max_risk, pop / 100)
        ror = annualised_return_on_risk(net_credit, max_risk, actual_dte)

        return CreditSpread(
            ticker=ticker,
            spread_type="BEAR_CALL",
            expiration=exp,
            dte=actual_dte,
            short_strike=short_strike,
            long_strike=actual_long_strike,
            short_premium=round(short_premium, 2),
            long_premium=round(long_premium, 2),
            net_credit=net_credit,
            max_risk=max_risk,
            max_reward=net_credit,
            risk_reward_ratio=rr_ratio,
            breakeven=breakeven,
            prob_profit=round(pop, 1),
            prob_touch=round(pot, 1),
            short_delta=round(short_delta_val, 3),
            width=width,
            annual_ror=ror,
            ev=ev,
        )

    except Exception as exc:
        logger.error(f"Bear call spread construction failed for {ticker}: {exc}")
        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _add_dte(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    today = pd.Timestamp.now().normalize()
    df["expiration_dt"] = pd.to_datetime(df["expiration"])
    df["dte"] = (df["expiration_dt"] - today).dt.days
    return df


def _pick_expiration(df: pd.DataFrame, target_dte: int) -> tuple[Optional[str], int]:
    """Find expiration closest to target_dte. Returns (expiration_str, actual_dte)."""
    if "dte" not in df.columns:
        return None, 0
    valid = df[df["dte"] >= 5].copy()   # At least 5 days
    if valid.empty:
        return None, 0
    idx = (valid["dte"] - target_dte).abs().idxmin()
    row = valid.loc[idx]
    return str(row["expiration"]), int(row["dte"])


def _pick_short_strike_put(
    exp_puts: pd.DataFrame,
    current_price: float,
    target_delta: float,
) -> Optional[pd.Series]:
    """Select short put strike: prefer delta-based, fall back to price-based."""
    puts_otm = exp_puts[exp_puts["strike"] < current_price]
    if puts_otm.empty:
        puts_otm = exp_puts

    # Delta-based selection
    if puts_otm["delta"].abs().sum() > 0:
        puts_otm = puts_otm.copy()
        puts_otm["delta_diff"] = (puts_otm["delta"].abs() - target_delta).abs()
        return puts_otm.loc[puts_otm["delta_diff"].idxmin()]

    # Price-based fallback: target ~1 ATR below (approximated as delta × price)
    target_strike = current_price * (1 - target_delta * 0.6)
    puts_otm = puts_otm.copy()
    puts_otm["strike_diff"] = (puts_otm["strike"] - target_strike).abs()
    return puts_otm.loc[puts_otm["strike_diff"].idxmin()]


def _pick_short_strike_call(
    exp_calls: pd.DataFrame,
    current_price: float,
    target_delta: float,
) -> Optional[pd.Series]:
    """Select short call strike: prefer delta-based, fall back to price-based."""
    calls_otm = exp_calls[exp_calls["strike"] > current_price]
    if calls_otm.empty:
        calls_otm = exp_calls

    if calls_otm["delta"].abs().sum() > 0:
        calls_otm = calls_otm.copy()
        calls_otm["delta_diff"] = (calls_otm["delta"].abs() - target_delta).abs()
        return calls_otm.loc[calls_otm["delta_diff"].idxmin()]

    target_strike = current_price * (1 + target_delta * 0.6)
    calls_otm = calls_otm.copy()
    calls_otm["strike_diff"] = (calls_otm["strike"] - target_strike).abs()
    return calls_otm.loc[calls_otm["strike_diff"].idxmin()]


def _snap_strike(
    target: float,
    available: np.ndarray,
    direction: int = -1,
) -> float:
    """Find the nearest available strike, preferring direction (-1=lower, +1=higher)."""
    if direction == -1:
        valid = available[available < target + 0.01]
        if len(valid) == 0:
            return float(available[0])
        return float(valid[np.argmin(np.abs(valid - target))])
    else:
        valid = available[available > target - 0.01]
        if len(valid) == 0:
            return float(available[-1])
        return float(valid[np.argmin(np.abs(valid - target))])


def _get_row_by_strike(
    df: pd.DataFrame,
    strike: float,
    tolerance: float = 0.51,
) -> Optional[pd.Series]:
    """Get a row by strike price within tolerance."""
    mask = (df["strike"] - strike).abs() <= tolerance
    subset = df[mask]
    if subset.empty:
        return None
    return subset.iloc[0]
