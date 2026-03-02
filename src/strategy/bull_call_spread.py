"""
src/strategy/bull_call_spread.py
================================
Bull Call Spread (debit): Buy ATM/near-ATM call, Sell OTM call.
Best when: UPTREND or STRONG_UPTREND + low/normal IV.
Max profit at short call strike. Max loss = net debit paid.

Usage:
    from src.strategy.bull_call_spread import construct_bull_call_spread

    trade = construct_bull_call_spread("AAPL", current_price, chain, target_dte=21)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.options_analytics import (
    debit_spread_ev, RISK_FREE_RATE, prob_otm,
)
from src.strategy.credit_spread import (
    _add_dte, _pick_expiration, _get_row_by_strike, _snap_strike,
)

logger = logging.getLogger(__name__)


@dataclass
class DebitSpread:
    ticker: str
    spread_type: str           # "BULL_CALL" or "BEAR_PUT"
    expiration: str
    dte: int
    long_strike: float
    short_strike: float
    long_premium: float
    short_premium: float
    net_debit: float           # Maximum risk
    max_profit: float          # Width − debit
    max_risk: float            # = net_debit
    risk_reward_ratio: float   # max_profit / net_debit (higher = better)
    breakeven: float
    prob_profit: float         # % probability stock ends above breakeven
    long_delta: float
    width: float
    ev: float
    notes: str = ""


def construct_bull_call_spread(
    ticker: str,
    current_price: float,
    chain: pd.DataFrame,
    target_dte: int = 21,
    long_delta: float = 0.50,   # Buy near-ATM call
    spread_width: float = 5.0,
) -> Optional[DebitSpread]:
    """
    Bull Call Spread:
    - Buy call at ~0.50 delta (near ATM)
    - Sell call spread_width higher

    Args:
        ticker: Symbol
        current_price: Latest close
        chain: Filtered options chain
        target_dte: Target DTE (will find nearest available)
        long_delta: Target delta for the long call (0.45–0.55 = ATM)
        spread_width: Strike separation in dollars

    Returns:
        DebitSpread or None if no valid setup found.
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

        # ── Long Call (ATM / near-ATM) ────────────────────────────────────────
        long_row = _pick_long_call(exp_calls, current_price, long_delta)
        if long_row is None:
            return None

        long_strike = float(long_row["strike"])
        short_target = long_strike + spread_width
        short_strike = _snap_strike(short_target, exp_calls["strike"].values, direction=1)

        short_row = _get_row_by_strike(exp_calls, short_strike)
        if short_row is None:
            higher = exp_calls[exp_calls["strike"] > long_strike]
            if higher.empty:
                return None
            short_row = higher.iloc[0]

        actual_short_strike = float(short_row["strike"])
        width = round(actual_short_strike - long_strike, 2)

        if width <= 0:
            return None

        # ── Pricing ───────────────────────────────────────────────────────────
        long_premium  = float(long_row.get("mid", long_row.get("ask", 0)))
        short_premium = float(short_row.get("mid", short_row.get("bid", 0)))

        if long_premium <= 0:
            long_premium = float(long_row.get("ask", 0))

        net_debit = round(long_premium - short_premium, 2)
        if net_debit <= 0.20:
            return None

        max_profit = round(width - net_debit, 2)
        if max_profit <= 0:
            return None

        breakeven = round(long_strike + net_debit, 2)
        rr_ratio  = round(max_profit / net_debit, 2)

        # ── Probability of Profit ─────────────────────────────────────────────
        iv = float(long_row.get("implied_volatility", 0) or 0)
        long_delta_val = float(long_row.get("delta", long_delta) or long_delta)

        if iv > 0 and actual_dte > 0:
            T = actual_dte / 365
            if iv > 1:
                iv = iv / 100
            # P(stock > breakeven at expiry)
            pop = (1 - prob_otm(current_price, breakeven, T, iv, RISK_FREE_RATE, "call")) * 100
        else:
            pop = round(long_delta_val * 100, 1)

        ev = debit_spread_ev(net_debit, max_profit, pop / 100)

        return DebitSpread(
            ticker=ticker,
            spread_type="BULL_CALL",
            expiration=exp,
            dte=actual_dte,
            long_strike=long_strike,
            short_strike=actual_short_strike,
            long_premium=round(long_premium, 2),
            short_premium=round(short_premium, 2),
            net_debit=net_debit,
            max_profit=max_profit,
            max_risk=net_debit,
            risk_reward_ratio=rr_ratio,
            breakeven=breakeven,
            prob_profit=round(pop, 1),
            long_delta=round(long_delta_val, 3),
            width=width,
            ev=ev,
        )

    except Exception as exc:
        logger.error(f"Bull call spread construction failed for {ticker}: {exc}")
        return None


def _pick_long_call(
    exp_calls: pd.DataFrame,
    current_price: float,
    target_delta: float = 0.50,
) -> Optional[pd.Series]:
    """Select the long call strike — ATM or slight OTM based on delta."""
    if exp_calls["delta"].abs().sum() > 0:
        exp_calls = exp_calls.copy()
        exp_calls["delta_diff"] = (exp_calls["delta"].abs() - target_delta).abs()
        return exp_calls.loc[exp_calls["delta_diff"].idxmin()]

    # Fallback: strike closest to current price
    exp_calls = exp_calls.copy()
    exp_calls["strike_diff"] = (exp_calls["strike"] - current_price).abs()
    return exp_calls.loc[exp_calls["strike_diff"].idxmin()]
