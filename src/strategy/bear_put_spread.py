"""
src/strategy/bear_put_spread.py
===============================
Bear Put Spread (debit): Buy ATM/near-ATM put, Sell lower-strike OTM put.
Best when: DOWNTREND or STRONG_DOWNTREND + low/normal IV.
Max profit = width − debit. Max loss = net debit.

Usage:
    from src.strategy.bear_put_spread import construct_bear_put_spread

    trade = construct_bear_put_spread("NVDA", current_price, chain, target_dte=21)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.options_analytics import (
    debit_spread_ev, RISK_FREE_RATE, prob_otm,
)
from src.strategy.bull_call_spread import DebitSpread
from src.strategy.credit_spread import (
    _add_dte, _pick_expiration, _get_row_by_strike, _snap_strike,
)

logger = logging.getLogger(__name__)


def construct_bear_put_spread(
    ticker: str,
    current_price: float,
    chain: pd.DataFrame,
    target_dte: int = 21,
    long_delta: float = 0.50,   # Buy near-ATM put (|delta| ≈ 0.50)
    spread_width: float = 5.0,
) -> Optional[DebitSpread]:
    """
    Bear Put Spread:
    - Buy put at ~0.50 delta (near ATM)
    - Sell put spread_width lower

    Args:
        ticker: Symbol
        current_price: Latest close
        chain: Filtered options chain
        target_dte: Target DTE
        long_delta: Absolute delta for long put (0.45–0.55 = ATM)
        spread_width: Strike separation in dollars

    Returns:
        DebitSpread or None.
    """
    try:
        puts = chain[chain["type"] == "put"].copy()
        if puts.empty:
            return None

        puts = _add_dte(puts)
        exp, actual_dte = _pick_expiration(puts, target_dte)
        if exp is None:
            return None

        exp_puts = puts[puts["expiration"] == exp].sort_values("strike")
        if len(exp_puts) < 2:
            return None

        # ── Long Put (ATM / near-ATM) ─────────────────────────────────────────
        long_row = _pick_long_put(exp_puts, current_price, long_delta)
        if long_row is None:
            return None

        long_strike = float(long_row["strike"])
        short_target = long_strike - spread_width
        short_strike = _snap_strike(short_target, exp_puts["strike"].values, direction=-1)

        short_row = _get_row_by_strike(exp_puts, short_strike)
        if short_row is None:
            lower = exp_puts[exp_puts["strike"] < long_strike]
            if lower.empty:
                return None
            short_row = lower.iloc[-1]   # Highest strike below long

        actual_short_strike = float(short_row["strike"])
        width = round(long_strike - actual_short_strike, 2)

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

        breakeven = round(long_strike - net_debit, 2)
        rr_ratio  = round(max_profit / net_debit, 2)

        # ── Probability of Profit ─────────────────────────────────────────────
        iv = float(long_row.get("implied_volatility", 0) or 0)
        long_delta_val = abs(float(long_row.get("delta", long_delta) or long_delta))

        if iv > 0 and actual_dte > 0:
            T = actual_dte / 365
            if iv > 1:
                iv = iv / 100
            # P(stock < breakeven at expiry)
            pop = prob_otm(current_price, breakeven, T, iv, RISK_FREE_RATE, "call") * 100
        else:
            pop = round(long_delta_val * 100, 1)

        ev = debit_spread_ev(net_debit, max_profit, pop / 100)

        return DebitSpread(
            ticker=ticker,
            spread_type="BEAR_PUT",
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
        logger.error(f"Bear put spread construction failed for {ticker}: {exc}")
        return None


def _pick_long_put(
    exp_puts: pd.DataFrame,
    current_price: float,
    target_delta: float = 0.50,
) -> Optional[pd.Series]:
    """Select the long put strike — ATM or slight OTM based on |delta|."""
    if exp_puts["delta"].abs().sum() > 0:
        exp_puts = exp_puts.copy()
        exp_puts["delta_diff"] = (exp_puts["delta"].abs() - target_delta).abs()
        return exp_puts.loc[exp_puts["delta_diff"].idxmin()]

    # Fallback: strike closest to current price
    exp_puts = exp_puts.copy()
    exp_puts["strike_diff"] = (exp_puts["strike"] - current_price).abs()
    return exp_puts.loc[exp_puts["strike_diff"].idxmin()]
