"""
src/strategy/butterfly.py
==========================
Long Call Butterfly: Buy 1 ATM call, Sell 2 OTM calls, Buy 1 further OTM call.
Best when: RANGE_BOUND or SQUEEZE + low/normal IV.
Max profit if stock pins at body (short) strike at expiry.

Also implements Long Put Butterfly for bearish-neutral setups.

Usage:
    from src.strategy.butterfly import construct_long_butterfly

    fly = construct_long_butterfly("AAPL", current_price, chain, target_dte=30)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.analysis.options_analytics import (
    debit_spread_ev, prob_between, RISK_FREE_RATE,
)
from src.strategy.credit_spread import (
    _add_dte, _pick_expiration, _get_row_by_strike, _snap_strike,
)

logger = logging.getLogger(__name__)


# ─── Butterfly Dataclass ──────────────────────────────────────────────────────

@dataclass
class Butterfly:
    ticker: str
    spread_type: str           # "LONG_CALL_BUTTERFLY" or "LONG_PUT_BUTTERFLY"
    expiration: str
    dte: int
    lower_wing: float          # Long strike (lower)
    body: float                # Short strikes (×2)
    upper_wing: float          # Long strike (upper)
    wing_width: float          # Body − lower_wing (should equal upper_wing − body)
    lower_premium: float       # Long lower wing premium
    body_premium: float        # Short body premium (each)
    upper_premium: float       # Long upper wing premium
    net_debit: float           # Total cost (max risk)
    max_profit: float          # Wing width − net_debit (at expiry, stock = body)
    max_risk: float            # = net_debit
    risk_reward_ratio: float   # max_profit / net_debit
    lower_breakeven: float     # Lower breakeven = lower_wing + net_debit
    upper_breakeven: float     # Upper breakeven = upper_wing − net_debit
    prob_profit: float         # P(stock between breakevens at expiry)
    prob_max_profit: float     # P(stock near body at expiry)
    ev: float
    notes: str = ""


# ─── Main Constructor ─────────────────────────────────────────────────────────

def construct_long_butterfly(
    ticker: str,
    current_price: float,
    chain: pd.DataFrame,
    target_dte: int = 30,
    body_delta: float = 0.50,    # Body (sold) near ATM
    wing_width: float = 5.0,
    option_type: str = "call",
) -> Optional[Butterfly]:
    """
    Construct a long call (or put) butterfly.

    Body (short ×2) near current price.
    Wings (long ×1 each) wing_width away on either side.

    Args:
        ticker: Symbol
        current_price: Latest close
        chain: Filtered options chain
        target_dte: Target DTE
        body_delta: Absolute delta for body strike (~0.50 = ATM)
        wing_width: Distance between body and each wing in dollars
        option_type: "call" or "put"

    Returns:
        Butterfly or None.
    """
    try:
        opts = chain[chain["type"] == option_type].copy()
        if opts.empty:
            return None

        opts = _add_dte(opts)
        exp, actual_dte = _pick_expiration(opts, target_dte)
        if exp is None:
            return None

        exp_opts = opts[opts["expiration"] == exp].sort_values("strike")
        if len(exp_opts) < 3:
            return None

        # ── Body Strike (ATM) ─────────────────────────────────────────────────
        body_row = _pick_body_strike(exp_opts, current_price, body_delta)
        if body_row is None:
            return None

        body_strike = float(body_row["strike"])

        # ── Wing Strikes ──────────────────────────────────────────────────────
        lower_target = body_strike - wing_width
        upper_target = body_strike + wing_width

        lower_strike = _snap_strike(lower_target, exp_opts["strike"].values, direction=-1)
        upper_strike = _snap_strike(upper_target, exp_opts["strike"].values, direction=1)

        lower_row = _get_row_by_strike(exp_opts, lower_strike)
        upper_row = _get_row_by_strike(exp_opts, upper_strike)

        if lower_row is None or upper_row is None:
            return None

        actual_lower = float(lower_row["strike"])
        actual_upper = float(upper_row["strike"])

        # Wings should be equidistant from body
        lower_width = body_strike - actual_lower
        upper_width = actual_upper - body_strike

        if lower_width <= 0 or upper_width <= 0:
            return None

        actual_wing_width = min(lower_width, upper_width)

        # ── Pricing ───────────────────────────────────────────────────────────
        lower_prem = float(lower_row.get("mid", lower_row.get("ask", 0)))
        body_prem  = float(body_row.get("mid", body_row.get("bid", 0)))
        upper_prem = float(upper_row.get("mid", upper_row.get("ask", 0)))

        if lower_prem <= 0:
            lower_prem = float(lower_row.get("ask", 0))
        if upper_prem <= 0:
            upper_prem = float(upper_row.get("ask", 0))

        # Net debit = buy lower + buy upper − sell 2× body
        net_debit = round(lower_prem + upper_prem - 2 * body_prem, 2)

        if net_debit <= 0.05:   # Minimum $0.05 debit
            logger.debug(f"{ticker}: butterfly debit too small ({net_debit})")
            return None

        max_profit = round(actual_wing_width - net_debit, 2)
        if max_profit <= 0:
            return None

        rr_ratio = round(max_profit / net_debit, 2)

        if rr_ratio < 1.0:   # Butterflies need at least 1:1 R/R
            logger.debug(f"{ticker}: butterfly R/R too low ({rr_ratio:.1f})")
            return None

        lower_be = round(actual_lower + net_debit, 2)
        upper_be = round(actual_upper - net_debit, 2)

        # ── Probability Calculations ──────────────────────────────────────────
        iv = float(body_row.get("implied_volatility", 0) or 0)

        if iv > 0 and actual_dte > 0:
            T = actual_dte / 365
            if iv > 1:
                iv = iv / 100
            pop = prob_between(
                current_price, lower_be, upper_be,
                T, iv, RISK_FREE_RATE,
            ) * 100
            # Max profit: stock within 1 wing-width of body
            p_max = prob_between(
                current_price,
                body_strike - actual_wing_width * 0.3,
                body_strike + actual_wing_width * 0.3,
                T, iv, RISK_FREE_RATE,
            ) * 100
        else:
            pop = 50.0
            p_max = 20.0

        ev = debit_spread_ev(net_debit, max_profit, pop / 100)

        spread_type = (
            "LONG_CALL_BUTTERFLY" if option_type == "call"
            else "LONG_PUT_BUTTERFLY"
        )

        return Butterfly(
            ticker=ticker,
            spread_type=spread_type,
            expiration=exp,
            dte=actual_dte,
            lower_wing=actual_lower,
            body=body_strike,
            upper_wing=actual_upper,
            wing_width=actual_wing_width,
            lower_premium=round(lower_prem, 2),
            body_premium=round(body_prem, 2),
            upper_premium=round(upper_prem, 2),
            net_debit=net_debit,
            max_profit=max_profit,
            max_risk=net_debit,
            risk_reward_ratio=rr_ratio,
            lower_breakeven=lower_be,
            upper_breakeven=upper_be,
            prob_profit=round(pop, 1),
            prob_max_profit=round(p_max, 1),
            ev=ev,
        )

    except Exception as exc:
        logger.error(f"Butterfly construction failed for {ticker}: {exc}")
        return None


def _pick_body_strike(
    exp_opts: pd.DataFrame,
    current_price: float,
    target_delta: float = 0.50,
) -> Optional[pd.Series]:
    """Select body (sold) strike closest to ATM."""
    if exp_opts["delta"].abs().sum() > 0:
        exp_opts = exp_opts.copy()
        exp_opts["delta_diff"] = (exp_opts["delta"].abs() - target_delta).abs()
        return exp_opts.loc[exp_opts["delta_diff"].idxmin()]

    exp_opts = exp_opts.copy()
    exp_opts["strike_diff"] = (exp_opts["strike"] - current_price).abs()
    return exp_opts.loc[exp_opts["strike_diff"].idxmin()]
