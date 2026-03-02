"""
src/strategy/iron_condor.py
============================
Iron Condor construction.
Combines a Bull Put Spread + Bear Call Spread on the same underlying.
Best when: RANGE_BOUND or SQUEEZE + high IV.

Structure:
  Sell OTM put  (short put, ~0.16 delta)
  Buy  lower OTM put  (long put, protection)
  Sell OTM call (short call, ~0.16 delta)
  Buy  higher OTM call (long call, protection)

Usage:
    from src.strategy.iron_condor import construct_iron_condor

    ic = construct_iron_condor("SPY", current_price, chain, target_dte=45)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.analysis.options_analytics import (
    credit_spread_ev, prob_between, annualised_return_on_risk,
    RISK_FREE_RATE,
)
from src.strategy.credit_spread import (
    construct_bull_put_spread, construct_bear_call_spread, CreditSpread,
)

logger = logging.getLogger(__name__)


# ─── IronCondor Dataclass ─────────────────────────────────────────────────────

@dataclass
class IronCondor:
    ticker: str
    expiration: str
    dte: int

    # Put wing (bull put spread)
    short_put: float
    long_put: float
    put_credit: float

    # Call wing (bear call spread)
    short_call: float
    long_call: float
    call_credit: float

    # Combined metrics
    total_credit: float        # Sum of both credits
    max_risk: float            # Max wing width − total credit
    max_reward: float          # = total_credit
    risk_reward_ratio: float
    put_breakeven: float       # Short put − total credit
    call_breakeven: float      # Short call + total credit
    profit_zone_width: float   # call breakeven − put breakeven
    prob_profit: float         # % probability stock stays between breakevens
    prob_max_profit: float     # % probability stock stays between short strikes
    annual_ror: float
    ev: float
    put_wing: CreditSpread     # Full put spread details
    call_wing: CreditSpread    # Full call spread details
    notes: str = ""


# ─── Main Constructor ─────────────────────────────────────────────────────────

def construct_iron_condor(
    ticker: str,
    current_price: float,
    chain: pd.DataFrame,
    target_dte: int = 45,
    wing_delta: float = 0.16,     # Each wing at ~0.16 delta
    spread_width: float = 5.0,
    min_credit_ratio: float = 0.10,  # Min credit as % of max risk (>=10%)
) -> Optional[IronCondor]:
    """
    Construct an Iron Condor from an options chain.

    Both wings use the same expiration. Wings are built independently
    using the credit_spread constructors.

    Args:
        ticker: Symbol
        current_price: Latest close price
        chain: Options chain (pre-filtered for liquidity)
        target_dte: Target DTE (will find nearest)
        wing_delta: Delta target for both short strikes (0.15–0.20 typical)
        spread_width: Strike width for each wing
        min_credit_ratio: Reject if total_credit / max_risk < this threshold

    Returns:
        IronCondor or None if construction fails or credits are too thin.
    """
    try:
        # Build put wing
        put_wing = construct_bull_put_spread(
            ticker, current_price, chain,
            target_dte=target_dte,
            target_delta=wing_delta,
            spread_width=spread_width,
        )
        if put_wing is None:
            logger.debug(f"{ticker}: IC put wing failed")
            return None

        # Filter chain to same expiration for call wing
        chain_same_exp = chain[chain["expiration"] == put_wing.expiration]
        if chain_same_exp.empty:
            chain_same_exp = chain

        # Build call wing
        call_wing = construct_bear_call_spread(
            ticker, current_price, chain_same_exp,
            target_dte=target_dte,
            target_delta=wing_delta,
            spread_width=spread_width,
        )
        if call_wing is None:
            logger.debug(f"{ticker}: IC call wing failed")
            return None

        # Validate wings don't overlap
        if call_wing.short_strike <= put_wing.short_strike:
            logger.debug(f"{ticker}: IC wings overlap")
            return None

        # ── Combined Metrics ──────────────────────────────────────────────────
        total_credit = round(put_wing.net_credit + call_wing.net_credit, 2)

        # Max risk = wider wing width − total credit
        put_width  = put_wing.width
        call_width = call_wing.width
        max_wing_width = max(put_width, call_width)
        max_risk = round(max_wing_width - total_credit, 2)

        if max_risk <= 0:
            return None

        # Reject if credit is too thin (poor R/R)
        if total_credit / (max_wing_width) < min_credit_ratio:
            logger.debug(
                f"{ticker}: IC credit too thin "
                f"({total_credit:.2f} / {max_wing_width:.0f} = "
                f"{total_credit/max_wing_width:.1%})"
            )
            return None

        rr_ratio = round(max_risk / total_credit, 2)

        put_breakeven  = round(put_wing.short_strike  - total_credit, 2)
        call_breakeven = round(call_wing.short_strike + total_credit, 2)
        profit_zone_width = round(call_breakeven - put_breakeven, 2)

        # ── Probability Calculations ──────────────────────────────────────────
        iv = _estimate_iv(chain, put_wing.expiration, current_price)
        actual_dte = put_wing.dte

        if iv > 0 and actual_dte > 0:
            T = actual_dte / 365
            if iv > 1:
                iv = iv / 100
            # P(stock between put and call breakevens at expiry)
            pop = prob_between(
                current_price,
                put_breakeven, call_breakeven,
                T, iv, RISK_FREE_RATE,
            ) * 100
            # P(max profit zone: between short strikes)
            p_max = prob_between(
                current_price,
                put_wing.short_strike, call_wing.short_strike,
                T, iv, RISK_FREE_RATE,
            ) * 100
        else:
            # Rough estimate from wing deltas
            pop = round((1 - put_wing.short_delta - call_wing.short_delta) * 100, 1)
            p_max = max(pop - 10, 0)

        ev = credit_spread_ev(total_credit, max_risk, pop / 100)
        ror = annualised_return_on_risk(total_credit, max_risk, actual_dte)

        return IronCondor(
            ticker=ticker,
            expiration=put_wing.expiration,
            dte=actual_dte,
            short_put=put_wing.short_strike,
            long_put=put_wing.long_strike,
            put_credit=put_wing.net_credit,
            short_call=call_wing.short_strike,
            long_call=call_wing.long_strike,
            call_credit=call_wing.net_credit,
            total_credit=total_credit,
            max_risk=max_risk,
            max_reward=total_credit,
            risk_reward_ratio=rr_ratio,
            put_breakeven=put_breakeven,
            call_breakeven=call_breakeven,
            profit_zone_width=profit_zone_width,
            prob_profit=round(pop, 1),
            prob_max_profit=round(p_max, 1),
            annual_ror=ror,
            ev=ev,
            put_wing=put_wing,
            call_wing=call_wing,
        )

    except Exception as exc:
        logger.error(f"Iron condor construction failed for {ticker}: {exc}")
        return None


# ─── Helper ───────────────────────────────────────────────────────────────────

def _estimate_iv(
    chain: pd.DataFrame,
    expiration: str,
    current_price: float,
) -> float:
    """Estimate ATM IV from the chain at the target expiration."""
    exp_chain = chain[chain["expiration"] == expiration]
    if exp_chain.empty:
        exp_chain = chain
    atm = exp_chain.iloc[(exp_chain["strike"] - current_price).abs().argsort()[:4]]
    iv_vals = atm["implied_volatility"].replace(0, float("nan")).dropna()
    if len(iv_vals) == 0:
        return 0.0
    return float(iv_vals.mean())
