"""
src/strategy/long_call.py
=========================
Long Call constructor: Buy a single call option.
Best when: STRONG_UPTREND + LOW IV + strong technical confirmation.

Risk: 100% of premium paid.
Reward: Theoretically unlimited.
Breakeven: Strike + premium.

Usage:
    from src.strategy.long_call import construct_long_call
    trade = construct_long_call("AAPL", current_price, chain, target_dte=35)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.analysis.options_analytics import bs_greeks, RISK_FREE_RATE
from src.strategy.credit_spread import _add_dte, _pick_expiration
from config.settings import (
    LONG_OPTION_DELTA,
    LONG_OPTION_MAX_THETA_RATE,
)

logger = logging.getLogger(__name__)


@dataclass
class LongOption:
    """A single long call or long put position."""
    ticker: str
    option_type: str           # "LONG_CALL" or "LONG_PUT"
    expiration: str
    dte: int
    strike: float
    premium: float             # Cost to buy (= max risk per contract)
    max_risk: float            # = premium
    breakeven: float           # strike + premium (call) or strike - premium (put)
    delta: float
    gamma: float
    theta: float               # Daily theta in dollars (negative)
    vega: float
    iv: float                  # Implied volatility of the contract (%)
    theta_rate: float          # |theta| / premium — daily decay as fraction of cost
    prob_profit: float         # Probability of stock exceeding breakeven at expiry
    ev: float                  # Expected value estimate
    implied_move_1sd: float    # 1-SD implied move in dollars
    notes: str = ""


def construct_long_call(
    ticker: str,
    current_price: float,
    chain: pd.DataFrame,
    target_dte: int = 35,
    target_delta: float = LONG_OPTION_DELTA,
    max_theta_rate: float = LONG_OPTION_MAX_THETA_RATE,
) -> Optional[LongOption]:
    """
    Construct a long call position.

    Selects a slightly ITM call (delta ~0.65) to maximize probability
    of profit while maintaining leverage. Filters out contracts where
    daily theta exceeds max_theta_rate of the premium.

    Args:
        ticker: Symbol
        current_price: Latest close
        chain: Filtered options chain DataFrame
        target_dte: Target days to expiration
        target_delta: Target delta (0.60-0.70 recommended)
        max_theta_rate: Maximum daily theta / premium ratio

    Returns:
        LongOption or None if no suitable contract found
    """
    try:
        calls = chain[chain["type"] == "call"].copy()
        if calls.empty:
            return None

        calls = _add_dte(calls)
        exp, actual_dte = _pick_expiration(calls, target_dte)
        if exp is None or actual_dte < 10:
            return None

        exp_calls = calls[calls["expiration"] == exp].sort_values("strike")
        if exp_calls.empty:
            return None

        selected = _pick_call_by_delta(exp_calls, current_price, target_delta)
        if selected is None:
            return None

        strike = float(selected["strike"])
        mid = selected.get("mid", None)
        ask = selected.get("ask", None)

        def _safe_float_val(v, default=0.0):
            if v is None:
                return default
            try:
                val = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
                return val if val > 0 else default
            except Exception:
                return default

        premium = _safe_float_val(mid) or _safe_float_val(ask)
        if premium <= 0:
            return None

        def _g(key, default=0.0):
            v = selected.get(key, default)
            return _safe_float_val(v, default)

        iv = _g("implied_volatility")
        delta_val = _g("delta", target_delta) or target_delta
        gamma_val = _g("gamma")
        theta_val = _g("theta")
        vega_val = _g("vega")

        if iv > 0 and (abs(delta_val) < 0.001 or abs(theta_val) < 0.0001):
            iv_dec = iv / 100 if iv > 1 else iv
            T = actual_dte / 365
            greeks = bs_greeks(current_price, strike, T, RISK_FREE_RATE, iv_dec, "call")
            if abs(delta_val) < 0.001:
                delta_val = greeks.delta
            if abs(theta_val) < 0.0001:
                theta_val = greeks.theta
            if gamma_val == 0:
                gamma_val = greeks.gamma
            if vega_val == 0:
                vega_val = greeks.vega

        theta_rate = abs(theta_val) / premium if premium > 0 else 999.0
        if theta_rate > max_theta_rate:
            logger.debug(
                f"{ticker}: long call rejected — theta rate {theta_rate:.3f} "
                f"> max {max_theta_rate}"
            )
            return None

        breakeven = round(strike + premium, 2)

        iv_dec = iv / 100 if iv > 1 else iv
        if iv_dec > 0.001 and actual_dte > 0:
            from src.analysis.options_analytics import prob_otm
            T = actual_dte / 365
            prob_profit = (1 - prob_otm(
                current_price, breakeven, T, iv_dec, RISK_FREE_RATE, "call"
            )) * 100
            implied_move = round(current_price * iv_dec * math.sqrt(T), 2)
        else:
            prob_profit = round(abs(delta_val) * 100, 1)
            implied_move = 0.0

        p = prob_profit / 100
        if implied_move > 0:
            expected_gain = implied_move * 0.4
            ev = round(p * expected_gain - (1 - p) * premium, 2)
        else:
            ev = round(p * premium * 2 - (1 - p) * premium, 2)

        iv_display = round(iv * 100 if iv < 1 else iv, 1)

        return LongOption(
            ticker=ticker,
            option_type="LONG_CALL",
            expiration=exp,
            dte=actual_dte,
            strike=round(strike, 2),
            premium=round(premium, 2),
            max_risk=round(premium, 2),
            breakeven=breakeven,
            delta=round(float(delta_val), 3),
            gamma=round(float(gamma_val), 5),
            theta=round(float(theta_val), 3),
            vega=round(float(vega_val), 3),
            iv=iv_display,
            theta_rate=round(theta_rate, 4),
            prob_profit=round(prob_profit, 1),
            ev=ev,
            implied_move_1sd=implied_move,
        )

    except Exception as exc:
        logger.error(f"Long call construction failed for {ticker}: {exc}")
        return None


def _pick_call_by_delta(
    exp_calls: pd.DataFrame,
    current_price: float,
    target_delta: float,
) -> Optional[pd.Series]:
    """Select the call closest to target delta, with strike-based fallback."""
    exp_calls = exp_calls.copy()

    # Coerce object/mixed dtype safely before fillna to avoid pandas downcasting warning.
    delta_col = pd.to_numeric(exp_calls["delta"], errors="coerce").fillna(0.0)

    if delta_col.abs().sum() > 0.01:
        exp_calls["_delta_diff"] = (delta_col.abs() - target_delta).abs()
        idx = exp_calls["_delta_diff"].idxmin()
        return exp_calls.loc[idx]

    target_strike = current_price * (1.0 - (target_delta - 0.50) * 0.20)
    exp_calls["_strike_diff"] = (pd.to_numeric(exp_calls["strike"], errors="coerce") - target_strike).abs()
    idx = exp_calls["_strike_diff"].idxmin()
    return exp_calls.loc[idx]
