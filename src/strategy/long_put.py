"""
src/strategy/long_put.py
========================
Long Put constructor: Buy a single put option.
Best when: STRONG_DOWNTREND + LOW IV + strong technical confirmation.

Risk: 100% of premium paid.
Reward: Strike - premium (floor at zero).
Breakeven: Strike - premium.

Usage:
    from src.strategy.long_put import construct_long_put
    trade = construct_long_put("NVDA", current_price, chain, target_dte=35)
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from src.analysis.options_analytics import bs_greeks, RISK_FREE_RATE
from src.strategy.credit_spread import _add_dte, _pick_expiration
from src.strategy.long_call import LongOption
from config.settings import (
    LONG_OPTION_DELTA, LONG_OPTION_MAX_THETA_RATE,
)

logger = logging.getLogger(__name__)


def construct_long_put(
    ticker: str,
    current_price: float,
    chain: pd.DataFrame,
    target_dte: int = 35,
    target_delta: float = LONG_OPTION_DELTA,
    max_theta_rate: float = LONG_OPTION_MAX_THETA_RATE,
) -> Optional[LongOption]:
    """
    Construct a long put position.

    Selects a slightly ITM put (|delta| ~0.65) to maximize probability
    of profit while maintaining leverage.

    Args:
        ticker: Symbol
        current_price: Latest close
        chain: Filtered options chain DataFrame
        target_dte: Target days to expiration
        target_delta: Target |delta| (0.60-0.70 recommended)
        max_theta_rate: Maximum daily theta / premium ratio

    Returns:
        LongOption or None if no suitable contract found
    """
    try:
        puts = chain[chain["type"] == "put"].copy()
        if puts.empty:
            return None

        puts = _add_dte(puts)
        exp, actual_dte = _pick_expiration(puts, target_dte)
        if exp is None or actual_dte < 10:
            return None

        exp_puts = puts[puts["expiration"] == exp].sort_values("strike")
        if exp_puts.empty:
            return None

        # ── Select Strike by Delta ────────────────────────────────────────────
        selected = _pick_put_by_delta(exp_puts, current_price, target_delta)
        if selected is None:
            return None

        strike = float(selected["strike"])

        def _safe_float_val(v, default=0.0):
            if v is None:
                return default
            try:
                val = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
                return val if val > 0 else default
            except Exception:
                return default

        mid = selected.get("mid", None)
        ask = selected.get("ask", None)
        premium = _safe_float_val(mid) or _safe_float_val(ask)
        if premium <= 0:
            return None

        # ── Greeks ────────────────────────────────────────────────────────────
        def _g(key, default=0.0):
            v = selected.get(key, default)
            return _safe_float_val(v, default)

        iv = _g("implied_volatility")
        delta_raw = selected.get("delta", 0)
        delta_val = abs(_safe_float_val(delta_raw, target_delta)) or target_delta
        gamma_val = _g("gamma")
        theta_val = _g("theta")
        vega_val = _g("vega")

        if iv > 0 and (delta_val < 0.001 or abs(theta_val) < 0.0001):
            iv_dec = iv / 100 if iv > 1 else iv
            T = actual_dte / 365
            greeks = bs_greeks(current_price, strike, T, RISK_FREE_RATE, iv_dec, "put")
            if delta_val < 0.001:
                delta_val = abs(greeks.delta)
            if abs(theta_val) < 0.0001:
                theta_val = greeks.theta
            if gamma_val == 0:
                gamma_val = greeks.gamma
            if vega_val == 0:
                vega_val = greeks.vega

        # ── Theta Rate Filter ─────────────────────────────────────────────────
        theta_rate = abs(theta_val) / premium if premium > 0 else 999.0
        if theta_rate > max_theta_rate:
            logger.debug(
                f"{ticker}: long put rejected — theta rate {theta_rate:.3f} "
                f"> max {max_theta_rate}"
            )
            return None

        # ── Breakeven & Probability ───────────────────────────────────────────
        breakeven = round(strike - premium, 2)

        iv_dec = iv / 100 if iv > 1 else iv
        if iv_dec > 0.001 and actual_dte > 0:
            from src.analysis.options_analytics import prob_otm
            T = actual_dte / 365
            # P(put profitable) = P(stock < breakeven) = N(-d1) at the breakeven strike
            prob_profit = prob_otm(
                current_price, breakeven, T, iv_dec, RISK_FREE_RATE, "call"
            ) * 100
            implied_move = round(current_price * iv_dec * math.sqrt(T), 2)
        else:
            prob_profit = round(delta_val * 100, 1)
            implied_move = 0.0

        # ── Expected Value ────────────────────────────────────────────────────
        p = prob_profit / 100
        if implied_move > 0:
            expected_gain = implied_move * 0.4
            ev = round(p * expected_gain - (1 - p) * premium, 2)
        else:
            ev = round(p * premium * 2 - (1 - p) * premium, 2)

        iv_display = round(iv * 100 if iv < 1 else iv, 1)

        return LongOption(
            ticker=ticker,
            option_type="LONG_PUT",
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
        logger.error(f"Long put construction failed for {ticker}: {exc}")
        return None


def _pick_put_by_delta(
    exp_puts: pd.DataFrame,
    current_price: float,
    target_delta: float,
) -> Optional[pd.Series]:
    """Select the put closest to target |delta|, with strike-based fallback."""
    exp_puts = exp_puts.copy()

    delta_col = exp_puts["delta"].fillna(0)
    if delta_col.abs().sum() > 0.01:
        exp_puts["_delta_diff"] = (delta_col.abs() - target_delta).abs()
        idx = exp_puts["_delta_diff"].idxmin()
        return exp_puts.loc[idx]

    # Fallback: slightly ITM put (above current price)
    target_strike = current_price * (1.0 + (target_delta - 0.50) * 0.20)
    exp_puts["_strike_diff"] = (exp_puts["strike"] - target_strike).abs()
    idx = exp_puts["_strike_diff"].idxmin()
    return exp_puts.loc[idx]
