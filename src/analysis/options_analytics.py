"""
src/analysis/options_analytics.py
==================================
Options pricing utilities: Black-Scholes Greeks, probability calculations,
spread P&L analysis, expected value.

Uses scipy for Black-Scholes — no external C library required.

Usage:
    from src.analysis.options_analytics import (
        bs_greeks, prob_otm, expected_value_credit_spread
    )
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

from config.settings import RISK_FREE_RATE  # P2 FIX #8: now configurable via .env

logger = logging.getLogger(__name__)

# RISK_FREE_RATE is imported from config.settings (set via RISK_FREE_RATE env var)


# ─── Black-Scholes Greeks ─────────────────────────────────────────────────────

@dataclass
class BSGreeks:
    """Black-Scholes option price and Greeks."""
    price: float
    delta: float    # dV/dS
    gamma: float    # d²V/dS²
    theta: float    # dV/dt (per calendar day, in dollar terms)
    vega: float     # dV/dσ (per 1% move in IV)
    rho: float      # dV/dr
    option_type: str
    iv: float


def bs_greeks(
    S: float,       # Current stock price
    K: float,       # Strike price
    T: float,       # Time to expiration (years)
    r: float,       # Risk-free rate
    sigma: float,   # Implied volatility (annual, decimal e.g. 0.25)
    option_type: str = "call",
) -> BSGreeks:
    """
    Compute Black-Scholes price and Greeks for a European option.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years (e.g. 30/365)
        r: Risk-free rate (annual decimal)
        sigma: Implied volatility (annual decimal)
        option_type: "call" or "put"

    Returns:
        BSGreeks dataclass
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return BSGreeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, option_type, sigma)

    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type.lower() == "call":
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        theta = (
            -(S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))) -
            r * K * math.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)
        ) / 365   # Per calendar day

        vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # Per 1% change in IV

        return BSGreeks(
            price=round(price, 4),
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 4),
            vega=round(vega, 4),
            rho=round(rho, 4),
            option_type=option_type,
            iv=sigma,
        )

    except Exception as exc:
        logger.debug(f"BS greeks error (S={S}, K={K}, T={T}, σ={sigma}): {exc}")
        return BSGreeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, option_type, sigma)


# ─── Probability Calculations ─────────────────────────────────────────────────

def prob_otm(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = RISK_FREE_RATE,
    option_type: str = "call",
) -> float:
    """
    Probability that an option expires OTM (out of the money).
    = Probability of profit for a short option position.

    Based on risk-neutral (log-normal) assumption.
    """
    if T <= 0 or sigma <= 0:
        return 1.0 if (option_type == "call" and S < K) or (option_type == "put" and S > K) else 0.0

    try:
        d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if option_type.lower() == "call":
            return round(float(norm.cdf(-d2)), 4)   # P(S_T < K)
        else:
            return round(float(norm.cdf(d2)), 4)    # P(S_T > K)
    except Exception:
        return 0.5


def prob_between(
    S: float,
    K_low: float,
    K_high: float,
    T: float,
    sigma: float,
    r: float = RISK_FREE_RATE,
) -> float:
    """
    Probability that stock price ends between K_low and K_high at expiry.
    Used for Iron Condor and Butterfly profit zone estimation.
    """
    if T <= 0 or sigma <= 0:
        return float(K_low <= S <= K_high)

    try:
        # d2 = (log(S/K) + (r - σ²/2)T) / (σ√T)
        # P(S_T < K) = N(-d2)  (log-normal risk-neutral measure)
        # P(K_low < S_T < K_high) = P(S_T < K_high) - P(S_T < K_low)
        #                         = N(-d2_high) - N(-d2_low)
        d2_high = (math.log(S / K_high) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2_low  = (math.log(S / K_low)  + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return round(float(norm.cdf(-d2_high) - norm.cdf(-d2_low)), 4)
    except Exception:
        return 0.5


def prob_touch(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = RISK_FREE_RATE,
) -> float:
    """
    Approximate probability that stock TOUCHES a level K before expiry.
    Uses the reflection principle: P(touch) ≈ 2 × P(expire beyond K).
    """
    d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if K > S:
        p_expire_above = float(norm.cdf(-d2))
        return round(min(2 * p_expire_above, 1.0), 4)
    else:
        p_expire_below = float(norm.cdf(d2))
        return round(min(2 * p_expire_below, 1.0), 4)


# ─── Spread P&L Analysis ──────────────────────────────────────────────────────

def credit_spread_ev(
    net_credit: float,
    max_risk: float,
    prob_profit: float,
) -> float:
    """
    Expected value of a credit spread.
    EV = (prob_profit × max_reward) − (prob_loss × max_risk)

    Args:
        net_credit: Credit received (= max reward)
        max_risk: Maximum possible loss (width − credit)
        prob_profit: Probability of full profit (0-1)

    Returns:
        Expected value per contract (dollar)
    """
    prob_loss = 1.0 - prob_profit
    return round(prob_profit * net_credit - prob_loss * max_risk, 3)


def debit_spread_ev(
    net_debit: float,
    max_profit: float,
    prob_profit: float,
) -> float:
    """Expected value of a debit spread."""
    prob_loss = 1.0 - prob_profit
    return round(prob_profit * max_profit - prob_loss * net_debit, 3)


def annualised_return_on_risk(
    net_credit: float,
    max_risk: float,
    dte: int,
) -> float:
    """
    Annualised return on risk for a credit spread.
    = (credit / max_risk) × (365 / dte) × 100%
    """
    if max_risk <= 0 or dte <= 0:
        return 0.0
    return round((net_credit / max_risk) * (365 / dte) * 100, 1)


# ─── Implied Move Estimate ────────────────────────────────────────────────────

def implied_move(
    current_price: float,
    iv: float,          # Annual IV (decimal, e.g. 0.25)
    dte: int,
) -> float:
    """
    Expected 1-SD price move by expiry based on IV.
    = S × IV × sqrt(DTE / 365)

    Returns:
        Dollar amount of 1 standard deviation move.
    """
    if iv <= 0 or dte <= 0:
        return 0.0
    return round(current_price * iv * math.sqrt(dte / 365), 2)


def implied_move_pct(
    current_price: float,
    iv: float,
    dte: int,
) -> float:
    """Implied move as a percentage of current price."""
    move = implied_move(current_price, iv, dte)
    return round(move / current_price * 100, 2) if current_price > 0 else 0.0
