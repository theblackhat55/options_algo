"""
src/risk/position_sizer.py
==========================
Position sizing using Kelly Criterion (fractional) and fixed-risk approach.
Ensures no single trade exceeds MAX_RISK_PER_TRADE_PCT of account.

Usage:
    from src.risk.position_sizer import size_position, PositionSize
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import MAX_RISK_PER_TRADE_PCT, MAX_POSITIONS

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    ticker: str
    contracts: int
    max_risk_per_contract: float
    total_risk: float
    total_credit_or_debit: float
    account_risk_pct: float
    sizing_method: str


def size_position(
    ticker: str,
    max_risk_per_contract: float,   # Max loss per contract (in $)
    net_credit_or_debit: float,     # Credit received or debit paid per contract
    account_size: float,            # Total account equity
    prob_profit: float,             # 0-100 probability of profit
    confidence: float = 0.5,        # Strategy confidence score 0-1
    method: str = "fixed_pct",      # "fixed_pct" | "fractional_kelly" | "min"
    max_risk_pct: float = None,
) -> PositionSize:
    """
    Calculate position size in number of contracts.

    Options contracts control 100 shares, so:
    - Max risk per contract = max_risk_per_contract × 100

    Args:
        ticker: Symbol
        max_risk_per_contract: Dollar risk per 1 contract (× 100 shares)
        net_credit_or_debit: Premium per contract
        account_size: Total account equity
        prob_profit: Probability of profit (0-100)
        confidence: Strategy confidence 0-1 (used for Kelly scaling)
        method: Sizing algorithm to use
        max_risk_pct: Override MAX_RISK_PER_TRADE_PCT if provided

    Returns:
        PositionSize with number of contracts and risk metrics
    """
    if max_risk_pct is None:
        max_risk_pct = MAX_RISK_PER_TRADE_PCT

    # Dollar risk per contract (broker convention: 1 contract = 100 shares)
    risk_per_contract_dollars = max_risk_per_contract * 100
    if risk_per_contract_dollars <= 0:
        return _zero_size(ticker, "zero_risk")

    # Maximum dollar risk allowed on this trade
    max_dollar_risk = account_size * (max_risk_pct / 100)

    if method == "fractional_kelly":
        contracts = _fractional_kelly(
            account_size, risk_per_contract_dollars,
            net_credit_or_debit * 100, prob_profit / 100,
            confidence, max_dollar_risk,
        )
    else:
        # Simple fixed percentage
        contracts = int(max_dollar_risk / risk_per_contract_dollars)

    # Hard floor and ceiling
    contracts = max(1, min(contracts, 10))   # 1–10 contracts

    total_risk = round(contracts * risk_per_contract_dollars, 2)
    total_col   = round(contracts * net_credit_or_debit * 100, 2)

    return PositionSize(
        ticker=ticker,
        contracts=contracts,
        max_risk_per_contract=max_risk_per_contract,
        total_risk=total_risk,
        total_credit_or_debit=total_col,
        account_risk_pct=round(total_risk / account_size * 100, 2),
        sizing_method=method,
    )


def _fractional_kelly(
    account_size: float,
    risk_per_contract: float,
    reward_per_contract: float,
    p: float,
    confidence: float,
    max_dollar_risk: float,
    kelly_fraction: float = 0.25,   # Quarter-Kelly for safety
) -> int:
    """Kelly Criterion scaled by confidence and quarter-Kelly."""
    if reward_per_contract <= 0 or risk_per_contract <= 0:
        return 1

    b = reward_per_contract / risk_per_contract   # Win/loss ratio
    q = 1 - p

    kelly_pct = (b * p - q) / b if b > 0 else 0

    # Apply fraction and confidence scaling
    kelly_pct = kelly_pct * kelly_fraction * confidence

    if kelly_pct <= 0:
        return 1

    kelly_dollars = account_size * kelly_pct
    capped = min(kelly_dollars, max_dollar_risk)
    contracts = int(capped / risk_per_contract)
    return max(1, contracts)


def _zero_size(ticker: str, reason: str) -> PositionSize:
    return PositionSize(
        ticker=ticker,
        contracts=0,
        max_risk_per_contract=0,
        total_risk=0,
        total_credit_or_debit=0,
        account_risk_pct=0,
        sizing_method=reason,
    )
