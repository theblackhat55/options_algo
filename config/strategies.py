"""
config/strategies.py
====================
Strategy definitions, parameters, and entry/exit rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config.settings import (
    DEFAULT_DTE_PREMIUM_SELL, DEFAULT_DTE_DIRECTIONAL,
    DEFAULT_DTE_BUTTERFLY, DEFAULT_DTE_IC,
    DEFAULT_SPREAD_WIDTH, DEFAULT_SHORT_DELTA,
    IC_WING_DELTA, PROFIT_TARGET_PCT, STOP_LOSS_PCT,
    DEFAULT_DTE_LONG_OPTION, LONG_OPTION_DELTA,
    LONG_OPTION_PROFIT_TARGET_PCT, LONG_OPTION_STOP_LOSS_PCT,
)


@dataclass
class StrategyConfig:
    """Configuration for a single strategy type."""
    name: str
    display_name: str
    category: str            # "debit" or "credit"
    default_dte: int
    target_short_delta: float
    spread_width: float
    profit_target_pct: float
    stop_loss_pct: float
    min_credit: Optional[float] = None    # Minimum credit to accept (credit spreads)
    min_rr_ratio: Optional[float] = None  # Minimum reward/risk (debit spreads)
    notes: str = ""


STRATEGY_CONFIGS: dict[str, StrategyConfig] = {
    "BULL_CALL_SPREAD": StrategyConfig(
        name="BULL_CALL_SPREAD",
        display_name="Bull Call Spread",
        category="debit",
        default_dte=DEFAULT_DTE_DIRECTIONAL,
        target_short_delta=0.50,   # Long ATM call
        spread_width=DEFAULT_SPREAD_WIDTH,
        profit_target_pct=50.0,    # Close at 50% of max profit
        stop_loss_pct=50.0,        # Close if debit doubles
        min_rr_ratio=1.0,          # At least 1:1 reward/risk
        notes="Use in uptrend + low/normal IV. Long ATM call, short OTM call.",
    ),
    "BEAR_PUT_SPREAD": StrategyConfig(
        name="BEAR_PUT_SPREAD",
        display_name="Bear Put Spread",
        category="debit",
        default_dte=DEFAULT_DTE_DIRECTIONAL,
        target_short_delta=0.50,   # Long ATM put
        spread_width=DEFAULT_SPREAD_WIDTH,
        profit_target_pct=50.0,
        stop_loss_pct=50.0,
        min_rr_ratio=1.0,
        notes="Use in downtrend + low/normal IV. Long ATM put, short OTM put.",
    ),
    "BULL_PUT_SPREAD": StrategyConfig(
        name="BULL_PUT_SPREAD",
        display_name="Bull Put Spread (Credit)",
        category="credit",
        default_dte=DEFAULT_DTE_PREMIUM_SELL,
        target_short_delta=DEFAULT_SHORT_DELTA,   # 0.25 delta short put
        spread_width=DEFAULT_SPREAD_WIDTH,
        profit_target_pct=PROFIT_TARGET_PCT,       # 50%
        stop_loss_pct=STOP_LOSS_PCT,               # 100% of credit (2x credit received)
        min_credit=0.50,
        notes="Use in uptrend + high IV. Sell OTM put, buy further OTM put. "
              "Theta decay is primary profit driver.",
    ),
    "BEAR_CALL_SPREAD": StrategyConfig(
        name="BEAR_CALL_SPREAD",
        display_name="Bear Call Spread (Credit)",
        category="credit",
        default_dte=DEFAULT_DTE_PREMIUM_SELL,
        target_short_delta=DEFAULT_SHORT_DELTA,
        spread_width=DEFAULT_SPREAD_WIDTH,
        profit_target_pct=PROFIT_TARGET_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        min_credit=0.50,
        notes="Use in downtrend + high IV. Sell OTM call, buy further OTM call.",
    ),
    "IRON_CONDOR": StrategyConfig(
        name="IRON_CONDOR",
        display_name="Iron Condor",
        category="credit",
        default_dte=DEFAULT_DTE_IC,
        target_short_delta=IC_WING_DELTA,   # 0.16 delta on each wing
        spread_width=DEFAULT_SPREAD_WIDTH,
        profit_target_pct=50.0,
        stop_loss_pct=200.0,                # Close if loss = 2x credit received
        min_credit=1.00,                    # Min $1.00 total credit
        notes="Use in range-bound + high IV. Sell OTM call spread + OTM put spread.",
    ),
    "LONG_BUTTERFLY": StrategyConfig(
        name="LONG_BUTTERFLY",
        display_name="Long Call Butterfly",
        category="debit",
        default_dte=DEFAULT_DTE_BUTTERFLY,
        target_short_delta=0.50,            # Body at ATM
        spread_width=DEFAULT_SPREAD_WIDTH,
        profit_target_pct=40.0,             # Butterflies: take 40% of max
        stop_loss_pct=50.0,
        min_rr_ratio=3.0,                   # Butterflies need good R/R
        notes="Use in range-bound + low IV + Bollinger squeeze. "
              "Buy 1 ATM call, sell 2 OTM calls, buy 1 further OTM call.",
    ),
    # P2 FIX #9: add LONG_CALL and LONG_PUT entries so get_strategy_config() works
    "LONG_CALL": StrategyConfig(
        name="LONG_CALL",
        display_name="Long Call",
        category="debit",
        default_dte=DEFAULT_DTE_LONG_OPTION,
        target_short_delta=LONG_OPTION_DELTA,   # 0.65 delta (in-the-money)
        spread_width=0.0,                       # single-leg, no spread
        profit_target_pct=LONG_OPTION_PROFIT_TARGET_PCT,   # 100%
        stop_loss_pct=LONG_OPTION_STOP_LOSS_PCT,           # 50%
        notes="Long ITM call for bullish directional. Low IV rank only (<40%).",
    ),
    "LONG_PUT": StrategyConfig(
        name="LONG_PUT",
        display_name="Long Put",
        category="debit",
        default_dte=DEFAULT_DTE_LONG_OPTION,
        target_short_delta=LONG_OPTION_DELTA,   # 0.65 delta
        spread_width=0.0,
        profit_target_pct=LONG_OPTION_PROFIT_TARGET_PCT,
        stop_loss_pct=LONG_OPTION_STOP_LOSS_PCT,
        notes="Long ITM put for bearish directional. Low IV rank only (<40%).",
    ),
}


def get_strategy_config(strategy_name: str) -> StrategyConfig:
    """Get configuration for a strategy, with fallback to defaults."""
    return STRATEGY_CONFIGS.get(strategy_name, STRATEGY_CONFIGS["BULL_PUT_SPREAD"])
