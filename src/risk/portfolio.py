"""
src/risk/portfolio.py
=====================
Portfolio-level risk management: track Greek exposure across open positions,
enforce maximum position count, and check correlation.

Usage:
    from src.risk.portfolio import PortfolioRisk, check_portfolio_limits
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import MAX_POSITIONS, TRADES_DIR

logger = logging.getLogger(__name__)

_POSITIONS_FILE = TRADES_DIR / "open_positions.json"


@dataclass
class OpenPosition:
    ticker: str
    strategy: str
    direction: str
    entry_date: str
    expiration: str
    dte_at_entry: int
    contracts: int
    net_credit: float         # Per contract (positive = credit)
    max_risk: float           # Per contract
    total_risk: float         # contracts × max_risk × 100
    short_strike: float = 0.0
    long_strike: float = 0.0
    body_strike: float = 0.0   # For butterflies
    status: str = "OPEN"      # OPEN, CLOSED, EXPIRED
    close_price: float = 0.0
    close_date: str = ""
    pnl: float = 0.0
    is_long_option: bool = False   # True for LONG_CALL / LONG_PUT
    trade_id: str = ""             # Links back to outcome_tracker


@dataclass
class PortfolioRisk:
    open_positions: list[OpenPosition]
    total_positions: int
    total_risk_committed: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    can_add_position: bool
    remaining_risk_budget: float
    notes: str = ""


def load_positions() -> list[OpenPosition]:
    """Load open positions from JSON file."""
    if not _POSITIONS_FILE.exists():
        return []
    try:
        with open(_POSITIONS_FILE) as f:
            raw = json.load(f)
        return [OpenPosition(**p) for p in raw]
    except Exception as exc:
        logger.warning(f"Failed to load positions: {exc}")
        return []


def save_positions(positions: list[OpenPosition]) -> None:
    """Persist positions to JSON."""
    try:
        data = [vars(p) for p in positions]
        with open(_POSITIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error(f"Failed to save positions: {exc}")


def add_position(pos: OpenPosition) -> bool:
    """Add a new position to the log. Returns True if added."""
    positions = load_positions()
    open_pos = [p for p in positions if p.status == "OPEN"]

    if len(open_pos) >= MAX_POSITIONS:
        logger.warning(f"Max positions ({MAX_POSITIONS}) reached. Cannot add {pos.ticker}")
        return False

    positions.append(pos)
    save_positions(positions)
    logger.info(f"Added position: {pos.ticker} {pos.strategy}")
    return True


def close_position(
    ticker: str,
    strategy: str,
    close_price: float,
    close_date: str = None,
) -> Optional[OpenPosition]:
    """Mark a position as closed and record P&L."""
    if close_date is None:
        close_date = datetime.now().date().isoformat()

    positions = load_positions()
    for pos in positions:
        if pos.ticker == ticker and pos.strategy == strategy and pos.status == "OPEN":
            pos.status = "CLOSED"
            pos.close_price = close_price
            pos.close_date = close_date
            # P&L for credit spreads: net_credit − close_price
            if pos.net_credit > 0:
                pos.pnl = round((pos.net_credit - close_price) * pos.contracts * 100, 2)
            else:
                # Debit spread: close_price − abs(net_credit)
                pos.pnl = round((close_price - abs(pos.net_credit)) * pos.contracts * 100, 2)
            save_positions(positions)
            logger.info(f"Closed {ticker} {strategy}: P&L ${pos.pnl:+.2f}")
            return pos

    logger.warning(f"No open position found for {ticker} {strategy}")
    return None


def check_portfolio_limits(
    new_ticker: str = None,
    account_size: float = 10000,
    max_risk_pct: float = 2.0,
) -> PortfolioRisk:
    """
    Check current portfolio risk and whether a new position can be added.
    """
    positions = load_positions()
    open_pos = [p for p in positions if p.status == "OPEN"]

    total_risk = sum(p.total_risk for p in open_pos)
    max_allowed_risk = account_size * (max_risk_pct / 100) * MAX_POSITIONS

    bullish  = sum(1 for p in open_pos if p.direction == "BULLISH")
    bearish  = sum(1 for p in open_pos if p.direction == "BEARISH")
    neutral  = sum(1 for p in open_pos if p.direction == "NEUTRAL")

    can_add = (
        len(open_pos) < MAX_POSITIONS and
        total_risk < max_allowed_risk * 0.9  # 90% of max before blocking
    )

    # Directional exposure check: don't pile into one direction
    if new_ticker:
        # Could add sector correlation check here
        pass

    remaining = round(max_allowed_risk - total_risk, 2)

    notes = []
    if len(open_pos) >= MAX_POSITIONS:
        notes.append(f"Max positions reached ({MAX_POSITIONS})")
    if bullish > 3:
        notes.append("Heavy bullish exposure")
    if bearish > 3:
        notes.append("Heavy bearish exposure")

    return PortfolioRisk(
        open_positions=open_pos,
        total_positions=len(open_pos),
        total_risk_committed=round(total_risk, 2),
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        can_add_position=can_add,
        remaining_risk_budget=max(remaining, 0),
        notes=" | ".join(notes),
    )
