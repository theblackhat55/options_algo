"""
src/pipeline/outcome_tracker.py
================================
Track paper trade entries and exits to build the ML training dataset.
Records every trade with its entry/exit details and outcome (win/loss).

Usage:
    from src.pipeline.outcome_tracker import record_entry, record_exit, load_outcomes
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import TRADES_DIR

logger = logging.getLogger(__name__)

_OUTCOMES_FILE = TRADES_DIR / "trade_outcomes.jsonl"   # JSON Lines for easy append
_PAPER_LOG = TRADES_DIR / "paper_trades.json"


@dataclass
class TradeOutcome:
    """Complete record of a paper/live trade for ML training."""
    trade_id: str
    ticker: str
    strategy: str
    direction: str
    regime: str
    iv_regime: str
    iv_rank: float
    iv_hv_ratio: float
    adx: float
    rsi: float
    trend_strength: float
    direction_score: float
    rs_rank: float
    sector: str
    dte_at_entry: int
    spread_width: float
    short_delta: float
    entry_date: str
    expiration: str
    short_strike: float
    long_strike: float
    net_credit_or_debit: float
    max_risk: float
    prob_profit: float
    confidence: float

    # Outcome fields (filled on close)
    exit_date: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0          # % of max profit realised
    won: Optional[bool] = None    # True = profitable, False = loss
    outcome: str = "OPEN"         # OPEN, WIN, LOSS, EXPIRED, STOPPED_OUT
    days_held: int = 0
    close_reason: str = ""        # 50% profit, stop loss, expiry, etc.

    # ── V2: Market context at entry (for ML feature set) ─────────────────────
    entry_vix: float = 0.0        # VIX level when trade was entered
    entry_vix_tier: str = ""      # NORMAL / CAUTION / DEFENSIVE / LIQUIDATION
    entry_spy_5d: float = 0.0     # SPY 5-day return at entry (%)
    entry_roc_3d: float = 0.0     # Stock 3-day ROC at entry (%) — momentum context
    entry_beta: float = 1.0       # 60-day beta vs SPY at entry
    entry_sector: str = ""        # GICS sector at entry

    # ── IBKR Real-time Features at Entry ─────────────────────────────────────
    options_flow_score: float = 0.0    # 0-100 unusual options activity at entry
    dominant_flow: str = ""            # CALLS / PUTS / NEUTRAL at entry
    put_call_volume_ratio: float = 1.0 # put vol / call vol at entry
    volume_pace: float = 1.0           # intraday volume / 20d avg at entry
    live_iv_at_entry: float = 0.0      # real-time ATM IV at entry (%)
    iv_skew_at_entry: float = 0.0      # put IV - call IV at entry


def record_entry(
    ticker: str,
    recommendation: dict,
    trade: dict,
    context: dict,
) -> str:
    """
    Record a new paper trade entry.

    Args:
        ticker: Stock symbol
        recommendation: From nightly_scan recommendation dict
        trade: From nightly_scan trade dict
        context: From nightly_scan context dict

    Returns:
        trade_id (UUID string)
    """
    trade_id = str(uuid.uuid4())[:8]
    iv = context.get("iv_detail", {})
    reg = context.get("regime_detail", {})
    rs = context.get("rs_detail", {})
    mkt = context.get("market_snapshot", {})   # V2: market context at entry
    flow = context.get("options_flow", {})     # IBKR real-time flow at entry

    # Handle both credit and debit spreads
    credit = trade.get("net_credit") or -(trade.get("net_debit", 0))
    max_r = trade.get("max_risk") or trade.get("net_debit", 0)
    short_s = trade.get("short_strike") or trade.get("body", 0)
    long_s = trade.get("long_strike") or trade.get("lower_wing", 0)
    width = trade.get("width") or trade.get("wing_width", 0)

    outcome = TradeOutcome(
        trade_id=trade_id,
        ticker=ticker,
        strategy=recommendation.get("strategy", ""),
        direction=recommendation.get("direction", ""),
        regime=recommendation.get("regime", ""),
        iv_regime=recommendation.get("iv_regime", ""),
        iv_rank=iv.get("iv_rank", 0),
        iv_hv_ratio=iv.get("iv_hv_ratio", 0),
        adx=reg.get("adx", 0),
        rsi=reg.get("rsi", 0),
        trend_strength=reg.get("trend_strength", 0),
        direction_score=reg.get("direction_score", 0),
        rs_rank=rs.get("rs_rank") or 50,
        sector=context.get("sector", ""),
        dte_at_entry=trade.get("dte") or recommendation.get("target_dte", 0),
        spread_width=width,
        short_delta=trade.get("short_delta") or trade.get("long_delta", 0),
        entry_date=date.today().isoformat(),
        expiration=trade.get("expiration", ""),
        short_strike=short_s,
        long_strike=long_s,
        net_credit_or_debit=credit,
        max_risk=max_r,
        prob_profit=trade.get("prob_profit", 0),
        confidence=recommendation.get("confidence", 0),
        # V2 market context fields
        entry_vix=float(mkt.get("vix", 0)),
        entry_vix_tier=str(mkt.get("vix_tier", "")),
        entry_spy_5d=float(mkt.get("spy_5d_return", 0)),
        entry_roc_3d=float(reg.get("roc_3d", 0)),
        entry_beta=float(context.get("beta", 1.0)),
        entry_sector=str(context.get("sector", "")),
        # IBKR real-time fields at entry
        options_flow_score=float(flow.get("flow_score", 0.0)),
        dominant_flow=str(flow.get("dominant_side", "")),
        put_call_volume_ratio=float(flow.get("put_call_volume_ratio", 1.0)),
        volume_pace=float(flow.get("volume_pace", 1.0)),
        live_iv_at_entry=float(flow.get("live_iv") or 0.0),
        iv_skew_at_entry=float(iv.get("skew", 0.0)),
    )

    _append_outcome(outcome)
    logger.info(f"Recorded entry: {trade_id} {ticker} {outcome.strategy}")
    return trade_id


def record_exit(
    trade_id: str,
    exit_price: float,
    close_reason: str = "manual",
) -> Optional[TradeOutcome]:
    """Update a trade outcome with exit details."""
    outcomes = _load_all_outcomes()

    for out in outcomes:
        if out.trade_id == trade_id and out.outcome == "OPEN":
            out.exit_date = date.today().isoformat()
            out.exit_price = exit_price
            out.days_held = (
                datetime.strptime(out.exit_date, "%Y-%m-%d") -
                datetime.strptime(out.entry_date, "%Y-%m-%d")
            ).days

            # P&L calculation
            if out.net_credit_or_debit > 0:
                # Credit spread: profit = credit − exit_price
                out.pnl = round((out.net_credit_or_debit - exit_price), 2)
            else:
                # Debit spread: profit = exit_price − |debit|
                out.pnl = round((exit_price - abs(out.net_credit_or_debit)), 2)

            # P&L as % of max profit
            max_profit = abs(out.net_credit_or_debit) if out.net_credit_or_debit > 0 else out.max_risk
            out.pnl_pct = round(out.pnl / max_profit * 100 if max_profit > 0 else 0, 1)

            out.won = out.pnl > 0
            out.outcome = "WIN" if out.won else "LOSS"
            out.close_reason = close_reason

            _save_all_outcomes(outcomes)
            logger.info(f"Closed {trade_id}: P&L {out.pnl:+.2f} ({out.outcome})")
            return out

    logger.warning(f"Trade {trade_id} not found or already closed")
    return None


def load_outcomes(only_closed: bool = True) -> pd.DataFrame:
    """
    Load all trade outcomes as a DataFrame for ML training.

    Args:
        only_closed: If True, return only closed/expired trades (not OPEN)
    """
    outcomes = _load_all_outcomes()
    if only_closed:
        outcomes = [o for o in outcomes if o.outcome != "OPEN"]

    if not outcomes:
        return pd.DataFrame()

    return pd.DataFrame([asdict(o) for o in outcomes])


def get_win_rate(strategy: str = None) -> dict:
    """Compute win rate statistics, optionally filtered by strategy."""
    df = load_outcomes(only_closed=True)
    if df.empty:
        return {"win_rate": None, "count": 0}

    if strategy:
        df = df[df["strategy"] == strategy]

    if df.empty:
        return {"win_rate": None, "count": 0, "strategy": strategy}

    wins = (df["won"] == True).sum()
    total = len(df)
    avg_pnl = df["pnl"].mean()
    avg_pnl_pct = df["pnl_pct"].mean()

    return {
        "strategy": strategy or "ALL",
        "count": total,
        "wins": int(wins),
        "losses": total - int(wins),
        "win_rate": round(wins / total * 100, 1),
        "avg_pnl": round(avg_pnl, 2),
        "avg_pnl_pct": round(avg_pnl_pct, 1),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _append_outcome(outcome: TradeOutcome) -> None:
    """Append a single trade outcome to JSONL file."""
    try:
        with open(_OUTCOMES_FILE, "a") as f:
            f.write(json.dumps(asdict(outcome)) + "\n")
    except Exception as exc:
        logger.error(f"Failed to append outcome: {exc}")


def _load_all_outcomes() -> list[TradeOutcome]:
    """Load all trade outcomes from JSONL file."""
    if not _OUTCOMES_FILE.exists():
        return []
    outcomes = []
    try:
        with open(_OUTCOMES_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    outcomes.append(TradeOutcome(**json.loads(line)))
    except Exception as exc:
        logger.warning(f"Failed to load outcomes: {exc}")
    return outcomes


def _save_all_outcomes(outcomes: list[TradeOutcome]) -> None:
    """Rewrite the entire JSONL file (used after updates)."""
    try:
        with open(_OUTCOMES_FILE, "w") as f:
            for o in outcomes:
                f.write(json.dumps(asdict(o)) + "\n")
    except Exception as exc:
        logger.error(f"Failed to save outcomes: {exc}")
