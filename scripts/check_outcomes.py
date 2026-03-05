#!/usr/bin/env python3
"""
scripts/check_outcomes.py
=========================
Daily outcome checker — runs each morning to update open trades.
Checks if price moved through strikes, if DTE expired, or if
profit target / stop loss was hit.

Note: record_exit(trade_id, exit_price, close_reason) handles P&L
calculation internally based on credit/debit and exit_price.
"""
import sys
import logging
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import LOG_LEVEL
from src.pipeline.outcome_tracker import _load_all_outcomes, record_exit
from src.data.stock_fetcher import download_universe

logger = logging.getLogger(__name__)

PROFIT_TARGET_PCT = 50
STOP_LOSS_PCT = 100


def check_all_outcomes():
    """Check all open trades and update outcomes."""
    outcomes = _load_all_outcomes()
    open_trades = [o for o in outcomes if o.outcome == "OPEN"]

    if not open_trades:
        print("No open trades to check.")
        return []

    print(f"Checking {len(open_trades)} open trades...")
    today = date.today()
    tickers = list(set(o.ticker for o in open_trades))

    data = download_universe(tickers, period="1mo")

    alerts = []

    for trade in open_trades:
        ticker = trade.ticker
        if ticker not in data or data[ticker].empty:
            print(f"  {ticker}: No price data, skipping")
            continue

        df = data[ticker]
        current_price = float(df["close"].iloc[-1])

        try:
            exp_date = date.fromisoformat(trade.expiration)
        except (ValueError, TypeError):
            exp_date = None

        entry_date = date.fromisoformat(trade.entry_date)
        days_held = (today - entry_date).days

        close_reason = ""
        should_close = False

        # 1. Expired
        if exp_date and today >= exp_date:
            should_close = True
            if trade.strategy in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "IRON_CONDOR"):
                if trade.strategy == "BULL_PUT_SPREAD":
                    won = current_price > trade.short_strike
                elif trade.strategy == "BEAR_CALL_SPREAD":
                    won = current_price < trade.short_strike
                else:
                    won = trade.long_strike < current_price < trade.short_strike
                close_reason = "Expired in profit zone" if won else "Expired — breached strike"
            else:
                close_reason = "Expired"

        # 2. Profit target (50% after 7+ days)
        elif days_held >= 7 and trade.strategy in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"):
            if trade.strategy == "BULL_PUT_SPREAD":
                distance = (current_price - trade.short_strike) / trade.short_strike * 100
                if distance > 3:
                    should_close = True
                    close_reason = f"Profit target ({PROFIT_TARGET_PCT}%) — price ${current_price:.2f} above short ${trade.short_strike:.2f}"
            elif trade.strategy == "BEAR_CALL_SPREAD":
                distance = (trade.short_strike - current_price) / trade.short_strike * 100
                if distance > 3:
                    should_close = True
                    close_reason = f"Profit target ({PROFIT_TARGET_PCT}%) — price ${current_price:.2f} below short ${trade.short_strike:.2f}"

        # 3. Stop loss — price breached short strike
        elif trade.strategy == "BULL_PUT_SPREAD" and current_price < trade.short_strike:
            should_close = True
            close_reason = f"Stop loss — price ${current_price:.2f} below short ${trade.short_strike:.2f}"
        elif trade.strategy == "BEAR_CALL_SPREAD" and current_price > trade.short_strike:
            should_close = True
            close_reason = f"Stop loss — price ${current_price:.2f} above short ${trade.short_strike:.2f}"

        if should_close:
            try:
                # record_exit signature: (trade_id, exit_price, close_reason)
                # It computes P&L internally from credit/debit and exit_price
                result = record_exit(
                    trade_id=trade.trade_id,
                    exit_price=current_price,
                    close_reason=close_reason,
                )
                if result:
                    icon = "✅" if result.won else "❌"
                    print(f"  {icon} {ticker} [{trade.strategy}] → {result.outcome} | P&L: ${result.pnl:+.2f} | {close_reason}")
                    alerts.append({
                        "ticker": ticker,
                        "strategy": trade.strategy,
                        "outcome": result.outcome,
                        "pnl": result.pnl,
                        "reason": close_reason,
                        "days_held": days_held,
                    })
                else:
                    print(f"  ⚠️ {ticker}: record_exit returned None (already closed?)")
            except Exception as e:
                print(f"  ⚠️ Failed to record exit for {ticker}: {e}")
        else:
            dte_remaining = (exp_date - today).days if exp_date else "?"
            print(f"  ⏳ {ticker} [{trade.strategy}] — OPEN | Price: ${current_price:.2f} | DTE: {dte_remaining} | Held: {days_held}d")

    closed = [a for a in alerts]
    if closed:
        wins = sum(1 for a in closed if a["outcome"] == "WIN")
        total_pnl = sum(a["pnl"] for a in closed)
        print(f"\n{'='*50}")
        print(f"CLOSED TODAY: {len(closed)} trades | Wins: {wins}/{len(closed)} | P&L: ${total_pnl:+.2f}")
        print(f"{'='*50}")

    return alerts


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    check_all_outcomes()
