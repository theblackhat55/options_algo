#!/usr/bin/env python3
"""
scripts/check_outcomes.py
=========================
Daily outcome checker — runs each morning to update open trades.
Checks if price moved through strikes, if DTE expired, or if
profit target / stop loss was hit.
"""
import sys
import logging
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import TRADES_DIR, LOG_LEVEL
from src.pipeline.outcome_tracker import load_outcomes, record_exit, _load_all_outcomes
from src.data.stock_fetcher import download_universe

logger = logging.getLogger(__name__)

PROFIT_TARGET_PCT = 50   # Close at 50% of max profit
STOP_LOSS_PCT = 100      # Close at 100% of credit received (2x loss)


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
    
    # Fetch latest prices
    data = download_universe(tickers, period="5d")
    
    alerts = []
    
    for trade in open_trades:
        ticker = trade.ticker
        if ticker not in data or data[ticker].empty:
            print(f"  {ticker}: No price data, skipping")
            continue
        
        df = data[ticker]
        current_price = df["close"].iloc[-1]
        
        # Check expiration
        try:
            exp_date = date.fromisoformat(trade.expiration)
        except (ValueError, TypeError):
            exp_date = None
        
        entry_date = date.fromisoformat(trade.entry_date)
        days_held = (today - entry_date).days
        
        # Determine outcome
        close_reason = ""
        pnl = 0.0
        won = None
        outcome = "OPEN"
        
        # 1. Expired
        if exp_date and today >= exp_date:
            if trade.strategy in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "IRON_CONDOR"):
                # Credit spread — check if price stayed in profit zone
                if trade.strategy == "BULL_PUT_SPREAD":
                    won = current_price > trade.short_strike
                elif trade.strategy == "BEAR_CALL_SPREAD":
                    won = current_price < trade.short_strike
                elif trade.strategy == "IRON_CONDOR":
                    won = trade.long_strike < current_price < trade.short_strike
                
                if won:
                    pnl = abs(trade.net_credit_or_debit) * 100  # Full credit kept
                    outcome = "WIN"
                    close_reason = "Expired in profit zone"
                else:
                    pnl = -trade.max_risk * 100  # Max loss
                    outcome = "LOSS"
                    close_reason = "Expired — breached strike"
            else:
                # Debit spread
                won = False  # Simplified — debit expired worthless
                pnl = -abs(trade.net_credit_or_debit) * 100
                outcome = "LOSS"
                close_reason = "Expired worthless"
        
        # 2. Profit target (50% of max profit after holding > 7 days)
        elif days_held >= 7 and trade.strategy in ("BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"):
            credit = abs(trade.net_credit_or_debit)
            if trade.strategy == "BULL_PUT_SPREAD":
                distance_from_short = (current_price - trade.short_strike) / trade.short_strike * 100
                if distance_from_short > 3:  # Price moved 3%+ away from short strike
                    pnl = credit * PROFIT_TARGET_PCT / 100 * 100
                    won = True
                    outcome = "WIN"
                    close_reason = f"Profit target ({PROFIT_TARGET_PCT}%) — price ${current_price:.2f} well above short strike ${trade.short_strike:.2f}"
            elif trade.strategy == "BEAR_CALL_SPREAD":
                distance_from_short = (trade.short_strike - current_price) / trade.short_strike * 100
                if distance_from_short > 3:
                    pnl = credit * PROFIT_TARGET_PCT / 100 * 100
                    won = True
                    outcome = "WIN"
                    close_reason = f"Profit target ({PROFIT_TARGET_PCT}%) — price ${current_price:.2f} well below short strike ${trade.short_strike:.2f}"
        
        # 3. Stop loss — price breached short strike
        elif trade.strategy == "BULL_PUT_SPREAD" and current_price < trade.short_strike:
            pnl = -trade.max_risk * 100
            won = False
            outcome = "LOSS"
            close_reason = f"Stop loss — price ${current_price:.2f} below short strike ${trade.short_strike:.2f}"
        elif trade.strategy == "BEAR_CALL_SPREAD" and current_price > trade.short_strike:
            pnl = -trade.max_risk * 100
            won = False
            outcome = "LOSS"
            close_reason = f"Stop loss — price ${current_price:.2f} above short strike ${trade.short_strike:.2f}"
        
        # Record exit if trade closed
        if outcome != "OPEN":
            try:
                record_exit(
                    trade_id=trade.trade_id,
                    exit_price=current_price,
                    pnl=pnl,
                    outcome=outcome,
                    close_reason=close_reason,
                )
                status_icon = "✅" if won else "❌"
                print(f"  {status_icon} {ticker} [{trade.strategy}] → {outcome} | P&L: ${pnl:+,.0f} | {close_reason}")
                alerts.append({
                    "ticker": ticker,
                    "strategy": trade.strategy,
                    "outcome": outcome,
                    "pnl": pnl,
                    "reason": close_reason,
                    "days_held": days_held,
                })
            except Exception as e:
                print(f"  ⚠️ Failed to record exit for {ticker}: {e}")
        else:
            dte_remaining = (exp_date - today).days if exp_date else "?"
            print(f"  ⏳ {ticker} [{trade.strategy}] — OPEN | Price: ${current_price:.2f} | DTE: {dte_remaining} | Held: {days_held}d")
    
    # Summary
    closed = [a for a in alerts if a["outcome"] != "OPEN"]
    if closed:
        wins = sum(1 for a in closed if a["outcome"] == "WIN")
        total_pnl = sum(a["pnl"] for a in closed)
        print(f"\n{'='*50}")
        print(f"CLOSED TODAY: {len(closed)} trades | Wins: {wins}/{len(closed)} | P&L: ${total_pnl:+,.0f}")
        print(f"{'='*50}")
    
    return alerts


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    check_all_outcomes()
