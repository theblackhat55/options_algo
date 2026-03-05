"""
scripts/backtest_long_options.py
=================================
Walk-forward backtest for LONG_CALL and LONG_PUT strategies.

Simulates the full lifecycle:
  • Entry: buy option at premium + delta filter
  • Exit rules:
    - Profit target : gain ≥ LONG_OPTION_PROFIT_TARGET_PCT (default 100%)
    - Stop loss     : loss ≥ LONG_OPTION_STOP_LOSS_PCT     (default 50%)
    - Time stop     : DTE ≤ LONG_OPTION_TIME_STOP_DTE      (default 10)
    - Expiry        : held to expiration

Output: CSV report saved to data/processed/backtest_long_options_<date>.csv

Usage:
    python scripts/backtest_long_options.py
    python scripts/backtest_long_options.py --ticker AAPL --start 2023-01-01
    python scripts/backtest_long_options.py --output /tmp/my_backtest.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    LONG_OPTION_PROFIT_TARGET_PCT,
    LONG_OPTION_STOP_LOSS_PCT,
    LONG_OPTION_TIME_STOP_DTE,
    LONG_OPTION_DELTA,
    DEFAULT_DTE_LONG_OPTION,
    DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = DATA_DIR / "processed"

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    ticker: str
    option_type: str          # CALL or PUT
    entry_date: str
    expiry_date: str
    dte_at_entry: int
    strike: float
    entry_premium: float      # Debit paid per contract (×100 for $ value)
    exit_date: str = ""
    exit_premium: float = 0.0
    exit_reason: str = ""     # PROFIT_TARGET, STOP_LOSS, TIME_STOP, EXPIRY
    pnl_per_contract: float = 0.0   # (exit_premium - entry_premium) × 100
    pnl_pct: float = 0.0            # (exit_premium - entry_premium) / entry_premium
    won: bool = False
    days_held: int = 0
    entry_price: float = 0.0  # Underlying price at entry
    exit_price: float = 0.0   # Underlying price at exit
    delta_at_entry: float = 0.0
    iv_at_entry: float = 0.0


@dataclass
class BacktestReport:
    tickers: list[str]
    start_date: str
    end_date: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    win_rate: float = 0.0
    avg_days_held: float = 0.0
    profit_target_exits: int = 0
    stop_loss_exits: int = 0
    time_stop_exits: int = 0
    expiry_exits: int = 0
    trades: list[BacktestTrade] = field(default_factory=list)


# ── Option Pricing (Black-Scholes approximation) ──────────────────────────────

def _black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Simplified Black-Scholes for backtest premium estimation."""
    from math import log, sqrt, exp
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0:
            # Intrinsic value only
            if option_type == "call":
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:
            return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    except Exception:
        # Fallback: intrinsic + rough time value
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        time_val = sigma * S * (T**0.5) * 0.4  # rough approximation
        return max(round(intrinsic + time_val, 2), 0.01)


def _estimate_delta(S: float, K: float, T: float, sigma: float, option_type: str) -> float:
    """Estimate option delta via Black-Scholes."""
    try:
        from math import log, sqrt
        from scipy.stats import norm
        if T <= 0 or sigma <= 0:
            return 1.0 if (S > K and option_type == "call") else 0.0
        d1 = (log(S / K) + (0.02 + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        if option_type == "call":
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)
    except Exception:
        return LONG_OPTION_DELTA


# ── Walk-Forward Simulation ───────────────────────────────────────────────────

def _simulate_ticker(
    ticker: str,
    df: pd.DataFrame,
    option_type: str = "CALL",
    target_dte: int = DEFAULT_DTE_LONG_OPTION,
    target_delta: float = LONG_OPTION_DELTA,
    profit_target_pct: float = LONG_OPTION_PROFIT_TARGET_PCT,
    stop_loss_pct: float = LONG_OPTION_STOP_LOSS_PCT,
    time_stop_dte: int = LONG_OPTION_TIME_STOP_DTE,
    entry_frequency_days: int = 21,  # enter a new trade every ~21 days
) -> list[BacktestTrade]:
    """
    Simulate a rolling long option strategy for one ticker.

    Entry rule: every entry_frequency_days, enter a new position if not already in one.
    Exit rules: profit target / stop loss / time stop / expiry.
    """
    if df is None or len(df) < 60:
        logger.warning(f"{ticker}: insufficient history ({len(df) if df is not None else 0} rows)")
        return []

    df = df.sort_index()
    prices = df["Close"].values
    dates_idx = df.index

    trades: list[BacktestTrade] = []
    i = 60  # start after warm-up
    r = 0.02  # risk-free rate

    while i < len(df) - target_dte - 2:
        entry_price = float(prices[i])
        entry_date = dates_idx[i].date()

        # Compute HV-20 as IV proxy
        log_returns = np.diff(np.log(prices[max(0, i - 22):i + 1]))
        sigma = float(np.std(log_returns) * np.sqrt(252)) if len(log_returns) > 2 else 0.30

        T = target_dte / 252.0

        # Select strike near target_delta
        if option_type == "CALL":
            # Start at ATM, find strike closest to target_delta
            atm_strike = round(entry_price / 5) * 5  # nearest $5 increment
            strike_candidates = [atm_strike - 10, atm_strike - 5, atm_strike,
                                  atm_strike + 5, atm_strike + 10]
            best_strike = atm_strike
            best_delta_diff = float("inf")
            for sk in strike_candidates:
                if sk <= 0:
                    continue
                d = _estimate_delta(entry_price, sk, T, sigma, "call")
                if abs(d - target_delta) < best_delta_diff:
                    best_delta_diff = abs(d - target_delta)
                    best_strike = sk
            strike = best_strike
        else:
            atm_strike = round(entry_price / 5) * 5
            strike_candidates = [atm_strike - 10, atm_strike - 5, atm_strike,
                                  atm_strike + 5, atm_strike + 10]
            best_strike = atm_strike
            best_delta_diff = float("inf")
            target_abs_delta = abs(target_delta)
            for sk in strike_candidates:
                if sk <= 0:
                    continue
                d = _estimate_delta(entry_price, sk, T, sigma, "put")
                if abs(abs(d) - target_abs_delta) < best_delta_diff:
                    best_delta_diff = abs(abs(d) - target_abs_delta)
                    best_strike = sk
            strike = best_strike

        entry_premium = _black_scholes_price(entry_price, strike, T, r, sigma, option_type.lower())
        if entry_premium < 0.05:
            i += entry_frequency_days
            continue

        delta_entry = _estimate_delta(entry_price, strike, T, sigma, option_type.lower())
        expiry_idx = min(i + target_dte, len(df) - 1)
        expiry_date = dates_idx[expiry_idx].date()

        # Walk forward day by day to check exit rules
        exit_date = expiry_date
        exit_premium = 0.0
        exit_reason = "EXPIRY"
        exit_price = float(prices[expiry_idx])

        for j in range(i + 1, expiry_idx + 1):
            current_price = float(prices[j])
            current_date = dates_idx[j].date()
            dte_now = (expiry_date - current_date).days

            T_now = max(dte_now / 252.0, 1e-6)
            current_premium = _black_scholes_price(current_price, strike, T_now, r, sigma, option_type.lower())
            gain_pct = (current_premium - entry_premium) / entry_premium * 100

            # Time stop
            if 0 < dte_now <= time_stop_dte:
                exit_date = current_date
                exit_premium = current_premium
                exit_reason = "TIME_STOP"
                exit_price = current_price
                break

            # Profit target
            if gain_pct >= profit_target_pct:
                exit_date = current_date
                exit_premium = current_premium
                exit_reason = "PROFIT_TARGET"
                exit_price = current_price
                break

            # Stop loss
            if gain_pct <= -stop_loss_pct:
                exit_date = current_date
                exit_premium = current_premium
                exit_reason = "STOP_LOSS"
                exit_price = current_price
                break

            # Expiry
            if j == expiry_idx:
                exit_premium = current_premium
                exit_price = current_price
                break

        pnl = (exit_premium - entry_premium) * 100
        pnl_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        days_held = (exit_date - entry_date).days

        trade = BacktestTrade(
            ticker=ticker,
            option_type=option_type,
            entry_date=entry_date.isoformat(),
            expiry_date=expiry_date.isoformat(),
            dte_at_entry=target_dte,
            strike=strike,
            entry_premium=round(entry_premium, 2),
            exit_date=exit_date.isoformat(),
            exit_premium=round(exit_premium, 2),
            exit_reason=exit_reason,
            pnl_per_contract=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            won=pnl > 0,
            days_held=days_held,
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            delta_at_entry=round(delta_entry, 3),
            iv_at_entry=round(sigma * 100, 1),
        )
        trades.append(trade)
        i += max(entry_frequency_days, days_held + 1)

    return trades


def run_backtest(
    tickers: list[str],
    start_date: str = "2022-01-01",
    end_date: str | None = None,
    option_type: str = "CALL",
    output_path: Path | None = None,
) -> BacktestReport:
    """
    Run walk-forward backtest for given tickers and return a BacktestReport.
    Also saves CSV to output_path.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance required: pip install yfinance")
        return BacktestReport(tickers=tickers, start_date=start_date, end_date=end_date or date.today().isoformat())

    if end_date is None:
        end_date = date.today().isoformat()

    all_trades: list[BacktestTrade] = []
    logger.info(f"Backtest: {len(tickers)} tickers | {start_date} → {end_date} | {option_type}")

    for ticker in tickers:
        try:
            logger.info(f"  Downloading {ticker}...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty or len(df) < 60:
                logger.warning(f"  {ticker}: insufficient data")
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            trades = _simulate_ticker(ticker, df, option_type=option_type)
            logger.info(f"  {ticker}: {len(trades)} trades simulated")
            all_trades.extend(trades)
        except Exception as exc:
            logger.error(f"  {ticker} failed: {exc}")

    if not all_trades:
        logger.warning("No trades generated.")
        return BacktestReport(tickers=tickers, start_date=start_date, end_date=end_date)

    # Build report
    total = len(all_trades)
    wins = sum(1 for t in all_trades if t.won)
    total_pnl = sum(t.pnl_per_contract for t in all_trades)
    avg_pnl_pct = sum(t.pnl_pct for t in all_trades) / total
    avg_days = sum(t.days_held for t in all_trades) / total

    report = BacktestReport(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        total_trades=total,
        winning_trades=wins,
        losing_trades=total - wins,
        total_pnl=round(total_pnl, 2),
        avg_pnl_pct=round(avg_pnl_pct, 2),
        win_rate=round(wins / total * 100, 1),
        avg_days_held=round(avg_days, 1),
        profit_target_exits=sum(1 for t in all_trades if t.exit_reason == "PROFIT_TARGET"),
        stop_loss_exits=sum(1 for t in all_trades if t.exit_reason == "STOP_LOSS"),
        time_stop_exits=sum(1 for t in all_trades if t.exit_reason == "TIME_STOP"),
        expiry_exits=sum(1 for t in all_trades if t.exit_reason == "EXPIRY"),
        trades=all_trades,
    )

    # Save CSV
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"backtest_long_options_{date.today()}.csv"

    _save_csv(all_trades, output_path)
    logger.info(f"Report saved: {output_path}")
    logger.info(
        f"Summary: {total} trades | Win rate {report.win_rate}% | "
        f"Total P&L ${report.total_pnl:,.0f} | Avg {avg_pnl_pct:.1f}%"
    )

    return report


def _save_csv(trades: list[BacktestTrade], path: Path) -> None:
    """Write backtest trades to CSV."""
    if not trades:
        logger.warning("No trades to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(trades[0].__dict__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trades:
            writer.writerow(t.__dict__)
    logger.info(f"CSV written: {path} ({len(trades)} rows)")


def print_summary(report: BacktestReport) -> None:
    """Print human-readable backtest summary."""
    print("\n" + "=" * 60)
    print(f"BACKTEST REPORT — Long {report.trades[0].option_type if report.trades else '?'}")
    print(f"Period: {report.start_date} → {report.end_date}")
    print(f"Tickers: {', '.join(report.tickers)}")
    print("=" * 60)
    print(f"Total trades   : {report.total_trades}")
    print(f"Win rate       : {report.win_rate:.1f}%")
    print(f"Wins / Losses  : {report.winning_trades} / {report.losing_trades}")
    print(f"Total P&L      : ${report.total_pnl:,.2f}")
    print(f"Avg P&L %      : {report.avg_pnl_pct:.1f}%")
    print(f"Avg days held  : {report.avg_days_held:.1f}")
    print("-" * 60)
    print(f"Exit reasons:")
    print(f"  Profit target : {report.profit_target_exits}")
    print(f"  Stop loss     : {report.stop_loss_exits}")
    print(f"  Time stop     : {report.time_stop_exits}")
    print(f"  Expiry        : {report.expiry_exits}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward long option backtest")
    parser.add_argument("--ticker", nargs="+", default=["AAPL", "MSFT", "TSLA", "NVDA"],
                        help="Tickers to backtest")
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--type", choices=["CALL", "PUT"], default="CALL",
                        help="Option type to backtest")
    parser.add_argument("--output", default=None, help="Output CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = Path(args.output) if args.output else None
    report = run_backtest(
        tickers=args.ticker,
        start_date=args.start,
        end_date=args.end,
        option_type=args.type,
        output_path=output,
    )
    print_summary(report)
