"""
src/pipeline/position_monitor.py
==================================
Intraday position monitor: checks open positions against exit rules.
Sends alerts when 50% profit target or 100% stop-loss is hit.

Runs hourly during market hours (or on demand).

Usage:
    python -m src.pipeline.position_monitor

    from src.pipeline.position_monitor import monitor_positions
    alerts = monitor_positions()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

import yfinance as yf

from config.settings import (
    PROFIT_TARGET_PCT, STOP_LOSS_PCT, SIGNALS_DIR,
    VIX_DEFENSIVE_LEVEL, VIX_SPIKE_WINDOW, VIX_SPIKE_THRESHOLD_PCT,
)
from src.risk.portfolio import load_positions, close_position, OpenPosition
from src.pipeline.outcome_tracker import record_exit

logger = logging.getLogger(__name__)


@dataclass
class MonitorAlert:
    trade_id: str = ""
    ticker: str = ""
    strategy: str = ""
    alert_type: str = ""          # "PROFIT_TARGET", "STOP_LOSS", "EXPIRING_SOON"
    message: str = ""
    current_price: float = 0.0
    entry_credit: float = 0.0
    current_spread_value: float = 0.0
    pnl_pct: float = 0.0
    action: str = ""              # "CLOSE", "REVIEW"
    priority: int = 1             # 1=urgent, 2=watch


def monitor_positions(
    profit_target_pct: float = PROFIT_TARGET_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
) -> list[MonitorAlert]:
    """
    Check all open positions against exit rules.

    Args:
        profit_target_pct: Close when P&L reaches this % of max profit
        stop_loss_pct: Close when loss reaches this % of credit received

    Returns:
        List of MonitorAlert objects (one per position needing action)
    """
    positions = [p for p in load_positions() if p.status == "OPEN"]

    if not positions:
        logger.info("No open positions to monitor")
        return []

    logger.info(f"Monitoring {len(positions)} open positions")
    alerts = []
    today = date.today()

    # ── VIX Spike Pre-Check ───────────────────────────────────────────────────
    # Pull current VIX before checking individual positions.  If VIX is at
    # DEFENSIVE level or above, flag all open credit spreads for review.
    _vix_spike_alerts = _check_vix_spike_for_positions(positions)
    alerts.extend(_vix_spike_alerts)

    for pos in positions:
        try:
            alert = _check_position(pos, profit_target_pct, stop_loss_pct, today)
            if alert:
                alerts.append(alert)
        except Exception as exc:
            logger.error(f"Monitor failed for {pos.ticker}: {exc}")

    if alerts:
        logger.info(f"Generated {len(alerts)} alerts")
        for a in alerts:
            logger.info(f"  [{a.alert_type}] {a.ticker} {a.strategy}: {a.message}")

    return alerts


def _check_position(
    pos: OpenPosition,
    profit_target_pct: float,
    stop_loss_pct: float,
    today: date,
) -> Optional[MonitorAlert]:
    """Check a single position for exit signals."""

    # ── Expiry Check ──────────────────────────────────────────────────────────
    try:
        exp_date = datetime.strptime(pos.expiration, "%Y-%m-%d").date()
        dte_remaining = (exp_date - today).days

        if dte_remaining <= 0:
            return MonitorAlert(
                ticker=pos.ticker,
                strategy=pos.strategy,
                alert_type="EXPIRED",
                message=f"Option EXPIRED on {pos.expiration}. Record outcome.",
                action="CLOSE",
                priority=1,
            )
        elif dte_remaining <= 5:
            return MonitorAlert(
                ticker=pos.ticker,
                strategy=pos.strategy,
                alert_type="EXPIRING_SOON",
                message=f"Only {dte_remaining} DTE remaining. Consider rolling or closing.",
                current_price=0,
                action="REVIEW",
                priority=2,
            )
    except Exception as exc:
        logger.debug(f"Expiry parse error for {pos.ticker}: {exc}")

    # ── Price-based P&L Check ──────────────────────────────────────────────────
    # Get current stock price to estimate spread value
    try:
        ticker_obj = yf.Ticker(pos.ticker)
        hist = ticker_obj.history(period="1d")
        if hist.empty:
            return None
        current_price = float(hist["Close"].iloc[-1])
    except Exception:
        return None

    # For credit spreads: estimate current spread value from intrinsic value
    if pos.net_credit > 0:
        estimated_spread_value = _estimate_credit_spread_value(
            pos, current_price
        )
        current_value = estimated_spread_value
        entry_value = pos.net_credit

        pnl_pct = (entry_value - current_value) / entry_value * 100 if entry_value > 0 else 0

        # Profit target: position decayed to 50% of credit
        if pnl_pct >= profit_target_pct:
            return MonitorAlert(
                ticker=pos.ticker,
                strategy=pos.strategy,
                alert_type="PROFIT_TARGET",
                message=(
                    f"PROFIT TARGET HIT: {pnl_pct:.0f}% of max profit. "
                    f"Close for ${entry_value - current_value:.2f} credit. "
                    f"Stock at ${current_price:.2f}"
                ),
                current_price=current_price,
                entry_credit=entry_value,
                current_spread_value=current_value,
                pnl_pct=pnl_pct,
                action="CLOSE",
                priority=1,
            )

        # Stop loss: spread expanded to 2x credit received
        loss_pct = (current_value - entry_value) / entry_value * 100
        if loss_pct >= stop_loss_pct:
            return MonitorAlert(
                ticker=pos.ticker,
                strategy=pos.strategy,
                alert_type="STOP_LOSS",
                message=(
                    f"STOP LOSS HIT: Spread at ${current_value:.2f} "
                    f"vs entry ${entry_value:.2f}. "
                    f"Loss: {loss_pct:.0f}% of credit. "
                    f"Stock at ${current_price:.2f}"
                ),
                current_price=current_price,
                entry_credit=entry_value,
                current_spread_value=current_value,
                pnl_pct=-loss_pct,
                action="CLOSE",
                priority=1,
            )

    return None


def _check_vix_spike_for_positions(positions) -> list[MonitorAlert]:
    """
    Pull current VIX; generate VIX_SPIKE alerts for open credit spreads when:
    (a) VIX ≥ VIX_DEFENSIVE_LEVEL, or
    (b) VIX spiked > VIX_SPIKE_THRESHOLD_PCT above its recent rolling average.
    """
    alerts: list[MonitorAlert] = []
    try:
        vix_hist = yf.Ticker("^VIX").history(period="15d")
        if vix_hist.empty or len(vix_hist) < 2:
            return alerts
        vix_now = round(float(vix_hist["Close"].iloc[-1]), 1)
        window = min(VIX_SPIKE_WINDOW, len(vix_hist))
        vix_5d = round(float(vix_hist["Close"].tail(window).mean()), 1)
    except Exception as exc:
        logger.warning(f"VIX fetch for spike check failed: {exc}")
        return alerts

    spike_pct = (vix_now - vix_5d) / vix_5d * 100 if vix_5d > 0 else 0
    is_spike = spike_pct > VIX_SPIKE_THRESHOLD_PCT
    is_defensive = vix_now >= VIX_DEFENSIVE_LEVEL

    if not is_defensive and not is_spike:
        return alerts  # nothing to flag

    spike_note = f" (spike +{spike_pct:.0f}% vs 5d avg)" if is_spike else ""
    logger.warning(
        f"VIX DEFENSIVE/SPIKE: {vix_now:.1f}{spike_note} — "
        "reviewing all credit spread positions"
    )

    today = date.today()
    credit_strategies = {
        "BULL_PUT_SPREAD", "BULL_PUT",
        "BEAR_CALL_SPREAD", "BEAR_CALL",
        "IRON_CONDOR",
    }
    for pos in positions:
        if pos.strategy.upper() not in credit_strategies:
            continue
        try:
            exp_date = datetime.strptime(pos.expiration, "%Y-%m-%d").date()
            dte = (exp_date - today).days
        except Exception:
            dte = 0
        if dte > 0:
            alerts.append(MonitorAlert(
                trade_id=getattr(pos, "trade_id", ""),
                ticker=pos.ticker,
                strategy=pos.strategy,
                alert_type="VIX_SPIKE",
                message=(
                    f"VIX at {vix_now:.0f}{spike_note} "
                    f"(≥ DEFENSIVE level if applicable). "
                    f"{dte} DTE remaining. "
                    "Review: consider closing losing credit spreads."
                ),
                action="REVIEW",
                priority=1,
            ))
    if alerts:
        logger.warning(
            f"VIX alert: generated {len(alerts)} VIX_SPIKE alert(s)"
        )
    return alerts


def _estimate_credit_spread_value(
    pos: OpenPosition,
    current_price: float,
) -> float:
    """
    Rough estimate of current spread value based on intrinsic value.
    This is a proxy — real monitoring needs live options quotes.
    """
    if pos.strategy in ("BULL_PUT_SPREAD", "BULL_PUT"):
        # Bull put: value = max(short_strike - current_price, 0) - max(long_strike - current_price, 0)
        intrinsic = max(pos.short_strike - current_price, 0) - max(pos.long_strike - current_price, 0)
        # Add time value estimate (decays linearly — rough proxy)
        return max(round(intrinsic + pos.net_credit * 0.1, 2), 0.01)

    elif pos.strategy in ("BEAR_CALL_SPREAD", "BEAR_CALL"):
        intrinsic = max(current_price - pos.short_strike, 0) - max(current_price - pos.long_strike, 0)
        return max(round(intrinsic + pos.net_credit * 0.1, 2), 0.01)

    # Default: return entry credit (no information)
    return pos.net_credit


def format_monitor_report(alerts: list[MonitorAlert]) -> str:
    """Format alerts into a WhatsApp-friendly message."""
    if not alerts:
        return f"OPTIONS MONITOR — {date.today()}\nAll positions nominal. No action needed."

    lines = [f"OPTIONS MONITOR — {date.today()}", "ALERTS:"]

    for a in alerts:
        emoji = "🔴" if a.priority == 1 else "🟡"
        lines.append(f"\n{emoji} [{a.alert_type}] {a.ticker}")
        lines.append(f"Strategy: {a.strategy}")
        lines.append(a.message)
        if a.action:
            lines.append(f"ACTION: {a.action}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    alerts = monitor_positions()
    print(format_monitor_report(alerts))
