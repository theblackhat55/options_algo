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
    LONG_OPTION_PROFIT_TARGET_PCT, LONG_OPTION_STOP_LOSS_PCT,
    LONG_OPTION_TIME_STOP_DTE,
)
from src.risk.portfolio import load_positions, close_position, OpenPosition
from src.pipeline.outcome_tracker import record_exit

try:
    from src.data.ibkr_client import connect_ibkr, disconnect_ibkr, fetch_stock_snapshot
except Exception:
    connect_ibkr = None  # type: ignore
    disconnect_ibkr = None  # type: ignore
    fetch_stock_snapshot = None  # type: ignore

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

    ib_monitor = None
    if connect_ibkr:
        try:
            ib_monitor = connect_ibkr()
            if ib_monitor:
                logger.info("Connected to IBKR for live position monitoring")
        except Exception as exc:
            logger.warning(f"IBKR monitor connection failed: {exc}")
            ib_monitor = None

    try:
        # ── VIX Spike Pre-Check ───────────────────────────────────────────────
        _vix_spike_alerts = _check_vix_spike_for_positions(positions)
        alerts.extend(_vix_spike_alerts)

        for pos in positions:
            try:
                alert = _check_position(pos, profit_target_pct, stop_loss_pct, today, ib_monitor)
                if alert:
                    alerts.append(alert)
            except Exception as exc:
                logger.error(f"Monitor failed for {pos.ticker}: {exc}")

    finally:
        if ib_monitor and disconnect_ibkr:
            try:
                disconnect_ibkr(ib_monitor)
            except Exception:
                pass

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
    ib_monitor=None,
) -> Optional[MonitorAlert]:
    """Check a single position for exit signals."""

    # ── Expiry Check (hard expired only — EXPIRING_SOON moved below P&L) ─────
    dte_remaining = None
    try:
        exp_date = datetime.strptime(pos.expiration, "%Y-%m-%d").date()
        dte_remaining = (exp_date - today).days

        if dte_remaining <= 0:
            return MonitorAlert(
                trade_id=getattr(pos, "trade_id", ""),
                ticker=pos.ticker,
                strategy=pos.strategy,
                alert_type="EXPIRED",
                message=f"Option EXPIRED on {pos.expiration}. Record outcome.",
                action="CLOSE",
                priority=1,
            )
    except Exception as exc:
        logger.debug(f"Expiry parse error for {pos.ticker}: {exc}")

    # ── Price-based P&L Check ──────────────────────────────────────────────────
    current_price = _get_current_price(pos.ticker, ib_monitor)
    if current_price is None:
        # FIX #3: Still emit EXPIRING_SOON if we can't get a price but DTE is low
        if dte_remaining is not None and 0 < dte_remaining <= 5:
            return MonitorAlert(
                ticker=pos.ticker,
                strategy=pos.strategy,
                alert_type="EXPIRING_SOON",
                message=f"Only {dte_remaining} DTE remaining. Consider rolling or closing.",
                current_price=0,
                action="REVIEW",
                priority=2,
            )
        return None

    # ── Long Option Route ─────────────────────────────────────────────────────
    is_long = getattr(pos, "is_long_option", False) or pos.strategy in ("LONG_CALL", "LONG_PUT")
    if is_long:
        alert = _check_long_option(pos, current_price, today)
        if alert:
            # P0 FIX #2: record exit AND close position for ALL auto-close events
            if alert.action == "CLOSE":
                _auto_close_position(pos, alert)
        return alert

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
            _alert = MonitorAlert(
                trade_id=getattr(pos, "trade_id", ""),
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
            _auto_close_position(pos, _alert)  # P0 FIX #2
            return _alert

        # Stop loss: spread expanded to 2x credit received
        loss_pct = (current_value - entry_value) / entry_value * 100
        if loss_pct >= stop_loss_pct:
            _alert = MonitorAlert(
                trade_id=getattr(pos, "trade_id", ""),
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
            _auto_close_position(pos, _alert)  # P0 FIX #2
            return _alert

    # Debit spread: estimate value and compare to entry debit
    if pos.net_credit < 0:
        estimated_value = _estimate_debit_spread_value(pos, current_price)
        entry_debit = abs(pos.net_credit)
        gain_pct = (estimated_value - entry_debit) / entry_debit * 100 if entry_debit > 0 else 0
        loss_pct = -gain_pct

        if gain_pct >= profit_target_pct:
            _alert = MonitorAlert(
                trade_id=getattr(pos, "trade_id", ""),
                ticker=pos.ticker,
                strategy=pos.strategy,
                alert_type="PROFIT_TARGET",
                message=(
                    f"DEBIT SPREAD PROFIT TARGET +{gain_pct:.0f}%: "
                    f"spread ≈ ${estimated_value:.2f} vs debit ${entry_debit:.2f}. "
                    f"Stock at ${current_price:.2f}"
                ),
                current_price=current_price,
                entry_credit=entry_debit,
                current_spread_value=estimated_value,
                pnl_pct=gain_pct,
                action="CLOSE",
                priority=1,
            )
            _auto_close_position(pos, _alert)  # P0 FIX #2
            return _alert
        if loss_pct >= stop_loss_pct:
            _alert = MonitorAlert(
                trade_id=getattr(pos, "trade_id", ""),
                ticker=pos.ticker,
                strategy=pos.strategy,
                alert_type="STOP_LOSS",
                message=(
                    f"DEBIT SPREAD STOP LOSS -{loss_pct:.0f}%: "
                    f"spread ≈ ${estimated_value:.2f} vs debit ${entry_debit:.2f}. "
                    f"Stock at ${current_price:.2f}"
                ),
                current_price=current_price,
                entry_credit=entry_debit,
                current_spread_value=estimated_value,
                pnl_pct=-loss_pct,
                action="CLOSE",
                priority=1,
            )
            _auto_close_position(pos, _alert)  # P0 FIX #2
            return _alert

    # FIX #3: EXPIRING_SOON check AFTER P&L checks so profit/stop alerts take priority
    if dte_remaining is not None and 0 < dte_remaining <= 5:
        return MonitorAlert(
            trade_id=getattr(pos, "trade_id", ""),
            ticker=pos.ticker,
            strategy=pos.strategy,
            alert_type="EXPIRING_SOON",
            message=f"Only {dte_remaining} DTE remaining. Consider rolling or closing.",
            current_price=current_price,
            action="REVIEW",
            priority=2,
        )

    return None


def _check_vix_spike_for_positions(positions) -> list[MonitorAlert]:
    """
    Pull current VIX; generate VIX_SPIKE alerts for open credit spreads when:
    (a) VIX >= VIX_DEFENSIVE_LEVEL, or
    (b) VIX spiked > VIX_SPIKE_THRESHOLD_PCT above its recent rolling average.
    """
    alerts: list[MonitorAlert] = []
    try:
        vix_hist = yf.Ticker("^VIX").history(period="15d")
        if vix_hist.empty or len(vix_hist) < 2:
            return alerts
        vix_now = round(float(vix_hist["Close"].iloc[-1]), 1)
        # FIX #6: Exclude today's value from the rolling average so a spike
        # today isn't diluted by including itself in the baseline.
        window = min(VIX_SPIKE_WINDOW, len(vix_hist) - 1)
        if window < 1:
            return alerts
        vix_prior_avg = round(float(vix_hist["Close"].iloc[-(window + 1):-1].mean()), 1)
    except Exception as exc:
        logger.warning(f"VIX fetch for spike check failed: {exc}")
        return alerts

    spike_pct = (vix_now - vix_prior_avg) / vix_prior_avg * 100 if vix_prior_avg > 0 else 0
    is_spike = spike_pct > VIX_SPIKE_THRESHOLD_PCT
    is_defensive = vix_now >= VIX_DEFENSIVE_LEVEL

    if not is_defensive and not is_spike:
        return alerts  # nothing to flag

    spike_note = f" (spike +{spike_pct:.0f}% vs {window}d avg)" if is_spike else ""
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
                    f"(>= DEFENSIVE level if applicable). "
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


def _get_current_price(ticker: str, ib_monitor=None) -> Optional[float]:
    """Fetch current stock price from IBKR, fallback to yfinance."""
    price = None
    if ib_monitor and fetch_stock_snapshot:
        try:
            snap = fetch_stock_snapshot(ib_monitor, ticker)
            if snap and snap.get("last", 0) > 0:
                price = float(snap["last"])
        except Exception as exc:
            logger.debug(f"{ticker}: IBKR snapshot failed ({exc}) -- falling back to yfinance")

    if price is not None:
        return price

    return _yfinance_fallback(ticker)


def _yfinance_fallback(ticker: str) -> Optional[float]:
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.debug(f"{ticker}: yfinance fallback failed: {exc}")
        return None


def _check_long_option(
    pos: "OpenPosition",
    current_price: float,
    today,
    profit_target_pct: float = LONG_OPTION_PROFIT_TARGET_PCT,
    stop_loss_pct: float = LONG_OPTION_STOP_LOSS_PCT,
    time_stop_dte: int = LONG_OPTION_TIME_STOP_DTE,
) -> Optional[MonitorAlert]:
    """
    Exit rules for LONG_CALL / LONG_PUT positions.

    Profit target : premium gained >= 100% of entry debit (2x debit)
    Stop loss     : premium lost   >=  50% of entry debit (0.5x debit)
    Time stop     : <= 10 DTE remaining -> close to avoid gamma/theta drain
    """
    try:
        exp_date = datetime.strptime(pos.expiration, "%Y-%m-%d").date()
        dte_remaining = (exp_date - today).days
    except Exception as exc:
        logger.debug(f"{pos.ticker}: expiry parse error: {exc}")
        dte_remaining = 999

    # Expired check
    if dte_remaining <= 0:
        return MonitorAlert(
            trade_id=getattr(pos, "trade_id", ""),
            ticker=pos.ticker,
            strategy=pos.strategy,
            alert_type="EXPIRED",
            message=f"Long option EXPIRED on {pos.expiration}. Record outcome.",
            current_price=current_price,
            action="CLOSE",
            priority=1,
        )

    # debit paid is stored as |net_credit| (negative net_credit means debit)
    entry_debit = abs(pos.net_credit) if pos.net_credit != 0 else pos.max_risk
    if entry_debit <= 0:
        logger.debug(f"{pos.ticker}: entry_debit=0, skipping long-option P&L check")
        return None

    # Estimate current option value
    current_option_value = _estimate_long_option_value(pos, current_price, dte_remaining)
    gain = current_option_value - entry_debit
    gain_pct = gain / entry_debit * 100

    # FIX #8: Check profit target and stop loss BEFORE time-stop so that
    # a position at 9 DTE that has already hit its target gets the correct exit reason.

    # Profit target: +100% gain (option doubled)
    if gain_pct >= profit_target_pct:
        return MonitorAlert(
            trade_id=getattr(pos, "trade_id", ""),
            ticker=pos.ticker,
            strategy=pos.strategy,
            alert_type="PROFIT_TARGET",
            message=(
                f"LONG OPTION PROFIT TARGET +{gain_pct:.0f}%: "
                f"option ~= ${current_option_value:.2f} vs debit ${entry_debit:.2f}. "
                f"Stock at ${current_price:.2f}"
            ),
            current_price=current_price,
            entry_credit=entry_debit,
            current_spread_value=current_option_value,
            pnl_pct=gain_pct,
            action="CLOSE",
            priority=1,
        )

    # Stop loss: -50% loss
    if gain_pct <= -stop_loss_pct:
        return MonitorAlert(
            trade_id=getattr(pos, "trade_id", ""),
            ticker=pos.ticker,
            strategy=pos.strategy,
            alert_type="STOP_LOSS",
            message=(
                f"LONG OPTION STOP LOSS {gain_pct:.0f}%: "
                f"option ~= ${current_option_value:.2f} vs debit ${entry_debit:.2f}. "
                f"Stock at ${current_price:.2f}"
            ),
            current_price=current_price,
            entry_credit=entry_debit,
            current_spread_value=current_option_value,
            pnl_pct=gain_pct,
            action="CLOSE",
            priority=1,
        )

    # Time-stop check AFTER profit/stop (FIX #8)
    if 0 < dte_remaining <= time_stop_dte:
        return MonitorAlert(
            trade_id=getattr(pos, "trade_id", ""),
            ticker=pos.ticker,
            strategy=pos.strategy,
            alert_type="TIME_STOP",
            message=(
                f"LONG OPTION TIME-STOP: {dte_remaining} DTE <= {time_stop_dte}. "
                f"P&L: {gain_pct:+.0f}%. "
                "Close to avoid accelerated theta decay."
            ),
            current_price=current_price,
            entry_credit=entry_debit,
            current_spread_value=current_option_value,
            pnl_pct=gain_pct,
            action="CLOSE",
            priority=1,
        )

    return None


def _estimate_long_option_value(
    pos: "OpenPosition",
    current_price: float,
    dte_remaining: int = -1,
) -> float:
    """
    Rough intrinsic + decaying time-value estimate for a long option.
    Real monitoring should use live options quotes.
    """
    # FIX #1: For long options the bought strike is stored in long_strike.
    # Fall back to short_strike only if long_strike is missing (legacy data).
    # FIX #15: Clean up is_call comparison.
    is_call = pos.strategy == "LONG_CALL"
    strike = pos.long_strike if pos.long_strike > 0 else pos.short_strike
    if strike <= 0:
        return abs(pos.net_credit)  # no strike info -- return entry cost

    if is_call:
        intrinsic = max(current_price - strike, 0)
    else:
        intrinsic = max(strike - current_price, 0)

    # FIX #2: Time value decays linearly from 20% of entry debit at open
    # to 0 at expiration, instead of a constant 20%.
    entry_debit = abs(pos.net_credit) if pos.net_credit != 0 else pos.max_risk
    dte_at_entry = getattr(pos, "dte_at_entry", 0) or 35  # default if missing
    if dte_remaining >= 0 and dte_at_entry > 0:
        decay_fraction = max(dte_remaining / dte_at_entry, 0.0)
    else:
        decay_fraction = 0.5  # unknown -- assume midpoint
    time_val = entry_debit * 0.20 * decay_fraction
    return round(intrinsic + time_val, 2)


def _estimate_debit_spread_value(pos: "OpenPosition", current_price: float) -> float:
    """
    Rough estimate of current debit-spread value from intrinsic value.
    Used for BULL_CALL_SPREAD and BEAR_PUT_SPREAD.
    """
    if pos.strategy in ("BULL_CALL_SPREAD", "BULL_CALL"):
        long_val = max(current_price - pos.long_strike, 0)
        short_val = max(current_price - pos.short_strike, 0)
        return max(round(long_val - short_val, 2), 0.01)
    elif pos.strategy in ("BEAR_PUT_SPREAD", "BEAR_PUT"):
        long_val = max(pos.long_strike - current_price, 0)
        short_val = max(pos.short_strike - current_price, 0)
        return max(round(long_val - short_val, 2), 0.01)
    return abs(pos.net_credit)


def _estimate_credit_spread_value(
    pos: OpenPosition,
    current_price: float,
) -> float:
    """
    Rough estimate of current spread value based on intrinsic value.
    This is a proxy -- real monitoring needs live options quotes.
    """
    if pos.strategy in ("BULL_PUT_SPREAD", "BULL_PUT"):
        intrinsic = max(pos.short_strike - current_price, 0) - max(pos.long_strike - current_price, 0)
        return max(round(intrinsic + pos.net_credit * 0.1, 2), 0.01)

    elif pos.strategy in ("BEAR_CALL_SPREAD", "BEAR_CALL"):
        intrinsic = max(current_price - pos.short_strike, 0) - max(current_price - pos.long_strike, 0)
        return max(round(intrinsic + pos.net_credit * 0.1, 2), 0.01)

    # Default: return entry credit (no information)
    return pos.net_credit


def _auto_close_position(pos: "OpenPosition", alert: "MonitorAlert") -> None:
    """
    P0 FIX #2 helper: atomically record the exit in outcome_tracker AND
    call close_position() in portfolio so the position is marked CLOSED.
    Called for every alert with action=="CLOSE" regardless of strategy type.
    """
    trade_id = getattr(pos, "trade_id", "")
    exit_price = getattr(alert, "current_spread_value", 0.0) or 0.0
    close_reason = getattr(alert, "alert_type", "MONITOR_AUTO_CLOSE")

    # 1. Record the exit in outcome_tracker (updates JSONL with P&L)
    if trade_id:
        try:
            record_exit(trade_id, exit_price=exit_price, close_reason=close_reason)
            logger.info(
                f"Auto-closed {pos.ticker} [{close_reason}]: "
                f"trade_id={trade_id} exit=${exit_price:.2f}"
            )
        except Exception as exc:
            logger.warning(f"record_exit failed for {pos.ticker} ({trade_id}): {exc}")

    # 2. Mark position as CLOSED in portfolio (removes from open positions)
    try:
        close_position(
            ticker=pos.ticker,
            strategy=pos.strategy,
            exit_price=exit_price,
            trade_id=trade_id or None,
        )
        logger.info(f"Portfolio position closed for {pos.ticker} via monitor")
    except Exception as exc:
        logger.warning(f"close_position failed for {pos.ticker}: {exc}")


def format_monitor_report(alerts: list[MonitorAlert]) -> str:
    """Format alerts into a WhatsApp-friendly message."""
    if not alerts:
        return f"OPTIONS MONITOR -- {date.today()}\nAll positions nominal. No action needed."

    lines = [f"OPTIONS MONITOR -- {date.today()}", "ALERTS:"]

    for a in alerts:
        emoji = "X" if a.priority == 1 else "!"
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
