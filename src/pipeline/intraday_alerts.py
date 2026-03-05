"""
src/pipeline/intraday_alerts.py
================================
Intraday alert engine — runs every 15 minutes during market hours (9:30–16:00 ET).

Responsibilities:
  1. VIX spike / defensive check — warn if VIX crosses defensive threshold
  2. Position monitor — route to position_monitor.py for stop/target/time-stop
  3. Breakout scanner — scan universe for fresh breakout above S/R levels
  4. Return formatted alert strings for downstream dispatch (WhatsApp, Slack, etc.)

Cron entry (every 15 min, Mon-Fri, 9:30–16:00 ET):
    */15 9-15 * * 1-5  /path/.venv/bin/python -m src.pipeline.intraday_alerts \
                           >> /var/log/intraday_alerts.log 2>&1

Usage:
    python -m src.pipeline.intraday_alerts
    from src.pipeline.intraday_alerts import run_intraday_alerts
    alerts = run_intraday_alerts()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import yfinance as yf

from config.settings import (
    VIX_DEFENSIVE_LEVEL,
    VIX_SPIKE_WINDOW,
    VIX_SPIKE_THRESHOLD_PCT,
    BREAKOUT_VOLUME_MULTIPLIER,
    SR_LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class IntradayAlert:
    alert_type: str             # VIX_SPIKE | POSITION_MONITOR | BREAKOUT | BREAKDOWN
    ticker: str = ""
    message: str = ""
    priority: int = 2           # 1=urgent, 2=normal
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_intraday_alerts(
    universe: list[str] | None = None,
    run_position_monitor: bool = True,
    run_breakout_scan: bool = True,
) -> list[IntradayAlert]:
    """
    Run the full intraday alert pipeline and return a list of IntradayAlert objects.

    Args:
        universe:             Optional list of tickers to scan for breakouts.
                              If None, uses the configured universe (or a compact set).
        run_position_monitor: Whether to invoke position_monitor for open positions.
        run_breakout_scan:    Whether to scan for fresh level breakouts.

    Returns:
        List of IntradayAlert objects sorted by priority (urgent first).
    """
    alerts: list[IntradayAlert] = []

    # ── 1. VIX Defensive Check ─────────────────────────────────────────────────
    try:
        vix_alerts = _check_vix_spike()
        alerts.extend(vix_alerts)
    except Exception as exc:
        logger.warning(f"VIX check failed: {exc}")

    # ── 2. Position Monitor ────────────────────────────────────────────────────
    if run_position_monitor:
        try:
            pos_alerts = _run_position_monitor()
            alerts.extend(pos_alerts)
        except Exception as exc:
            logger.warning(f"Position monitor failed: {exc}")

    # ── 3. Breakout Scanner ────────────────────────────────────────────────────
    if run_breakout_scan and universe:
        try:
            breakout_alerts = _scan_breakouts(universe)
            alerts.extend(breakout_alerts)
        except Exception as exc:
            logger.warning(f"Breakout scan failed: {exc}")

    # Sort: priority 1 first, then by timestamp descending
    alerts.sort(key=lambda a: (a.priority, a.timestamp))

    if alerts:
        logger.info(f"Intraday alerts: {len(alerts)} generated")
        for a in alerts:
            logger.info(f"  [{a.priority}] {a.alert_type} {a.ticker}: {a.message[:80]}")
    else:
        logger.debug("Intraday: no alerts this cycle")

    return alerts


# ── VIX Spike / Defensive Check ───────────────────────────────────────────────

def _check_vix_spike() -> list[IntradayAlert]:
    """
    Pull current VIX. Generate an alert if:
      (a) VIX ≥ VIX_DEFENSIVE_LEVEL, or
      (b) VIX spiked > VIX_SPIKE_THRESHOLD_PCT above its rolling average.
    """
    alerts: list[IntradayAlert] = []
    try:
        vix_hist = yf.Ticker("^VIX").history(period="15d")
        if vix_hist.empty or len(vix_hist) < 2:
            return alerts
        vix_now = round(float(vix_hist["Close"].iloc[-1]), 1)
        # FIX #6: Exclude today's value from rolling average so a spike today
        # isn't diluted by including itself in the baseline.
        window = min(VIX_SPIKE_WINDOW, len(vix_hist) - 1)
        if window < 1:
            return alerts
        vix_avg = round(float(vix_hist["Close"].iloc[-(window + 1):-1].mean()), 1)
    except Exception as exc:
        logger.warning(f"VIX intraday fetch failed: {exc}")
        return alerts

    spike_pct = (vix_now - vix_avg) / vix_avg * 100 if vix_avg > 0 else 0
    is_spike = spike_pct > VIX_SPIKE_THRESHOLD_PCT
    is_defensive = vix_now >= VIX_DEFENSIVE_LEVEL

    if is_defensive:
        alerts.append(IntradayAlert(
            alert_type="VIX_DEFENSIVE",
            message=(
                f"VIX at {vix_now:.1f} ≥ DEFENSIVE level ({VIX_DEFENSIVE_LEVEL}). "
                "Consider closing credit spreads. Avoid new directional entries."
            ),
            priority=1,
        ))
    elif is_spike:
        alerts.append(IntradayAlert(
            alert_type="VIX_SPIKE",
            message=(
                f"VIX spiked to {vix_now:.1f} (+{spike_pct:.0f}% vs {window}d avg {vix_avg:.1f}). "
                "Review open credit positions for stop-loss proximity."
            ),
            priority=1,
        ))

    return alerts


# ── Position Monitor Bridge ────────────────────────────────────────────────────

def _run_position_monitor() -> list[IntradayAlert]:
    """Invoke position_monitor and convert MonitorAlert → IntradayAlert."""
    from src.pipeline.position_monitor import monitor_positions, MonitorAlert

    monitor_alerts = monitor_positions()
    intraday: list[IntradayAlert] = []

    for ma in monitor_alerts:
        intraday.append(IntradayAlert(
            alert_type=ma.alert_type,
            ticker=ma.ticker,
            message=(
                f"[{ma.strategy}] {ma.message} | "
                f"Price: ${ma.current_price:.2f} | Action: {ma.action}"
            ),
            priority=ma.priority,
        ))

    return intraday


# ── Breakout Scanner ──────────────────────────────────────────────────────────

def _scan_breakouts(universe: list[str]) -> list[IntradayAlert]:
    """
    For each ticker in the universe, check if today's price has broken
    above/below a key S/R level from the volume profile, with confirmation
    from elevated volume (> BREAKOUT_VOLUME_MULTIPLIER × 20-day avg).
    """
    alerts: list[IntradayAlert] = []

    for ticker in universe[:50]:  # cap at 50 to stay within rate limits
        try:
            alert = _check_breakout(ticker)
            if alert:
                alerts.append(alert)
        except Exception as exc:
            logger.debug(f"Breakout check failed for {ticker}: {exc}")

    return alerts


def _check_breakout(ticker: str) -> Optional[IntradayAlert]:
    """
    Check a single ticker for a volume-confirmed breakout or breakdown.

    FIX #13: Use analyze_levels() from src.analysis.levels for S/R levels
    consistent with the nightly scan, falling back to quantile method.
    """
    try:
        df = yf.Ticker(ticker).history(period=f"{SR_LOOKBACK_DAYS + 5}d")
        if df.empty or len(df) < 30:
            return None
    except Exception as exc:
        logger.debug(f"{ticker}: yfinance fetch failed: {exc}")
        return None

    current_price = float(df["Close"].iloc[-1])
    current_vol = float(df["Volume"].iloc[-1])
    avg_vol_20d = float(df["Volume"].tail(20).mean())

    # Volume confirmation
    vol_elevated = avg_vol_20d > 0 and (current_vol / avg_vol_20d) >= BREAKOUT_VOLUME_MULTIPLIER
    if not vol_elevated:
        return None

    # Try to use the levels module for consistent S/R with nightly scan
    resistance = None
    support = None
    try:
        from src.analysis.levels import analyze_levels
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]
        levels = analyze_levels(ticker, df_lower)
        if levels is not None:
            if getattr(levels, "nearest_resistance", None) and levels.nearest_resistance > 0:
                resistance = levels.nearest_resistance
            if getattr(levels, "nearest_support", None) and levels.nearest_support > 0:
                support = levels.nearest_support
    except Exception as exc:
        logger.debug(f"{ticker}: levels module unavailable ({exc}), using quantile fallback")

    # Fallback to quantile method if levels module didn't produce results
    if resistance is None or support is None:
        lookback_df = df.tail(SR_LOOKBACK_DAYS)
        if resistance is None:
            resistance = float(lookback_df["High"].quantile(0.90))
        if support is None:
            support = float(lookback_df["Low"].quantile(0.10))

    # Breakout above resistance
    if current_price > resistance:
        return IntradayAlert(
            alert_type="BREAKOUT",
            ticker=ticker,
            message=(
                f"BREAKOUT: {ticker} at ${current_price:.2f} > "
                f"resistance ${resistance:.2f}. "
                f"Volume {current_vol / avg_vol_20d:.1f}x 20d avg. "
                "Potential LONG_CALL setup."
            ),
            priority=2,
        )

    # Breakdown below support
    if current_price < support:
        return IntradayAlert(
            alert_type="BREAKDOWN",
            ticker=ticker,
            message=(
                f"BREAKDOWN: {ticker} at ${current_price:.2f} < "
                f"support ${support:.2f}. "
                f"Volume {current_vol / avg_vol_20d:.1f}x 20d avg. "
                "Potential LONG_PUT setup."
            ),
            priority=2,
        )

    return None


# ── Formatter ─────────────────────────────────────────────────────────────────

def format_intraday_report(alerts: list[IntradayAlert]) -> str:
    """Format alerts into a compact, human-readable message."""
    if not alerts:
        return f"INTRADAY MONITOR — {datetime.now(timezone.utc).strftime('%H:%M UTC')} — No alerts."

    lines = [f"INTRADAY ALERTS — {datetime.now(timezone.utc).strftime('%H:%M UTC')}"]
    for a in alerts:
        emoji = "🔴" if a.priority == 1 else "🟡"
        lines.append(f"\n{emoji} [{a.alert_type}]" + (f" {a.ticker}" if a.ticker else ""))
        lines.append(a.message)
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Quick universe from env or argv
    from config.universe import get_universe
    try:
        universe = get_universe()[:30]  # limit to 30 for quick intraday run
    except Exception:
        universe = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META"]

    alerts = run_intraday_alerts(universe=universe)
    print(format_intraday_report(alerts))
