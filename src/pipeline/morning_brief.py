"""
src/pipeline/morning_brief.py
=============================
Format the nightly scan signal into a WhatsApp-friendly message.
Delivered via OpenClaw at 9:00 AM ET.

Usage:
    python -m src.pipeline.morning_brief

    from src.pipeline.morning_brief import format_morning_brief
    msg = format_morning_brief()
    print(msg)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import SIGNALS_DIR

logger = logging.getLogger(__name__)

# WhatsApp-friendly formatting: no markdown, plain text
MAX_LINES = 50
SEPARATOR = "─" * 30


def format_morning_brief(signal: dict = None) -> str:
    """
    Format nightly scan results into a WhatsApp message.

    Args:
        signal: Signal dict (from nightly_scan). If None, loads latest file.

    Returns:
        Formatted plain-text string.
    """
    if signal is None:
        signal = _load_latest_signal()
        if signal is None:
            return "OPTIONS ALGO: No signal found. Run nightly scan first."

    picks = signal.get("top_picks", [])
    scan_date = signal.get("scan_date", "unknown")
    mkt = signal.get("market_context", {})
    regime_dist = signal.get("regime_distribution", {})

    lines = [
        f"OPTIONS ALGO PICKS — {scan_date}",
        SEPARATOR,
    ]

    # Market overview
    market_regime = mkt.get("market_regime", "?")
    spy_ret_5d = mkt.get("spy_return_5d", 0)
    vix = mkt.get("vix_level", 0)
    breadth = mkt.get("breadth_score", 0)
    leaders = ", ".join(mkt.get("sector_leaders", [])[:2])
    laggards = ", ".join(mkt.get("sector_laggards", [])[:2])

    lines.append(f"Market: {market_regime} | SPY 5d: {spy_ret_5d:+.1f}%")
    lines.append(f"VIX: {vix:.0f} | Breadth: {breadth:.0%}")
    lines.append(f"Leaders: {leaders}")
    lines.append(f"Laggards: {laggards}")

    if mkt.get("notes"):
        lines.append(f"NOTE: {mkt['notes']}")

    lines.append(SEPARATOR)

    if not picks:
        lines.append("No high-confidence picks today.")
        lines.append("Conditions: " + str(regime_dist))
        return "\n".join(lines)

    lines.append(f"{len(picks)} PICKS:")
    lines.append("")

    for p in picks:
        rec = p.get("recommendation", {})
        trade = p.get("trade", {})
        ctx = p.get("context", {})
        iv = ctx.get("iv_detail", {})
        reg = ctx.get("regime_detail", {})
        rs = ctx.get("rs_detail", {})

        ticker = rec.get("ticker", "?")
        strategy = rec.get("strategy", "?").replace("_", " ")
        direction = rec.get("direction", "?")
        conf = rec.get("confidence", 0)
        rationale = rec.get("rationale", "")
        price = ctx.get("price", 0)
        sector = ctx.get("sector", "")
        priority = p.get("priority", "?")

        lines.append(f"#{priority} {ticker}  [{sector}]")
        lines.append(f"Strategy: {strategy}")
        lines.append(f"Signal:   {direction} | Conf: {conf:.0%}")
        lines.append(f"Price:    ${price:.2f}")

        # IV context
        iv_rank = iv.get("iv_rank", 0)
        iv_hv = iv.get("iv_hv_ratio", 0)
        iv_trend = iv.get("iv_trend", "")
        lines.append(f"IV Rank:  {iv_rank:.0f}% | IV/HV: {iv_hv:.2f} | Trend: {iv_trend}")

        # Technical context
        adx = reg.get("adx", 0)
        rsi = reg.get("rsi", 0)
        ema_align = reg.get("ema_alignment", "")
        squeeze = reg.get("bb_squeeze", False)
        vol_trend = reg.get("volume_trend", "")
        squeeze_txt = " | SQUEEZE" if squeeze else ""
        lines.append(f"Tech:     ADX={adx:.0f} RSI={rsi:.0f} EMA={ema_align}{squeeze_txt} Vol={vol_trend}")

        # RS
        rs_rank = rs.get("rs_rank")
        rs_trend = rs.get("rs_trend")
        if rs_rank is not None:
            lines.append(f"RS Rank:  {rs_rank:.0f}th pctile | {rs_trend}")

        # Trade specifics
        if not trade.get("dry_run"):
            lines.append(_format_trade(trade, rec.get("strategy", "")))

        lines.append(f"Rationale: {rationale}")
        lines.append(SEPARATOR)

    # Scan stats
    lines.append(
        f"Scan: {signal.get('universe_size')} tickers → "
        f"{signal.get('qualified')} qualified → "
        f"{signal.get('after_event_filter')} safe → "
        f"{len(picks)} picks"
    )
    lines.append(f"Generated: {signal.get('generated_at', '')[:16]}")

    return "\n".join(lines)


def _format_trade(trade: dict, strategy_name: str) -> str:
    """Format the specific trade details depending on strategy type."""
    spread_type = trade.get("spread_type", strategy_name)

    if spread_type in ("BULL_PUT", "BEAR_CALL"):
        return (
            f"Trade:    Sell {trade.get('short_strike', 0):.1f} / "
            f"Buy {trade.get('long_strike', 0):.1f} "
            f"({trade.get('expiration', '?')}, {trade.get('dte', 0)}d) | "
            f"Credit ${trade.get('net_credit', 0):.2f} | "
            f"Risk ${trade.get('max_risk', 0):.2f} | "
            f"PoP {trade.get('prob_profit', 0):.0f}%"
        )
    elif spread_type in ("BULL_CALL", "BEAR_PUT"):
        return (
            f"Trade:    Buy {trade.get('long_strike', 0):.1f} / "
            f"Sell {trade.get('short_strike', 0):.1f} "
            f"({trade.get('expiration', '?')}, {trade.get('dte', 0)}d) | "
            f"Debit ${trade.get('net_debit', 0):.2f} | "
            f"Max Profit ${trade.get('max_profit', 0):.2f} | "
            f"PoP {trade.get('prob_profit', 0):.0f}%"
        )
    elif "BUTTERFLY" in spread_type.upper() or "BUTTERFLY" in strategy_name.upper():
        return (
            f"Trade:    Buy {trade.get('lower_wing', 0):.1f} / "
            f"Sell 2x {trade.get('body', 0):.1f} / "
            f"Buy {trade.get('upper_wing', 0):.1f} "
            f"({trade.get('expiration', '?')}, {trade.get('dte', 0)}d) | "
            f"Debit ${trade.get('net_debit', 0):.2f} | "
            f"Max ${trade.get('max_profit', 0):.2f}"
        )
    elif "IRON_CONDOR" in strategy_name.upper() or "CONDOR" in spread_type.upper():
        return (
            f"Trade:    Put {trade.get('long_put', 0):.1f}/{trade.get('short_put', 0):.1f} | "
            f"Call {trade.get('short_call', 0):.1f}/{trade.get('long_call', 0):.1f} "
            f"({trade.get('expiration', '?')}, {trade.get('dte', 0)}d) | "
            f"Credit ${trade.get('total_credit', 0):.2f} | "
            f"Zone {trade.get('put_breakeven', 0):.1f}-{trade.get('call_breakeven', 0):.1f} | "
            f"PoP {trade.get('prob_profit', 0):.0f}%"
        )
    else:
        return f"Trade:    {spread_type} (see signal JSON for details)"


def _load_latest_signal() -> Optional[dict]:
    """Load the most recent signal JSON."""
    path = SIGNALS_DIR / "options_signal_latest.json"
    if not path.exists():
        # Try last N days
        from datetime import date, timedelta
        for i in range(7):
            d = (date.today() - timedelta(days=i)).isoformat()
            p = SIGNALS_DIR / f"options_signal_{d}.json"
            if p.exists():
                path = p
                break
        else:
            return None

    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.error(f"Failed to load signal: {exc}")
        return None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.WARNING)
    msg = format_morning_brief()
    print(msg)
