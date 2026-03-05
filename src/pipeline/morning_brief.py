"""
src/pipeline/morning_brief.py
=============================
Format the nightly scan signal into a WhatsApp-friendly message.
Delivered via OpenClaw at 9:00 AM ET.

At market open, re-enriches top picks with LIVE IBKR data (options flow,
live IV, volume pace) — replacing the empty pre-market values from the
nightly scan. Enriched data is also saved back to the signal JSON so
the Streamlit dashboard reflects live context.

Usage:
    python -m src.pipeline.morning_brief

    from src.pipeline.morning_brief import format_morning_brief
    msg = format_morning_brief()
    print(msg)
"""
from __future__ import annotations

import json
import logging
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import SIGNALS_DIR, IBKR_HOST, IBKR_PORT

logger = logging.getLogger(__name__)

# WhatsApp-friendly formatting: no markdown, plain text
MAX_LINES = 50
SEPARATOR = "─" * 30


def _ibkr_reachable() -> bool:
    """Quick TCP check — avoids hanging if IB Gateway is down."""
    try:
        with socket.create_connection((IBKR_HOST, IBKR_PORT), timeout=3):
            return True
    except OSError:
        return False


def _enrich_picks_live(picks: list[dict]) -> tuple[list[dict], str]:
    """
    Re-fetch IBKR options flow / live IV for each pick at market-open time.

    Returns:
        (enriched_picks, ibkr_status_note)
        ibkr_status_note is a short string for the brief footer:
          "IBKR: live data ✓ (3/3 tickers)"
          "IBKR: partial (1/3 tickers)"
          "IBKR: unavailable — pre-market data shown"
    """
    if not picks:
        return picks, ""

    if not _ibkr_reachable():
        logger.warning("Morning brief: IBKR not reachable — skipping live enrichment")
        return picks, "IBKR: unavailable — pre-market flow data shown"

    try:
        from src.data.ibkr_client import connect_ibkr, disconnect_ibkr
        from src.data.ibkr_realtime import fetch_realtime_enrichment_fast as fetch_realtime_enrichment
    except ImportError as e:
        logger.warning(f"Morning brief: IBKR modules not importable — {e}")
        return picks, "IBKR: module error — pre-market flow data shown"

    ib = connect_ibkr()
    if not ib:
        return picks, "IBKR: connection failed — pre-market flow data shown"

    enriched_count = 0
    try:
        for p in picks:
            ticker = p.get("recommendation", {}).get("ticker")
            if not ticker:
                continue
            try:
                rt = fetch_realtime_enrichment(ib, ticker)
                if rt and rt.get("data_quality") != "NONE":
                    # Merge live data into context.options_flow
                    p.setdefault("context", {})["options_flow"] = {
                        "flow_score":           rt.get("flow_score", 0),
                        "dominant_side":        rt.get("dominant_side", "NEUTRAL"),
                        "put_call_volume_ratio": rt.get("put_call_volume_ratio", 1.0),
                        "volume_pace":          rt.get("volume_pace", 1.0),
                        "live_iv":              rt.get("iv_pct"),
                        "iv_skew":              rt.get("skew"),
                        "source":               "ibkr_live_morning",
                        "timestamp":            rt.get("timestamp"),
                    }
                    # Confidence adjustment: boost/penalize based on flow alignment
                    direction = p.get("recommendation", {}).get("direction", "NEUTRAL")
                    dominant = rt.get("dominant_side", "NEUTRAL")
                    flow_score = rt.get("flow_score", 0)
                    if flow_score > 60:
                        conf = p.get("recommendation", {}).get("confidence", 0.5)
                        if (dominant == "CALLS" and direction == "BULLISH") or \
                           (dominant == "PUTS" and direction == "BEARISH"):
                            p["recommendation"]["confidence"] = min(conf * 1.15, 0.95)
                        elif (dominant == "CALLS" and direction == "BEARISH") or \
                             (dominant == "PUTS" and direction == "BULLISH"):
                            p["recommendation"]["confidence"] = conf * 0.85
                    enriched_count += 1
                    logger.info(f"  {ticker}: live enrichment OK "
                                f"(flow={rt.get('flow_score',0):.0f}, "
                                f"iv={rt.get('iv_pct')}%)")
            except Exception as exc:
                logger.warning(f"  {ticker}: live enrichment failed — {exc}")
    finally:
        try:
            disconnect_ibkr(ib)
        except Exception:
            pass

    total = len(picks)
    if enriched_count == total:
        note = f"IBKR: live data OK ({enriched_count}/{total} tickers)"
    elif enriched_count > 0:
        note = f"IBKR: partial ({enriched_count}/{total} tickers)"
    else:
        note = "IBKR: no live data — check market hours"

    return picks, note


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

    # ── Live IBKR enrichment at market open ──────────────────────────────────
    ibkr_note = ""
    if picks:
        logger.info("Morning brief: fetching live IBKR enrichment...")
        picks, ibkr_note = _enrich_picks_live(picks)
        signal["top_picks"] = picks
        signal["ibkr_live_enrichment"] = ibkr_note
        signal["ibkr_enriched_at"] = datetime.utcnow().isoformat()
        # Save enriched signal back to disk so dashboard reflects live data
        _save_enriched_signal(signal)

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

        # Options flow (if available from IBKR real-time enrichment)
        flow = ctx.get("options_flow", {})
        flow_score = flow.get("flow_score", 0)
        dominant = flow.get("dominant_side", "")
        vol_pace = flow.get("volume_pace", 1.0)
        if flow_score > 0 or vol_pace != 1.0:
            flow_txt = f"Flow:     Score={flow_score:.0f} | {dominant} | Vol pace={vol_pace:.1f}x"
            if flow.get("live_iv"):
                flow_txt += f" | Live IV={flow.get('live_iv'):.1f}%"
            lines.append(flow_txt)

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
    if ibkr_note:
        lines.append(ibkr_note)

    return "\n".join(lines)


def _save_enriched_signal(signal: dict) -> None:
    """Save the IBKR-enriched signal back to disk (overwrites latest + dated file)."""
    try:
        scan_date = signal.get("scan_date", datetime.utcnow().date().isoformat())
        for path in [
            SIGNALS_DIR / "options_signal_latest.json",
            SIGNALS_DIR / f"options_signal_{scan_date}.json",
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(signal, f, indent=2, default=str)
        logger.info("Morning brief: enriched signal saved to disk")
    except Exception as exc:
        logger.warning(f"Morning brief: failed to save enriched signal — {exc}")


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
