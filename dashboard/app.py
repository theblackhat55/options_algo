"""
dashboard/app.py
================
Streamlit dashboard for options_algo.

Pages:
  1. Today's Picks     — Top signals from latest scan
  2. IV Heatmap        — Universe ranked by IV rank
  3. Regime Map        — Color-coded regime distribution
  4. Trade Log         — Paper trade history + P&L
  5. Strategy Stats    — Win rates per strategy
  6. Run Scan          — On-demand nightly scan trigger

Usage:
    streamlit run dashboard/app.py
    streamlit run dashboard/app.py --server.port 8501
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SIGNALS_DIR, TRADES_DIR
from src.pipeline.outcome_tracker import load_outcomes, get_win_rate

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Algo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Options Algo")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Navigation",
    ["Today's Picks", "IV Heatmap", "Regime Map", "Trade Log", "Strategy Stats", "Run Scan"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Date: {date.today()}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_latest_signal() -> dict | None:
    path = SIGNALS_DIR / "options_signal_latest.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_all_signals(days: int = 30) -> list[dict]:
    signals = []
    for i in range(days):
        d = (date.today() - timedelta(days=i)).isoformat()
        p = SIGNALS_DIR / f"options_signal_{d}.json"
        if p.exists():
            try:
                with open(p) as f:
                    signals.append(json.load(f))
            except Exception:
                pass
    return signals


def regime_color(regime: str) -> str:
    colors = {
        "STRONG_UPTREND": "#00C853",
        "UPTREND": "#69F0AE",
        "RANGE_BOUND": "#FFD740",
        "DOWNTREND": "#FF6D00",
        "STRONG_DOWNTREND": "#D50000",
        "SQUEEZE": "#AA00FF",
        "REVERSAL_UP": "#00B0FF",
        "REVERSAL_DOWN": "#FF4081",
    }
    return colors.get(regime, "#9E9E9E")


def direction_badge(direction: str) -> str:
    if direction == "BULLISH":
        return "🟢 BULLISH"
    elif direction == "BEARISH":
        return "🔴 BEARISH"
    return "🟡 NEUTRAL"


# ─── Page: Today's Picks ──────────────────────────────────────────────────────

def page_todays_picks():
    st.title("📈 Today's Picks")

    signal = load_latest_signal()
    if signal is None:
        st.warning("No signal found. Run the nightly scan first.")
        st.code("python -m src.pipeline.nightly_scan --dry-run")
        return

    scan_date = signal.get("scan_date", "?")
    st.caption(f"Scan date: {scan_date} | Generated: {signal.get('generated_at', '')[:16]}")

    # Market context banner
    mkt = signal.get("market_context", {})
    if mkt:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Market Regime", mkt.get("market_regime", "?"))
        c2.metric("VIX Proxy", f"{mkt.get('vix_level', 0):.0f}")
        c3.metric("SPY 5d", f"{mkt.get('spy_return_5d', 0):+.1f}%")
        c4.metric("Breadth", f"{mkt.get('breadth_score', 0):.0%}")
        c5.metric("SPY Trend", mkt.get("spy_trend", "?"))

        if mkt.get("notes"):
            st.warning(f"⚠️ {mkt['notes']}")

    st.markdown("---")

    picks = signal.get("top_picks", [])
    if not picks:
        st.info("No picks generated today.")
        return

    st.subheader(f"Top {len(picks)} Picks")

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
        sector = ctx.get("sector", "")
        price = ctx.get("price", 0)

        with st.expander(
            f"#{p.get('priority', '?')}  {ticker}  [{sector}]  —  {strategy}  "
            f"({direction_badge(direction)})  |  Conf: {conf:.0%}",
            expanded=(p.get("priority", 10) <= 3),
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Setup**")
                st.write(f"Price: **${price:.2f}**")
                st.write(f"Regime: `{rec.get('regime', '?')}`")
                st.write(f"Direction Score: {reg.get('direction_score', 0):.2f}")
                st.write(f"ADX: {reg.get('adx', 0):.0f} | RSI: {reg.get('rsi', 0):.0f}")
                st.write(f"EMA: {reg.get('ema_alignment', '?')} | Vol: {reg.get('volume_trend', '?')}")
                if reg.get("bb_squeeze"):
                    st.success("BB Squeeze Active")

            with col2:
                st.markdown("**Volatility**")
                st.write(f"IV Rank: **{iv.get('iv_rank', 0):.0f}%**")
                st.write(f"IV Pctile: {iv.get('iv_percentile', 0):.0f}%")
                st.write(f"Current IV: {iv.get('current_iv', 0):.1f}%")
                st.write(f"HV-20: {iv.get('hv_20', 0):.1f}%")
                st.write(f"IV/HV: **{iv.get('iv_hv_ratio', 0):.2f}**")
                st.write(f"IV Trend: {iv.get('iv_trend', '?')}")
                st.write(f"Action: **{iv.get('premium_action', '?')}**")

            with col3:
                st.markdown("**Trade**")
                if not trade.get("dry_run"):
                    spread_type = trade.get("spread_type", "")
                    if spread_type in ("BULL_PUT", "BEAR_CALL"):
                        st.write(f"Sell **{trade.get('short_strike', 0):.1f}** / Buy **{trade.get('long_strike', 0):.1f}**")
                        st.write(f"Expiry: {trade.get('expiration', '?')} ({trade.get('dte', 0)}d)")
                        st.write(f"Credit: **${trade.get('net_credit', 0):.2f}**")
                        st.write(f"Max Risk: ${trade.get('max_risk', 0):.2f}")
                        st.write(f"R/R Ratio: {trade.get('risk_reward_ratio', 0):.1f}x")
                        st.write(f"Breakeven: ${trade.get('breakeven', 0):.2f}")
                        st.write(f"PoP: **{trade.get('prob_profit', 0):.0f}%**")
                    elif "BUTTERFLY" in str(trade.get("spread_type", "")):
                        st.write(f"Buy {trade.get('lower_wing', 0):.1f} / Sell 2× {trade.get('body', 0):.1f} / Buy {trade.get('upper_wing', 0):.1f}")
                        st.write(f"Expiry: {trade.get('expiration', '?')} ({trade.get('dte', 0)}d)")
                        st.write(f"Debit: **${trade.get('net_debit', 0):.2f}**")
                        st.write(f"Max Profit: ${trade.get('max_profit', 0):.2f}")
                        st.write(f"PoP: **{trade.get('prob_profit', 0):.0f}%**")
                    elif "CONDOR" in rec.get("strategy", "") or "IRON" in rec.get("strategy", ""):
                        st.write(f"Put: {trade.get('long_put', 0):.1f}/{trade.get('short_put', 0):.1f}")
                        st.write(f"Call: {trade.get('short_call', 0):.1f}/{trade.get('long_call', 0):.1f}")
                        st.write(f"Expiry: {trade.get('expiration', '?')} ({trade.get('dte', 0)}d)")
                        st.write(f"Total Credit: **${trade.get('total_credit', 0):.2f}**")
                        st.write(f"Zone: {trade.get('put_breakeven', 0):.1f} — {trade.get('call_breakeven', 0):.1f}")
                        st.write(f"PoP: **{trade.get('prob_profit', 0):.0f}%**")
                    else:
                        st.write(f"Buy {trade.get('long_strike', 0):.1f} / Sell {trade.get('short_strike', 0):.1f}")
                        st.write(f"Expiry: {trade.get('expiration', '?')} ({trade.get('dte', 0)}d)")
                        st.write(f"Debit: **${trade.get('net_debit', 0):.2f}**")
                        st.write(f"Max Profit: ${trade.get('max_profit', 0):.2f}")
                        st.write(f"PoP: **{trade.get('prob_profit', 0):.0f}%**")
                else:
                    st.info("Dry-run mode — no trade details")

                if rs.get("rs_rank") is not None:
                    st.write(f"RS Rank: {rs.get('rs_rank', 0):.0f}th pctile ({rs.get('rs_trend', '?')})")

            st.caption(f"Rationale: {rec.get('rationale', '')}")
            st.caption(f"Composite score: {p.get('composite_score', 0):.4f}")

    # Scan stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Universe", signal.get("universe_size", 0))
    col2.metric("Qualified", signal.get("qualified", 0))
    col3.metric("After Events", signal.get("after_event_filter", 0))
    col4.metric("Elapsed", f"{signal.get('elapsed_seconds', 0):.0f}s")

    # Regime distribution
    regime_dist = signal.get("regime_distribution", {})
    if regime_dist:
        st.markdown("### Regime Distribution")
        df_reg = pd.DataFrame(
            list(regime_dist.items()), columns=["Regime", "Count"]
        ).sort_values("Count", ascending=False)
        fig = px.bar(
            df_reg, x="Regime", y="Count",
            color="Regime",
            color_discrete_map={r: regime_color(r) for r in df_reg["Regime"]},
            title="Universe Regime Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── Page: IV Heatmap ─────────────────────────────────────────────────────────

def page_iv_heatmap():
    st.title("🌡️ IV Heatmap")

    signal = load_latest_signal()
    if signal is None:
        st.warning("No signal data. Run nightly scan first.")
        return

    recs = signal.get("all_recommendations", [])
    if not recs:
        st.info("No recommendation data in latest signal.")
        return

    # Build IV table from signal context
    picks = signal.get("top_picks", [])
    rows = []
    for p in picks:
        rec = p.get("recommendation", {})
        ctx = p.get("context", {})
        iv = ctx.get("iv_detail", {})
        rows.append({
            "Ticker": rec.get("ticker", "?"),
            "Sector": ctx.get("sector", "?"),
            "IV Rank": iv.get("iv_rank", 0),
            "IV Pctile": iv.get("iv_percentile", 0),
            "Current IV": iv.get("current_iv", 0),
            "HV-20": iv.get("hv_20", 0),
            "IV/HV": iv.get("iv_hv_ratio", 0),
            "IV Trend": iv.get("iv_trend", "?"),
            "Action": iv.get("premium_action", "?"),
            "Strategy": rec.get("strategy", "?"),
        })

    if not rows:
        st.info("Detailed IV data only available for top picks.")
        return

    df = pd.DataFrame(rows).sort_values("IV Rank", ascending=False)

    # Color-coded table
    st.dataframe(
        df.style.background_gradient(subset=["IV Rank"], cmap="RdYlGn_r")
               .background_gradient(subset=["IV/HV"], cmap="RdYlGn_r"),
        use_container_width=True,
        height=400,
    )

    # Scatter: IV Rank vs IV/HV
    fig = px.scatter(
        df,
        x="IV Rank", y="IV/HV",
        color="Action",
        text="Ticker",
        size=[10] * len(df),
        title="IV Rank vs IV/HV Ratio",
        color_discrete_map={"SELL": "#D50000", "NEUTRAL": "#FFD740", "BUY": "#00C853"},
    )
    fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="IV High (70)")
    fig.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="IV Low (30)")
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="IV=HV")
    st.plotly_chart(fig, use_container_width=True)


# ─── Page: Regime Map ─────────────────────────────────────────────────────────

def page_regime_map():
    st.title("🗺️ Regime Map")

    signal = load_latest_signal()
    if signal is None:
        st.warning("No signal data.")
        return

    recs = signal.get("all_recommendations", [])
    if not recs:
        st.info("No recommendation data.")
        return

    df = pd.DataFrame(recs)

    # Regime distribution pie
    regime_counts = df["regime"].value_counts().reset_index()
    regime_counts.columns = ["Regime", "Count"]

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            regime_counts, names="Regime", values="Count",
            title="Regime Distribution",
            color="Regime",
            color_discrete_map={r: regime_color(r) for r in regime_counts["Regime"]},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Strategy distribution
        strat_counts = df["strategy"].value_counts().reset_index()
        strat_counts.columns = ["Strategy", "Count"]
        fig_bar = px.bar(
            strat_counts, x="Count", y="Strategy",
            orientation="h", title="Strategy Distribution",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed table
    st.markdown("### All Recommendations")
    display_df = df[["ticker", "regime", "iv_regime", "strategy", "direction", "confidence"]].copy()
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.0%}")
    st.dataframe(display_df, use_container_width=True, height=400)


# ─── Page: Trade Log ─────────────────────────────────────────────────────────

def page_trade_log():
    st.title("📋 Trade Log")

    try:
        df = load_outcomes(only_closed=False)
    except Exception as e:
        st.error(f"Failed to load outcomes: {e}")
        return

    if df.empty:
        st.info("No trades recorded yet. Trades are logged automatically during nightly scan.")
        return

    # Summary metrics
    closed = df[df["outcome"] != "OPEN"]
    open_pos = df[df["outcome"] == "OPEN"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades", len(df))
    c2.metric("Open", len(open_pos))
    c3.metric("Closed", len(closed))

    if not closed.empty:
        wins = (closed["won"] == True).sum()
        win_rate = wins / len(closed) * 100
        total_pnl = closed["pnl"].sum()
        c4.metric("Win Rate", f"{win_rate:.0f}%")
        c5.metric("Total P&L", f"${total_pnl:+.2f}")

    st.markdown("---")

    # Open positions
    if not open_pos.empty:
        st.subheader("Open Positions")
        open_display = open_pos[[
            "trade_id", "ticker", "strategy", "direction",
            "entry_date", "expiration", "net_credit_or_debit",
            "max_risk", "prob_profit", "confidence"
        ]].copy()
        st.dataframe(open_display, use_container_width=True)

    # Closed trades
    if not closed.empty:
        st.subheader("Closed Trades")
        closed_display = closed[[
            "trade_id", "ticker", "strategy", "direction",
            "entry_date", "exit_date", "net_credit_or_debit",
            "pnl", "pnl_pct", "outcome", "close_reason"
        ]].copy()
        closed_display["pnl"] = closed_display["pnl"].apply(lambda x: f"${x:+.2f}")
        closed_display["pnl_pct"] = closed_display["pnl_pct"].apply(lambda x: f"{x:+.1f}%")

        st.dataframe(
            closed_display.sort_values("exit_date", ascending=False),
            use_container_width=True,
        )

        # P&L chart
        if "exit_date" in closed.columns and "pnl" in closed.columns:
            pnl_ts = closed.sort_values("exit_date").copy()
            pnl_ts["cumulative_pnl"] = pnl_ts["pnl"].cumsum()
            fig = px.line(
                pnl_ts, x="exit_date", y="cumulative_pnl",
                title="Cumulative P&L",
                labels={"exit_date": "Date", "cumulative_pnl": "Cumulative P&L ($)"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)


# ─── Page: Strategy Stats ─────────────────────────────────────────────────────

def page_strategy_stats():
    st.title("📊 Strategy Stats")

    try:
        df = load_outcomes(only_closed=True)
    except Exception as e:
        st.error(f"Failed to load outcomes: {e}")
        return

    if df.empty:
        st.info("No closed trades yet.")
        return

    strategies = df["strategy"].unique()

    rows = []
    for strat in strategies:
        stats = get_win_rate(strat)
        rows.append(stats)

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        st.info("No strategy stats available.")
        return

    st.dataframe(stats_df, use_container_width=True)

    # Win rate bar chart
    fig = px.bar(
        stats_df, x="strategy", y="win_rate",
        title="Win Rate by Strategy",
        color="win_rate",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50%")
    st.plotly_chart(fig, use_container_width=True)

    # Avg P&L by strategy
    if "avg_pnl" in stats_df.columns:
        fig2 = px.bar(
            stats_df, x="strategy", y="avg_pnl",
            title="Average P&L by Strategy ($)",
            color="avg_pnl",
            color_continuous_scale="RdYlGn",
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)


# ─── Page: Run Scan ───────────────────────────────────────────────────────────

def page_run_scan():
    st.title("▶️ Run Scan")
    st.info(
        "Trigger a nightly scan on demand. "
        "Use **Dry Run** to test without fetching live options data."
    )

    dry_run = st.checkbox("Dry Run (skip live options data)", value=True)

    custom_tickers_raw = st.text_input(
        "Custom tickers (comma-separated, leave blank for full universe)",
        placeholder="AAPL, MSFT, NVDA, SPY"
    )

    if st.button("🚀 Run Nightly Scan", type="primary"):
        custom_tickers = None
        if custom_tickers_raw.strip():
            custom_tickers = [t.strip().upper() for t in custom_tickers_raw.split(",") if t.strip()]

        with st.spinner("Running scan... this may take 1–3 minutes"):
            try:
                import logging
                logging.basicConfig(level=logging.INFO)
                from src.pipeline.nightly_scan import run_nightly_scan
                signal = run_nightly_scan(
                    universe_override=custom_tickers,
                    dry_run=dry_run,
                )
                st.success(
                    f"Scan complete! "
                    f"{len(signal.get('top_picks', []))} picks generated in "
                    f"{signal.get('elapsed_seconds', 0):.1f}s"
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Scan failed: {exc}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")
    st.markdown("### OpenClaw Cron Setup")
    st.code("""
# Nightly scan (9:30 PM ET = 02:30 UTC) Tue-Sat
openclaw cron add \\
  --name "Options nightly scan" \\
  --cron "30 2 * * 2-6" \\
  --tz UTC --exact \\
  --session isolated \\
  --message "Run the options algo nightly scan. Execute: sudo /root/options_algo/.venv/bin/python3 -m src.pipeline.nightly_scan. Show the top 5 picks with strategy, strikes, credit/debit, max risk, and probability of profit." \\
  --announce --channel whatsapp --to "your_whatsapp_number_here" \\
  --timeout-seconds 600

# Morning brief (9:00 AM ET = 14:00 UTC) Mon-Fri
openclaw cron add \\
  --name "Options morning brief" \\
  --cron "0 14 * * 1-5" \\
  --tz UTC --exact \\
  --session isolated \\
  --message "Run: sudo /root/options_algo/.venv/bin/python3 -m src.pipeline.morning_brief. Send the formatted options picks for today via WhatsApp." \\
  --announce --channel whatsapp --to "your_whatsapp_number_here" \\
  --timeout-seconds 120
""", language="bash")


# ─── Router ───────────────────────────────────────────────────────────────────

if page == "Today's Picks":
    page_todays_picks()
elif page == "IV Heatmap":
    page_iv_heatmap()
elif page == "Regime Map":
    page_regime_map()
elif page == "Trade Log":
    page_trade_log()
elif page == "Strategy Stats":
    page_strategy_stats()
elif page == "Run Scan":
    page_run_scan()
