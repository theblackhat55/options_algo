"""
trading_dashboard/app.py
========================
Unified Trading Dashboard — SPX Iron Condor + Options Algo
Single Streamlit app on one port.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Project Roots ─────────────────────────────────────────────────────
SPX_ROOT = Path("/root/spx_algo")
OPT_ROOT = Path("/root/options_algo")

# Add both to sys.path so we can import from either
for p in [str(SPX_ROOT), str(OPT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Command Center",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SPX Paths ─────────────────────────────────────────────────────────
SPX_RAW = SPX_ROOT / "data" / "raw"
SPX_SIGNALS = SPX_ROOT / "output" / "signals"
SPX_TRADES = SPX_ROOT / "output" / "trades" / "paper_trade_log.csv"
SPX_REPORTS = SPX_ROOT / "output" / "reports"

# ── Options Paths ─────────────────────────────────────────────────────
OPT_SIGNALS = OPT_ROOT / "output" / "signals"
OPT_TRADES = OPT_ROOT / "output" / "trades"


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADERS — SPX
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def spx_load_data():
    df = pd.read_parquet(SPX_RAW / "spx_daily.parquet")
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=300)
def spx_load_vix():
    df = pd.read_parquet(SPX_RAW / "vix_daily.parquet")
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=300)
def spx_load_signal():
    p = SPX_SIGNALS / "latest_signal.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return None

@st.cache_data(ttl=300)
def spx_load_replay():
    for name in ["replay_jan_feb_2026_v2.csv", "replay_jan_feb_2026.csv"]:
        p = SPX_REPORTS / name
        if p.exists():
            df = pd.read_csv(p)
            df["date"] = pd.to_datetime(df["date"])
            rename_map = {"h_err": "h_err_pct", "l_err": "l_err_pct",
                          "dir_ok": "dir_correct", "net_pnl": "net_pnl_dollars"}
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def spx_load_paper_log():
    if SPX_TRADES.exists():
        df = pd.read_csv(SPX_TRADES)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def spx_load_market_intel():
    p = SPX_ROOT / "data" / "processed" / "market_intel.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return None

@st.cache_data(ttl=300)
def spx_load_es_levels():
    p = SPX_SIGNALS / "es_levels_latest.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return None

@st.cache_data(ttl=60)
def load_cron_jobs():
    p = Path("/home/openclaw/.openclaw/cron/jobs.json")
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        return data.get("jobs", [])
    return []


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADERS — OPTIONS ALGO
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def opt_load_signal():
    p = OPT_SIGNALS / "options_signal_latest.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════

st.sidebar.title("🏦 Trading Command Center")
st.sidebar.markdown("---")

section = st.sidebar.selectbox("Section", ["SPX Iron Condor", "Options Algo", "System"])

if section == "SPX Iron Condor":
    page = st.sidebar.radio("Navigate", [
        "🏠 Overview",
        "🔮 Latest Signal",
        "📊 ES/MES Levels",
        "📈 Backtest",
        "📋 Paper Trades",
        "⚡ Risk & Intel",
    ])
elif section == "Options Algo":
    page = st.sidebar.radio("Navigate", [
        "🔍 Ticker Analysis",
        "📈 Today's Picks",
        "📋 Trade Tracker",
        "📊 Strategy Stats",
        "🌡️ IV Heatmap",
        "🗺️ Regime Map",
        "▶️ Run Scan",
    ])
else:
    page = st.sidebar.radio("Navigate", [
        "📅 Cron Jobs",
    ])

st.sidebar.markdown("---")
st.sidebar.caption(f"Date: {date.today()}")


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

REGIME_COLORS = {
    "STRONG_UPTREND": "#00C853", "UPTREND": "#69F0AE",
    "RANGE_BOUND": "#FFD740", "DOWNTREND": "#FF6D00",
    "STRONG_DOWNTREND": "#D50000", "SQUEEZE": "#AA00FF",
    "REVERSAL_UP": "#00B0FF", "REVERSAL_DOWN": "#FF4081",
}

def direction_badge(d):
    return {"BULLISH": "🟢 BULLISH", "BEARISH": "🔴 BEARISH"}.get(d, "🟡 NEUTRAL")


# ══════════════════════════════════════════════════════════════════════
#  SPX PAGES
# ══════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("SPX Iron-Condor Algo — Overview")

    try:
        spx = spx_load_data()
        vix = spx_load_vix()
    except Exception:
        st.warning("SPX/VIX data not found. Run daily_cron.sh first.")
        spx, vix = pd.DataFrame(), pd.DataFrame()

    sig = spx_load_signal()
    replay = spx_load_replay()

    if not spx.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("SPX Close", f"{spx['Close'].iloc[-1]:,.2f}",
                   f"{spx['Close'].iloc[-1] - spx['Close'].iloc[-2]:+.2f}")
        c2.metric("VIX", f"{vix['Close'].iloc[-1]:.2f}" if not vix.empty else "N/A")
        if sig:
            c3.metric("Regime", sig.get("regime", "N/A"))
            c4.metric("Direction", sig.get("direction", "N/A"),
                       f"{sig.get('direction_prob', 0)*100:.1f}%")
            c5.metric("Tradeable", "✅" if sig.get("tradeable") else "❌")

    if not replay.empty and "net_pnl_dollars" in replay.columns:
        st.markdown("---")
        st.subheader("Jan-Feb 2026 Replay Summary")
        m1, m2, m3, m4 = st.columns(4)
        wins = len(replay[replay["condor"] == "WIN"])
        total = len(replay)
        m1.metric("Win Rate", f"{wins/total*100:.1f}%")
        m2.metric("Total P&L", f"${replay['net_pnl_dollars'].sum():,.0f}")
        equity = replay["net_pnl_dollars"].cumsum()
        m3.metric("Max DD", f"${(equity - equity.cummax()).min():,.0f}")
        m4.metric("Trades", total)

    if not spx.empty:
        st.markdown("---")
        st.subheader("SPX Last 60 Days")
        recent = spx.tail(60)
        fig = go.Figure(data=[go.Candlestick(
            x=recent.index, open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"], name="SPX"
        )])
        fig.update_layout(height=400, xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width="stretch")


elif page == "🔮 Latest Signal":
    st.title("SPX — Latest Signal")
    sig = spx_load_signal()
    if sig:
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Date", sig.get("signal_date", "N/A"))
        h2.metric("Regime", sig.get("regime", "N/A"))
        h3.metric("Direction", sig.get("direction", "N/A"), f"{sig.get('direction_prob',0)*100:.1f}%")
        h4.metric("VIX", f"{sig.get('vix_spot', 0):.2f}")
        h5.metric("Tradeable", "✅" if sig.get("tradeable") else "❌")

        st.markdown("---")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("**Predictions**")
            st.write(f"Prior Close: {sig.get('prior_close',0):,.2f}")
            st.write(f"Pred High: {sig.get('predicted_high',0):,.2f}")
            st.write(f"Pred Low: {sig.get('predicted_low',0):,.2f}")
        with p2:
            st.markdown("**IC Strikes**")
            st.write(f"Short Call: {sig.get('ic_short_call',0):,.2f}")
            st.write(f"Short Put: {sig.get('ic_short_put',0):,.2f}")
            st.write(f"Long Call: {sig.get('ic_long_call',0):,.2f}")
            st.write(f"Long Put: {sig.get('ic_long_put',0):,.2f}")
        with p3:
            st.markdown("**Conformal Bands**")
            st.write(f"68% High: [{sig.get('conf_68_high_lo',0):,.2f}, {sig.get('conf_68_high_hi',0):,.2f}]")
            st.write(f"90% High: [{sig.get('conf_90_high_lo',0):,.2f}, {sig.get('conf_90_high_hi',0):,.2f}]")

        with st.expander("Raw JSON"):
            st.json(sig)
    else:
        st.warning("No signal found.")


elif page == "📊 ES/MES Levels":
    st.title("ES/MES Futures Trading Levels")
    es = spx_load_es_levels()
    if es:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Prior Close", f"{es.get('prior_close', 0):,.2f}")
        c2.metric("Regime", es.get("regime", "N/A"))
        c3.metric("VIX", f"{es.get('vix', 0):.2f}")
        c4.metric("Direction", es.get("direction", "N/A"))
        c5.metric("Risk", f"{es.get('risk_score', 0)}/5")

        status = es.get("status", "")
        if "DO NOT TRADE" in status:
            st.error(f"⛔ {status}")
        elif "CAUTION" in status:
            st.warning(f"⚠️ {status}")
        else:
            st.success(f"✅ {status}")

        st.markdown("---")
        st.subheader("Key Levels")
        l1, l2 = st.columns(2)
        l1.metric("Upside Wall", f"{es.get('upside_wall', 0):,.2f}")
        l2.metric("Downside Wall", f"{es.get('downside_wall', 0):,.2f}")

        with st.expander("Full ES Levels JSON"):
            st.json(es)
    else:
        st.warning("No ES levels found. Run es_levels.py first.")


elif page == "📈 Backtest":
    st.title("SPX — Backtest Results")
    replay = spx_load_replay()
    if not replay.empty and "net_pnl_dollars" in replay.columns:
        equity = replay["net_pnl_dollars"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=replay["date"], y=equity, mode="lines",
                                  name="Cumulative P&L", fill="tozeroy"))
        fig.update_layout(height=400, title="Equity Curve")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(replay, width="stretch", height=400)
    else:
        st.warning("No replay data found.")


elif page == "📋 Paper Trades":
    st.title("SPX — Paper Trade Log")
    paper = spx_load_paper_log()
    if not paper.empty:
        st.dataframe(paper, width="stretch", height=500)
    else:
        st.info("No paper trades yet.")


elif page == "⚡ Risk & Intel":
    st.title("Market Risk & Intelligence")
    intel = spx_load_market_intel()
    if intel:
        risk = intel.get("risk_score", 0)
        risk_icons = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "⛔"}
        st.markdown(f"## {risk_icons.get(risk, '❓')} Risk: {risk}/5")
        if intel.get("tail_risk_flag"):
            st.error("⚠️ TAIL RISK FLAG — DO NOT TRADE")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", f"{risk}/5")
        c2.metric("Tail Risk", "🔴 YES" if intel.get("tail_risk_flag") else "🟢 No")
        c3.metric("Regime", intel.get("regime", "N/A"))

        events = intel.get("key_events", [])
        if events:
            st.markdown("### Key Events")
            for ev in events:
                if isinstance(ev, dict):
                    st.write(f"• **{ev.get('event','')}** — {ev.get('impact','')}")
                else:
                    st.write(f"• {ev}")
        with st.expander("Raw Intel"):
            st.json(intel)
    else:
        st.info("No market intel. Updates at 19:30 UTC.")


# ══════════════════════════════════════════════════════════════════════
#  OPTIONS ALGO PAGES
# ══════════════════════════════════════════════════════════════════════

elif page == "🔍 Ticker Analysis":
    st.title("🔍 Ticker Analysis")
    st.caption("Type any ticker for full regime, IV, and options strategy analysis.")

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        ticker = st.text_input("Ticker", value="", placeholder="CRM, AAPL, NVDA...").strip().upper()
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Analyze", type="primary", width="stretch")

    if ticker and run:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                from src.data.stock_fetcher import download_universe
                from src.analysis.technical import classify_regime
                from src.analysis.volatility import analyze_iv
                from src.strategy.selector import select_strategy

                data = download_universe([ticker], period="2y")
                if ticker not in data or data[ticker].empty:
                    st.error(f"No data for {ticker}")
                else:
                    df = data[ticker]  # lowercase columns: close, high, low, open, volume
                    price = df["close"].iloc[-1]
                    prev = df["close"].iloc[-2] if len(df) > 1 else price
                    chg = (price - prev) / prev * 100

                    # ── Price Metrics ──
                    st.markdown("---")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Price", f"${price:.2f}", f"{chg:+.2f}%")
                    c2.metric("High", f"${df['high'].iloc[-1]:.2f}")
                    c3.metric("Low", f"${df['low'].iloc[-1]:.2f}")
                    c4.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}")

                    # ── Regime ──
                    regime = classify_regime(ticker, df)
                    if regime:
                        st.markdown("---")
                        st.subheader("Technical Regime")
                        r1, r2, r3, r4, r5, r6 = st.columns(6)
                        r1.metric("Regime", regime.regime.value)
                        r2.metric("Trend Score", f"{regime.direction_score:+.2f}")
                        r3.metric("ADX", f"{regime.adx:.1f}")
                        r4.metric("RSI", f"{regime.rsi:.1f}")
                        r5.metric("ATR %", f"{regime.atr_pct:.2f}%")
                        r6.metric("Volume", regime.volume_trend)

                        # Regime context
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**EMA Alignment:** {regime.ema_alignment}")
                            st.write(f"**Volatility State:** {regime.volatility_state}")
                            st.write(f"**Trend Strength:** {regime.trend_strength:.2f}")
                        with col_b:
                            st.write(f"**Support:** ${regime.support:.2f}")
                            st.write(f"**Resistance:** ${regime.resistance:.2f}")
                            st.write(f"**ATR:** ${regime.atr:.2f}")
                        if regime.bb_squeeze:
                            st.success("📦 Bollinger Band Squeeze Active — potential breakout")
                    else:
                        st.warning("Could not classify regime (insufficient data).")

                    # ── IV Analysis ──
                    iv = analyze_iv(ticker, df)
                    if iv:
                        st.markdown("---")
                        st.subheader("Volatility Profile")
                        v1, v2, v3, v4, v5 = st.columns(5)
                        v1.metric("IV Rank", f"{iv.iv_rank:.0f}%")
                        v2.metric("IV Percentile", f"{iv.iv_percentile:.0f}%")
                        v3.metric("Current IV", f"{iv.current_iv:.1f}%")
                        v4.metric("HV-20", f"{iv.hv_20:.1f}%")
                        v5.metric("IV/HV Ratio", f"{iv.iv_hv_ratio:.2f}")

                        vc1, vc2, vc3, vc4 = st.columns(4)
                        vc1.metric("IV Regime", iv.iv_regime)
                        vc2.metric("IV Trend", iv.iv_trend)
                        vc3.metric("HV-60", f"{iv.hv_60:.1f}%")
                        vc4.metric("Skew", f"{iv.skew:.3f}")

                        if iv.premium_action == "SELL":
                            st.error(f"💰 **SELL PREMIUM** — IV Rank {iv.iv_rank:.0f}% (elevated vs history)")
                        elif iv.premium_action == "BUY":
                            st.success(f"🎯 **BUY PREMIUM** — IV Rank {iv.iv_rank:.0f}% (depressed)")
                        else:
                            st.info(f"⚖️ **NEUTRAL** — IV Rank {iv.iv_rank:.0f}%")
                    else:
                        st.warning("Could not compute IV analysis.")

                    # ── Strategy Recommendation ──
                    if regime and iv:
                        strat = select_strategy(regime, iv)
                        st.markdown("---")
                        st.subheader("Recommended Strategy")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Strategy", strat.strategy.replace("_", " "))
                        s2.metric("Direction", strat.direction)
                        s3.metric("Confidence", f"{strat.confidence:.0%}")
                        s4.metric("Target DTE", f"{strat.target_dte}d")

                        rationale = getattr(strat, "rationale", "")
                        if rationale:
                            st.info(f"**Rationale:** {rationale}")

                        # Trade note — constructors need live options chain
                        st.caption("⚠️ Full trade construction (strikes, credit, PoP) requires Polygon.io options chain data. "
                                   "Subscribe to Polygon Options Starter ($29/mo) to enable live strike selection.")

                    # ── Summary Box ──
                    if regime and iv:
                        st.markdown("---")
                        st.subheader("Summary")

                        direction_icon = "🟢" if regime.direction_score > 0.2 else "🔴" if regime.direction_score < -0.2 else "🟡"
                        iv_icon = "🔴" if iv.iv_rank > 70 else "🟢" if iv.iv_rank < 30 else "🟡"

                        st.markdown(f"""
| Metric | Value | Signal |
|--------|-------|--------|
| **Regime** | {regime.regime.value} | {direction_icon} Score: {regime.direction_score:+.2f} |
| **Trend** | ADX {regime.adx:.0f} / RSI {regime.rsi:.0f} | {"Strong" if regime.adx > 25 else "Weak"} trend |
| **IV Rank** | {iv.iv_rank:.0f}% | {iv_icon} {iv.premium_action} premium |
| **IV/HV** | {iv.iv_hv_ratio:.2f}x | {"Overpriced" if iv.iv_hv_ratio > 1.1 else "Underpriced" if iv.iv_hv_ratio < 0.9 else "Fair"} |
| **Strategy** | {strat.strategy.replace("_", " ")} | Conf: {strat.confidence:.0%} |
| **Setup** | {strat.direction} / {strat.target_dte}d DTE | EMA: {regime.ema_alignment} |
""")

                    # ── Chart ──
                    st.markdown("---")
                    st.subheader("Price Chart (90 days)")
                    chart_df = df.tail(90)
                    fig = go.Figure(data=[go.Candlestick(
                        x=chart_df.index,
                        open=chart_df["open"], high=chart_df["high"],
                        low=chart_df["low"], close=chart_df["close"]
                    )])
                    # Add support/resistance lines if regime exists
                    if regime:
                        fig.add_hline(y=regime.support, line_dash="dash", line_color="#FF9800",
                                      annotation_text=f"Support ${regime.support:.2f}")
                        fig.add_hline(y=regime.resistance, line_dash="dash", line_color="#2196F3",
                                      annotation_text=f"Resistance ${regime.resistance:.2f}")
                    fig.update_layout(height=500, xaxis_rangeslider_visible=False,
                                      template="plotly_dark",
                                      margin=dict(l=50, r=150, t=30, b=30))
                    st.plotly_chart(fig, width="stretch")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Enter a ticker and click **Analyze**.")
        if "analysis_history" in st.session_state and st.session_state.get("analysis_history"):
            st.markdown("---")
            st.markdown("### Recent Analyses")
            for h in reversed(st.session_state.analysis_history[-5:]):
                st.caption(f"**{h['ticker']}** — {h['strategy']} | Conf: {h['confidence']:.0%} | IV Rank: {h['iv_rank']:.0f}%")


elif page == "📈 Today's Picks":
    st.title("Options Algo — Today's Picks")
    signal = opt_load_signal()
    if signal:
        mkt = signal.get("market_context", {})
        if mkt:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Market", mkt.get("market_regime", "?"))
            c2.metric("VIX", f"{mkt.get('vix_level', 0):.0f}")
            c3.metric("SPY Trend", mkt.get("spy_trend", "?"))
            c4.metric("Breadth", f"{mkt.get('breadth_score', 0):.0%}")

        picks = signal.get("top_picks", [])
        if picks:
            for p in picks:
                rec = p.get("recommendation", {})
                ctx = p.get("context", {})
                iv_d = ctx.get("iv_detail", {})
                with st.expander(
                    f"#{p.get('priority','?')} {rec.get('ticker','?')} — "
                    f"{rec.get('strategy','?').replace('_',' ')} | "
                    f"{direction_badge(rec.get('direction',''))} | "
                    f"Conf: {rec.get('confidence',0):.0%}",
                    expanded=(p.get("priority", 10) <= 3),
                ):
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        st.write(f"Price: **${ctx.get('price',0):.2f}**")
                        st.write(f"Regime: `{rec.get('regime','?')}`")
                        st.write(f"IV Rank: **{iv_d.get('iv_rank',0):.0f}%** | IV/HV: {iv_d.get('iv_hv_ratio',0):.2f}")
                    with tc2:
                        trade = p.get("trade", {})
                        if not trade.get("dry_run"):
                            for k in ["net_credit", "max_risk", "prob_profit", "breakeven"]:
                                if k in trade:
                                    label = k.replace("_", " ").title()
                                    val = trade[k]
                                    st.write(f"{label}: **{'$' if 'credit' in k or 'risk' in k or 'break' in k else ''}{val:.2f}{'%' if 'prob' in k else ''}**")
                        else:
                            st.info("Dry-run — no trade details")
        else:
            st.info("No picks today.")
    else:
        st.warning("No signal. Run nightly scan first.")


elif page == "📋 Trade Tracker":
    st.title("📋 Trade Tracker")
    st.caption("All paper trades — open positions and closed outcomes.")

    try:
        trades_file = OPT_ROOT / "output" / "trades" / "trade_outcomes.jsonl"
        if trades_file.exists():
            import json
            trades = []
            with open(trades_file) as f:
                for line in f:
                    if line.strip():
                        trades.append(json.loads(line))

            if trades:
                df = pd.DataFrame(trades)

                # Summary metrics
                open_trades = df[df["outcome"] == "OPEN"]
                closed_trades = df[df["outcome"] != "OPEN"]

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Trades", len(df))
                c2.metric("Open", len(open_trades))
                c3.metric("Closed", len(closed_trades))
                if not closed_trades.empty:
                    wins = (closed_trades["won"] == True).sum()
                    wr = wins / len(closed_trades) * 100
                    total_pnl = closed_trades["pnl"].sum()
                    c4.metric("Win Rate", f"{wr:.0f}%")
                    c5.metric("Total P&L", f"${total_pnl:+,.0f}")

                # Open positions
                if not open_trades.empty:
                    st.markdown("---")
                    st.subheader(f"Open Positions ({len(open_trades)})")
                    open_display = open_trades[[
                        "trade_id", "ticker", "strategy", "direction",
                        "entry_date", "expiration", "short_strike", "long_strike",
                        "net_credit_or_debit", "max_risk", "confidence", "iv_rank"
                    ]].copy()
                    open_display.columns = [
                        "ID", "Ticker", "Strategy", "Dir",
                        "Entry", "Expiry", "Short", "Long",
                        "Credit/Debit", "Max Risk", "Conf", "IV Rank"
                    ]
                    st.dataframe(open_display, width="stretch")

                # Closed trades
                if not closed_trades.empty:
                    st.markdown("---")
                    st.subheader(f"Closed Trades ({len(closed_trades)})")
                    closed_display = closed_trades[[
                        "trade_id", "ticker", "strategy", "direction",
                        "entry_date", "exit_date", "outcome", "pnl",
                        "days_held", "close_reason"
                    ]].copy()
                    closed_display.columns = [
                        "ID", "Ticker", "Strategy", "Dir",
                        "Entry", "Exit", "Outcome", "P&L",
                        "Days", "Reason"
                    ]
                    closed_display["P&L"] = closed_display["P&L"].apply(lambda x: f"${x:+,.0f}" if pd.notna(x) else "")

                    st.dataframe(closed_display.sort_values("Exit", ascending=False),
                                 width="stretch")

                    # Equity curve
                    st.markdown("---")
                    st.subheader("Equity Curve")
                    pnl_series = closed_trades.sort_values("exit_date")
                    pnl_series["cumulative_pnl"] = pnl_series["pnl"].cumsum()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pnl_series["exit_date"], y=pnl_series["cumulative_pnl"],
                        mode="lines+markers", fill="tozeroy", name="Cumulative P&L",
                        line=dict(color="#2196F3", width=2),
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(height=400, template="plotly_dark",
                                      yaxis_title="Cumulative P&L ($)")
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No trades recorded yet. Trades are logged automatically during nightly scan.")
        else:
            st.info("No trade outcomes file found. Trades will be logged after the first nightly scan runs without --dry-run.")
            st.code("python -m src.pipeline.nightly_scan  # (without --dry-run)")
    except Exception as e:
        st.error(f"Failed to load trades: {e}")
        import traceback
        st.code(traceback.format_exc())


elif page == "📊 Strategy Stats":
    st.title("📊 Strategy Performance")
    st.caption("Win rates and P&L breakdown by strategy type.")

    try:
        trades_file = OPT_ROOT / "output" / "trades" / "trade_outcomes.jsonl"
        if trades_file.exists():
            import json
            trades = []
            with open(trades_file) as f:
                for line in f:
                    if line.strip():
                        trades.append(json.loads(line))

            df = pd.DataFrame(trades)
            closed = df[df["outcome"] != "OPEN"]

            if not closed.empty:
                strategies = closed["strategy"].unique()
                rows = []
                for strat in strategies:
                    s = closed[closed["strategy"] == strat]
                    wins = (s["won"] == True).sum()
                    losses = (s["won"] == False).sum()
                    total = len(s)
                    total_pnl = s["pnl"].sum()
                    avg_pnl = s["pnl"].mean()
                    avg_days = s["days_held"].mean() if "days_held" in s.columns else 0
                    avg_conf = s["confidence"].mean()
                    rows.append({
                        "Strategy": strat.replace("_", " "),
                        "Trades": total,
                        "Wins": wins,
                        "Losses": losses,
                        "Win Rate": f"{wins/total*100:.0f}%",
                        "Total P&L": f"${total_pnl:+,.0f}",
                        "Avg P&L": f"${avg_pnl:+,.0f}",
                        "Avg Days": f"{avg_days:.0f}",
                        "Avg Conf": f"{avg_conf:.0%}",
                    })

                stats_df = pd.DataFrame(rows)
                st.dataframe(stats_df, width="stretch")

                # Win rate chart
                chart_data = pd.DataFrame([{
                    "Strategy": r["Strategy"],
                    "Win Rate": int(r["Win Rate"].replace("%", "")),
                    "Trades": r["Trades"],
                } for r in rows])

                fig = px.bar(chart_data, x="Strategy", y="Win Rate",
                             color="Win Rate", color_continuous_scale="RdYlGn",
                             range_color=[0, 100], title="Win Rate by Strategy")
                fig.add_hline(y=50, line_dash="dash", line_color="white",
                              annotation_text="50%")
                fig.update_layout(height=400, template="plotly_dark")
                st.plotly_chart(fig, width="stretch")

                # P&L by regime
                if "regime" in closed.columns:
                    st.markdown("---")
                    st.subheader("P&L by Regime")
                    regime_pnl = closed.groupby("regime").agg(
                        Trades=("pnl", "count"),
                        Total_PnL=("pnl", "sum"),
                        Win_Rate=("won", "mean"),
                    ).round(2)
                    regime_pnl["Win_Rate"] = (regime_pnl["Win_Rate"] * 100).round(0).astype(str) + "%"
                    regime_pnl["Total_PnL"] = regime_pnl["Total_PnL"].apply(lambda x: f"${x:+,.0f}")
                    st.dataframe(regime_pnl, width="stretch")
            else:
                st.info("No closed trades yet. Check back after trades expire or hit targets.")
        else:
            st.info("No trade data found.")
    except Exception as e:
        st.error(f"Failed to load stats: {e}")


elif page == "🌡️ IV Heatmap":
    st.title("Options Algo — IV Heatmap")
    signal = opt_load_signal()
    if signal:
        picks = signal.get("top_picks", [])
        rows = []
        for p in picks:
            rec = p.get("recommendation", {})
            iv_d = p.get("context", {}).get("iv_detail", {})
            rows.append({
                "Ticker": rec.get("ticker", "?"),
                "IV Rank": iv_d.get("iv_rank", 0),
                "IV/HV": iv_d.get("iv_hv_ratio", 0),
                "Action": iv_d.get("premium_action", "?"),
                "Strategy": rec.get("strategy", "?"),
            })
        if rows:
            df = pd.DataFrame(rows).sort_values("IV Rank", ascending=False)
            st.dataframe(df,
                         width="stretch")
        else:
            st.info("No IV data in latest signal.")
    else:
        st.warning("No signal data.")


elif page == "🗺️ Regime Map":
    st.title("Options Algo — Regime Map")
    signal = opt_load_signal()
    if signal:
        dist = signal.get("regime_distribution", {})
        if dist:
            df_r = pd.DataFrame(list(dist.items()), columns=["Regime", "Count"])
            fig = px.pie(df_r, names="Regime", values="Count", title="Regime Distribution",
                         color="Regime", color_discrete_map=REGIME_COLORS)
            st.plotly_chart(fig, width="stretch")

        recs = signal.get("all_recommendations", [])
        if recs:
            st.dataframe(pd.DataFrame(recs)[["ticker", "regime", "strategy", "direction", "confidence"]],
                         width="stretch", height=400)
    else:
        st.warning("No signal data.")


elif page == "▶️ Run Scan":
    st.title("Options Algo — Run Scan")
    dry = st.checkbox("Dry Run", value=True)
    tickers_raw = st.text_input("Custom tickers (comma-separated, blank = full universe)",
                                 placeholder="AAPL, MSFT, NVDA")
    if st.button("🚀 Run Scan", type="primary"):
        custom = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()] if tickers_raw.strip() else None
        with st.spinner("Running..."):
            try:
                from src.pipeline.nightly_scan import run_nightly_scan
                result = run_nightly_scan(universe_override=custom, dry_run=dry)
                st.success(f"Done! {len(result.get('top_picks',[]))} picks in {result.get('elapsed_seconds',0):.0f}s")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(str(e))
                import traceback
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════
#  SYSTEM PAGES
# ══════════════════════════════════════════════════════════════════════

elif page == "📅 Cron Jobs":
    st.title("Scheduled Jobs")
    jobs = load_cron_jobs()
    if jobs:
        for job in jobs:
            name = job.get("name", "Unknown")
            sched = job.get("schedule", {})
            expr = sched.get("expr", "N/A") if isinstance(sched, dict) else str(sched)
            state = job.get("state", {})
            status = state.get("lastRunStatus", "never")
            icon = "🟢" if status == "ok" else "🔴" if status == "error" else "⚪"
            with st.expander(f"{icon} {name} — `{expr}`"):
                c1, c2 = st.columns(2)
                c1.metric("Status", status)
                c2.metric("Errors", state.get("consecutiveErrors", 0))
    else:
        st.warning("No cron jobs found.")
