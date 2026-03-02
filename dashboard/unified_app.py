"""
trading_dashboard/app.py
========================
Unified Trading Dashboard — SPX Iron Condor + Options Algo
Single Streamlit app on port 8503.

Verified against:
  - stock_fetcher.download_universe() returns lowercase columns
  - classify_regime(ticker, df) — 2 args
  - analyze_iv(ticker, df) — 2 args
  - select_strategy(regime, iv) returns StrategyRecommendation
  - StrategyRecommendation fields: .strategy (StrategyType enum), .direction,
    .confidence, .target_dte, .rationale, .risk_reward, .ticker
  - StockRegime fields: .regime (Regime enum), .direction_score, .trend_strength,
    .adx, .rsi, .bb_squeeze, .ema_alignment, .support, .resistance,
    .atr, .atr_pct, .price, .volume_trend, .volatility_state
  - IVAnalysis fields: .iv_rank, .iv_percentile, .current_iv, .hv_20, .hv_60,
    .iv_hv_ratio, .iv_regime, .premium_action, .iv_trend, .iv_30d_avg, .skew
  - SPX data uses UPPERCASE columns (Open, High, Low, Close, Volume)
  - Options algo data uses lowercase columns (open, high, low, close, volume)
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

# ── Paths ─────────────────────────────────────────────────────────────
SPX_RAW       = SPX_ROOT / "data" / "raw"
SPX_SIGNALS   = SPX_ROOT / "output" / "signals"
SPX_TRADES    = SPX_ROOT / "output" / "trades" / "paper_trade_log.csv"
SPX_REPORTS   = SPX_ROOT / "output" / "reports"
OPT_SIGNALS   = OPT_ROOT / "output" / "signals"
OPT_TRADES    = OPT_ROOT / "output" / "trades"


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def spx_load_data():
    p = SPX_RAW / "spx_daily.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=300)
def spx_load_vix():
    p = SPX_RAW / "vix_daily.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=300)
def spx_load_signal():
    p = SPX_SIGNALS / "latest_signal.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(ttl=300)
def spx_load_replay():
    for name in ["replay_jan_feb_2026_v2.csv", "replay_jan_feb_2026.csv"]:
        p = SPX_REPORTS / name
        if p.exists():
            df = pd.read_csv(p)
            df["date"] = pd.to_datetime(df["date"])
            rename = {"h_err": "h_err_pct", "l_err": "l_err_pct",
                      "dir_ok": "dir_correct", "net_pnl": "net_pnl_dollars"}
            df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
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
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(ttl=300)
def spx_load_es_levels():
    p = SPX_SIGNALS / "es_levels_latest.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(ttl=300)
def opt_load_signal():
    p = OPT_SIGNALS / "options_signal_latest.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

def opt_load_trades():
    """Load trade outcomes from JSONL file. Returns list of dicts."""
    p = OPT_TRADES / "trade_outcomes.jsonl"
    if not p.exists():
        return []
    trades = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
    except Exception:
        pass
    return trades

@st.cache_data(ttl=60)
def load_cron_jobs():
    p = Path("/home/openclaw/.openclaw/cron/jobs.json")
    if not p.exists():
        return []
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get("jobs", [])
    except Exception:
        return []


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

    spx = spx_load_data()
    vix = spx_load_vix()
    sig = spx_load_signal()
    replay = spx_load_replay()

    if not spx.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        # SPX data has uppercase columns
        c1.metric("SPX Close", f"{spx['Close'].iloc[-1]:,.2f}",
                   f"{spx['Close'].iloc[-1] - spx['Close'].iloc[-2]:+.2f}")
        c2.metric("VIX", f"{vix['Close'].iloc[-1]:.2f}" if not vix.empty else "N/A")
        if sig:
            c3.metric("Regime", sig.get("regime", "N/A"))
            c4.metric("Direction", sig.get("direction", "N/A"),
                       f"{sig.get('direction_prob', 0)*100:.1f}%")
            c5.metric("Tradeable", "✅" if sig.get("tradeable") else "❌")
    else:
        st.warning("SPX data not found. Run daily pipeline first.")

    if not replay.empty and "net_pnl_dollars" in replay.columns:
        st.markdown("---")
        st.subheader("Jan-Feb 2026 Replay Summary")
        m1, m2, m3, m4 = st.columns(4)
        wins = len(replay[replay["condor"] == "WIN"])
        total = len(replay)
        m1.metric("Win Rate", f"{wins/total*100:.1f}%" if total > 0 else "N/A")
        m2.metric("Total P&L", f"${replay['net_pnl_dollars'].sum():,.0f}")
        equity = replay["net_pnl_dollars"].cumsum()
        m3.metric("Max DD", f"${(equity - equity.cummax()).min():,.0f}")
        m4.metric("Trades", total)

    if not spx.empty:
        st.markdown("---")
        st.subheader("SPX Last 60 Days")
        recent = spx.tail(60)
        fig = go.Figure(data=[go.Candlestick(
            x=recent.index,
            open=recent["Open"], high=recent["High"],
            low=recent["Low"], close=recent["Close"],
            name="SPX"
        )])
        fig.update_layout(height=400, xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)


elif page == "🔮 Latest Signal":
    st.title("SPX — Latest Signal")
    sig = spx_load_signal()
    if sig:
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Date", sig.get("signal_date", "N/A"))
        h2.metric("Regime", sig.get("regime", "N/A"))
        h3.metric("Direction", sig.get("direction", "N/A"),
                   f"{sig.get('direction_prob', 0)*100:.1f}%")
        h4.metric("VIX", f"{sig.get('vix_spot', 0):.2f}")
        h5.metric("Tradeable", "✅" if sig.get("tradeable") else "❌")

        st.markdown("---")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("**Predictions**")
            st.write(f"Prior Close: {sig.get('prior_close', 0):,.2f}")
            st.write(f"Pred High: {sig.get('predicted_high', 0):,.2f}")
            st.write(f"Pred Low: {sig.get('predicted_low', 0):,.2f}")
        with p2:
            st.markdown("**IC Strikes**")
            st.write(f"Short Call: {sig.get('ic_short_call', 0):,.2f}")
            st.write(f"Short Put: {sig.get('ic_short_put', 0):,.2f}")
            st.write(f"Long Call: {sig.get('ic_long_call', 0):,.2f}")
            st.write(f"Long Put: {sig.get('ic_long_put', 0):,.2f}")
        with p3:
            st.markdown("**Conformal Bands**")
            st.write(f"68%: [{sig.get('conf_68_high_lo', 0):,.2f}, {sig.get('conf_68_high_hi', 0):,.2f}]")
            st.write(f"90%: [{sig.get('conf_90_high_lo', 0):,.2f}, {sig.get('conf_90_high_hi', 0):,.2f}]")

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
        l1, l2 = st.columns(2)
        l1.metric("Upside Wall", f"{es.get('upside_wall', 0):,.2f}")
        l2.metric("Downside Wall", f"{es.get('downside_wall', 0):,.2f}")

        with st.expander("Full ES Levels JSON"):
            st.json(es)
    else:
        st.warning("No ES levels found.")


elif page == "📈 Backtest":
    st.title("SPX — Backtest Results")
    replay = spx_load_replay()
    if not replay.empty and "net_pnl_dollars" in replay.columns:
        equity = replay["net_pnl_dollars"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=replay["date"], y=equity, mode="lines",
                                  name="Cumulative P&L", fill="tozeroy"))
        fig.update_layout(height=400, title="Equity Curve")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(replay, use_container_width=True, height=400)
    else:
        st.warning("No replay data found.")


elif page == "📋 Paper Trades":
    st.title("SPX — Paper Trade Log")
    paper = spx_load_paper_log()
    if not paper.empty:
        st.dataframe(paper, use_container_width=True, height=500)
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
                    st.write(f"• **{ev.get('event', '')}** — {ev.get('impact', '')}")
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
        run = st.button("Analyze", type="primary", use_container_width=True)

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
                    df = data[ticker]  # lowercase: close, high, low, open, volume
                    price = float(df["close"].iloc[-1])
                    prev = float(df["close"].iloc[-2]) if len(df) > 1 else price
                    chg = (price - prev) / prev * 100

                    # ── Price Metrics ──
                    st.markdown("---")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Price", f"${price:.2f}", f"{chg:+.2f}%")
                    c2.metric("High", f"${float(df['high'].iloc[-1]):.2f}")
                    c3.metric("Low", f"${float(df['low'].iloc[-1]):.2f}")
                    c4.metric("Volume", f"{int(df['volume'].iloc[-1]):,}")

                    # ── Regime ── (classify_regime needs ticker AND df)
                    regime = classify_regime(ticker, df)
                    if regime:
                        st.markdown("---")
                        st.subheader("📊 Technical Regime")
                        r1, r2, r3, r4, r5, r6 = st.columns(6)
                        r1.metric("Regime", regime.regime.value)
                        r2.metric("Trend Score", f"{regime.direction_score:+.2f}")
                        r3.metric("ADX", f"{regime.adx:.1f}")
                        r4.metric("RSI", f"{regime.rsi:.1f}")
                        r5.metric("ATR %", f"{regime.atr_pct:.2f}%")
                        r6.metric("Volume", regime.volume_trend)

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
                        st.warning("Could not classify regime.")

                    # ── IV Analysis ── (analyze_iv needs ticker AND df)
                    iv = analyze_iv(ticker, df)
                    if iv:
                        st.markdown("---")
                        st.subheader("📈 Volatility Profile")
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
                            st.error(f"💰 **SELL PREMIUM** — IV Rank {iv.iv_rank:.0f}%")
                        elif iv.premium_action == "BUY":
                            st.success(f"🎯 **BUY PREMIUM** — IV Rank {iv.iv_rank:.0f}%")
                        else:
                            st.info(f"⚖️ **NEUTRAL** — IV Rank {iv.iv_rank:.0f}%")
                    else:
                        st.warning("Could not compute IV analysis.")

                    # ── Strategy ──
                    # select_strategy returns StrategyRecommendation with:
                    #   .strategy = StrategyType enum (use .value for string)
                    #   .direction, .confidence, .target_dte, .rationale
                    if regime and iv:
                        strat = select_strategy(regime, iv)
                        st.markdown("---")
                        st.subheader("🎯 Recommended Strategy")
                        s1, s2, s3, s4 = st.columns(4)
                        # .strategy is a StrategyType enum — use .value
                        strat_name = strat.strategy.value if hasattr(strat.strategy, 'value') else str(strat.strategy)
                        s1.metric("Strategy", strat_name.replace("_", " "))
                        s2.metric("Direction", strat.direction)
                        s3.metric("Confidence", f"{strat.confidence:.0%}")
                        s4.metric("Target DTE", f"{strat.target_dte}d")

                        if strat.rationale:
                            st.info(f"**Rationale:** {strat.rationale}")

                        st.caption("⚠️ Full trade construction requires Polygon.io options data ($29/mo).")

                        # ── Summary Table ──
                        st.markdown("---")
                        st.subheader("📋 Summary")
                        dir_icon = "🟢" if regime.direction_score > 0.2 else "🔴" if regime.direction_score < -0.2 else "🟡"
                        iv_icon = "🔴" if iv.iv_rank > 70 else "🟢" if iv.iv_rank < 30 else "🟡"

                        summary_data = {
                            "Metric": ["Regime", "Trend", "IV Rank", "IV/HV", "Strategy", "Setup"],
                            "Value": [
                                regime.regime.value,
                                f"ADX {regime.adx:.0f} / RSI {regime.rsi:.0f}",
                                f"{iv.iv_rank:.0f}%",
                                f"{iv.iv_hv_ratio:.2f}x",
                                strat_name.replace("_", " "),
                                f"{strat.direction} / {strat.target_dte}d DTE",
                            ],
                            "Signal": [
                                f"{dir_icon} Score: {regime.direction_score:+.2f}",
                                "Strong" if regime.adx > 25 else "Weak",
                                f"{iv_icon} {iv.premium_action}",
                                "Overpriced" if iv.iv_hv_ratio > 1.1 else "Underpriced" if iv.iv_hv_ratio < 0.9 else "Fair",
                                f"Conf: {strat.confidence:.0%}",
                                f"EMA: {regime.ema_alignment}",
                            ],
                        }
                        st.table(pd.DataFrame(summary_data))

                    # ── Price Chart ──
                    st.markdown("---")
                    st.subheader("📉 Price Chart (90 days)")
                    chart_df = df.tail(90)
                    fig = go.Figure(data=[go.Candlestick(
                        x=chart_df.index,
                        open=chart_df["open"], high=chart_df["high"],
                        low=chart_df["low"], close=chart_df["close"]
                    )])
                    if regime:
                        fig.add_hline(y=regime.support, line_dash="dash", line_color="#FF9800",
                                      annotation_text=f"Support ${regime.support:.2f}")
                        fig.add_hline(y=regime.resistance, line_dash="dash", line_color="#2196F3",
                                      annotation_text=f"Resistance ${regime.resistance:.2f}")
                    fig.update_layout(height=500, xaxis_rangeslider_visible=False,
                                      template="plotly_dark",
                                      margin=dict(l=50, r=150, t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)

                    # Save to session history
                    if "analysis_history" not in st.session_state:
                        st.session_state.analysis_history = []
                    strat_val = ""
                    conf_val = 0.0
                    if regime and iv:
                        strat_val = strat_name
                        conf_val = strat.confidence
                    st.session_state.analysis_history.append({
                        "ticker": ticker, "price": price,
                        "regime": regime.regime.value if regime else "?",
                        "iv_rank": iv.iv_rank if iv else 0,
                        "strategy": strat_val, "confidence": conf_val,
                        "direction": strat.direction if regime and iv else "?",
                        "date": date.today().isoformat(),
                    })

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
                st.caption(
                    f"**{h['ticker']}** — {h['strategy'].replace('_', ' ')} | "
                    f"Conf: {h['confidence']:.0%} | IV Rank: {h['iv_rank']:.0f}%"
                )


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

            if mkt.get("notes"):
                st.warning(f"⚠️ {mkt['notes']}")

        picks = signal.get("top_picks", [])
        if picks:
            st.subheader(f"Top {len(picks)} Picks")
            for p in picks:
                rec = p.get("recommendation", {})
                ctx = p.get("context", {})
                iv_d = ctx.get("iv_detail", {})
                reg_d = ctx.get("regime_detail", {})
                trade = p.get("trade", {})

                with st.expander(
                    f"#{p.get('priority', '?')} {rec.get('ticker', '?')} — "
                    f"{rec.get('strategy', '?').replace('_', ' ')} | "
                    f"{direction_badge(rec.get('direction', ''))} | "
                    f"Conf: {rec.get('confidence', 0):.0%}",
                    expanded=(p.get("priority", 10) <= 3),
                ):
                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        st.markdown("**Setup**")
                        st.write(f"Price: **${ctx.get('price', 0):.2f}**")
                        st.write(f"Regime: `{rec.get('regime', '?')}`")
                        st.write(f"ADX: {reg_d.get('adx', 0):.0f} | RSI: {reg_d.get('rsi', 0):.0f}")
                        st.write(f"EMA: {reg_d.get('ema_alignment', '?')} | Vol: {reg_d.get('volume_trend', '?')}")
                        if reg_d.get("bb_squeeze"):
                            st.success("BB Squeeze")
                    with tc2:
                        st.markdown("**Volatility**")
                        st.write(f"IV Rank: **{iv_d.get('iv_rank', 0):.0f}%**")
                        st.write(f"IV/HV: **{iv_d.get('iv_hv_ratio', 0):.2f}**")
                        st.write(f"IV Trend: {iv_d.get('iv_trend', '?')}")
                        st.write(f"Action: **{iv_d.get('premium_action', '?')}**")
                    with tc3:
                        st.markdown("**Trade**")
                        if not trade.get("dry_run"):
                            credit = trade.get("net_credit") or trade.get("total_credit")
                            if credit:
                                st.write(f"Credit: **${credit:.2f}**")
                            debit = trade.get("net_debit")
                            if debit:
                                st.write(f"Debit: **${debit:.2f}**")
                            if trade.get("max_risk"):
                                st.write(f"Max Risk: ${trade['max_risk']:.2f}")
                            if trade.get("prob_profit"):
                                st.write(f"PoP: **{trade['prob_profit']:.0f}%**")
                            if trade.get("breakeven"):
                                st.write(f"Breakeven: ${trade['breakeven']:.2f}")
                        else:
                            st.info("Dry-run — no trade details")

                    st.caption(f"Rationale: {rec.get('rationale', '')}")
                    st.caption(f"Score: {p.get('composite_score', 0):.4f}")

            # Scan stats
            st.markdown("---")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Universe", signal.get("universe_size", 0))
            sc2.metric("Qualified", signal.get("qualified", 0))
            sc3.metric("After Events", signal.get("after_event_filter", 0))
            sc4.metric("Elapsed", f"{signal.get('elapsed_seconds', 0):.0f}s")

            # Regime distribution bar
            regime_dist = signal.get("regime_distribution", {})
            if regime_dist:
                st.markdown("### Regime Distribution")
                df_reg = pd.DataFrame(list(regime_dist.items()), columns=["Regime", "Count"])
                df_reg = df_reg.sort_values("Count", ascending=False)
                fig = px.bar(df_reg, x="Regime", y="Count", color="Regime",
                             color_discrete_map=REGIME_COLORS,
                             title="Universe Regime Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No picks today.")
    else:
        st.warning("No signal. Run nightly scan first.")


elif page == "📋 Trade Tracker":
    st.title("📋 Trade Tracker")
    st.caption("Paper trades — open positions and closed outcomes.")

    trades = opt_load_trades()
    if trades:
        df = pd.DataFrame(trades)
        open_trades = df[df["outcome"] == "OPEN"]
        closed_trades = df[df["outcome"] != "OPEN"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total", len(df))
        c2.metric("Open", len(open_trades))
        c3.metric("Closed", len(closed_trades))
        if not closed_trades.empty:
            wins = (closed_trades["won"] == True).sum()
            wr = wins / len(closed_trades) * 100
            c4.metric("Win Rate", f"{wr:.0f}%")
            c5.metric("Total P&L", f"${closed_trades['pnl'].sum():+,.2f}")

        if not open_trades.empty:
            st.markdown("---")
            st.subheader(f"Open Positions ({len(open_trades)})")
            cols = ["trade_id", "ticker", "strategy", "direction", "entry_date",
                    "expiration", "short_strike", "long_strike", "net_credit_or_debit",
                    "max_risk", "confidence", "iv_rank"]
            available = [c for c in cols if c in open_trades.columns]
            st.dataframe(open_trades[available], use_container_width=True)

        if not closed_trades.empty:
            st.markdown("---")
            st.subheader(f"Closed Trades ({len(closed_trades)})")
            cols = ["trade_id", "ticker", "strategy", "direction", "entry_date",
                    "exit_date", "outcome", "pnl", "pnl_pct", "days_held", "close_reason"]
            available = [c for c in cols if c in closed_trades.columns]
            st.dataframe(
                closed_trades[available].sort_values("exit_date", ascending=False),
                use_container_width=True,
            )

            st.markdown("---")
            st.subheader("Equity Curve")
            pnl_series = closed_trades.sort_values("exit_date").copy()
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
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet. Trades are logged during nightly scan (non dry-run).")
        st.code("python -m src.pipeline.nightly_scan  # without --dry-run")


elif page == "📊 Strategy Stats":
    st.title("📊 Strategy Performance")

    trades = opt_load_trades()
    if trades:
        df = pd.DataFrame(trades)
        closed = df[df["outcome"] != "OPEN"]

        if not closed.empty:
            strategies = closed["strategy"].unique()
            rows = []
            for strat in strategies:
                s = closed[closed["strategy"] == strat]
                wins = int((s["won"] == True).sum())
                total = len(s)
                rows.append({
                    "Strategy": strat.replace("_", " "),
                    "Trades": total,
                    "Wins": wins,
                    "Losses": total - wins,
                    "Win Rate %": round(wins / total * 100, 0) if total > 0 else 0,
                    "Total P&L": round(s["pnl"].sum(), 2),
                    "Avg P&L": round(s["pnl"].mean(), 2),
                    "Avg Days": round(s["days_held"].mean(), 0) if "days_held" in s.columns else 0,
                })

            stats_df = pd.DataFrame(rows)
            st.dataframe(stats_df, use_container_width=True)

            fig = px.bar(stats_df, x="Strategy", y="Win Rate %",
                         color="Win Rate %", color_continuous_scale="RdYlGn",
                         range_color=[0, 100], title="Win Rate by Strategy")
            fig.add_hline(y=50, line_dash="dash", line_color="white", annotation_text="50%")
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            if "regime" in closed.columns:
                st.markdown("---")
                st.subheader("P&L by Regime")
                regime_stats = closed.groupby("regime").agg(
                    Trades=("pnl", "count"),
                    Total_PnL=("pnl", "sum"),
                    Avg_PnL=("pnl", "mean"),
                    Win_Rate=("won", "mean"),
                ).round(2)
                regime_stats["Win_Rate"] = (regime_stats["Win_Rate"] * 100).round(0).astype(int).astype(str) + "%"
                st.dataframe(regime_stats, use_container_width=True)
        else:
            st.info("No closed trades yet.")
    else:
        st.info("No trade data found.")


elif page == "🌡️ IV Heatmap":
    st.title("🌡️ IV Heatmap")
    signal = opt_load_signal()
    if signal:
        picks = signal.get("top_picks", [])
        rows = []
        for p in picks:
            rec = p.get("recommendation", {})
            ctx = p.get("context", {})
            iv_d = ctx.get("iv_detail", {})
            rows.append({
                "Ticker": rec.get("ticker", "?"),
                "Sector": ctx.get("sector", "?"),
                "IV Rank": round(float(iv_d.get("iv_rank", 0)), 1),
                "IV Pctile": round(float(iv_d.get("iv_percentile", 0)), 1),
                "Current IV": round(float(iv_d.get("current_iv", 0)), 1),
                "HV-20": round(float(iv_d.get("hv_20", 0)), 1),
                "IV/HV": round(float(iv_d.get("iv_hv_ratio", 0)), 2),
                "IV Trend": iv_d.get("iv_trend", "?"),
                "Action": iv_d.get("premium_action", "?"),
                "Strategy": rec.get("strategy", "?").replace("_", " "),
            })

        if rows:
            iv_df = pd.DataFrame(rows).sort_values("IV Rank", ascending=False)

            # Visual heatmap
            heatmap_data = iv_df[["IV Rank", "IV Pctile", "HV-20", "IV/HV"]].values
            fig = px.imshow(
                heatmap_data,
                x=["IV Rank", "IV Pctile", "HV-20", "IV/HV"],
                y=iv_df["Ticker"].tolist(),
                color_continuous_scale="RdYlGn_r",
                aspect="auto",
                text_auto=".1f",
            )
            fig.update_layout(
                title="IV Heatmap — Top Picks",
                height=max(300, len(rows) * 80 + 100),
                template="plotly_dark",
                margin=dict(l=80, r=20, t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Action summary
            sell_count = sum(1 for r in rows if r["Action"] == "SELL")
            buy_count = sum(1 for r in rows if r["Action"] == "BUY")
            neutral_count = len(rows) - sell_count - buy_count
            st.markdown(f"**Summary:** {sell_count} SELL premium | {buy_count} BUY premium | {neutral_count} NEUTRAL")

            # Scatter: IV Rank vs IV/HV
            fig2 = px.scatter(
                iv_df, x="IV Rank", y="IV/HV", color="Action", text="Ticker",
                title="IV Rank vs IV/HV Ratio",
                color_discrete_map={"SELL": "#D50000", "NEUTRAL": "#FFD740", "BUY": "#00C853"},
            )
            fig2.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="IV High")
            fig2.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="IV Low")
            fig2.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="IV=HV")
            fig2.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

            # Detail table
            st.markdown("---")
            st.subheader("Detail Table")
            st.dataframe(iv_df, use_container_width=True)
        else:
            st.info("No IV data in latest signal.")
    else:
        st.warning("No signal data. Run scan first.")


elif page == "🗺️ Regime Map":
    st.title("🗺️ Regime Map")
    signal = opt_load_signal()
    if signal:
        dist = signal.get("regime_distribution", {})
        if dist:
            col1, col2 = st.columns(2)
            with col1:
                df_r = pd.DataFrame(list(dist.items()), columns=["Regime", "Count"])
                fig = px.pie(df_r, names="Regime", values="Count",
                             title="Regime Distribution",
                             color="Regime", color_discrete_map=REGIME_COLORS)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                recs = signal.get("all_recommendations", [])
                if recs:
                    strat_counts = pd.DataFrame(recs)["strategy"].value_counts().reset_index()
                    strat_counts.columns = ["Strategy", "Count"]
                    fig2 = px.bar(strat_counts, x="Count", y="Strategy",
                                  orientation="h", title="Strategy Distribution")
                    fig2.update_layout(template="plotly_dark")
                    st.plotly_chart(fig2, use_container_width=True)

        recs = signal.get("all_recommendations", [])
        if recs:
            st.markdown("### All Recommendations")
            rec_df = pd.DataFrame(recs)
            display_cols = ["ticker", "regime", "iv_regime", "strategy", "direction", "confidence"]
            available = [c for c in display_cols if c in rec_df.columns]
            display = rec_df[available].copy()
            if "confidence" in display.columns:
                display["confidence"] = display["confidence"].apply(lambda x: f"{x:.0%}")
            st.dataframe(display, use_container_width=True, height=400)
    else:
        st.warning("No signal data.")


elif page == "▶️ Run Scan":
    st.title("▶️ Run Scan")
    st.info("Trigger the nightly scan on demand. Use **Dry Run** to skip live options data.")

    dry = st.checkbox("Dry Run (recommended until Polygon subscription)", value=True)
    tickers_raw = st.text_input(
        "Custom tickers (comma-separated, blank = full universe)",
        placeholder="AAPL, MSFT, NVDA"
    )

    if st.button("🚀 Run Scan", type="primary"):
        custom = None
        if tickers_raw.strip():
            custom = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        with st.spinner("Running scan... this may take 1-3 minutes"):
            try:
                from src.pipeline.nightly_scan import run_nightly_scan
                result = run_nightly_scan(universe_override=custom, dry_run=dry)
                n_picks = len(result.get("top_picks", []))
                elapsed = result.get("elapsed_seconds", 0)
                st.success(f"Scan complete! {n_picks} picks in {elapsed:.0f}s")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Scan failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════
#  SYSTEM PAGES
# ══════════════════════════════════════════════════════════════════════

elif page == "📅 Cron Jobs":
    st.title("📅 Scheduled Jobs")
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
        st.info("No cron jobs found or unable to read jobs file.")
