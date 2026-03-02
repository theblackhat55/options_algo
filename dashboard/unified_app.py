"""
trading_dashboard/app.py — Unified Trading Command Center
SPX Iron Condor + Options Algo — All pages verified.
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

SPX_ROOT = Path("/root/spx_algo")
OPT_ROOT = Path("/root/options_algo")
for p in [str(SPX_ROOT), str(OPT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

st.set_page_config(page_title="Trading Command Center", page_icon="🏦",
                    layout="wide", initial_sidebar_state="expanded")

SPX_RAW     = SPX_ROOT / "data" / "raw"
SPX_SIGNALS = SPX_ROOT / "output" / "signals"
SPX_TRADES  = SPX_ROOT / "output" / "trades" / "paper_trade_log.csv"
SPX_REPORTS = SPX_ROOT / "output" / "reports"
SPX_MON     = SPX_ROOT / "output" / "monitoring"
OPT_SIGNALS = OPT_ROOT / "output" / "signals"
OPT_TRADES  = OPT_ROOT / "output" / "trades"

# ══════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def spx_load_data():
    p = SPX_RAW / "spx_daily.parquet"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p); df.index = pd.to_datetime(df.index); return df

@st.cache_data(ttl=300)
def spx_load_vix():
    p = SPX_RAW / "vix_daily.parquet"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p); df.index = pd.to_datetime(df.index); return df

@st.cache_data(ttl=300)
def spx_load_signal():
    p = SPX_SIGNALS / "latest_signal.json"
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

@st.cache_data(ttl=300)
def spx_load_all_signals():
    signals = []
    for p in sorted(SPX_SIGNALS.glob("signal_*.json")):
        try:
            with open(p) as f: signals.append(json.load(f))
        except Exception: pass
    return signals

@st.cache_data(ttl=300)
def spx_load_replay():
    for name in ["replay_jan_feb_2026_v2.csv", "replay_jan_feb_2026.csv"]:
        p = SPX_REPORTS / name
        if p.exists():
            df = pd.read_csv(p); df["date"] = pd.to_datetime(df["date"])
            rename = {"h_err":"h_err_pct","l_err":"l_err_pct","dir_ok":"dir_correct","net_pnl":"net_pnl_dollars"}
            df.rename(columns={k:v for k,v in rename.items() if k in df.columns}, inplace=True)
            return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def spx_load_paper_log():
    if SPX_TRADES.exists():
        df = pd.read_csv(SPX_TRADES); df["date"] = pd.to_datetime(df["date"]); return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def spx_load_market_intel():
    p = SPX_ROOT / "data" / "processed" / "market_intel.json"
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

@st.cache_data(ttl=300)
def spx_load_es_levels():
    p = SPX_SIGNALS / "es_levels_latest.json"
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

@st.cache_data(ttl=300)
def spx_load_error_history():
    p = SPX_MON / "error_history.csv"
    if not p.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(p, parse_dates=["date"], index_col="date"); return df.sort_index()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=300)
def spx_load_calibration_health():
    p = SPX_MON / "calibration_health.json"
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

@st.cache_data(ttl=300)
def spx_load_correction_log():
    p = SPX_MON / "correction_log.csv"
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=300)
def spx_load_retrain_report():
    p = SPX_REPORTS / "weekly_retrain_latest.json"
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

@st.cache_data(ttl=300)
def opt_load_signal():
    p = OPT_SIGNALS / "options_signal_latest.json"
    if not p.exists(): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

def opt_load_trades():
    p = OPT_TRADES / "trade_outcomes.jsonl"
    if not p.exists(): return []
    trades = []
    try:
        with open(p) as f:
            for line in f:
                if line.strip(): trades.append(json.loads(line))
    except Exception: pass
    return trades

@st.cache_data(ttl=60)
def load_cron_jobs():
    p = Path("/home/openclaw/.openclaw/cron/jobs.json")
    if not p.exists(): return []
    try:
        with open(p) as f: return json.load(f).get("jobs", [])
    except Exception: return []

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
st.sidebar.title("🏦 Trading Command Center")
st.sidebar.markdown("---")
section = st.sidebar.selectbox("Section", ["SPX Iron Condor", "Options Algo", "System"])

if section == "SPX Iron Condor":
    page = st.sidebar.radio("Navigate", [
        "🏠 Overview", "📈 Backtest", "🎯 Replay Detail",
        "📋 Paper Trades", "🔮 Latest Signal", "📊 ES/MES Levels",
        "🧠 Model Health", "⚡ Risk & Intel",
    ])
elif section == "Options Algo":
    page = st.sidebar.radio("Navigate", [
        "🔍 Ticker Analysis", "📈 Today's Picks", "📋 Trade Tracker",
        "📊 Strategy Stats", "🌡️ IV Heatmap", "🗺️ Regime Map", "▶️ Run Scan",
    ])
else:
    page = st.sidebar.radio("Navigate", ["📅 Cron Jobs"])

st.sidebar.markdown("---")
st.sidebar.caption(f"Date: {date.today()}")

REGIME_COLORS = {"STRONG_UPTREND":"#00C853","UPTREND":"#69F0AE","RANGE_BOUND":"#FFD740",
    "DOWNTREND":"#FF6D00","STRONG_DOWNTREND":"#D50000","SQUEEZE":"#AA00FF",
    "REVERSAL_UP":"#00B0FF","REVERSAL_DOWN":"#FF4081"}
def direction_badge(d):
    return {"BULLISH":"🟢 BULLISH","BEARISH":"🔴 BEARISH"}.get(d,"🟡 NEUTRAL")

# ══════════════════════════════════════════════════════════════════════
#  SPX: OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("SPX Iron-Condor Algo — Overview")
    spx = spx_load_data(); vix = spx_load_vix()
    sig = spx_load_signal(); replay = spx_load_replay()

    if not spx.empty:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("SPX Close",f"{spx['Close'].iloc[-1]:,.2f}",f"{spx['Close'].iloc[-1]-spx['Close'].iloc[-2]:+.2f}")
        c2.metric("VIX",f"{vix['Close'].iloc[-1]:.2f}" if not vix.empty else "N/A")
        if sig:
            c3.metric("Regime",sig.get("regime","N/A"))
            c4.metric("Direction",sig.get("direction","N/A"),f"{sig.get('direction_prob',0)*100:.1f}%")
            c5.metric("Tradeable","✅" if sig.get("tradeable") else "❌")
    else:
        st.warning("SPX data not found.")

    if not replay.empty and "net_pnl_dollars" in replay.columns:
        st.markdown("---"); st.subheader("Jan-Feb 2026 Replay Summary")
        m1,m2,m3,m4 = st.columns(4)
        wins = len(replay[replay["condor"]=="WIN"]); total = len(replay)
        m1.metric("Win Rate",f"{wins/total*100:.1f}%" if total>0 else "N/A")
        m2.metric("Total P&L",f"${replay['net_pnl_dollars'].sum():,.0f}")
        equity = replay["net_pnl_dollars"].cumsum()
        m3.metric("Max DD",f"${(equity-equity.cummax()).min():,.0f}")
        m4.metric("Trades",total)

    if not spx.empty:
        st.markdown("---"); st.subheader("SPX Last 60 Days")
        recent = spx.tail(60)
        fig = go.Figure(data=[go.Candlestick(x=recent.index,open=recent["Open"],high=recent["High"],
            low=recent["Low"],close=recent["Close"],name="SPX")])
        fig.update_layout(height=400,xaxis_rangeslider_visible=False,margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
#  SPX: BACKTEST — Predicted vs Actual (the chart you want restored)
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Backtest":
    st.title("Backtest Dashboard — Predictions vs Actuals")
    replay = spx_load_replay()

    if not replay.empty:
        st.subheader("Prediction vs Actual (Jan–Feb 2026)")

        # Ensure dates are datetime
        replay["date"] = pd.to_datetime(replay["date"])

        # All needed columns already exist in replay_jan_feb_2026_v2.csv
        dates = replay["date"]

        # ── Summary Metrics ──
        m1, m2, m3, m4, m5 = st.columns(5)
        wins = len(replay[replay["condor"] == "WIN"])
        total = len(replay)
        total_pnl = replay["net_pnl_dollars"].sum()
        equity = replay["net_pnl_dollars"].cumsum()
        max_dd = (equity - equity.cummax()).min()
        avg_h_err = replay["h_err_pct"].mean()
        avg_l_err = replay["l_err_pct"].mean()

        m1.metric("Win Rate", f"{wins/total*100:.1f}%", f"{wins}/{total} trades")
        m2.metric("Total P&L", f"${total_pnl:,.0f}")
        m3.metric("Max Drawdown", f"${max_dd:,.0f}")
        m4.metric("Avg High Err", f"{avg_h_err:.3f}%")
        m5.metric("Avg Low Err", f"{avg_l_err:.3f}%")

        st.markdown("---")

        # ── 3-Panel Chart ──
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=("SPX: Predicted vs Actual High/Low", "Prediction Error (%)", "Equity Curve ($)"),
            row_heights=[0.45, 0.25, 0.30])

        # Panel 1: Actual high/low as range area + predicted as dotted lines
        # Actual range shaded area
        fig.add_trace(go.Scatter(
            x=dates, y=replay["actual_high"], mode="lines",
            name="Actual High", line=dict(color="#26a69a", width=1.5),
            legendgroup="actual"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=replay["actual_low"], mode="lines",
            name="Actual Low", line=dict(color="#26a69a", width=1.5),
            fill="tonexty", fillcolor="rgba(38,166,154,0.15)",
            legendgroup="actual"
        ), row=1, col=1)

        # Actual close as solid line
        fig.add_trace(go.Scatter(
            x=dates, y=replay["actual_close"], mode="lines+markers",
            name="Actual Close", line=dict(color="white", width=1.5),
            marker=dict(size=4, color="white")
        ), row=1, col=1)

        # Predicted high — blue dotted
        fig.add_trace(go.Scatter(
            x=dates, y=replay["pred_high"], mode="lines+markers",
            name="Predicted High",
            line=dict(color="#2196F3", width=2.5, dash="dot"),
            marker=dict(size=6, symbol="circle", color="#2196F3")
        ), row=1, col=1)

        # Predicted low — red/orange dotted
        fig.add_trace(go.Scatter(
            x=dates, y=replay["pred_low"], mode="lines+markers",
            name="Predicted Low",
            line=dict(color="#FF5722", width=2.5, dash="dot"),
            marker=dict(size=6, symbol="circle", color="#FF5722")
        ), row=1, col=1)

        # Conformal bands if available
        if "conf_90_high_hi" in replay.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=replay["conf_90_high_hi"], mode="lines",
                name="90% Conf Upper", line=dict(color="rgba(33,150,243,0.3)", width=1, dash="dash"),
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=dates, y=replay["conf_90_low_lo"], mode="lines",
                name="90% Conf Lower", line=dict(color="rgba(255,87,34,0.3)", width=1, dash="dash"),
                fill="tonexty", fillcolor="rgba(100,100,100,0.05)",
                showlegend=False
            ), row=1, col=1)

        # Panel 2: Error bars
        colors_h = ["#ef5350" if e > 0.5 else "#26a69a" for e in replay["h_err_pct"]]
        colors_l = ["#ef5350" if e > 0.5 else "#FF9800" for e in replay["l_err_pct"]]
        fig.add_trace(go.Bar(x=dates, y=replay["h_err_pct"], name="High Error %",
            marker_color=colors_h, opacity=0.7), row=2, col=1)
        fig.add_trace(go.Bar(x=dates, y=-replay["l_err_pct"], name="Low Error %",
            marker_color=colors_l, opacity=0.7), row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

        # Panel 3: Equity curve
        equity_vals = replay["net_pnl_dollars"].cumsum()
        fig.add_trace(go.Scatter(x=dates, y=equity_vals, mode="lines",
            name="Cumulative P&L", line=dict(color="#2196F3", width=2.5),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.1)"), row=3, col=1)

        loss_mask = replay["condor"] == "LOSS"
        if loss_mask.any():
            fig.add_trace(go.Scatter(x=dates[loss_mask], y=equity_vals[loss_mask],
                mode="markers", name="Loss Days",
                marker=dict(color="red", size=10, symbol="x")), row=3, col=1)

        win_mask = replay["condor"] == "WIN"
        if win_mask.any():
            fig.add_trace(go.Scatter(x=dates[win_mask], y=equity_vals[win_mask],
                mode="markers", name="Win Days",
                marker=dict(color="#26a69a", size=6, symbol="circle")), row=3, col=1)

        fig.update_layout(
            height=950, showlegend=True,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=20, t=60, b=30),
            xaxis_rangeslider_visible=False
        )
        fig.update_yaxes(title_text="SPX Price", row=1, col=1)
        fig.update_yaxes(title_text="Error %", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative $", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # ── Day-by-Day Table ──
        st.markdown("---")
        st.subheader("Day-by-Day Comparison")
        table_cols = ["date"]
        display_names = {"date": "Date"}
        for col, name in [("prior_close","Prior Close"), ("pred_high","Pred High"),
            ("actual_high","Actual High"), ("h_err_pct","High Err %"),
            ("pred_low","Pred Low"), ("actual_low","Actual Low"), ("l_err_pct","Low Err %"),
            ("actual_close","Actual Close"), ("dir_correct","Dir OK"), ("regime","Regime"),
            ("condor","Condor"), ("net_pnl_dollars","P&L ($)")]:
            if col in replay.columns:
                table_cols.append(col)
                display_names[col] = name

        display_df = replay[table_cols].copy().rename(columns=display_names)
        for c in ["Prior Close","Pred High","Actual High","Pred Low","Actual Low","Actual Close"]:
            if c in display_df.columns:
                display_df[c] = display_df[c].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        if "High Err %" in display_df.columns:
            display_df["High Err %"] = display_df["High Err %"].map(lambda x: f"{x:.3f}%")
        if "Low Err %" in display_df.columns:
            display_df["Low Err %"] = display_df["Low Err %"].map(lambda x: f"{x:.3f}%")
        if "P&L ($)" in display_df.columns:
            display_df["P&L ($)"] = display_df["P&L ($)"].map(lambda x: f"${x:+,.0f}")
        if "Dir OK" in display_df.columns:
            display_df["Dir OK"] = display_df["Dir OK"].map(lambda x: "✓" if x else "✗")
        if "Date" in display_df.columns:
            display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, use_container_width=True, height=600)

        # ── Coverage Heatmap ──
        cov_cols = ["cov68h", "cov68l", "cov90h", "cov90l"]
        if all(c in replay.columns for c in cov_cols):
            st.markdown("---")
            st.subheader("Conformal Coverage Heatmap")
            cov_data = replay[["date"] + cov_cols].copy()
            cov_matrix = cov_data[cov_cols].astype(int).T
            cov_matrix.columns = cov_data["date"].dt.strftime("%Y-%m-%d")
            cov_matrix.index = ["68% High", "68% Low", "90% High", "90% Low"]

            fig_cov = go.Figure(data=go.Heatmap(
                z=cov_matrix.values, x=cov_matrix.columns.tolist(),
                y=cov_matrix.index.tolist(),
                colorscale=[[0, "#ef5350"], [1, "#26a69a"]],
                showscale=False, text=cov_matrix.values,
                texttemplate="%{text}", textfont=dict(size=10)
            ))
            fig_cov.update_layout(height=200, margin=dict(l=100, r=20, t=20, b=40),
                template="plotly_dark", xaxis=dict(tickangle=45))
            st.plotly_chart(fig_cov, use_container_width=True)

            # Coverage summary
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("68% High Coverage", f"{replay['cov68h'].mean()*100:.1f}%")
            c2.metric("68% Low Coverage", f"{replay['cov68l'].mean()*100:.1f}%")
            c3.metric("90% High Coverage", f"{replay['cov90h'].mean()*100:.1f}%")
            c4.metric("90% Low Coverage", f"{replay['cov90l'].mean()*100:.1f}%")
    else:
        st.warning("No replay data found. Run the replay script first.")


elif page == "🎯 Replay Detail":
    st.title("Replay Detail — Jan-Feb 2026")
    replay = spx_load_replay(); spx = spx_load_data()

    if not replay.empty:
        selected_date = st.selectbox("Select trading day",replay["date"].dt.strftime("%Y-%m-%d").tolist(),index=len(replay)-1)
        row = replay[replay["date"]==pd.Timestamp(selected_date)].iloc[0]
        td = pd.Timestamp(selected_date)
        spx_row = spx.loc[td] if td in spx.index else None

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Regime",row.get("regime","N/A") if "regime" in row.index else "N/A")
        dir_correct = row.get("dir_correct",None) if "dir_correct" in row.index else None
        c2.metric("Direction","N/A","✓" if dir_correct else ("✗" if dir_correct is not None else ""))
        condor = row.get("condor","N/A") if "condor" in row.index else "N/A"
        pnl = row.get("net_pnl_dollars",0) if "net_pnl_dollars" in row.index else 0
        c3.metric("Condor",condor,f"${pnl:+,.0f}" if pd.notna(pnl) else "")
        c4.metric("VIX",f"{row['vix']:.2f}" if "vix" in row.index and pd.notna(row.get("vix")) else "N/A")

        st.markdown("---")
        e1,e2,e3 = st.columns(3)
        if "h_err_pct" in row.index: e1.metric("High Error",f"{row['h_err_pct']:.3f}%")
        if "l_err_pct" in row.index: e2.metric("Low Error",f"{row['l_err_pct']:.3f}%")
        if "h_err_pct" in row.index and "l_err_pct" in row.index:
            e3.metric("Combined MAE",f"{(row['h_err_pct']+row['l_err_pct'])/2:.3f}%")

        # Coverage flags
        st.markdown("---"); st.subheader("Conformal Coverage")
        cv1,cv2,cv3,cv4 = st.columns(4)
        for col,label,container in [("cov68h","68% High",cv1),("cov68l","68% Low",cv2),
            ("cov90h","90% High",cv3),("cov90l","90% Low",cv4)]:
            if col in row.index: container.metric(label,"✅ Covered" if row[col] else "❌ Missed")

        # Price levels chart with predicted vs actual
        if spx_row is not None:
            st.markdown("---"); st.subheader("Price Levels — Predicted vs Actual")
            prior_loc = spx.index.get_loc(td)
            prior_close = float(spx["Close"].iloc[prior_loc-1]) if prior_loc>0 else np.nan
            actual_high = float(spx_row["High"]); actual_low = float(spx_row["Low"])

            pred_high = pred_low = np.nan
            if "h_err_pct" in row.index and pd.notna(prior_close):
                pred_high = actual_high + (row["h_err_pct"]/100*prior_close)
                pred_low = actual_low + (row["l_err_pct"]/100*prior_close)

            fig_ic = go.Figure()
            fig_ic.add_trace(go.Bar(x=["Day"],y=[actual_high-actual_low],base=[actual_low],
                name="Actual Range",marker_color="rgba(38,166,154,0.3)",width=0.4))
            fig_ic.add_hline(y=prior_close,line_color="gray",line_width=2,
                annotation_text=f"Prior Close: {prior_close:,.2f}")
            fig_ic.add_hline(y=actual_high,line_color="#26a69a",line_dash="solid",
                annotation_text=f"Actual High: {actual_high:,.2f}")
            fig_ic.add_hline(y=actual_low,line_color="#ef5350",line_dash="solid",
                annotation_text=f"Actual Low: {actual_low:,.2f}")
            if pd.notna(pred_high):
                fig_ic.add_hline(y=pred_high,line_color="#2196F3",line_dash="dot",
                    annotation_text=f"Pred High: {pred_high:,.2f}")
                fig_ic.add_hline(y=pred_low,line_color="#FF9800",line_dash="dot",
                    annotation_text=f"Pred Low: {pred_low:,.2f}")
            fig_ic.update_layout(height=450,yaxis_title="SPX Price",
                margin=dict(l=50,r=180,t=30,b=30),showlegend=False)
            st.plotly_chart(fig_ic,use_container_width=True)

        with st.expander("Raw row data"):
            st.json({k:(float(v) if isinstance(v,(np.integer,np.floating)) else v) for k,v in row.to_dict().items()})
    else:
        st.warning("No replay data found.")

# ══════════════════════════════════════════════════════════════════════
#  SPX: PAPER TRADES
# ══════════════════════════════════════════════════════════════════════
elif page == "📋 Paper Trades":
    st.title("SPX — Paper Trade Log")
    paper = spx_load_paper_log()
    if not paper.empty:
        completed = paper[paper["actual_close"].notna()] if "actual_close" in paper.columns else pd.DataFrame()
        pending = paper[paper["actual_close"].isna()] if "actual_close" in paper.columns else paper
        if not pending.empty:
            st.info(f"📌 {len(pending)} pending signal(s)")
            st.dataframe(pending,use_container_width=True)
        if not completed.empty:
            st.markdown("---"); st.subheader("Completed Trades")
            if "condor_pnl" in completed.columns:
                eq = pd.to_numeric(completed["condor_pnl"],errors="coerce").cumsum()
                fig = go.Figure(); fig.add_trace(go.Scatter(x=completed["date"],y=eq,mode="lines+markers",fill="tozeroy"))
                fig.update_layout(height=300,title="Paper Trade Equity"); st.plotly_chart(fig,use_container_width=True)
            st.dataframe(completed,use_container_width=True,height=400)
    else:
        st.info("No paper trades yet.")

# ══════════════════════════════════════════════════════════════════════
#  SPX: LATEST SIGNAL (with strike map)
# ══════════════════════════════════════════════════════════════════════
elif page == "🔮 Latest Signal":
    st.title("SPX — Latest Signal")
    sig = spx_load_signal()
    if sig:
        h1,h2,h3,h4,h5 = st.columns(5)
        h1.metric("Date",sig.get("signal_date","N/A"))
        h2.metric("Regime",sig.get("regime","N/A"))
        h3.metric("Direction",sig.get("direction","N/A"),f"{sig.get('direction_prob',0)*100:.1f}%")
        h4.metric("VIX",f"{sig.get('vix_spot',0):.2f}")
        h5.metric("Tradeable","✅" if sig.get("tradeable") else "❌")

        st.markdown("---")
        p1,p2,p3 = st.columns(3)
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
            st.write(f"68%: [{sig.get('conf_68_high_lo',0):,.2f}, {sig.get('conf_68_high_hi',0):,.2f}]")
            st.write(f"90%: [{sig.get('conf_90_high_lo',0):,.2f}, {sig.get('conf_90_high_hi',0):,.2f}]")

        # Strike Map
        st.markdown("---"); st.subheader("Strike Map")
        pc = sig.get("prior_close",0)
        fig = go.Figure()
        fig.add_hrect(y0=sig.get("conf_90_low_lo",0),y1=sig.get("conf_90_high_hi",0),fillcolor="rgba(33,150,243,0.05)",line_width=0)
        fig.add_hrect(y0=sig.get("conf_68_low_lo",0),y1=sig.get("conf_68_high_hi",0),fillcolor="rgba(33,150,243,0.1)",line_width=0)
        fig.add_hline(y=pc,line_color="gray",line_width=2,annotation_text=f"Prior Close: {pc:,.2f}")
        fig.add_hline(y=sig.get("predicted_high",0),line_color="#2196F3",line_dash="dot",annotation_text=f"Pred High: {sig.get('predicted_high',0):,.2f}")
        fig.add_hline(y=sig.get("predicted_low",0),line_color="#FF9800",line_dash="dot",annotation_text=f"Pred Low: {sig.get('predicted_low',0):,.2f}")
        fig.add_hline(y=sig.get("ic_short_call",0),line_color="#e91e63",line_dash="dash",annotation_text=f"Short Call: {sig.get('ic_short_call',0):,.2f}")
        fig.add_hline(y=sig.get("ic_short_put",0),line_color="#9c27b0",line_dash="dash",annotation_text=f"Short Put: {sig.get('ic_short_put',0):,.2f}")
        fig.update_layout(height=500,yaxis_title="SPX Price",margin=dict(l=50,r=200,t=30,b=30))
        st.plotly_chart(fig,use_container_width=True)

        with st.expander("Raw JSON"): st.json(sig)
    else:
        st.warning("No signal found.")

# ══════════════════════════════════════════════════════════════════════
#  SPX: ES/MES LEVELS
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 ES/MES Levels":
    st.title("ES/MES Futures Trading Levels")
    es = spx_load_es_levels()
    if es:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Prior Close",f"{es.get('prior_close',0):,.2f}")
        c2.metric("Regime",es.get("regime","N/A"))
        c3.metric("VIX",f"{es.get('vix',0):.2f}")
        c4.metric("Direction",es.get("direction","N/A"))
        c5.metric("Risk",f"{es.get('risk_score',0)}/5")
        status = es.get("status","")
        if "DO NOT TRADE" in status: st.error(f"⛔ {status}")
        elif "CAUTION" in status: st.warning(f"⚠️ {status}")
        else: st.success(f"✅ {status}")
        st.markdown("---")
        l1,l2 = st.columns(2)
        l1.metric("Upside Wall",f"{es.get('upside_wall',0):,.2f}")
        l2.metric("Downside Wall",f"{es.get('downside_wall',0):,.2f}")
        with st.expander("Full JSON"): st.json(es)
    else:
        st.warning("No ES levels found.")

# ══════════════════════════════════════════════════════════════════════
#  SPX: MODEL HEALTH
# ══════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Health":
    st.title("Model Health & Learning")
    tab1,tab2,tab3 = st.tabs(["Error Correction","Conformal Calibration","Weekly Retrain"])

    with tab1:
        st.subheader("Level 3: Error Correction Layer")
        error_df = spx_load_error_history()
        if error_df.empty:
            st.info("Error correction activates after 20 trading days. Morning recon records errors daily.")
        else:
            st.metric("Days of error data",len(error_df))
            active = len(error_df) >= 20
            st.metric("Status","🟢 Active" if active else f"🟡 Collecting ({len(error_df)}/20)")
            if "error_high" in error_df.columns:
                fig = make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=("High Error","Low Error"))
                fig.add_trace(go.Scatter(x=error_df.index,y=error_df["error_high"],mode="lines+markers",
                    name="High Error",line=dict(color="#2196F3")),row=1,col=1)
                fig.add_trace(go.Scatter(x=error_df.index,y=error_df["error_low"],mode="lines+markers",
                    name="Low Error",line=dict(color="#FF9800")),row=2,col=1)
                fig.add_hline(y=0,line_color="gray",line_dash="dash",row=1,col=1)
                fig.add_hline(y=0,line_color="gray",line_dash="dash",row=2,col=1)
                fig.update_layout(height=500); st.plotly_chart(fig,use_container_width=True)
            if len(error_df) >= 5:
                st.markdown("---"); st.subheader("Rolling Bias (5-day)")
                recent = error_df.tail(5); bc1,bc2 = st.columns(2)
                if "error_high" in recent.columns:
                    bc1.metric("High Bias",f"{recent['error_high'].mean():+.4f}%")
                if "error_low" in recent.columns:
                    bc2.metric("Low Bias",f"{recent['error_low'].mean():+.4f}%")
            corr_log = spx_load_correction_log()
            if not corr_log.empty:
                st.markdown("---"); st.subheader("Recent Corrections")
                st.dataframe(corr_log.tail(10),use_container_width=True)

    with tab2:
        st.subheader("Level 2: Adaptive Conformal Calibration")
        cal = spx_load_calibration_health()
        if cal is None:
            st.info("Calibration data appears after next signal generation.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Date",cal.get("date","N/A"))
            c2.metric("Samples",cal.get("n_samples",0))
            c3.metric("Half-life",f"{cal.get('half_life',15)} days")
            st.metric("Auto-widen","🔴 YES" if cal.get("widen_active") else "🟢 No")
            st.markdown("---"); st.subheader("Quantile Widths by Regime")
            regime_data = []
            for regime in ["GREEN","YELLOW","RED"]:
                r = {"Regime":regime}
                for q in [68,90]:
                    val = cal.get(f"q{q}_{regime}")
                    r[f"{q}% Width"] = f"{val*100:.3f}%" if val else "N/A"
                regime_data.append(r)
            all_r = {"Regime":"ALL"}
            for q in [68,90]:
                val = cal.get(f"q{q}_all")
                all_r[f"{q}% Width"] = f"{val*100:.3f}%" if val else "N/A"
            regime_data.append(all_r)
            st.table(pd.DataFrame(regime_data))

    with tab3:
        st.subheader("Level 1: Weekly Model Retrain")
        retrain = spx_load_retrain_report()
        if retrain is None:
            st.info("Retrain runs Sundays at 10:00 UTC.")
        else:
            st.metric("Last Retrain",retrain.get("timestamp","N/A")[:10])
            st.metric("Status",retrain.get("status","N/A"))
            st.metric("Decision",retrain.get("decision","N/A"))
            if "old_metrics" in retrain:
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown("**Current Model**"); old = retrain["old_metrics"]
                    st.write(f"Win Rate: {old.get('win_rate',0):.1f}%")
                    st.write(f"P&L: ${old.get('total_pnl',0):,.2f}")
                    st.write(f"Sharpe: {old.get('sharpe',0):.2f}")
                with col2:
                    if "new_metrics" in retrain:
                        st.markdown("**Retrained Model**"); new = retrain["new_metrics"]
                        st.write(f"Win Rate: {new.get('win_rate',0):.1f}%")
                        st.write(f"P&L: ${new.get('total_pnl',0):,.2f}")
                        st.write(f"Sharpe: {new.get('sharpe',0):.2f}")

# ══════════════════════════════════════════════════════════════════════
#  SPX: RISK & INTEL
# ══════════════════════════════════════════════════════════════════════
elif page == "⚡ Risk & Intel":
    st.title("Market Risk & Intelligence")
    intel = spx_load_market_intel()
    if intel:
        risk = intel.get("risk_score",0)
        icons = {1:"🟢",2:"🟡",3:"🟠",4:"🔴",5:"⛔"}
        labels = {1:"LOW",2:"MODERATE",3:"ELEVATED",4:"HIGH",5:"EXTREME"}
        st.markdown(f"## {icons.get(risk,'❓')} Risk: {risk}/5 — {labels.get(risk,'UNKNOWN')}")
        if intel.get("tail_risk_flag"): st.error("⚠️ TAIL RISK FLAG — DO NOT TRADE")
        c1,c2,c3 = st.columns(3)
        c1.metric("Risk Score",f"{risk}/5")
        c2.metric("Tail Risk","🔴 YES" if intel.get("tail_risk_flag") else "🟢 No")
        c3.metric("Regime",intel.get("regime","N/A"))
        events = intel.get("key_events",[])
        if events:
            st.markdown("### Key Events")
            for ev in events:
                if isinstance(ev,dict): st.write(f"• **{ev.get('event','')}** — {ev.get('impact','')}")
                else: st.write(f"• {ev}")
        with st.expander("Raw Intel"): st.json(intel)
    else:
        st.info("No market intel. Updates at 19:30 UTC.")


# ══════════════════════════════════════════════════════════════════════
#  OPTIONS ALGO PAGES (all verified — same as previous fix)
# ══════════════════════════════════════════════════════════════════════

elif page == "🔍 Ticker Analysis":
    st.title("🔍 Ticker Analysis")
    st.caption("Type any ticker for regime, IV, and strategy analysis.")
    col_in,col_btn = st.columns([3,1])
    with col_in: ticker = st.text_input("Ticker",value="",placeholder="CRM, AAPL, NVDA...").strip().upper()
    with col_btn: st.markdown("<br>",unsafe_allow_html=True); run = st.button("Analyze",type="primary",use_container_width=True)

    if ticker and run:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                from src.data.stock_fetcher import download_universe
                from src.analysis.technical import classify_regime
                from src.analysis.volatility import analyze_iv
                from src.strategy.selector import select_strategy
                data = download_universe([ticker],period="2y")
                if ticker not in data or data[ticker].empty: st.error(f"No data for {ticker}")
                else:
                    df = data[ticker]; price = float(df["close"].iloc[-1])
                    prev = float(df["close"].iloc[-2]) if len(df)>1 else price
                    chg = (price-prev)/prev*100
                    st.markdown("---")
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Price",f"${price:.2f}",f"{chg:+.2f}%")
                    c2.metric("High",f"${float(df['high'].iloc[-1]):.2f}")
                    c3.metric("Low",f"${float(df['low'].iloc[-1]):.2f}")
                    c4.metric("Volume",f"{int(df['volume'].iloc[-1]):,}")

                    regime = classify_regime(ticker,df)
                    if regime:
                        st.markdown("---"); st.subheader("📊 Technical Regime")
                        r1,r2,r3,r4,r5,r6 = st.columns(6)
                        r1.metric("Regime",regime.regime.value); r2.metric("Score",f"{regime.direction_score:+.2f}")
                        r3.metric("ADX",f"{regime.adx:.1f}"); r4.metric("RSI",f"{regime.rsi:.1f}")
                        r5.metric("ATR %",f"{regime.atr_pct:.2f}%"); r6.metric("Volume",regime.volume_trend)
                        ca,cb = st.columns(2)
                        with ca: st.write(f"**EMA:** {regime.ema_alignment} | **Vol State:** {regime.volatility_state}")
                        with cb: st.write(f"**Support:** ${regime.support:.2f} | **Resistance:** ${regime.resistance:.2f}")
                        if regime.bb_squeeze: st.success("📦 BB Squeeze Active")

                    iv = analyze_iv(ticker,df)
                    if iv:
                        st.markdown("---"); st.subheader("📈 Volatility Profile")
                        v1,v2,v3,v4,v5 = st.columns(5)
                        v1.metric("IV Rank",f"{iv.iv_rank:.0f}%"); v2.metric("IV Pctile",f"{iv.iv_percentile:.0f}%")
                        v3.metric("Current IV",f"{iv.current_iv:.1f}%"); v4.metric("HV-20",f"{iv.hv_20:.1f}%")
                        v5.metric("IV/HV",f"{iv.iv_hv_ratio:.2f}")
                        if iv.premium_action=="SELL": st.error(f"💰 SELL PREMIUM — IV Rank {iv.iv_rank:.0f}%")
                        elif iv.premium_action=="BUY": st.success(f"🎯 BUY PREMIUM — IV Rank {iv.iv_rank:.0f}%")
                        else: st.info(f"⚖️ NEUTRAL — IV Rank {iv.iv_rank:.0f}%")

                    if regime and iv:
                        strat = select_strategy(regime,iv)
                        st.markdown("---"); st.subheader("🎯 Strategy")
                        strat_name = strat.strategy.value if hasattr(strat.strategy,'value') else str(strat.strategy)
                        s1,s2,s3,s4 = st.columns(4)
                        s1.metric("Strategy",strat_name.replace("_"," ")); s2.metric("Direction",strat.direction)
                        s3.metric("Confidence",f"{strat.confidence:.0%}"); s4.metric("DTE",f"{strat.target_dte}d")
                        if strat.rationale: st.info(f"**Rationale:** {strat.rationale}")
                        st.caption("⚠️ Full trade construction requires Polygon.io ($29/mo).")

                    st.markdown("---"); st.subheader("📉 Chart (90d)")
                    cdf = df.tail(90)
                    fig = go.Figure(data=[go.Candlestick(x=cdf.index,open=cdf["open"],high=cdf["high"],low=cdf["low"],close=cdf["close"])])
                    if regime:
                        fig.add_hline(y=regime.support,line_dash="dash",line_color="#FF9800",annotation_text=f"S ${regime.support:.0f}")
                        fig.add_hline(y=regime.resistance,line_dash="dash",line_color="#2196F3",annotation_text=f"R ${regime.resistance:.0f}")
                    fig.update_layout(height=500,xaxis_rangeslider_visible=False,template="plotly_dark",margin=dict(l=50,r=150,t=30,b=30))
                    st.plotly_chart(fig,use_container_width=True)
            except Exception as e:
                st.error(f"Analysis failed: {e}"); import traceback; st.code(traceback.format_exc())
    else:
        st.info("Enter a ticker and click **Analyze**.")


elif page == "📈 Today's Picks":
    st.title("Options Algo — Today's Picks")
    signal = opt_load_signal()
    if signal:
        mkt = signal.get("market_context",{})
        if mkt:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Market",mkt.get("market_regime","?")); c2.metric("VIX",f"{mkt.get('vix_level',0):.0f}")
            c3.metric("SPY Trend",mkt.get("spy_trend","?")); c4.metric("Breadth",f"{mkt.get('breadth_score',0):.0%}")
        picks = signal.get("top_picks",[])
        if picks:
            for p in picks:
                rec=p.get("recommendation",{}); ctx=p.get("context",{}); iv_d=ctx.get("iv_detail",{})
                with st.expander(f"#{p.get('priority','?')} {rec.get('ticker','?')} — {rec.get('strategy','?').replace('_',' ')} | {direction_badge(rec.get('direction',''))} | Conf: {rec.get('confidence',0):.0%}",
                    expanded=(p.get("priority",10)<=3)):
                    tc1,tc2 = st.columns(2)
                    with tc1:
                        st.write(f"Price: **${ctx.get('price',0):.2f}** | Regime: `{rec.get('regime','?')}`")
                        st.write(f"IV Rank: **{iv_d.get('iv_rank',0):.0f}%** | IV/HV: {iv_d.get('iv_hv_ratio',0):.2f} | {iv_d.get('premium_action','?')}")
                    with tc2:
                        trade=p.get("trade",{})
                        if not trade.get("dry_run"):
                            for k in ["net_credit","max_risk","prob_profit"]:
                                if k in trade: st.write(f"{k.replace('_',' ').title()}: **${trade[k]:.2f}**" if 'prob' not in k else f"PoP: **{trade[k]:.0f}%**")
                        else: st.info("Dry-run")
                    st.caption(rec.get("rationale",""))
            st.markdown("---")
            regime_dist = signal.get("regime_distribution",{})
            if regime_dist:
                df_r = pd.DataFrame(list(regime_dist.items()),columns=["Regime","Count"]).sort_values("Count",ascending=False)
                fig = px.bar(df_r,x="Regime",y="Count",color="Regime",color_discrete_map=REGIME_COLORS)
                st.plotly_chart(fig,use_container_width=True)
        else: st.info("No picks today.")
    else: st.warning("No signal. Run scan first.")


elif page == "📋 Trade Tracker":
    st.title("📋 Trade Tracker")
    trades = opt_load_trades()
    if trades:
        df = pd.DataFrame(trades); ot = df[df["outcome"]=="OPEN"]; ct = df[df["outcome"]!="OPEN"]
        c1,c2,c3 = st.columns(3); c1.metric("Total",len(df)); c2.metric("Open",len(ot)); c3.metric("Closed",len(ct))
        if not ct.empty:
            wins = (ct["won"]==True).sum(); st.metric("Win Rate",f"{wins/len(ct)*100:.0f}%")
        if not ot.empty:
            st.markdown("---"); st.subheader("Open"); st.dataframe(ot,use_container_width=True)
        if not ct.empty:
            st.markdown("---"); st.subheader("Closed"); st.dataframe(ct.sort_values("exit_date",ascending=False),use_container_width=True)
    else: st.info("No trades yet.")


elif page == "📊 Strategy Stats":
    st.title("📊 Strategy Performance")
    trades = opt_load_trades()
    if trades:
        df = pd.DataFrame(trades); closed = df[df["outcome"]!="OPEN"]
        if not closed.empty:
            rows = []
            for s in closed["strategy"].unique():
                sub = closed[closed["strategy"]==s]; w = int((sub["won"]==True).sum()); t = len(sub)
                rows.append({"Strategy":s.replace("_"," "),"Trades":t,"Wins":w,"Win %":round(w/t*100) if t else 0,
                    "Total P&L":round(sub["pnl"].sum(),2),"Avg P&L":round(sub["pnl"].mean(),2)})
            st.dataframe(pd.DataFrame(rows),use_container_width=True)
        else: st.info("No closed trades.")
    else: st.info("No trade data.")


elif page == "🌡️ IV Heatmap":
    st.title("🌡️ IV Heatmap")
    signal = opt_load_signal()
    if signal:
        picks = signal.get("top_picks",[]); rows = []
        for p in picks:
            rec=p.get("recommendation",{}); iv_d=p.get("context",{}).get("iv_detail",{})
            rows.append({"Ticker":rec.get("ticker","?"),"IV Rank":round(float(iv_d.get("iv_rank",0)),1),
                "IV Pctile":round(float(iv_d.get("iv_percentile",0)),1),"HV-20":round(float(iv_d.get("hv_20",0)),1),
                "IV/HV":round(float(iv_d.get("iv_hv_ratio",0)),2),"Action":iv_d.get("premium_action","?"),
                "Strategy":rec.get("strategy","?").replace("_"," ")})
        if rows:
            iv_df = pd.DataFrame(rows).sort_values("IV Rank",ascending=False)
            fig = px.imshow(iv_df[["IV Rank","IV Pctile","HV-20","IV/HV"]].values,
                x=["IV Rank","IV Pctile","HV-20","IV/HV"],y=iv_df["Ticker"].tolist(),
                color_continuous_scale="RdYlGn_r",aspect="auto",text_auto=".1f")
            fig.update_layout(height=max(300,len(rows)*80+100),template="plotly_dark")
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(iv_df,use_container_width=True)
        else: st.info("No IV data.")
    else: st.warning("No signal.")


elif page == "🗺️ Regime Map":
    st.title("🗺️ Regime Map")
    signal = opt_load_signal()
    if signal:
        dist = signal.get("regime_distribution",{})
        if dist:
            c1,c2 = st.columns(2)
            with c1:
                df_r = pd.DataFrame(list(dist.items()),columns=["Regime","Count"])
                st.plotly_chart(px.pie(df_r,names="Regime",values="Count",color="Regime",color_discrete_map=REGIME_COLORS),use_container_width=True)
            with c2:
                recs = signal.get("all_recommendations",[])
                if recs:
                    sc = pd.DataFrame(recs)["strategy"].value_counts().reset_index(); sc.columns=["Strategy","Count"]
                    st.plotly_chart(px.bar(sc,x="Count",y="Strategy",orientation="h"),use_container_width=True)
        recs = signal.get("all_recommendations",[])
        if recs:
            rd = pd.DataFrame(recs)
            cols = [c for c in ["ticker","regime","strategy","direction","confidence"] if c in rd.columns]
            st.dataframe(rd[cols],use_container_width=True,height=400)
    else: st.warning("No signal.")


elif page == "▶️ Run Scan":
    st.title("▶️ Run Scan")
    dry = st.checkbox("Dry Run",value=True)
    tickers_raw = st.text_input("Custom tickers (comma-separated, blank=full universe)",placeholder="AAPL, MSFT")
    if st.button("🚀 Run Scan",type="primary"):
        custom = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()] if tickers_raw.strip() else None
        with st.spinner("Running..."):
            try:
                from src.pipeline.nightly_scan import run_nightly_scan
                result = run_nightly_scan(universe_override=custom,dry_run=dry)
                st.success(f"Done! {len(result.get('top_picks',[]))} picks in {result.get('elapsed_seconds',0):.0f}s")
                st.cache_data.clear(); st.rerun()
            except Exception as e:
                st.error(str(e)); import traceback; st.code(traceback.format_exc())


elif page == "📅 Cron Jobs":
    st.title("📅 Scheduled Jobs")
    jobs = load_cron_jobs()
    if jobs:
        for job in jobs:
            name=job.get("name","?"); sched=job.get("schedule",{})
            expr = sched.get("expr","N/A") if isinstance(sched,dict) else str(sched)
            state=job.get("state",{}); status=state.get("lastRunStatus","never")
            icon = "🟢" if status=="ok" else "🔴" if status=="error" else "⚪"
            with st.expander(f"{icon} {name} — `{expr}`"):
                c1,c2 = st.columns(2); c1.metric("Status",status); c2.metric("Errors",state.get("consecutiveErrors",0))
    else: st.info("No cron jobs found.")
