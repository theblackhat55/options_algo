"""
dashboard/ticker_analysis.py
=============================
Single-ticker deep analysis page for the Options Algo dashboard.
Type any ticker → get regime, IV, strategy, trade construction, and chart.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stock_fetcher import download_tickers
from src.analysis.technical import classify_regime
from src.analysis.volatility import analyze_iv
from src.analysis.relative_strength import compute_rs
from src.strategy.selector import select_strategy
from src.strategy.credit_spread import build_bull_put, build_bear_call
from src.strategy.bull_call_spread import build_bull_call
from src.strategy.bear_put_spread import build_bear_put
from src.strategy.iron_condor import build_iron_condor
from src.strategy.butterfly import build_butterfly


BUILDERS = {
    "BEAR_CALL_SPREAD": build_bear_call,
    "BULL_PUT_SPREAD": build_bull_put,
    "BULL_CALL_SPREAD": build_bull_call,
    "BEAR_PUT_SPREAD": build_bear_put,
    "IRON_CONDOR": build_iron_condor,
    "BUTTERFLY": build_butterfly,
}

REGIME_COLORS = {
    "STRONG_UPTREND": "#00C853",
    "UPTREND": "#69F0AE",
    "RANGE_BOUND": "#FFD740",
    "DOWNTREND": "#FF6D00",
    "STRONG_DOWNTREND": "#D50000",
    "SQUEEZE": "#AA00FF",
    "REVERSAL_UP": "#00B0FF",
    "REVERSAL_DOWN": "#FF4081",
}


def _build_price_chart(df: pd.DataFrame, ticker: str, regime) -> go.Figure:
    """Candlestick + EMA + volume chart."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{ticker} — {regime.regime.value}", "Volume"],
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
    ), row=1, col=1)

    for col in df.columns:
        if "EMA" in col or "SMA" in col:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                line=dict(width=1),
            ), row=1, col=1)

    if "BBL" in df.columns and "BBU" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BBU"], name="BB Upper",
            line=dict(width=0.5, dash="dot", color="gray"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BBL"], name="BB Lower",
            line=dict(width=0.5, dash="dot", color="gray"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.1)",
        ), row=1, col=1)

    colors = ["#EF5350" if c < o else "#26A69A" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=600, xaxis_rangeslider_visible=False,
        template="plotly_dark", margin=dict(t=40, b=20, l=50, r=20),
    )
    fig.update_xaxes(range=[df.index[-90], df.index[-1]], row=1, col=1)
    return fig


def _build_iv_chart(df: pd.DataFrame) -> go.Figure:
    """HV cone chart (20d, 40d, 60d)."""
    fig = go.Figure()
    for window, color in [(20, "#00B0FF"), (40, "#FFD740"), (60, "#FF6D00")]:
        col = f"HV_{window}"
        if col not in df.columns:
            log_ret = df["Close"].pct_change().apply(lambda x: (1 + x)).apply(lambda x: x if x > 0 else 0.0001)
            import numpy as np
            df[col] = log_ret.rolling(window).std() * (252 ** 0.5) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=f"HV-{window}",
            line=dict(color=color, width=1.5),
        ))

    fig.update_layout(
        title="Historical Volatility Cone",
        height=350, template="plotly_dark",
        margin=dict(t=40, b=20, l=50, r=20),
        yaxis_title="Annualized Vol (%)",
    )
    fig.update_xaxes(range=[df.index[-252], df.index[-1]])
    return fig


def render():
    """Main render function called from app.py router."""
    st.title("🔍 Ticker Analysis")
    st.caption("Type any ticker to get a full regime, IV, and options strategy analysis.")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker = st.text_input(
            "Ticker Symbol",
            value="",
            placeholder="e.g. CRM, AAPL, NVDA, MSFT",
            key="ticker_input",
        ).strip().upper()
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Analyze", type="primary", use_container_width=True)

    if not ticker or not run:
        st.info("Enter a ticker and click **Analyze** to begin.")
        # Show recent analyses from session state
        if "analysis_history" in st.session_state and st.session_state.analysis_history:
            st.markdown("---")
            st.markdown("### Recent Analyses")
            for h in reversed(st.session_state.analysis_history[-5:]):
                st.caption(f"**{h['ticker']}** — {h['strategy']} | Conf: {h['confidence']:.0%} | IV Rank: {h['iv_rank']:.0f}%")
        return

    # ── Fetch Data ──
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            data = download_tickers([ticker], period="2y")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return

    if ticker not in data or data[ticker].empty:
        st.error(f"No data found for **{ticker}**. Check the symbol and try again.")
        return

    df = data[ticker]
    price = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else price
    day_change = (price - prev_close) / prev_close * 100

    # ── Header Metrics ──
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"${price:.2f}", f"{day_change:+.2f}%")
    c2.metric("Day High", f"${df['High'].iloc[-1]:.2f}")
    c3.metric("Day Low", f"${df['Low'].iloc[-1]:.2f}")
    c4.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    c5.metric("Data", f"{len(df)} bars")

    # ── Regime Classification ──
    with st.spinner("Classifying regime..."):
        regime = classify_regime(df)

    if regime is None:
        st.error("Could not classify regime (insufficient data).")
        return

    regime_col = REGIME_COLORS.get(regime.regime.value, "#9E9E9E")

    st.markdown("---")
    st.subheader("Technical Regime")

    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
    rc1.metric("Regime", regime.regime.value)
    rc2.metric("Direction", regime.direction)
    rc3.metric("ADX", f"{regime.adx:.1f}")
    rc4.metric("RSI", f"{regime.rsi:.1f}")
    rc5.metric("Trend Score", f"{regime.direction_score:.2f}")
    rc6.metric("Volume Trend", getattr(regime, "volume_trend", "?"))

    ema_info = getattr(regime, "ema_alignment", "N/A")
    squeeze = getattr(regime, "bb_squeeze", False)
    if squeeze:
        st.success("📦 Bollinger Band Squeeze Active — potential breakout incoming")
    if ema_info:
        st.caption(f"EMA Alignment: {ema_info}")

    # ── IV Analysis ──
    with st.spinner("Analyzing implied volatility..."):
        iv = analyze_iv(df)

    if iv is None:
        st.error("Could not compute IV analysis (insufficient data).")
        return

    st.markdown("---")
    st.subheader("Volatility Profile")

    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
    vc1.metric("IV Rank", f"{iv.iv_rank:.0f}%")
    vc2.metric("IV Percentile", f"{iv.iv_percentile:.0f}%")
    vc3.metric("Current IV", f"{iv.current_iv:.1f}%")
    vc4.metric("HV-20", f"{iv.hv_20:.1f}%")
    vc5.metric("IV/HV Ratio", f"{iv.iv_hv_ratio:.2f}")

    iv_trend = getattr(iv, "iv_trend", "N/A")
    premium_action = iv.premium_action

    if premium_action == "SELL":
        st.error(f"💰 **SELL PREMIUM** — IV Rank {iv.iv_rank:.0f}% (elevated). IV trend: {iv_trend}")
    elif premium_action == "BUY":
        st.success(f"🎯 **BUY PREMIUM** — IV Rank {iv.iv_rank:.0f}% (depressed). IV trend: {iv_trend}")
    else:
        st.info(f"⚖️ **NEUTRAL** — IV Rank {iv.iv_rank:.0f}%. IV trend: {iv_trend}")

    # ── Strategy Recommendation ──
    with st.spinner("Selecting optimal strategy..."):
        strat = select_strategy(regime, iv)

    st.markdown("---")
    st.subheader("Recommended Strategy")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Strategy", strat.strategy.replace("_", " "))
    sc2.metric("Direction", strat.direction)
    sc3.metric("Confidence", f"{strat.confidence:.0%}")
    sc4.metric("Target DTE", f"{strat.dte}d")

    rationale = getattr(strat, "rationale", "")
    if rationale:
        st.caption(f"Rationale: {rationale}")

    # ── Trade Construction ──
    builder = BUILDERS.get(strat.strategy)
    if builder:
        st.markdown("---")
        st.subheader("Trade Construction")

        try:
            trade = builder(price, iv, strat.dte)
        except Exception as e:
            trade = None
            st.warning(f"Could not construct trade: {e}")

        if trade:
            trade_dict = trade.__dict__
            spread_type = trade_dict.get("spread_type", strat.strategy)

            tc1, tc2, tc3 = st.columns(3)

            with tc1:
                st.markdown("**Strikes**")
                if "short_strike" in trade_dict and "long_strike" in trade_dict:
                    st.write(f"Short: **${trade_dict['short_strike']:.1f}**")
                    st.write(f"Long: **${trade_dict['long_strike']:.1f}**")
                if "body" in trade_dict:
                    st.write(f"Body: **${trade_dict['body']:.1f}**")
                    st.write(f"Wings: ${trade_dict.get('lower_wing', 0):.1f} / ${trade_dict.get('upper_wing', 0):.1f}")
                if "short_put" in trade_dict:
                    st.write(f"Put: ${trade_dict.get('long_put', 0):.1f} / ${trade_dict['short_put']:.1f}")
                    st.write(f"Call: ${trade_dict.get('short_call', 0):.1f} / ${trade_dict.get('long_call', 0):.1f}")

                exp = trade_dict.get("expiration", "N/A")
                dte = trade_dict.get("dte", strat.dte)
                st.write(f"Expiry: **{exp}** ({dte}d)")

            with tc2:
                st.markdown("**Risk / Reward**")
                credit = trade_dict.get("net_credit", trade_dict.get("total_credit", 0))
                debit = trade_dict.get("net_debit", 0)
                max_risk = trade_dict.get("max_risk", 0)
                max_profit = trade_dict.get("max_profit", credit * 100 if credit else 0)

                if credit:
                    st.write(f"Credit: **${credit:.2f}**")
                if debit:
                    st.write(f"Debit: **${debit:.2f}**")
                st.write(f"Max Risk: **${max_risk:.2f}**")
                st.write(f"Max Profit: **${max_profit:.2f}**")
                rr = trade_dict.get("risk_reward_ratio", 0)
                if rr:
                    st.write(f"R/R Ratio: **{rr:.1f}x**")

            with tc3:
                st.markdown("**Probabilities**")
                pop = trade_dict.get("prob_profit", 0)
                be = trade_dict.get("breakeven", trade_dict.get("put_breakeven", 0))
                st.write(f"PoP: **{pop:.0f}%**")
                if be:
                    st.write(f"Breakeven: **${be:.2f}**")
                be2 = trade_dict.get("call_breakeven", 0)
                if be2:
                    st.write(f"Upper BE: **${be2:.2f}**")

                ev = trade_dict.get("expected_value", 0)
                if ev:
                    st.write(f"Expected Value: **${ev:.2f}**")
                ann_ror = trade_dict.get("annualized_ror", 0)
                if ann_ror:
                    st.write(f"Ann. RoR: **{ann_ror:.0f}%**")
    else:
        st.info(f"No automated trade builder available for {strat.strategy}. Manual construction recommended.")

    # ── Charts ──
    st.markdown("---")
    st.subheader("Charts")

    tab1, tab2 = st.tabs(["Price & Technicals", "Volatility Cone"])

    with tab1:
        fig_price = _build_price_chart(df.tail(252), ticker, regime)
        st.plotly_chart(fig_price, use_container_width=True)

    with tab2:
        fig_iv = _build_iv_chart(df)
        st.plotly_chart(fig_iv, use_container_width=True)

    # ── Save to session history ──
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    st.session_state.analysis_history.append({
        "ticker": ticker,
        "price": price,
        "regime": regime.regime.value,
        "iv_rank": iv.iv_rank,
        "strategy": strat.strategy,
        "confidence": strat.confidence,
        "direction": strat.direction,
        "date": date.today().isoformat(),
    })
