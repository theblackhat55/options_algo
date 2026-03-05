"""
dashboard/ticker_analysis.py
=============================
Single-ticker deep analysis page for the Options Algo dashboard.
Type any ticker → get regime, IV, strategy recommendation, and chart.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.stock_fetcher import download_universe
from src.analysis.technical import classify_regime
from src.analysis.volatility import analyze_iv
from src.strategy.selector import select_strategy


def _build_price_chart(df: pd.DataFrame, ticker: str, regime_label: str) -> go.Figure:
    """90-day candlestick + volume chart."""
    recent = df.tail(90).copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
        subplot_titles=[f"{ticker} — {regime_label}", "Volume"],
    )
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent["open"], high=recent["high"],
        low=recent["low"], close=recent["close"], name="Price",
    ), row=1, col=1)

    colors = ["#EF5350" if c < o else "#26A69A"
              for o, c in zip(recent["open"], recent["close"])]
    fig.add_trace(go.Bar(
        x=recent.index, y=recent["volume"], name="Volume",
        marker_color=colors, showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=550, xaxis_rangeslider_visible=False,
        template="plotly_dark", margin=dict(t=40, b=20, l=50, r=20),
    )
    return fig


def _build_hv_chart(df: pd.DataFrame) -> go.Figure:
    """Historical volatility cone (20d, 40d, 60d)."""
    fig = go.Figure()
    close = df["close"].astype(float)
    for window, color in [(20, "#00B0FF"), (40, "#FFD740"), (60, "#FF6D00")]:
        log_ret = np.log(close / close.shift(1))
        hv = log_ret.rolling(window).std() * np.sqrt(252) * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=hv, name=f"HV-{window}",
            line=dict(color=color, width=1.5),
        ))
    fig.update_layout(
        title="Historical Volatility Cone",
        height=350, template="plotly_dark",
        margin=dict(t=40, b=20, l=50, r=20),
        yaxis_title="Annualized Vol (%)",
    )
    recent_idx = df.index[-252] if len(df) >= 252 else df.index[0]
    fig.update_xaxes(range=[recent_idx, df.index[-1]])
    return fig


def render():
    """Main render function called from app.py router."""
    st.title("🔍 Ticker Analysis")
    st.caption("Type any ticker for regime, IV, and strategy analysis.")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker = st.text_input(
            "Ticker Symbol", value="", placeholder="e.g. CRM, AAPL, NVDA",
            key="ticker_input",
        ).strip().upper()
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Analyze", type="primary", use_container_width=True)

    if not ticker or not run:
        st.info("Enter a ticker and click **Analyze** to begin.")
        if "analysis_history" in st.session_state and st.session_state.analysis_history:
            st.markdown("---")
            st.markdown("### Recent Analyses")
            for h in reversed(st.session_state.analysis_history[-5:]):
                st.caption(
                    f"**{h['ticker']}** — {h['strategy']} | "
                    f"Conf: {h['confidence']:.0%} | IV Rank: {h['iv_rank']:.0f}%"
                )
        return

    # ── Fetch Data ──
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            data = download_universe([ticker], period="2y")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return

    if ticker not in data or data[ticker].empty:
        st.error(f"No data found for **{ticker}**.")
        return

    df = data[ticker]  # lowercase columns: close, high, low, open, volume

    price = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else price
    day_change = (price - prev_close) / prev_close * 100

    # ── Header Metrics ──
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"${price:.2f}", f"{day_change:+.2f}%")
    c2.metric("Day High", f"${float(df['high'].iloc[-1]):.2f}")
    c3.metric("Day Low", f"${float(df['low'].iloc[-1]):.2f}")
    c4.metric("Volume", f"{int(df['volume'].iloc[-1]):,}")
    c5.metric("Data", f"{len(df)} bars")

    # ── Regime Classification ──
    with st.spinner("Classifying regime..."):
        regime = classify_regime(ticker, df)

    if regime is None:
        st.error("Could not classify regime (insufficient data or error).")
        return

    st.markdown("---")
    st.subheader("📊 Technical Regime")

    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
    rc1.metric("Regime", regime.regime.value)
    rc2.metric("Direction", f"{'Bull' if regime.direction_score > 0 else 'Bear'} ({regime.direction_score:+.2f})")
    rc3.metric("ADX", f"{regime.adx:.1f}")
    rc4.metric("RSI", f"{regime.rsi:.1f}")
    rc5.metric("Trend Str.", f"{regime.trend_strength:.2f}")
    rc6.metric("Volume", regime.volume_trend)

    extra1, extra2, extra3, extra4 = st.columns(4)
    extra1.metric("ATR %", f"{regime.atr_pct:.2f}%")
    extra2.metric("EMA Align", regime.ema_alignment)
    extra3.metric("Support", f"${regime.support:.2f}")
    extra4.metric("Resistance", f"${regime.resistance:.2f}")

    if regime.bb_squeeze:
        st.success("📦 Bollinger Band Squeeze Active — potential breakout incoming")

    # ── TA Signals & Level Analysis ──────────────────────────────────────────
    try:
        from src.analysis.levels import analyze_levels
        from src.analysis.patterns import detect_patterns
        with st.spinner("Running TA analysis (levels + patterns)..."):
            levels = analyze_levels(ticker, df)
            patterns = detect_patterns(ticker, df)

        if levels is not None or patterns is not None:
            st.markdown("---")
            st.subheader("🔍 TA Signals & S/R Levels")

        if levels is not None:
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Nearest Support", f"${levels.nearest_support:.2f}" if levels.nearest_support else "N/A")
            lc2.metric("Nearest Resistance", f"${levels.nearest_resistance:.2f}" if levels.nearest_resistance else "N/A")
            lc3.metric("Dist to Support", f"{levels.distance_to_support_pct:.1f}%" if levels.distance_to_support_pct else "N/A")
            lc4.metric("Dist to Resistance", f"{levels.distance_to_resistance_pct:.1f}%" if levels.distance_to_resistance_pct else "N/A")

            flag_cols = st.columns(4)
            flag_cols[0].write(f"{'✅' if levels.near_support else '—'} Near Support")
            flag_cols[1].write(f"{'✅' if levels.near_resistance else '—'} Near Resistance")
            flag_cols[2].write(f"{'⬆' if levels.breakout_above else '—'} Breakout Above")
            flag_cols[3].write(f"{'⬇' if levels.breakdown_below else '—'} Breakdown Below")

            if levels.volume_profile_skew:
                st.caption(f"Volume profile skew: **{levels.volume_profile_skew}**")

        if patterns is not None:
            with st.expander("📡 Pattern Signals", expanded=True):
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Pattern Score", f"{patterns.pattern_score:+.2f}")
                pc2.metric(
                    "Divergence",
                    "Bullish" if patterns.bullish_divergence else ("Bearish" if patterns.bearish_divergence else "None")
                )
                pc3.metric("Squeeze Fired", patterns.squeeze_direction.upper() if patterns.squeeze_direction else "No")

                pat_flags = st.columns(4)
                pat_flags[0].write(f"{'✅' if patterns.bullish_divergence else '—'} Bull Divergence")
                pat_flags[1].write(f"{'✅' if patterns.bearish_divergence else '—'} Bear Divergence")
                pat_flags[2].write(f"{'✅' if patterns.inside_bar else '—'} Inside Bar")
                pat_flags[3].write(f"{'✅' if patterns.volume_climax else '—'} Vol Climax ({patterns.climax_direction})")

                vwap_flags = st.columns(2)
                vwap_flags[0].write(f"{'✅' if patterns.above_anchored_vwap else '—'} Above Anchored VWAP")
                vwap_flags[1].write(f"{'✅' if patterns.below_anchored_vwap else '—'} Below Anchored VWAP")

    except Exception as exc:
        st.caption(f"TA analysis unavailable: {exc}")

    # ── IV Analysis ──
    with st.spinner("Analyzing volatility..."):
        iv = analyze_iv(ticker, df)

    if iv is None:
        st.error("Could not compute IV analysis.")
        return

    st.markdown("---")
    st.subheader("📈 Volatility Profile")

    vc1, vc2, vc3, vc4, vc5 = st.columns(5)
    vc1.metric("IV Rank", f"{iv.iv_rank:.0f}%")
    vc2.metric("IV Percentile", f"{iv.iv_percentile:.0f}%")
    vc3.metric("Current IV", f"{iv.current_iv:.1f}%")
    vc4.metric("HV-20", f"{iv.hv_20:.1f}%")
    vc5.metric("IV/HV Ratio", f"{iv.iv_hv_ratio:.2f}")

    vc6, vc7, vc8 = st.columns(3)
    vc6.metric("HV-60", f"{iv.hv_60:.1f}%")
    vc7.metric("IV Trend", iv.iv_trend)
    vc8.metric("Skew", f"{iv.skew:.2f}")

    if iv.premium_action == "SELL":
        st.error(f"💰 **SELL PREMIUM** — IV Rank {iv.iv_rank:.0f}% (elevated). IV trend: {iv.iv_trend}")
    elif iv.premium_action == "BUY":
        st.success(f"🎯 **BUY PREMIUM** — IV Rank {iv.iv_rank:.0f}% (depressed). IV trend: {iv.iv_trend}")
    else:
        st.info(f"⚖️ **NEUTRAL** — IV Rank {iv.iv_rank:.0f}%. IV trend: {iv.iv_trend}")

    # ── Strategy Recommendation ──
    with st.spinner("Selecting strategy..."):
        strat = select_strategy(regime, iv)

    st.markdown("---")
    st.subheader("🎯 Recommended Strategy")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Strategy", strat.strategy.value.replace("_", " "))
    sc2.metric("Direction", strat.direction)
    sc3.metric("Confidence", f"{strat.confidence:.0%}")
    sc4.metric("Target DTE", f"{strat.target_dte}d")

    if strat.rationale:
        st.caption(f"Rationale: {strat.rationale}")

    st.info("⚠️ Full trade construction (exact strikes, credit, P&L) requires Polygon Options data. "
            "Use these signals to guide manual trade entry on your broker platform.")

    # ── Summary Table ──
    st.markdown("---")
    st.subheader("📋 Summary")
    summary = {
        "Ticker": ticker,
        "Price": f"${price:.2f}",
        "Regime": regime.regime.value,
        "ADX": f"{regime.adx:.1f}",
        "RSI": f"{regime.rsi:.1f}",
        "IV Rank": f"{iv.iv_rank:.0f}%",
        "IV/HV": f"{iv.iv_hv_ratio:.2f}",
        "Premium": iv.premium_action,
        "Strategy": strat.strategy.value.replace("_", " "),
        "Direction": strat.direction,
        "Confidence": f"{strat.confidence:.0%}",
        "DTE": f"{strat.target_dte}d",
    }
    st.table(pd.DataFrame([summary]))

    # ── Charts ──
    st.markdown("---")
    st.subheader("📉 Charts")

    tab1, tab2 = st.tabs(["Price & Technicals", "Volatility Cone"])
    with tab1:
        fig_price = _build_price_chart(df, ticker, regime.regime.value)
        st.plotly_chart(fig_price, use_container_width=True)
    with tab2:
        fig_hv = _build_hv_chart(df)
        st.plotly_chart(fig_hv, use_container_width=True)

    # ── Save to session history ──
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    st.session_state.analysis_history.append({
        "ticker": ticker,
        "price": price,
        "regime": regime.regime.value,
        "iv_rank": iv.iv_rank,
        "strategy": strat.strategy.value,
        "confidence": strat.confidence,
        "direction": strat.direction,
        "date": date.today().isoformat(),
    })
