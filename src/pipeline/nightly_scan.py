"""
src/pipeline/nightly_scan.py
============================
Main orchestrator: runs after market close.

Pipeline:
  1. Update OHLCV data for the full universe
  2. Pre-filter (price + volume)
  3. Classify technical regimes
  4. Compute market context (VIX proxy, breadth)
  5. Analyze implied volatility
  6. Select strategies per stock
  7. Event filter (earnings)
  8. Construct option trades for top candidates
  9. Rank by composite score
  10. Save signal JSON and return top picks

Usage:
    python -m src.pipeline.nightly_scan

    from src.pipeline.nightly_scan import run_nightly_scan
    signal = run_nightly_scan()
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.pipeline.outcome_tracker import record_entry

from config.settings import (
    SIGNALS_DIR, MAX_POSITIONS,
    MIN_STOCK_PRICE, MIN_AVG_VOLUME, LOG_LEVEL,
    VIX_CAUTION_LEVEL, VIX_DEFENSIVE_LEVEL, VIX_LIQUIDATION_LEVEL,
    MAX_PER_SECTOR, SPY_DIRECTIONAL_GATE_PCT, IBKR_ENABLED,
)
from config.universe import get_universe, get_sector, get_tradeable_universe

from src.data.stock_fetcher import update_universe
from src.data.options_fetcher import fetch_options_chain, filter_liquid_options
from src.data.market_context import get_market_context

from src.analysis.technical import classify_universe, Regime, get_regime_summary
from src.analysis.volatility import analyze_iv
from src.analysis.relative_strength import rank_universe_rs
from src.analysis.levels import analyze_universe_levels
from src.analysis.patterns import detect_universe_patterns

from src.strategy.selector import select_strategy, StrategyType
from src.strategy.credit_spread import construct_bull_put_spread, construct_bear_call_spread
from src.strategy.bull_call_spread import construct_bull_call_spread
from src.strategy.bear_put_spread import construct_bear_put_spread
from src.strategy.iron_condor import construct_iron_condor
from src.strategy.butterfly import construct_long_butterfly
from src.strategy.long_call import construct_long_call
from src.strategy.long_put import construct_long_put

from src.risk.event_filter import filter_safe_tickers

logger = logging.getLogger(__name__)

# ── Phase 5: ML Confidence Blending (optional) ────────────────────────────────
import os
import pickle
from pathlib import Path as _Path

_ML_MODEL_PATH = _Path(os.getenv("ML_MODEL_PATH", "models/lgb_win_predictor.pkl"))
_ML_BLEND_HEURISTIC = float(os.getenv("ML_BLEND_HEURISTIC", "0.60"))  # 60% heuristic
_ML_BLEND_ML = float(os.getenv("ML_BLEND_ML", "0.40"))                # 40% ML
_ml_model = None  # lazy-loaded

def _load_ml_model():
    global _ml_model
    if _ml_model is not None:
        return _ml_model
    if _ML_MODEL_PATH.exists():
        try:
            with open(_ML_MODEL_PATH, "rb") as _f:
                _ml_model = pickle.load(_f)
            logger.info(f"ML model loaded from {_ML_MODEL_PATH}")
        except Exception as _exc:
            logger.warning(f"ML model load failed: {_exc}")
            _ml_model = None
    return _ml_model


def _build_ml_feature_vector(rec, regime, iv_analysis, rs_analysis=None) -> list:
    """
    Build a numeric feature vector for the ML win predictor.
    Must match the feature order used when the model was trained
    (see src/models/features.py NUMERIC_FEATURES + BINARY_FEATURES).
    """
    rs_rank = rs_analysis.rank if rs_analysis and hasattr(rs_analysis, "rank") else 50.0
    ta_sigs = getattr(regime, "ta_signals", {}) or {}
    return [
        # numeric
        iv_analysis.iv_rank if iv_analysis else 50.0,
        iv_analysis.iv_hv_ratio if iv_analysis else 1.0,
        regime.adx,
        regime.rsi,
        regime.trend_strength,
        regime.direction_score,
        float(rs_rank),
        float(rec.target_dte),
        0.0,   # spread_width (N/A for long option)
        0.0,   # short_delta
        float(rec.confidence * 100),  # prob_profit proxy
        rec.confidence,
        0.0,   # options_flow_score
        1.0,   # put_call_volume_ratio
        1.0,   # volume_pace
        0.0,   # live_iv_at_entry
        0.0,   # iv_skew_at_entry
        ta_sigs.get("pattern_score", ta_sigs.get("ta_pattern_score", 0.0)) or 0.0,
        0.0,   # entry_theta_rate
        iv_analysis.iv_rank if iv_analysis else 50.0,  # entry_iv_rank
        # binary
        float(ta_sigs.get("breakout_above", False)),
        float(ta_sigs.get("bullish_divergence", False) or ta_sigs.get("bearish_divergence", False)),
        float(rec.strategy.value in ("LONG_CALL", "LONG_PUT")),
    ]


def _apply_ml_confidence(rec, regime, iv_analysis, rs_analysis=None) -> None:
    """
    If lgb_win_predictor.pkl exists, blend ML prediction with heuristic confidence.
    rec.confidence = 0.60 × heuristic + 0.40 × ML_probability
    """
    model = _load_ml_model()
    if model is None:
        return  # no model → use heuristic confidence as-is

    try:
        fv = _build_ml_feature_vector(rec, regime, iv_analysis, rs_analysis)
        import numpy as np
        X = np.array([fv], dtype=float)
        ml_prob = float(model.predict_proba(X)[0][1])  # prob of win
        heuristic = rec.confidence
        blended = _ML_BLEND_HEURISTIC * heuristic + _ML_BLEND_ML * ml_prob
        blended = round(min(max(blended, 0.01), 0.99), 4)
        logger.debug(
            f"ML blend {rec.ticker}: heuristic={heuristic:.3f} ml={ml_prob:.3f} "
            f"→ blended={blended:.3f}"
        )
        rec.confidence = blended
    except Exception as exc:
        logger.debug(f"ML confidence blend failed for {rec.ticker}: {exc}")




# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_nightly_scan(
    universe_override: list[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Full nightly scan pipeline. Returns a signal dictionary.

    Args:
        universe_override: Optional list of tickers to scan (overrides config)
        dry_run: If True, skip options chain fetching (faster, for testing)

    Returns:
        Signal dict with top_picks, metadata, and full recommendations list.
    """
    start_time = time.time()
    scan_date = datetime.now(timezone.utc).date().isoformat()
    logger.info("=" * 60)
    logger.info(f"OPTIONS ALGO — NIGHTLY SCAN  {scan_date}")
    logger.info("=" * 60)

    # ── Step 1: Download / Update Data ────────────────────────────────────────
    tickers = universe_override or get_universe()
    logger.info(f"Step 1: Updating data for {len(tickers)} tickers")
    data = update_universe(tickers)
    logger.info(f"  Loaded: {len(data)} tickers")

    # ── Step 2: Pre-filter (price + volume) ───────────────────────────────────
    qualified: dict = {}
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        price = float(df["close"].iloc[-1])
        avg_vol = float(df["volume"].tail(20).mean())
        if price >= MIN_STOCK_PRICE and avg_vol >= MIN_AVG_VOLUME:
            qualified[ticker] = df

    logger.info(f"Step 2: {len(qualified)} tickers pass price/volume filter")

    # ── Step 3: Market Context ─────────────────────────────────────────────────
    logger.info("Step 3: Computing market context")
    market_ctx = get_market_context(qualified)
    logger.info(
        f"  Market: {market_ctx.market_regime} | "
        f"VIX: {market_ctx.vix_level} (tier: {market_ctx.vix_tier}) | "
        f"SPY: {market_ctx.spy_trend} | "
        f"Breadth: {market_ctx.breadth_score:.0%}"
    )
    if market_ctx.vix_spike:
        logger.warning(
            f"  VIX SPIKE: {market_ctx.vix_level:.1f} vs 5d avg {market_ctx.vix_5d_avg:.1f}"
        )

    # ── VIX Circuit Breaker ────────────────────────────────────────────────────
    if market_ctx.vix_tier == "LIQUIDATION":
        logger.warning(
            f"VIX LIQUIDATION mode ({market_ctx.vix_level:.1f} ≥ {VIX_LIQUIDATION_LEVEL}) "
            "— halting all new trade generation"
        )
        return _build_empty_signal(
            scan_date, start_time, market_ctx,
            reason=f"VIX liquidation ({market_ctx.vix_level:.1f})",
            tickers=tickers, qualified=qualified,
        )

    if market_ctx.vix_tier == "DEFENSIVE":
        logger.warning(
            f"VIX DEFENSIVE mode ({market_ctx.vix_level:.1f} ≥ {VIX_DEFENSIVE_LEVEL}) "
            "— neutral/credit-only strategies"
        )

    # Gate: don't generate picks in crash conditions
    if market_ctx.market_regime == "CRASH":
        logger.warning("CRASH conditions detected. Generating bear plays only.")

    # ── Step 3b: Beta Map ─────────────────────────────────────────────────────
    # Compute rough 60-day beta vs SPY for every qualified ticker.
    # Used for net-exposure logging and trade metadata enrichment.
    beta_map: dict[str, float] = {}
    spy_close = qualified.get("SPY", pd.DataFrame()).get("close")
    if spy_close is not None and len(spy_close) >= 60:
        spy_ret = spy_close.pct_change().tail(60).dropna()
        for t, df in qualified.items():
            if t == "SPY" or df is None or len(df) < 60:
                beta_map[t] = 1.0
                continue
            try:
                stock_ret = df["close"].pct_change().tail(60).dropna()
                aligned = stock_ret.align(spy_ret, join="inner")
                cov = aligned[0].cov(aligned[1])
                var = aligned[1].var()
                beta_map[t] = round(cov / var, 2) if var > 0 else 1.0
            except Exception:
                beta_map[t] = 1.0
    else:
        beta_map = {t: 1.0 for t in qualified}

    # ── Step 4: Classify Regimes ──────────────────────────────────────────────
    logger.info("Step 4: Classifying regimes")
    regimes = classify_universe(qualified)
    regime_map = {r.ticker: r for r in regimes}
    regime_dist = get_regime_summary(regimes)
    logger.info(f"  Distribution: {regime_dist}")

    # ── Step 4b: TA Signals (levels + patterns) ────────────────────────────────
    logger.info("Step 4b: Computing TA signals (S/R levels + pattern detection)")
    try:
        level_map = analyze_universe_levels(qualified)
        pattern_map = detect_universe_patterns(qualified)

        ta_enriched = 0
        for ticker, regime in regime_map.items():
            signals: dict = {}

            # Attach level analysis
            la = level_map.get(ticker)
            if la is not None:
                signals.update({
                    "breakout_above": la.breakout_above,
                    "breakdown_below": la.breakdown_below,
                    "near_support": la.near_support,
                    "near_resistance": la.near_resistance,
                    "support_distance_pct": la.support_distance_pct,
                    "resistance_distance_pct": la.resistance_distance_pct,
                    "volume_profile_skew": la.volume_profile_skew,
                    "high_volume_node": la.high_volume_node,
                })

            # Attach pattern signals
            ps = pattern_map.get(ticker)
            if ps is not None:
                signals.update({
                    "bullish_divergence": ps.bullish_divergence,
                    "bearish_divergence": ps.bearish_divergence,
                    "divergence_strength": ps.divergence_strength,
                    "squeeze_fired": ps.squeeze_fired,
                    "squeeze_direction": ps.squeeze_direction,
                    "volume_climax": ps.volume_climax,
                    "climax_direction": ps.climax_direction,
                    "inside_bar": ps.inside_bar,
                    "above_anchored_vwap": ps.above_anchored_vwap,
                    "below_anchored_vwap": ps.below_anchored_vwap,
                    "pattern_score": ps.pattern_score,
                })

            if signals:
                regime.ta_signals = signals
                ta_enriched += 1

        logger.info(
            f"  TA signals: {ta_enriched} tickers enriched "
            f"({len(level_map)} levels, {len(pattern_map)} patterns)"
        )
    except Exception as exc:
        logger.warning(f"  Step 4b TA signals failed (non-fatal): {exc}")

    # ── Step 5: Relative Strength ──────────────────────────────────────────────
    logger.info("Step 5: Computing relative strength")
    rs_map = rank_universe_rs(qualified)

    # ── Step 6: IV Analysis ───────────────────────────────────────────────────
    logger.info("Step 6: Analyzing implied volatility")
    iv_map = {}
    for ticker in qualified:
        iv = analyze_iv(ticker, qualified[ticker])
        if iv is not None:
            iv_map[ticker] = iv
    logger.info(f"  IV analyzed: {len(iv_map)} tickers")

    # ── Step 6b: IBKR Real-time Enrichment ───────────────────────────────────
    logger.info("Step 6b: IBKR real-time enrichment (options flow, live IV)")
    rt_enrichment: dict[str, dict] = {}
    if not dry_run and IBKR_ENABLED:
        try:
            from src.data.ibkr_client import connect_ibkr, disconnect_ibkr
            from src.data.ibkr_realtime import fetch_realtime_enrichment

            ib_rt = connect_ibkr()
            if ib_rt:
                # Only enrich tickers that passed IV analysis (top candidates)
                tickers_to_enrich = [t for t in iv_map][:30]  # cap at 30 (IBKR pacing)
                for t in tickers_to_enrich:
                    stock_df = qualified.get(t)
                    rt = fetch_realtime_enrichment(ib_rt, t, stock_df=stock_df)
                    if rt:
                        rt_enrichment[t] = rt
                disconnect_ibkr(ib_rt)
                logger.info(f"  Enriched {len(rt_enrichment)} tickers with IBKR real-time data")
            else:
                logger.info("  IBKR not available — skipping real-time enrichment")
        except Exception as exc:
            logger.warning(f"  IBKR real-time enrichment failed (non-fatal): {exc}")

    # ── Step 7: Strategy Selection ────────────────────────────────────────────
    logger.info("Step 7: Selecting strategies")
    recommendations = []
    # Pass SPY 5-day return to the strategy selector so it can block directional
    # trades that fight the broad market tape (V2: SPY gate)
    spy_ret_5d = market_ctx.spy_return_5d
    logger.info(f"  SPY 5d return: {spy_ret_5d:+.2f}% (gate: ±{SPY_DIRECTIONAL_GATE_PCT:.1f}%)")

    for ticker in qualified:
        if ticker not in regime_map or ticker not in iv_map:
            continue
        rec = select_strategy(
            regime_map[ticker],
            iv_map[ticker],
            spy_return_5d=spy_ret_5d,   # V2: market context gate
        )
        if rec.strategy != StrategyType.SKIP:
            # Boost confidence if RS is strong/weak and aligned with direction
            rs = rs_map.get(ticker)
            if rs:
                _adjust_confidence_for_rs(rec, rs)

            # Adjust confidence based on real-time options flow
            rt = rt_enrichment.get(ticker, {})
            flow_score = rt.get("flow_score", 0)
            dominant_side = rt.get("dominant_side", "NEUTRAL")
            if flow_score > 60:  # Unusual activity threshold
                if (dominant_side == "CALLS" and rec.direction == "BULLISH") or \
                   (dominant_side == "PUTS" and rec.direction == "BEARISH"):
                    rec.confidence = min(rec.confidence * 1.15, 0.95)  # boost aligned flow
                elif (dominant_side == "CALLS" and rec.direction == "BEARISH") or \
                     (dominant_side == "PUTS" and rec.direction == "BULLISH"):
                    rec.confidence = rec.confidence * 0.85  # penalize counter-flow

            recommendations.append(rec)

    # ── VIX Tier Overlay ──────────────────────────────────────────────────────
    # DEFENSIVE mode: strip all directional recs, keep only NEUTRAL.
    # CAUTION mode: keep credit strategies and NEUTRAL only.
    if market_ctx.vix_tier == "DEFENSIVE":
        before = len(recommendations)
        recommendations = [r for r in recommendations if r.direction == "NEUTRAL"]
        logger.info(
            f"  DEFENSIVE filter: {before} → {len(recommendations)} neutral-only recs"
        )
    elif market_ctx.vix_tier == "CAUTION":
        before = len(recommendations)
        recommendations = [
            r for r in recommendations
            if r.direction == "NEUTRAL" or getattr(r, "risk_reward", "") == "CREDIT"
        ]
        logger.info(
            f"  CAUTION filter: {before} → {len(recommendations)} credit/neutral recs"
        )

    recommendations.sort(key=lambda r: r.confidence, reverse=True)
    logger.info(f"  {len(recommendations)} strategy recommendations")

    # ── Step 8: Event Filter ──────────────────────────────────────────────────
    logger.info("Step 8: Event filter (earnings)")
    top_for_filter = recommendations[:40]
    dte_map = {r.ticker: r.target_dte for r in top_for_filter}
    safety = filter_safe_tickers(
        [r.ticker for r in top_for_filter],
        dte_map,
        max_per_batch=40,
    )
    safe_recs = [r for r in top_for_filter if safety.get(r.ticker, None) and safety[r.ticker].safe]
    logger.info(f"  {len(safe_recs)} pass event filter")

    # ── Step 9: Construct Trades ──────────────────────────────────────────────
    logger.info(f"Step 9: Constructing trades for top {min(len(safe_recs), 15)} candidates")
    trades = []

    for rec in safe_recs[:15]:
        ticker = rec.ticker
        price = float(qualified[ticker]["close"].iloc[-1])

        if dry_run:
            # Skip real options fetching in dry_run mode
            trade_stub = _build_trade_stub(rec, qualified[ticker], regime_map, iv_map, rs_map, market_ctx, beta_map)
            _rt = rt_enrichment.get(ticker, {})
            trade_stub["context"]["options_flow"] = {
                "flow_score": _rt.get("flow_score", 0),
                "dominant_side": _rt.get("dominant_side", "NEUTRAL"),
                "put_call_volume_ratio": _rt.get("put_call_volume_ratio", 1.0),
                "volume_pace": _rt.get("volume_pace", 1.0),
                "live_iv": _rt.get("iv_pct"),
            }
            trades.append(trade_stub)
            continue

        # Fetch and filter options chain
        chain = fetch_options_chain(ticker, dte_min=7, dte_max=65)
        chain = filter_liquid_options(chain)

        if chain.empty:
            logger.debug(f"  {ticker}: no liquid options")
            continue

        trade_obj = _construct_trade(rec, ticker, price, chain)
        if trade_obj is None:
            logger.debug(f"  {ticker}: trade construction failed")
            continue

        trade_dict = _build_trade_dict(rec, trade_obj, qualified, regime_map, iv_map, rs_map, ticker, price, market_ctx, beta_map)
        _rt = rt_enrichment.get(ticker, {})
        trade_dict["context"]["options_flow"] = {
            "flow_score": _rt.get("flow_score", 0),
            "dominant_side": _rt.get("dominant_side", "NEUTRAL"),
            "put_call_volume_ratio": _rt.get("put_call_volume_ratio", 1.0),
            "volume_pace": _rt.get("volume_pace", 1.0),
            "live_iv": _rt.get("iv_pct"),
        }
        trades.append(trade_dict)

        time.sleep(0.3)  # Rate limiting between options fetches

    logger.info(f"  {len(trades)} trades constructed")

    # ── Step 9b: Sector Concentration Cap ────────────────────────────────────
    # Never put more than MAX_PER_SECTOR picks from the same GICS sector.
    sector_counts: dict[str, int] = {}
    sector_filtered: list[dict] = []
    for t in trades:
        sector = t["context"].get("sector", "Unknown")
        if sector_counts.get(sector, 0) < MAX_PER_SECTOR:
            sector_filtered.append(t)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        else:
            tk = t["recommendation"]["ticker"]
            logger.info(f"  Sector cap: skipped {tk} ({sector}) — {MAX_PER_SECTOR}/{MAX_PER_SECTOR} slots used")
    trades = sector_filtered

    # ── Step 10: Rank and Finalize ────────────────────────────────────────────
    logger.info("Step 10: Ranking picks")
    trades = _rank_trades(trades)
    top_picks = trades[:MAX_POSITIONS]

    # Net beta-weighted directional exposure (informational only)
    net_beta = sum(
        beta_map.get(t["recommendation"]["ticker"], 1.0)
        * (1 if t["recommendation"]["direction"] == "BULLISH" else
           -1 if t["recommendation"]["direction"] == "BEARISH" else 0)
        for t in top_picks
    )
    logger.info(f"  Net beta-weighted exposure: {net_beta:+.2f}")

    elapsed = round(time.time() - start_time, 1)

    signal = {
        "scan_date": scan_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "universe_size": len(tickers),
        "qualified": len(qualified),
        "regimes_classified": len(regimes),
        "recommendations": len(recommendations),
        "after_event_filter": len(safe_recs),
        "trades_constructed": len(trades),
        "top_picks": top_picks,
        "regime_distribution": regime_dist,
        "market_context": {
            "market_regime": market_ctx.market_regime,
            "vix_level": market_ctx.vix_level,
            "vix_regime": market_ctx.vix_regime,
            "vix_5d_avg": market_ctx.vix_5d_avg,
            "vix_spike": market_ctx.vix_spike,
            "vix_tier": market_ctx.vix_tier,
            "spy_trend": market_ctx.spy_trend,
            "spy_return_5d": market_ctx.spy_return_5d,
            "spy_return_20d": market_ctx.spy_return_20d,
            "breadth_score": market_ctx.breadth_score,
            "sector_leaders": market_ctx.sector_leaders,
            "sector_laggards": market_ctx.sector_laggards,
            "notes": market_ctx.notes,
        },
        "all_recommendations": [
            {
                "ticker": r.ticker,
                "strategy": r.strategy.value,
                "confidence": r.confidence,
                "regime": r.regime,
                "iv_regime": r.iv_regime,
                "direction": r.direction,
            }
            for r in recommendations[:25]
        ],
    }

    # ── Save Signal ───────────────────────────────────────────────────────────
    path = SIGNALS_DIR / f"options_signal_{scan_date}.json"
    latest = SIGNALS_DIR / "options_signal_latest.json"
    for p in [path, latest]:
        try:
            with open(p, "w") as f:
                json.dump(signal, f, indent=2, default=str)
        except Exception as exc:
            logger.error(f"Failed to save signal to {p}: {exc}")

    logger.info(f"=== SCAN COMPLETE: {len(top_picks)} picks in {elapsed}s ===")
    return signal


# ─── Empty Signal (VIX Halt) ─────────────────────────────────────────────────

def _build_empty_signal(
    scan_date: str,
    start_time: float,
    market_ctx,
    reason: str,
    tickers: list,
    qualified: dict,
) -> dict:
    """
    Return a zero-picks signal when a circuit breaker fires (e.g. VIX liquidation).
    Preserves full market context so operators know why no trades were generated.
    """
    import time as _time
    elapsed = round(_time.time() - start_time, 1)
    return {
        "scan_date": scan_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "universe_size": len(tickers),
        "qualified": len(qualified),
        "regimes_classified": 0,
        "recommendations": 0,
        "after_event_filter": 0,
        "trades_constructed": 0,
        "top_picks": [],
        "circuit_breaker": reason,
        "regime_distribution": {},
        "market_context": {
            "market_regime": market_ctx.market_regime,
            "vix_level": market_ctx.vix_level,
            "vix_regime": market_ctx.vix_regime,
            "vix_5d_avg": market_ctx.vix_5d_avg,
            "vix_spike": market_ctx.vix_spike,
            "vix_tier": market_ctx.vix_tier,
            "spy_trend": market_ctx.spy_trend,
            "spy_return_5d": market_ctx.spy_return_5d,
            "spy_return_20d": market_ctx.spy_return_20d,
            "breadth_score": market_ctx.breadth_score,
            "sector_leaders": market_ctx.sector_leaders,
            "sector_laggards": market_ctx.sector_laggards,
            "notes": market_ctx.notes,
        },
        "all_recommendations": [],
    }


# ─── Trade Construction Dispatch ─────────────────────────────────────────────

def _construct_trade(rec, ticker, price, chain):
    """Route to the appropriate strategy constructor."""
    strategy = rec.strategy
    dte = rec.target_dte

    from config.settings import DEFAULT_SPREAD_WIDTH, IC_WING_DELTA

    if strategy == StrategyType.BULL_PUT_SPREAD:
        return construct_bull_put_spread(ticker, price, chain, target_dte=dte)

    elif strategy == StrategyType.BEAR_CALL_SPREAD:
        return construct_bear_call_spread(ticker, price, chain, target_dte=dte)

    elif strategy == StrategyType.BULL_CALL_SPREAD:
        return construct_bull_call_spread(ticker, price, chain, target_dte=dte)

    elif strategy == StrategyType.BEAR_PUT_SPREAD:
        return construct_bear_put_spread(ticker, price, chain, target_dte=dte)

    elif strategy == StrategyType.IRON_CONDOR:
        return construct_iron_condor(ticker, price, chain,
                                     target_dte=dte, wing_delta=IC_WING_DELTA)

    elif strategy == StrategyType.LONG_BUTTERFLY:
        return construct_long_butterfly(ticker, price, chain, target_dte=dte)

    elif strategy == StrategyType.LONG_CALL:
        from config.settings import DEFAULT_DTE_LONG_OPTION
        return construct_long_call(ticker, price, chain,
                                   target_dte=rec.target_dte or DEFAULT_DTE_LONG_OPTION)

    elif strategy == StrategyType.LONG_PUT:
        from config.settings import DEFAULT_DTE_LONG_OPTION
        return construct_long_put(ticker, price, chain,
                                  target_dte=rec.target_dte or DEFAULT_DTE_LONG_OPTION)

    return None


def _build_trade_dict(rec, trade_obj, data, regime_map, iv_map, rs_map, ticker, price,
                      market_ctx=None, beta_map=None):
    """Convert trade object + recommendation into a signal dict."""
    regime = regime_map[ticker]
    iv = iv_map[ticker]
    rs = rs_map.get(ticker)

    # Serialize trade object (varies by type)
    trade_data = vars(trade_obj)
    # Remove non-serialisable nested objects (wing CreditSpreads in IC)
    for k, v in list(trade_data.items()):
        if hasattr(v, "__dict__"):
            trade_data[k] = vars(v)

    # Market snapshot for ML features / audit trail
    market_snapshot: dict = {}
    if market_ctx is not None:
        market_snapshot = {
            "vix": market_ctx.vix_level,
            "vix_tier": market_ctx.vix_tier,
            "vix_spike": market_ctx.vix_spike,
            "spy_5d_return": market_ctx.spy_return_5d,
            "breadth": market_ctx.breadth_score,
            "market_regime": market_ctx.market_regime,
        }

    beta = (beta_map or {}).get(ticker, 1.0)

    return {
        "recommendation": {
            "ticker": ticker,
            "strategy": rec.strategy.value,
            "direction": rec.direction,
            "regime": rec.regime,
            "iv_regime": rec.iv_regime,
            "confidence": rec.confidence,
            "rationale": rec.rationale,
            "target_dte": rec.target_dte,
        },
        "trade": trade_data,
        "context": {
            "price": round(price, 2),
            "beta": beta,
            "sector": get_sector(ticker),
            "market_snapshot": market_snapshot,
            "regime_detail": {
                "adx": regime.adx,
                "rsi": regime.rsi,
                "trend_strength": regime.trend_strength,
                "direction_score": regime.direction_score,
                "ema_alignment": regime.ema_alignment,
                "bb_squeeze": regime.bb_squeeze,
                "support": regime.support,
                "resistance": regime.resistance,
                "atr": regime.atr,
                "atr_pct": regime.atr_pct,
                "volume_trend": regime.volume_trend,
                "roc_3d": regime.roc_3d,
                "atr_move_5d": regime.atr_move_5d,
            },
            "iv_detail": {
                "iv_rank": iv.iv_rank,
                "iv_percentile": iv.iv_percentile,
                "current_iv": iv.current_iv,
                "hv_20": iv.hv_20,
                "iv_hv_ratio": iv.iv_hv_ratio,
                "iv_trend": iv.iv_trend,
                "premium_action": iv.premium_action,
                "iv_rv_spread": iv.iv_rv_spread,
                "premium_rich": iv.premium_rich,
            },
            "rs_detail": {
                "rs_vs_spy": rs.rs_vs_spy if rs else None,
                "rs_rank": rs.rs_rank if rs else None,
                "rs_trend": rs.rs_trend if rs else None,
            },
            # ── V3: TA signals from levels + patterns ────────────────────────────────
            "ta_signals": getattr(regime, "ta_signals", {}),
            # ── V3: Long option metadata ─────────────────────────────────────────
            "is_long_option": rec.strategy.value in ("LONG_CALL", "LONG_PUT"),
        },
    }


def _build_trade_stub(rec, df, regime_map, iv_map, rs_map,
                      market_ctx=None, beta_map=None):
    """Build a placeholder trade dict for dry_run mode."""
    ticker = rec.ticker
    price = float(df["close"].iloc[-1]) if not df.empty else 0

    market_snapshot: dict = {}
    if market_ctx is not None:
        market_snapshot = {
            "vix": market_ctx.vix_level,
            "vix_tier": market_ctx.vix_tier,
            "vix_spike": market_ctx.vix_spike,
            "spy_5d_return": market_ctx.spy_return_5d,
            "breadth": market_ctx.breadth_score,
            "market_regime": market_ctx.market_regime,
        }

    beta = (beta_map or {}).get(ticker, 1.0)

    return {
        "recommendation": {
            "ticker": ticker,
            "strategy": rec.strategy.value,
            "direction": rec.direction,
            "regime": rec.regime,
            "iv_regime": rec.iv_regime,
            "confidence": rec.confidence,
            "rationale": rec.rationale,
            "target_dte": rec.target_dte,
        },
        "trade": {"dry_run": True, "price": price},
        "context": {
            "price": round(price, 2),
            "beta": beta,
            "sector": get_sector(ticker),
            "market_snapshot": market_snapshot,
            "regime_detail": vars(regime_map[ticker]) if ticker in regime_map else {},
            "iv_detail": vars(iv_map[ticker]) if ticker in iv_map else {},
            "rs_detail": {},
        },
    }


def _rank_trades(trades: list[dict]) -> list[dict]:
    """
    Rank trades by composite score and assign priority.

    V2: Enforces directional balance — no more than MAX_SAME_DIRECTION_PCT
    of MAX_POSITIONS slots may be filled by the same direction (BULLISH/BEARISH).
    Neutral (iron condors, butterflies) are never limited.
    """
    from config.settings import MAX_SAME_DIRECTION_PCT

    for t in trades:
        trade = t.get("trade", {})
        rec = t.get("recommendation", {})

        conf = float(rec.get("confidence", 0.5))
        pop = float(trade.get("prob_profit", 60))
        rr = float(trade.get("risk_reward_ratio", 3.0))
        ev = float(trade.get("ev", 0.0))

        # Composite = confidence × (PoP/100) × (1/RR) × ev_bonus
        ev_bonus = 1.0 + max(ev / 100, 0)
        composite = conf * (pop / 100) * (1 / max(rr, 0.5)) * ev_bonus

        t["composite_score"] = round(composite, 4)

    trades.sort(key=lambda t: t.get("composite_score", 0), reverse=True)

    # ── V2: Directional balance gate ────────────────────────────────────────
    # Cap the number of same-direction trades at MAX_SAME_DIRECTION_PCT% of
    # MAX_POSITIONS so we never run 4 bear spreads + 1 bull spread again.
    max_same_dir = max(2, int(MAX_POSITIONS * MAX_SAME_DIRECTION_PCT / 100))
    balanced = []
    direction_counts: dict[str, int] = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}

    for t in trades:
        direction = t.get("recommendation", {}).get("direction", "NEUTRAL")
        current_count = direction_counts.get(direction, 0)

        if direction == "NEUTRAL" or current_count < max_same_dir:
            balanced.append(t)
            direction_counts[direction] = current_count + 1
        else:
            ticker = t.get("recommendation", {}).get("ticker", "?")
            logger.info(
                f"  Directional balance: skipped {ticker} ({direction}) — "
                f"{current_count}/{max_same_dir} {direction} slots filled"
            )

    # ── V3: Long-option allocation cap ─────────────────────────────────────
    # No more than LONG_OPTION_MAX_ALLOCATION_PCT% of MAX_POSITIONS may be
    # directional long options (they carry 100% premium-loss risk vs. capped
    # spread risk, so we keep them to a minority of the book).
    from config.settings import LONG_OPTION_MAX_ALLOCATION_PCT
    max_long_option_slots = max(1, int(MAX_POSITIONS * LONG_OPTION_MAX_ALLOCATION_PCT / 100))
    long_option_count = 0
    capped: list[dict] = []
    for t in balanced:
        strategy = t.get("recommendation", {}).get("strategy", "")
        if strategy in ("LONG_CALL", "LONG_PUT"):
            if long_option_count >= max_long_option_slots:
                ticker_name = t.get("recommendation", {}).get("ticker", "?")
                logger.info(
                    f"  Long-option cap: skipped {ticker_name} ({strategy}) — "
                    f"{long_option_count}/{max_long_option_slots} long-option slots filled"
                )
                continue
            long_option_count += 1
        capped.append(t)
    balanced = capped

    for i, t in enumerate(balanced):
        t["priority"] = i + 1

    return balanced


def _adjust_confidence_for_rs(rec, rs) -> None:
    """Boost/reduce confidence based on relative strength alignment."""
    if rec.direction == "BULLISH" and rs.outperforming_spy and rs.rs_trend == "IMPROVING":
        rec.confidence = min(rec.confidence + 0.05, 1.0)
    elif rec.direction == "BEARISH" and not rs.outperforming_spy and rs.rs_trend == "WEAKENING":
        rec.confidence = min(rec.confidence + 0.05, 1.0)
    elif rec.direction == "BULLISH" and not rs.outperforming_spy:
        rec.confidence = max(rec.confidence - 0.05, 0.0)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    dry = "--dry-run" in sys.argv
    result = run_nightly_scan(dry_run=dry)

    print("\n" + "=" * 60)
    print(f"TOP {len(result['top_picks'])} PICKS")
    print("=" * 60)

    for pick in result["top_picks"]:
        rec = pick["recommendation"]
        trade = pick["trade"]
        ctx = pick["context"]
        print(
            f"#{pick['priority']} {rec['ticker']:6s} | {rec['strategy']:20s} | "
            f"Conf={rec['confidence']:.0%} | {rec['direction']:7s} | "
            f"IV rank={ctx['iv_detail'].get('iv_rank', 0):.0f}% | "
            f"Price=${ctx['price']:.2f}"
        )
        if not trade.get("dry_run"):
            credit = trade.get("net_credit") or trade.get("total_credit")
            max_r = trade.get("max_risk")
            pop = trade.get("prob_profit")
            if credit:
                print(f"   Credit: ${credit} | Max Risk: ${max_r} | PoP: {pop}%")
        print(f"   {rec['rationale']}")

        # Log paper trade
        if not trade.get("dry_run"):
            try:
                tid = record_entry(
                    ticker=rec["ticker"],
                    recommendation=rec,
                    trade=trade,
                    context=ctx,
                )
                print(f"   📝 Trade logged: {tid}")
            except Exception as e:
                print(f"   ⚠️ Trade logging failed: {e}")
    print()
