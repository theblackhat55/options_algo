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

from src.pipeline.outcome_tracker import record_entry

from config.settings import (
    SIGNALS_DIR, MAX_POSITIONS,
    MIN_STOCK_PRICE, MIN_AVG_VOLUME, LOG_LEVEL,
)
from config.universe import get_universe, get_sector, get_tradeable_universe

from src.data.stock_fetcher import update_universe
from src.data.options_fetcher import fetch_options_chain, filter_liquid_options
from src.data.market_context import get_market_context

from src.analysis.technical import classify_universe, Regime, get_regime_summary
from src.analysis.volatility import analyze_iv
from src.analysis.relative_strength import rank_universe_rs

from src.strategy.selector import select_strategy, StrategyType
from src.strategy.credit_spread import construct_bull_put_spread, construct_bear_call_spread
from src.strategy.bull_call_spread import construct_bull_call_spread
from src.strategy.bear_put_spread import construct_bear_put_spread
from src.strategy.iron_condor import construct_iron_condor
from src.strategy.butterfly import construct_long_butterfly

from src.risk.event_filter import filter_safe_tickers

logger = logging.getLogger(__name__)


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
        f"VIX proxy: {market_ctx.vix_level} | "
        f"SPY: {market_ctx.spy_trend} | "
        f"Breadth: {market_ctx.breadth_score:.0%}"
    )

    # Gate: don't generate picks in crash conditions
    if market_ctx.market_regime == "CRASH":
        logger.warning("CRASH conditions detected. Generating bear plays only.")

    # ── Step 4: Classify Regimes ──────────────────────────────────────────────
    logger.info("Step 4: Classifying regimes")
    regimes = classify_universe(qualified)
    regime_map = {r.ticker: r for r in regimes}
    regime_dist = get_regime_summary(regimes)
    logger.info(f"  Distribution: {regime_dist}")

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

    # ── Step 7: Strategy Selection ────────────────────────────────────────────
    logger.info("Step 7: Selecting strategies")
    recommendations = []
    # Pass SPY 5-day return to the strategy selector so it can block directional
    # trades that fight the broad market tape (V2: SPY gate)
    spy_ret_5d = market_ctx.spy_return_5d
    logger.info(f"  SPY 5d return: {spy_ret_5d:+.2f}% (gate: ±{__import__('config.settings', fromlist=['SPY_DIRECTIONAL_GATE_PCT']).SPY_DIRECTIONAL_GATE_PCT:.1f}%)")

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
            recommendations.append(rec)

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
            trades.append(_build_trade_stub(rec, qualified[ticker], regime_map, iv_map, rs_map))
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

        trade_dict = _build_trade_dict(rec, trade_obj, qualified, regime_map, iv_map, rs_map, ticker, price)
        trades.append(trade_dict)

        time.sleep(0.3)  # Rate limiting between options fetches

    logger.info(f"  {len(trades)} trades constructed")

    # ── Step 10: Rank and Finalize ────────────────────────────────────────────
    logger.info("Step 10: Ranking picks")
    trades = _rank_trades(trades)
    top_picks = trades[:MAX_POSITIONS]

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

    return None


def _build_trade_dict(rec, trade_obj, data, regime_map, iv_map, rs_map, ticker, price):
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
            "sector": get_sector(ticker),
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
            },
            "iv_detail": {
                "iv_rank": iv.iv_rank,
                "iv_percentile": iv.iv_percentile,
                "current_iv": iv.current_iv,
                "hv_20": iv.hv_20,
                "iv_hv_ratio": iv.iv_hv_ratio,
                "iv_trend": iv.iv_trend,
                "premium_action": iv.premium_action,
            },
            "rs_detail": {
                "rs_vs_spy": rs.rs_vs_spy if rs else None,
                "rs_rank": rs.rs_rank if rs else None,
                "rs_trend": rs.rs_trend if rs else None,
            },
        },
    }


def _build_trade_stub(rec, df, regime_map, iv_map, rs_map):
    """Build a placeholder trade dict for dry_run mode."""
    ticker = rec.ticker
    price = float(df["close"].iloc[-1]) if not df.empty else 0

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
            "sector": get_sector(ticker),
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
