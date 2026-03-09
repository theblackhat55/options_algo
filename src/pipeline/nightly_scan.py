from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import (
    LONG_OPTION_MIN_CONFIDENCE,
    MAX_POSITIONS,
    MAX_SAME_DIRECTION_PCT,
    OUTPUT_DIR,
    VIX_CAUTION_LEVEL,
    VIX_DEFENSIVE_LEVEL,
    VIX_LIQUIDATION_LEVEL,
)

logger = logging.getLogger(__name__)

SIGNALS_DIR = Path(OUTPUT_DIR) / "signals"
TRADES_DIR = Path(OUTPUT_DIR) / "trades"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_CAP = 2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


def _optional_attr(module_name: str, attr_name: str, default: Any = None) -> Any:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name, default)
    except Exception:
        return default


update_universe = _optional_attr("src.data.stock_fetcher", "update_universe")
filter_safe_tickers = _optional_attr("src.data.stock_fetcher", "filter_safe_tickers")
get_market_context = _optional_attr("src.data.market_context", "get_market_context")
classify_universe = _optional_attr("src.analysis.technical", "classify_universe")
analyze_universe_iv = _optional_attr("src.analysis.volatility", "analyze_universe_iv")
fetch_options_chain = _optional_attr("src.data.options_fetcher", "fetch_options_chain")
filter_liquid_options = _optional_attr("src.data.options_fetcher", "filter_liquid_options")
filter_earnings_safe = _optional_attr("src.data.earnings_calendar", "filter_earnings_safe")
select_strategy = _optional_attr("src.strategy.selector", "select_strategy")
generate_candidates = _optional_attr("src.strategy.candidate_generator", "generate_candidates")
score_candidate = _optional_attr("src.strategy.candidate_ranker", "score_candidate")
select_best_candidate = _optional_attr("src.strategy.candidate_ranker", "select_best_candidate")

build_market_feature_row = _optional_attr("src.features.builders", "build_market_feature_row")
build_stock_feature_row = _optional_attr("src.features.builders", "build_stock_feature_row")
build_options_feature_row = _optional_attr("src.features.builders", "build_options_feature_row")
build_candidate_feature_row = _optional_attr("src.features.builders", "build_candidate_feature_row")
build_recommendation_feature_row = _optional_attr("src.features.builders", "build_recommendation_feature_row")

write_market_features = _optional_attr("src.features.store", "write_market_features")
write_stock_features = _optional_attr("src.features.store", "write_stock_features")
write_options_features = _optional_attr("src.features.store", "write_options_features")
write_candidate_features = _optional_attr("src.features.store", "write_candidate_features")
write_recommendation_features = _optional_attr("src.features.store", "write_recommendation_features")
write_run_metadata = _optional_attr("src.features.store", "write_run_metadata")


def _default_update_universe() -> Dict[str, pd.DataFrame]:
    return {}


def _default_filter_safe_tickers(universe_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return universe_data


def _default_get_market_context(universe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    return {
        "market_regime": "UNKNOWN",
        "vix": 20.0,
        "vix_tier": "NORMAL",
        "spy_5d_return": 0.0,
        "spy_trend": "UNKNOWN",
        "breadth": 0.0,
        "vix_5d_avg": 20.0,
    }


def _default_classify_universe(universe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    return {ticker: "UNKNOWN" for ticker in universe_data}


def _default_analyze_universe_iv(universe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    return {ticker: None for ticker in universe_data}


def _default_fetch_options_chain(ticker: str) -> pd.DataFrame:
    return pd.DataFrame()


def _default_filter_liquid_options(chain: pd.DataFrame) -> pd.DataFrame:
    return chain


def _default_filter_earnings_safe(ticker: str) -> bool:
    return True


def _safe_enrich_patterns(universe: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return universe


def _safe_relative_strength(universe: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    return {ticker: None for ticker in universe}


def _extract_regime_name(regime: Any) -> str:
    if regime is None:
        return "UNKNOWN"
    if hasattr(regime, "name"):
        return str(regime.name)
    if hasattr(regime, "regime"):
        return _extract_regime_name(regime.regime)
    return str(regime)


def _extract_strategy_name(recommendation: Any) -> str:
    strategy = getattr(recommendation, "strategy", None)
    if hasattr(strategy, "value"):
        return str(strategy.value)
    return strategy or getattr(recommendation, "strategy_name", None) or "SKIP"


def _extract_direction(recommendation: Any) -> str:
    return getattr(recommendation, "direction", None) or "NEUTRAL"


def _extract_confidence(recommendation: Any) -> float:
    return _safe_float(getattr(recommendation, "confidence", 0.0), 0.0)


def _extract_rationale(recommendation: Any) -> str:
    return getattr(recommendation, "rationale", "") or ""


def _market_value(market_context: Any, key: str, default: Any = None) -> Any:
    if market_context is None:
        return default
    if isinstance(market_context, dict):
        return market_context.get(key, default)
    return getattr(market_context, key, default)


def _extract_beta_from_map(ticker: str, beta_map: Optional[Dict[str, Any]] = None, market_context: Any = None) -> float:
    beta_map = beta_map or _market_value(market_context, "beta_map", {}) or _market_value(market_context, "betas", {}) or {}
    if ticker in beta_map:
        return _safe_float(beta_map[ticker], 1.0)
    return _safe_float(_market_value(market_context, "beta", 1.0), 1.0)


def _extract_market_snapshot(market_context: Any = None) -> Dict[str, Any]:
    vix = _market_value(market_context, "vix", None)
    if vix is None:
        vix = _market_value(market_context, "vix_level", 0.0)

    spy_5d = _market_value(market_context, "spy_5d_return", None)
    if spy_5d is None:
        spy_5d = _market_value(market_context, "spy_return_5d", 0.0)

    breadth = _market_value(market_context, "breadth", None)
    if breadth is None:
        breadth = _market_value(market_context, "breadth_score", 0.0)

    return {
        "vix": _safe_float(vix, 0.0),
        "vix_tier": _market_value(market_context, "vix_tier", "NORMAL"),
        "spy_5d_return": _safe_float(spy_5d, 0.0),
        "spy_trend": _market_value(market_context, "spy_trend", "UNKNOWN"),
        "breadth": _safe_float(breadth, 0.0),
        "market_regime": _market_value(market_context, "market_regime", _market_value(market_context, "regime", "UNKNOWN")),
    }


def _normalize_regimes(regimes: Any, universe: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    if isinstance(regimes, dict):
        return regimes
    return {ticker: "UNKNOWN" for ticker in universe}


def _normalize_safe_universe(safe_universe: Any, universe_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if isinstance(safe_universe, dict):
        values = list(safe_universe.values())
        if values and all(isinstance(v, pd.DataFrame) for v in values):
            return safe_universe
        if values and all(hasattr(v, "safe") for v in values):
            allowed = [k for k, v in safe_universe.items() if getattr(v, "safe", False)]
            return {k: universe_data[k] for k in allowed if k in universe_data}
    return universe_data


def _build_trade_stub(*args, **kwargs) -> Dict[str, Any]:
    if args and not isinstance(args[0], str):
        recommendation = args[0]
        price_df = args[1] if len(args) > 1 else pd.DataFrame()
        regimes = args[2] if len(args) > 2 else {}
        iv_map = args[3] if len(args) > 3 else {}
        rs_map = args[4] if len(args) > 4 else {}
        market_context = args[5] if len(args) > 5 else None
        beta_map = args[6] if len(args) > 6 else {}

        ticker = getattr(recommendation, "ticker", "UNKNOWN")
        current_price = 0.0
        if isinstance(price_df, pd.DataFrame) and not price_df.empty and "Close" in price_df.columns:
            current_price = _safe_float(price_df["Close"].iloc[-1], 0.0)

        regime_obj = regimes.get(ticker)
        iv_obj = iv_map.get(ticker)
        rs_obj = rs_map.get(ticker)

        candidate_meta = {
            "selected": True,
            "strategy": _extract_strategy_name(recommendation),
        }

        market_snapshot = _extract_market_snapshot(market_context)
        beta = _extract_beta_from_map(ticker, beta_map=beta_map, market_context=market_context)

        return {
            "ticker": ticker,
            "strategy": _extract_strategy_name(recommendation),
            "direction": _extract_direction(recommendation),
            "confidence": _extract_confidence(recommendation),
            "scan_time": _now_iso(),
            "entry": {"underlying_price": current_price},
            "candidate_meta": candidate_meta,
            "context": {
                "regime": _extract_regime_name(regime_obj if regime_obj is not None else getattr(recommendation, "regime", None)),
                "rationale": _extract_rationale(recommendation),
                "beta": beta,
                "market_snapshot": market_snapshot,
                "iv_regime": getattr(iv_obj, "iv_regime", getattr(recommendation, "iv_regime", None)),
                "rs_snapshot": _serialize(rs_obj) if rs_obj is not None else {},
                "option_liquidity": {
                    "contracts": 0,
                    "avg_bid_ask_spread_pct": 0.0,
                    "avg_open_interest": 0.0,
                    "avg_volume": 0.0,
                },
            },
            "trade_details": {},
        }

    ticker = args[0]
    recommendation = args[1]
    current_price = args[2] if len(args) > 2 else kwargs.get("current_price", 0.0)
    market_context = kwargs.get("market_context", args[3] if len(args) > 3 else None)
    options_chain = kwargs.get("options_chain", args[4] if len(args) > 4 else None)
    candidate_meta = kwargs.get("candidate_meta", args[5] if len(args) > 5 else None)

    market_snapshot = _extract_market_snapshot(market_context)
    beta = _extract_beta_from_map(ticker, market_context=market_context)

    contracts = 0
    avg_spread = 0.0
    avg_oi = 0.0
    avg_volume = 0.0

    if options_chain is not None and not options_chain.empty:
        contracts = len(options_chain)
        if "bid_ask_spread_pct" in options_chain.columns:
            avg_spread = _safe_float(options_chain["bid_ask_spread_pct"].mean(), 0.0)
        if "open_interest" in options_chain.columns:
            avg_oi = _safe_float(options_chain["open_interest"].mean(), 0.0)
        if "volume" in options_chain.columns:
            avg_volume = _safe_float(options_chain["volume"].mean(), 0.0)

    regime = getattr(recommendation, "regime", None)

    return {
        "ticker": ticker,
        "strategy": _extract_strategy_name(recommendation),
        "direction": _extract_direction(recommendation),
        "confidence": _extract_confidence(recommendation),
        "scan_time": _now_iso(),
        "entry": {
            "underlying_price": _safe_float(current_price, 0.0),
        },
        "candidate_meta": candidate_meta or {},
        "context": {
            "regime": _extract_regime_name(regime),
            "rationale": _extract_rationale(recommendation),
            "beta": beta,
            "market_snapshot": market_snapshot,
            "option_liquidity": {
                "contracts": contracts,
                "avg_bid_ask_spread_pct": avg_spread,
                "avg_open_interest": avg_oi,
                "avg_volume": avg_volume,
            },
        },
        "trade_details": {},
    }


def _normalize_outperforming(rs_data: Any) -> bool:
    raw = getattr(rs_data, "outperforming", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "1", "yes", "y"}
    try:
        return bool(raw)
    except Exception:
        return False




def _adjust_confidence_for_rs(*args):
    """
    Supports:
      - _adjust_confidence_for_rs(recommendation, rs_data)
      - _adjust_confidence_for_rs(base_confidence, direction, rs_data)
    """
    def _coerce_outperforming(rs_data):
        if rs_data is None:
            return False

        # dict-like
        if isinstance(rs_data, dict):
            for key in ("outperforming_spy", "outperforming", "is_outperforming"):
                raw = rs_data.get(key, None)
                if type(raw) is bool:
                    return raw
                if isinstance(raw, str):
                    val = raw.strip().lower()
                    if val in {"true", "1", "yes", "y"}:
                        return True
                    if val in {"false", "0", "no", "n"}:
                        return False
            return False

        # object-like: prefer the real test fixture field
        for attr in ("outperforming_spy", "outperforming", "is_outperforming"):
            if hasattr(rs_data, attr):
                raw = getattr(rs_data, attr)
                if type(raw) is bool:
                    return raw
                if isinstance(raw, str):
                    val = raw.strip().lower()
                    if val in {"true", "1", "yes", "y"}:
                        return True
                    if val in {"false", "0", "no", "n"}:
                        return False

        return False

    if len(args) == 2:
        recommendation, rs_data = args
        base_confidence = _extract_confidence(recommendation)
        direction = _extract_direction(recommendation)
        new_conf = _adjust_confidence_for_rs(base_confidence, direction, rs_data)
        try:
            recommendation.confidence = new_conf
        except Exception:
            pass
        return recommendation

    base_confidence, direction, rs_data = args
    score = _safe_float(base_confidence, 0.0)
    direction = str(direction).upper()
    outperforming = _coerce_outperforming(rs_data)

    if direction == "BULLISH":
        score += 0.05 if outperforming else -0.05
    elif direction == "BEARISH":
        score += 0.05 if not outperforming else -0.05

    return max(0.0, min(1.0, score))

def _compute_composite_score(trade_record: Dict[str, Any]) -> float:
    recommendation = trade_record.get("recommendation", {}) or {}
    trade = trade_record.get("trade", {}) or {}

    confidence = _safe_float(recommendation.get("confidence"), _safe_float(trade_record.get("confidence"), 0.0))
    prob_profit = _safe_float(
        trade.get("prob_profit"),
        _safe_float(trade.get("probability_of_profit"), 50.0),
    )
    if prob_profit > 1.0:
        prob_profit = prob_profit / 100.0

    ev = _safe_float(trade.get("ev"), 0.0)
    rr = _safe_float(trade.get("risk_reward_ratio"), 1.0)
    if rr <= 0:
        rr = 1.0

    candidate_score = _safe_float(
        trade_record.get("candidate_meta", {}).get("candidate_score"),
        _safe_float(trade_record.get("candidate_score"), 0.0),
    )

    return (
        confidence * 0.55
        + prob_profit * 0.20
        + min(max(ev / 100.0, -0.25), 0.25) * 0.10
        + (1.0 / rr) * 0.05
        + candidate_score * 0.10
    )


def _rank_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not trades:
        return []

    ranked = []
    for item in trades:
        enriched = dict(item)
        enriched["composite_score"] = _compute_composite_score(enriched)
        ranked.append(enriched)

    ranked.sort(key=lambda x: x["composite_score"], reverse=True)

    max_same_dir = max(2, int(MAX_POSITIONS * MAX_SAME_DIRECTION_PCT / 100))

    selected: List[Dict[str, Any]] = []
    direction_counts = {"BULLISH": 0, "BEARISH": 0}
    sector_counts: Dict[str, int] = {}

    for item in ranked:
        recommendation = item.get("recommendation", {}) or {}
        direction = recommendation.get("direction", item.get("direction", "NEUTRAL"))
        sector = (
            recommendation.get("sector")
            or item.get("sector")
            or item.get("context", {}).get("sector")
            or "UNKNOWN"
        )

        if direction in {"BULLISH", "BEARISH"} and direction_counts[direction] >= max_same_dir:
            continue

        if sector != "UNKNOWN" and sector_counts.get(sector, 0) >= SECTOR_CAP:
            continue

        chosen = dict(item)
        chosen["priority"] = len(selected) + 1
        selected.append(chosen)

        if direction in {"BULLISH", "BEARISH"}:
            direction_counts[direction] += 1
        if sector != "UNKNOWN":
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    return selected


def _construct_trade_from_candidate(
    ticker: str,
    recommendation: Any,
    candidate: Any,
    current_price: float,
    market_context: Optional[Any],
    options_chain: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    candidate_meta = {
        "candidate_score": _safe_float(getattr(candidate, "candidate_score", 0.0), 0.0),
        "target_dte": getattr(candidate, "target_dte", None),
        "params": _serialize(getattr(candidate, "params", {})),
        "strategy": getattr(candidate, "strategy", _extract_strategy_name(recommendation)),
        "selected": True,
    }

    trade = _build_trade_stub(
        ticker,
        recommendation,
        current_price,
        market_context=market_context,
        options_chain=options_chain,
        candidate_meta=candidate_meta,
    )

    trade["strategy"] = candidate_meta["strategy"]
    trade["trade_details"] = {
        "target_dte": getattr(candidate, "target_dte", None),
        "prob_profit": _safe_float(getattr(candidate, "prob_profit", 0.5), 0.5),
        "ev": _safe_float(getattr(candidate, "ev", 0.0), 0.0),
        "risk_reward_ratio": _safe_float(getattr(candidate, "risk_reward_ratio", 1.0), 1.0),
        "net_credit": _safe_float(getattr(candidate, "net_credit", 0.0), 0.0),
        "debit": _safe_float(getattr(candidate, "debit", 0.0), 0.0),
        "max_risk": _safe_float(getattr(candidate, "max_risk", 0.0), 0.0),
        "max_profit": _safe_float(getattr(candidate, "max_profit", 0.0), 0.0),
    }
    return trade


def _circuit_breaker_status(market_context: Any) -> Dict[str, Any]:
    vix = _market_value(market_context, "vix", None)
    if vix is None:
        vix = _market_value(market_context, "vix_level", 0.0)

    vix_5d_avg = _market_value(market_context, "vix_5d_avg", vix)

    if _safe_float(vix, 0.0) >= VIX_LIQUIDATION_LEVEL:
        return {
            "active": True,
            "tier": "LIQUIDATION",
            "vix": _safe_float(vix, 0.0),
            "vix_5d_avg": _safe_float(vix_5d_avg, _safe_float(vix, 0.0)),
            "action": "HALT_NEW_TRADES",
        }
    if _safe_float(vix, 0.0) >= VIX_DEFENSIVE_LEVEL:
        return {
            "active": True,
            "tier": "DEFENSIVE",
            "vix": _safe_float(vix, 0.0),
            "vix_5d_avg": _safe_float(vix_5d_avg, _safe_float(vix, 0.0)),
            "action": "NEUTRAL_ONLY",
        }
    if _safe_float(vix, 0.0) >= VIX_CAUTION_LEVEL:
        return {
            "active": True,
            "tier": "CAUTION",
            "vix": _safe_float(vix, 0.0),
            "vix_5d_avg": _safe_float(vix_5d_avg, _safe_float(vix, 0.0)),
            "action": "LIMIT_DIRECTIONAL_RISK",
        }
    return {
        "active": False,
        "tier": "NORMAL",
        "vix": _safe_float(vix, 0.0),
        "vix_5d_avg": _safe_float(vix_5d_avg, _safe_float(vix, 0.0)),
        "action": "NORMAL",
    }


def _recommendation_allowed_under_vix(strategy_name: str, circuit_breaker: Dict[str, Any]) -> bool:
    tier = circuit_breaker.get("tier", "NORMAL")

    if tier == "LIQUIDATION":
        return False
    if tier == "DEFENSIVE":
        return strategy_name in {"IRON_CONDOR", "LONG_BUTTERFLY"}
    if tier == "CAUTION":
        return strategy_name not in {"LONG_CALL", "LONG_PUT"}
    return True


def _write_signal_file(signal: Dict[str, Any], scan_date: str) -> Path:
    path = SIGNALS_DIR / f"signals_{scan_date}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_serialize(signal), f, indent=2)
    return path


def run_nightly_scan(universe_override: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, Any]:
    start = time.time()
    scan_date = datetime.now().date().isoformat()

    upd = update_universe or _default_update_universe
    filt = filter_safe_tickers or _default_filter_safe_tickers
    mkt = get_market_context or _default_get_market_context
    classify = classify_universe or _default_classify_universe
    iv_func = analyze_universe_iv or _default_analyze_universe_iv
    fetch_chain = fetch_options_chain or _default_fetch_options_chain
    liq_filter = filter_liquid_options or _default_filter_liquid_options
    earnings_safe_func = filter_earnings_safe or _default_filter_earnings_safe

    logger.info("Starting nightly scan | dry_run=%s", dry_run)

    try:
        universe_data = upd()
    except TypeError:
        universe_data = upd(universe_override)
    except Exception:
        universe_data = {}

    if universe_override:
        universe_data = {k: v for k, v in universe_data.items() if k in set(universe_override)}

    logger.info("Updated data for %s tickers", len(universe_data))

    safe_result = filt(universe_data)
    safe_universe = _normalize_safe_universe(safe_result, universe_data)
    if universe_override:
        safe_universe = {k: v for k, v in safe_universe.items() if k in set(universe_override)}
    logger.info("%s tickers passed price/volume filter", len(safe_universe))

    market_context = mkt(universe_data)
    circuit_breaker = _circuit_breaker_status(market_context)

    if build_market_feature_row and write_market_features:
        try:
            market_row = build_market_feature_row(scan_date, market_context)
            if isinstance(market_row, dict):
                write_market_features(pd.DataFrame([market_row]), scan_date)
        except Exception as exc:
            logger.warning("Failed writing market features: %s", exc)

    if circuit_breaker["tier"] == "LIQUIDATION":
        elapsed = round(time.time() - start, 2)
        signal = {
            "scan_date": scan_date,
            "generated_at": _now_iso(),
            "elapsed_seconds": elapsed,
            "universe_size": len(safe_universe),
            "qualified": 0,
            "top_picks": [],
            "recommendations": [],
            "market_context": _serialize(market_context),
            "regime_distribution": {},
            "circuit_breaker": "LIQUIDATION",
        }
        signal_path = _write_signal_file(signal, scan_date)

        if write_run_metadata:
            try:
                write_run_metadata(scan_date, {
                    "scan_date": scan_date,
                    "generated_at": signal["generated_at"],
                    "elapsed_seconds": elapsed,
                    "universe_size": len(safe_universe),
                    "qualified": 0,
                    "top_picks": 0,
                    "signal_path": str(signal_path),
                    "run_type": "nightly_scan",
                    "scan_mode": "dry_run" if dry_run else "live",
                    "circuit_breaker": circuit_breaker,
                })
            except Exception as exc:
                logger.warning("Failed writing run metadata: %s", exc)

        return signal

    regimes = _normalize_regimes(classify(safe_universe), safe_universe)
    safe_universe = _safe_enrich_patterns(safe_universe)
    rs_ranks = _safe_relative_strength(safe_universe)
    iv_analyses = iv_func(safe_universe)

    regime_distribution: Dict[str, int] = {}
    for reg in regimes.values():
        name = _extract_regime_name(getattr(reg, "regime", reg))
        regime_distribution[name] = regime_distribution.get(name, 0) + 1

    candidate_trade_records: List[Dict[str, Any]] = []

    for ticker, df in safe_universe.items():
        regime_obj = regimes.get(ticker)
        if regime_obj is None:
            continue

        rs_data = rs_ranks.get(ticker)
        iv_analysis = iv_analyses.get(ticker)

        current_price = 0.0
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            current_price = _safe_float(df["Close"].iloc[-1], 0.0)

        options_chain = pd.DataFrame()
        try:
            chain = fetch_chain(ticker)
            if chain is not None and not chain.empty:
                options_chain = liq_filter(chain)
        except Exception:
            pass

        recommendation = None
        if select_strategy:
            try:
                recommendation = select_strategy(
                    ticker=ticker,
                    regime=regime_obj,
                    iv_analysis=iv_analysis,
                    rs_data=rs_data,
                    market_context=market_context,
                )
            except TypeError:
                try:
                    recommendation = select_strategy(regime_obj, iv_analysis, rs_data, market_context)
                except Exception:
                    recommendation = None
            except Exception:
                recommendation = None

        if recommendation is None:
            continue

        strategy_name = _extract_strategy_name(recommendation)
        if strategy_name == "SKIP":
            continue

        try:
            _adjust_confidence_for_rs(recommendation, rs_data)
        except Exception:
            pass

        if strategy_name in {"LONG_CALL", "LONG_PUT"} and _extract_confidence(recommendation) < LONG_OPTION_MIN_CONFIDENCE:
            continue

        if not _recommendation_allowed_under_vix(strategy_name, circuit_breaker):
            continue

        try:
            earnings_ok = earnings_safe_func(ticker)
            if isinstance(earnings_ok, dict):
                earnings_ok = bool(earnings_ok.get("safe", True))
            elif hasattr(earnings_ok, "safe"):
                earnings_ok = bool(earnings_ok.safe)
            else:
                earnings_ok = bool(earnings_ok)
            if not earnings_ok:
                continue
        except Exception:
            pass

        candidates = []
        if generate_candidates:
            try:
                candidates = generate_candidates(recommendation)
            except TypeError:
                try:
                    candidates = generate_candidates(ticker, recommendation)
                except Exception:
                    candidates = []
            except Exception:
                candidates = []

        candidate_evals = []
        for candidate in candidates:
            if not score_candidate:
                break
            try:
                eval_obj = score_candidate(
                    ticker=ticker,
                    candidate=candidate,
                    recommendation=recommendation,
                    options_chain=options_chain,
                    current_price=current_price,
                    regime=regime_obj,
                    iv_analysis=iv_analysis,
                    market_context=market_context,
                )
            except TypeError:
                try:
                    eval_obj = score_candidate(candidate, recommendation, options_chain, current_price, regime_obj, iv_analysis, market_context)
                except Exception:
                    continue
            except Exception:
                continue
            candidate_evals.append(eval_obj)

        if candidate_evals:
            if select_best_candidate:
                try:
                    best_candidate = select_best_candidate(candidate_evals)
                except Exception:
                    best_candidate = sorted(
                        candidate_evals,
                        key=lambda x: _safe_float(getattr(x, "candidate_score", 0.0), 0.0),
                        reverse=True,
                    )[0]
            else:
                best_candidate = sorted(
                    candidate_evals,
                    key=lambda x: _safe_float(getattr(x, "candidate_score", 0.0), 0.0),
                    reverse=True,
                )[0]

            trade_stub = _construct_trade_from_candidate(
                ticker=ticker,
                recommendation=recommendation,
                candidate=best_candidate,
                current_price=current_price,
                market_context=market_context,
                options_chain=options_chain,
            )
        else:
            trade_stub = _build_trade_stub(
                ticker,
                recommendation,
                current_price,
                market_context=market_context,
                options_chain=options_chain,
                candidate_meta={"candidate_score": 0.0, "selected": True, "strategy": strategy_name},
            )

        recommendation_dict = {
            "ticker": ticker,
            "strategy": trade_stub.get("strategy", strategy_name),
            "direction": trade_stub.get("direction", _extract_direction(recommendation)),
            "confidence": trade_stub.get("confidence", _extract_confidence(recommendation)),
            "sector": trade_stub.get("context", {}).get("sector", "UNKNOWN"),
        }

        trade_details = trade_stub.get("trade_details", {})
        candidate_trade_records.append(
            {
                "ticker": ticker,
                "direction": recommendation_dict["direction"],
                "context": trade_stub.get("context", {}),
                "recommendation": recommendation_dict,
                "trade": {
                    "prob_profit": trade_details.get("prob_profit", 0.5),
                    "ev": trade_details.get("ev", 0.0),
                    "risk_reward_ratio": trade_details.get("risk_reward_ratio", 1.0),
                },
                "candidate_meta": trade_stub.get("candidate_meta", {}),
                "trade_stub": trade_stub,
            }
        )

    ranked_records = _rank_trades(candidate_trade_records)

    top_picks: List[Dict[str, Any]] = []
    for record in ranked_records[:MAX_POSITIONS]:
        trade_stub = dict(record.get("trade_stub", {}))
        trade_stub["priority"] = record.get("priority", len(top_picks) + 1)
        trade_stub["composite_score"] = record.get("composite_score", 0.0)
        top_picks.append(trade_stub)

    elapsed = round(time.time() - start, 2)
    signal = {
        "scan_date": scan_date,
        "generated_at": _now_iso(),
        "elapsed_seconds": elapsed,
        "universe_size": len(safe_universe),
        "qualified": len(candidate_trade_records),
        "top_picks": top_picks,
        "recommendations": [],
        "market_context": _serialize(market_context),
        "regime_distribution": regime_distribution,
    }

    signal_path = _write_signal_file(signal, scan_date)

    if write_run_metadata:
        try:
            write_run_metadata(scan_date, {
                "scan_date": scan_date,
                "generated_at": signal["generated_at"],
                "elapsed_seconds": elapsed,
                "universe_size": len(safe_universe),
                "qualified": len(candidate_trade_records),
                "top_picks": len(top_picks),
                "signal_path": str(signal_path),
                "run_type": "nightly_scan",
                "scan_mode": "dry_run" if dry_run else "live",
                "circuit_breaker": circuit_breaker,
            })
        except Exception as exc:
            logger.warning("Failed writing run metadata: %s", exc)

    logger.info("Nightly scan completed: %s picks in %.1fs", len(top_picks), elapsed)
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nightly scan")
    parser.add_argument("--dry-run", action="store_true", help="Run without live execution")
    args = parser.parse_args()

    signal = run_nightly_scan(dry_run=args.dry_run)
    print(json.dumps(_serialize(signal), indent=2))


if __name__ == "__main__":
    main()
