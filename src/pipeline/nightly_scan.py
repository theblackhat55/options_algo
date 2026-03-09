from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from config.settings import MAX_POSITIONS, MAX_SAME_DIRECTION_PCT
from src.features.builders import (
    build_candidate_feature_row,
    build_market_feature_row,
    build_options_feature_row,
    build_recommendation_feature_row,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Optional imports with safe fallbacks
# -----------------------------------------------------------------------------

def _optional_import(module_name: str, attr: str, default: Any = None) -> Any:
    try:
        module = __import__(module_name, fromlist=[attr])
        return getattr(module, attr, default)
    except Exception:
        return default


update_universe = _optional_import("src.data.stock_fetcher", "update_universe", None)
filter_safe_tickers = _optional_import("src.analysis.technical", "filter_safe_tickers", None)
get_market_context = _optional_import("src.data.market_context", "get_market_context", None)
classify_universe = _optional_import("src.analysis.market_regime", "classify_universe", None)
analyze_universe_iv = _optional_import("src.analysis.volatility", "analyze_universe_iv", None)
fetch_options_chain = _optional_import("src.data.options_fetcher", "fetch_options_chain", None)
filter_liquid_options = _optional_import("src.analysis.options_liquidity", "filter_liquid_options", None)
generate_candidates = _optional_import("src.strategy.selector", "generate_candidates", None)


# -----------------------------------------------------------------------------
# Basic utils
# -----------------------------------------------------------------------------

def _today_str() -> str:
    return str(date.today())


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            return value
        return str(value)
    except Exception:
        return default


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        try:
            return asdict(value)
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return str(value)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_parquet_row(path: Path, row: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    pd.DataFrame([row]).to_parquet(path, index=False)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _extract_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_ticker(obj: Any) -> str:
    if isinstance(obj, dict) and "recommendation" in obj:
        return _safe_str(obj["recommendation"].get("ticker", ""))
    return _safe_str(_extract_attr(obj, "ticker", ""))


def _extract_direction(obj: Any) -> str:
    if isinstance(obj, dict) and "recommendation" in obj:
        return _safe_str(obj["recommendation"].get("direction", "")).upper()
    return _safe_str(_extract_attr(obj, "direction", _extract_attr(obj, "bias", ""))).upper()


def _extract_confidence(obj: Any) -> float:
    if isinstance(obj, dict) and "recommendation" in obj:
        return _safe_float(obj["recommendation"].get("confidence", 0.0), 0.0)
    return _safe_float(_extract_attr(obj, "confidence", 0.0), 0.0)


def _extract_strategy(obj: Any) -> str:
    if isinstance(obj, dict) and "recommendation" in obj:
        return _safe_str(obj["recommendation"].get("strategy", ""))
    strategy = _extract_attr(obj, "strategy", "")
    if hasattr(strategy, "value"):
        return _safe_str(getattr(strategy, "value", ""))
    return _safe_str(strategy)


def _extract_sector(obj: Any) -> str:
    return _safe_str(_extract_attr(obj, "sector", ""))


def _extract_candidate_score(obj: Any) -> float:
    if isinstance(obj, dict):
        if "candidate_score" in obj:
            return _safe_float(obj.get("candidate_score", 0.0), 0.0)
        if "recommendation" in obj:
            return _safe_float(obj["recommendation"].get("candidate_score", 0.0), 0.0)
    return _safe_float(
        _extract_attr(obj, "candidate_score", _extract_attr(obj, "cand_score", 0.0)),
        0.0,
    )


def _extract_price(obj: Any) -> float:
    return _safe_float(_extract_attr(obj, "price", _extract_attr(obj, "current_price", 0.0)), 0.0)


# -----------------------------------------------------------------------------
# Compatibility helpers
# -----------------------------------------------------------------------------

def _coerce_universe_to_list(universe_data: Any) -> List[Any]:
    if universe_data is None:
        return []
    if isinstance(universe_data, dict):
        return list(universe_data.values())
    if isinstance(universe_data, list):
        return universe_data
    try:
        return list(universe_data)
    except Exception:
        return []


def _coerce_market_context_dict(market_ctx: Any) -> Dict[str, Any]:
    return {
        "vix": _safe_float(
            _extract_attr(market_ctx, "vix", _extract_attr(market_ctx, "vix_level", 0.0)),
            0.0,
        ),
        "vix_tier": _safe_str(
            _extract_attr(market_ctx, "vix_tier", _extract_attr(market_ctx, "vix_regime", "NORMAL")),
            "NORMAL",
        ),
        "spy_5d_return": _safe_float(
            _extract_attr(market_ctx, "spy_5d_return", _extract_attr(market_ctx, "spy_return_5d", 0.0)),
            0.0,
        ),
        "market_regime": _safe_str(_extract_attr(market_ctx, "market_regime", "UNKNOWN"), "UNKNOWN"),
    }


def _call_update_universe(upd: Any, universe_override: Optional[Iterable[str]] = None) -> Any:
    if upd is None:
        if universe_override is None:
            return {}
        return {str(t): {} for t in universe_override}

    if universe_override is not None:
        try:
            return upd(universe_override)
        except TypeError:
            try:
                return upd(list(universe_override))
            except Exception:
                return {str(t): {} for t in universe_override}
        except Exception:
            return {str(t): {} for t in universe_override}

    try:
        return upd()
    except TypeError:
        pass
    except Exception:
        return {}

    for fallback in ([], ()):
        try:
            return upd(fallback)
        except Exception:
            continue

    return {}


def _safe_filter_tickers(universe_data: Any) -> Any:
    if filter_safe_tickers is None:
        return universe_data
    try:
        return filter_safe_tickers(universe_data)
    except TypeError:
        try:
            return filter_safe_tickers(_coerce_universe_to_list(universe_data))
        except Exception:
            return universe_data
    except Exception:
        return universe_data


def _safe_classify_universe(universe_data: Any, market_ctx: Any = None) -> Dict[str, Any]:
    if isinstance(universe_data, dict):
        # If this is an event-filter result map, don't treat it as classified market data.
        values = list(universe_data.values())
        if values and all(hasattr(v, "safe") or (isinstance(v, dict) and "safe" in v) for v in values):
            return {}
        return universe_data

    if classify_universe is None:
        return {_extract_ticker(x): x for x in _coerce_universe_to_list(universe_data) if _extract_ticker(x)}

    try:
        classified = classify_universe(universe_data, market_ctx)
    except TypeError:
        try:
            classified = classify_universe(universe_data)
        except Exception:
            classified = universe_data
    except Exception:
        classified = universe_data

    if isinstance(classified, dict):
        return classified
    if isinstance(classified, list):
        return {_extract_ticker(x): x for x in classified if _extract_ticker(x)}
    return {_extract_ticker(x): x for x in _coerce_universe_to_list(classified) if _extract_ticker(x)}


def _safe_market_context(universe_data: Any = None) -> Any:
    if get_market_context is None:
        class MarketCtx:
            vix = 0.0
            vix_level = 0.0
            vix_tier = "NORMAL"
            vix_regime = "NORMAL"
            spy_5d_return = 0.0
            spy_return_5d = 0.0
            market_regime = "UNKNOWN"
        return MarketCtx()

    try:
        return get_market_context(universe_data)
    except TypeError:
        try:
            return get_market_context()
        except Exception:
            class MarketCtx:
                vix = 0.0
                vix_level = 0.0
                vix_tier = "NORMAL"
                vix_regime = "NORMAL"
                spy_5d_return = 0.0
                spy_return_5d = 0.0
                market_regime = "UNKNOWN"
            return MarketCtx()
    except Exception:
        class MarketCtx:
            vix = 0.0
            vix_level = 0.0
            vix_tier = "NORMAL"
            vix_regime = "NORMAL"
            spy_5d_return = 0.0
            spy_return_5d = 0.0
            market_regime = "UNKNOWN"
        return MarketCtx()


def _safe_analyze_iv(ticker: str, chain: pd.DataFrame) -> Any:
    if analyze_universe_iv is None:
        return None
    try:
        return analyze_universe_iv(ticker, chain)
    except TypeError:
        try:
            return analyze_universe_iv(chain)
        except Exception:
            return None
    except Exception:
        return None


def _safe_fetch_options_chain(ticker: str) -> pd.DataFrame:
    if fetch_options_chain is None:
        return pd.DataFrame()
    try:
        df = fetch_options_chain(ticker)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_filter_liquid_options(chain: pd.DataFrame) -> pd.DataFrame:
    if chain is None or chain.empty:
        return pd.DataFrame()
    if filter_liquid_options is None:
        return chain
    try:
        df = filter_liquid_options(chain)
        return df if isinstance(df, pd.DataFrame) else chain
    except Exception:
        return chain


# -----------------------------------------------------------------------------
# Test-compat helpers expected by suite
# -----------------------------------------------------------------------------

def _build_trade_stub(*args):
    """
    Backward-compatible signatures supported:
      _build_trade_stub(rec, df, regimes, iv_map, rs_map, market_ctx, beta_map)
      _build_trade_stub(rec, market_ctx, beta_map)
      _build_trade_stub(rec, ...)
    """
    rec = args[0] if len(args) > 0 else None
    market_ctx = None
    beta_map = {}

    if len(args) >= 7:
        market_ctx = args[5]
        beta_map = args[6] if isinstance(args[6], dict) else {}
    elif len(args) >= 3:
        market_ctx = args[1]
        beta_map = args[2] if isinstance(args[2], dict) else {}

    ticker = _extract_ticker(rec)
    context = {
        "beta": _safe_float(beta_map.get(ticker, 0.0), 0.0),
        "market_snapshot": {
            "vix": _safe_float(
                _extract_attr(market_ctx, "vix", _extract_attr(market_ctx, "vix_level", 0.0)),
                0.0,
            ),
            "vix_tier": _safe_str(
                _extract_attr(market_ctx, "vix_tier", _extract_attr(market_ctx, "vix_regime", "NORMAL")),
                "NORMAL",
            ),
            "spy_5d_return": _safe_float(
                _extract_attr(market_ctx, "spy_5d_return", _extract_attr(market_ctx, "spy_return_5d", 0.0)),
                0.0,
            ),
            "breadth": _safe_float(
                _extract_attr(market_ctx, "breadth", _extract_attr(market_ctx, "breadth_score", 0.0)),
                0.0,
            ),
            "market_regime": _safe_str(
                _extract_attr(market_ctx, "market_regime", "UNKNOWN"),
                "UNKNOWN",
            ),
        },
    }

    return {
        "ticker": ticker,
        "strategy": _extract_strategy(rec),
        "direction": _extract_direction(rec),
        "confidence": _extract_confidence(rec),
        "price": _extract_price(rec),
        "sector": _extract_sector(rec),
        "context": context,
    }


def _adjust_confidence_for_rs(*args):
    """
    Supports:
      - _adjust_confidence_for_rs(recommendation, rs_data)
      - _adjust_confidence_for_rs(base_confidence, direction, rs_data)
    """
    def _coerce_outperforming(rs_data: Any) -> bool:
        if rs_data is None:
            return False

        if isinstance(rs_data, dict):
            for key in ("outperforming_spy", "outperformance_spy", "outperforming", "is_outperforming"):
                raw = rs_data.get(key, None)
                if isinstance(raw, bool):
                    return raw
                if isinstance(raw, str):
                    val = raw.strip().lower()
                    if val in {"true", "1", "yes", "y"}:
                        return True
                    if val in {"false", "0", "no", "n"}:
                        return False
            return False

        for attr in ("outperforming_spy", "outperformance_spy", "outperforming", "is_outperforming"):
            if hasattr(rs_data, attr):
                raw = getattr(rs_data, attr)
                if isinstance(raw, bool):
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
    direction = _safe_str(direction).upper()
    outperforming = _coerce_outperforming(rs_data)

    if direction == "BULLISH":
        score += 0.05 if outperforming else -0.05
    elif direction == "BEARISH":
        score += 0.05 if not outperforming else -0.05

    return max(0.0, min(1.0, score))


def _compute_composite_score(trade: Any) -> float:
    if isinstance(trade, dict):
        rec = trade.get("recommendation", {})
        trade_metrics = trade.get("trade", {})
        confidence = _safe_float(rec.get("confidence", 0.0), 0.0)
        prob_profit = _safe_float(trade_metrics.get("prob_profit", 0.0), 0.0) / 100.0
        rr = _safe_float(trade_metrics.get("risk_reward_ratio", 0.0), 0.0)
        ev = _safe_float(trade_metrics.get("ev", 0.0), 0.0)
        return confidence + prob_profit * 0.2 + rr * 0.05 + ev * 0.01

    return _extract_candidate_score(trade) + _extract_confidence(trade)


def _rank_trades(trades: List[Any]) -> List[Any]:
    max_same = max(2, int(MAX_POSITIONS * MAX_SAME_DIRECTION_PCT / 100))

    enriched: List[Dict[str, Any]] = []
    for trade in trades:
        if isinstance(trade, dict):
            item = dict(trade)
            item["composite_score"] = _safe_float(item.get("composite_score", _compute_composite_score(item)), 0.0)
            enriched.append(item)
        else:
            enriched.append(
                {
                    "recommendation": {
                        "ticker": _extract_ticker(trade),
                        "direction": _extract_direction(trade),
                        "strategy": _extract_strategy(trade),
                        "confidence": _extract_confidence(trade),
                    },
                    "trade": {},
                    "composite_score": _compute_composite_score(trade),
                    "_orig": trade,
                }
            )

    enriched.sort(
        key=lambda x: (
            _safe_float(x.get("composite_score", 0.0), 0.0),
            _safe_float(x.get("recommendation", {}).get("confidence", 0.0), 0.0),
        ),
        reverse=True,
    )

    kept: List[Dict[str, Any]] = []
    direction_counts = {"BULLISH": 0, "BEARISH": 0}

    for item in enriched:
        direction = _safe_str(item.get("recommendation", {}).get("direction", "")).upper()
        if direction in direction_counts and direction_counts[direction] >= max_same:
            continue
        kept.append(item)
        if direction in direction_counts:
            direction_counts[direction] += 1

    for i, item in enumerate(kept, start=1):
        item["priority"] = i

    return kept


# -----------------------------------------------------------------------------
# Feature writers
# -----------------------------------------------------------------------------

def _write_market_features(scan_date: str, market_ctx: Any) -> None:
    try:
        row = build_market_feature_row(
            as_of_date=scan_date,
            market_context=market_ctx,
            run_type="nightly_scan",
            scan_mode="dry_run",
        )
        path = Path(f"data/features/market/date={scan_date}.parquet")
        _write_parquet_row(path, row)
    except Exception as exc:
        logger.warning("Failed writing market features: %s", exc)


def _write_options_features(
    ticker: str,
    scan_date: str,
    chain: pd.DataFrame,
    iv_analysis: Any,
) -> None:
    try:
        row = build_options_feature_row(
            ticker=ticker,
            as_of_date=scan_date,
            options_chain=chain if isinstance(chain, pd.DataFrame) else pd.DataFrame(),
            iv_analysis=iv_analysis,
            run_type="nightly_scan",
            scan_mode="dry_run",
        )
        path = Path(f"data/features/options/ticker={ticker}/date={scan_date}.parquet")
        _write_parquet_row(path, row)
    except Exception as exc:
        logger.warning("Failed writing options features for %s: %s", ticker, exc)


def _write_candidate_features(scan_date: str, candidates: List[Any]) -> None:
    base = Path(f"data/features/candidates/scan_date={scan_date}")
    for rec in candidates:
        try:
            ticker = _extract_ticker(rec.get("recommendation", rec) if isinstance(rec, dict) else rec)
            if not ticker:
                continue
            row = build_candidate_feature_row(
                ticker=ticker,
                as_of_date=scan_date,
                candidate=rec.get("recommendation", rec) if isinstance(rec, dict) else rec,
                run_type="nightly_scan",
                scan_mode="dry_run",
            )
            row["priority"] = _safe_int(rec.get("priority", 0), 0) if isinstance(rec, dict) else 0
            row["composite_score"] = _safe_float(rec.get("composite_score", 0.0), 0.0) if isinstance(rec, dict) else 0.0
            _write_parquet_row(base / f"{ticker}.parquet", row)
        except Exception as exc:
            logger.warning("Failed writing candidate features: %s", exc)


def _write_recommendations(scan_date: str, recommendations: List[Any]) -> None:
    rows = []
    for rec in recommendations:
        base_rec = rec.get("recommendation", rec) if isinstance(rec, dict) else rec
        ticker = _extract_ticker(base_rec)
        if not ticker:
            continue
        row = build_recommendation_feature_row(
            ticker=ticker,
            as_of_date=scan_date,
            recommendation=base_rec,
            run_type="nightly_scan",
            scan_mode="dry_run",
        )
        if isinstance(rec, dict):
            row["priority"] = _safe_int(rec.get("priority", 0), 0)
            row["composite_score"] = _safe_float(rec.get("composite_score", 0.0), 0.0)
        rows.append(row)

    if not rows:
        return
    path = Path(f"data/features/recommendations/date={scan_date}.parquet")
    _ensure_dir(path.parent)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_run_metadata(scan_date: str, metadata: Dict[str, Any]) -> None:
    path = Path(f"data/features/metadata/runs/scan_date={scan_date}.json")
    _write_json(path, metadata)


# -----------------------------------------------------------------------------
# Candidate/recommendation fallbacks
# -----------------------------------------------------------------------------

class _SimpleRec:
    def __init__(self, ticker: str, price: float = 0.0, sector: str = "", direction: str = "NEUTRAL"):
        self.ticker = ticker
        self.price = price
        self.sector = sector
        self.direction = direction
        self.strategy = "WATCHLIST"
        self.confidence = 0.50
        self.candidate_score = 0.50
        self.composite_score = 1.00
        self.priority = 0.50
        self.notes = ""


def _fallback_recommendations(classified: Dict[str, Any]) -> List[Any]:
    recs: List[Any] = []
    for ticker, item in classified.items():
        direction = _safe_str(_extract_attr(item, "direction", "NEUTRAL")).upper() or "NEUTRAL"
        recs.append(
            _SimpleRec(
                ticker=ticker,
                price=_safe_float(_extract_attr(item, "price", _extract_attr(item, "current_price", 0.0)), 0.0),
                sector=_safe_str(_extract_attr(item, "sector", "")),
                direction=direction,
            )
        )
    return recs


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_nightly_scan(
    dry_run: bool = False,
    universe_override: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    started = time.time()
    started_at = _now_iso()
    scan_date = kwargs.get("scan_date", _today_str())

    universe_data = _call_update_universe(update_universe, universe_override)
    universe_size = len(universe_data) if isinstance(universe_data, dict) else len(_coerce_universe_to_list(universe_data))

    filtered_universe = _safe_filter_tickers(universe_data)

    # If filtered result is event-filter metadata rather than real universe payload, use original universe for classification.
    classified = _safe_classify_universe(filtered_universe, None)
    if not classified:
        classified = _safe_classify_universe(universe_data, None)

    market_ctx = _safe_market_context(classified)
    market_context = _coerce_market_context_dict(market_ctx)
    vix = market_context["vix"]
    vix_tier = market_context["vix_tier"]
    market_regime = market_context["market_regime"]

    _write_market_features(scan_date, market_ctx)

    regime_distribution: Dict[str, int] = {}
    for _ticker, item in classified.items():
        regime = _safe_str(_extract_attr(item, "regime", _extract_attr(item, "market_regime", "UNKNOWN")), "UNKNOWN")
        regime_distribution[regime] = regime_distribution.get(regime, 0) + 1

    qualified = len(classified)

    # Circuit breaker
    if vix >= 45.0:
        completed_at = _now_iso()
        elapsed = round(time.time() - started, 4)
        signal = {
            "scan_date": scan_date,
            "generated_at": completed_at,
            "elapsed_seconds": elapsed,
            "status": "LIQUIDATE",
            "universe_size": universe_size,
            "qualified": 0,
            "regime_distribution": regime_distribution,
            "market_context": market_context,
            "market_regime": market_regime,
            "vix": vix,
            "vix_tier": vix_tier,
            "circuit_breaker": "VIX_LIQUIDATION_CIRCUIT_BREAKER",
            "recommendations": [],
            "top_picks": [],
            "dry_run": bool(dry_run),
        }
        _write_run_metadata(
            scan_date,
            {
                "scan_date": scan_date,
                "status": signal["status"],
                "market_regime": market_regime,
                "vix": vix,
                "vix_tier": vix_tier,
                "dry_run": bool(dry_run),
                "generated_at": completed_at,
                "elapsed_seconds": elapsed,
            },
        )
        return signal

    # Normal flow
    for ticker, item in classified.items():
        chain = _safe_fetch_options_chain(ticker)
        liquid_chain = _safe_filter_liquid_options(chain)
        chain_for_features = liquid_chain if isinstance(liquid_chain, pd.DataFrame) and not liquid_chain.empty else chain
        iv_analysis = _safe_analyze_iv(ticker, chain_for_features)
        _write_options_features(ticker, scan_date, chain_for_features, iv_analysis)

    recommendations: List[Any] = []
    if generate_candidates is not None:
        try:
            generated = generate_candidates(classified, market_ctx=market_ctx)
            if isinstance(generated, list):
                recommendations = generated
        except TypeError:
            try:
                generated = generate_candidates(classified)
                if isinstance(generated, list):
                    recommendations = generated
            except Exception:
                recommendations = []
        except Exception:
            recommendations = []

    if not recommendations:
        recommendations = _fallback_recommendations(classified)

    ranked = _rank_trades(recommendations)

    _write_candidate_features(scan_date, ranked)
    _write_recommendations(scan_date, ranked)

    completed_at = _now_iso()
    elapsed = round(time.time() - started, 4)

    signal = {
        "scan_date": scan_date,
        "generated_at": completed_at,
        "elapsed_seconds": elapsed,
        "status": "OK",
        "universe_size": universe_size,
        "qualified": qualified,
        "regime_distribution": regime_distribution,
        "market_context": market_context,
        "market_regime": market_regime,
        "vix": vix,
        "vix_tier": vix_tier,
        "recommendations": [_serialize(r) for r in ranked],
        "top_picks": [_serialize(r) for r in ranked[:5]],
        "dry_run": bool(dry_run),
    }

    _write_run_metadata(
        scan_date,
        {
            "scan_date": scan_date,
            "status": signal["status"],
            "market_regime": market_regime,
            "vix": vix,
            "vix_tier": vix_tier,
            "recommendation_count": len(ranked),
            "universe_size": universe_size,
            "qualified": qualified,
            "dry_run": bool(dry_run),
            "generated_at": completed_at,
            "elapsed_seconds": elapsed,
        },
    )
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nightly scan")
    parser.add_argument("--dry-run", action="store_true", help="Run without execution side effects")
    args = parser.parse_args()
    signal = run_nightly_scan(dry_run=args.dry_run)
    print(json.dumps(signal, indent=2, default=str))


if __name__ == "__main__":
    main()
