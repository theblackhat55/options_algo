from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import pandas as pd

from src.analysis.options_surface import analyze_options_surface


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


def _extract(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_strategy(obj: Any) -> str:
    strategy = _extract(obj, "strategy", "")
    if hasattr(strategy, "value"):
        return _safe_str(getattr(strategy, "value", ""))
    return _safe_str(strategy, "")


def _common_fields(
    ticker: str,
    as_of_date: str,
    run_type: str,
    scan_mode: str,
) -> Dict[str, Any]:
    return {
        "ticker": _safe_str(ticker),
        "as_of_date": _safe_str(as_of_date),
        "run_type": _safe_str(run_type),
        "scan_mode": _safe_str(scan_mode),
    }


def build_market_feature_row(
    as_of_date: str,
    market_context: Any,
    run_type: str = "nightly_scan",
    scan_mode: str = "standard",
) -> Dict[str, Any]:
    row = {
        "as_of_date": _safe_str(as_of_date),
        "run_type": _safe_str(run_type),
        "scan_mode": _safe_str(scan_mode),
    }

    if market_context is None:
        return row

    attrs = (
        "market_regime",
        "vix",
        "vix_level",
        "vix_tier",
        "spy_price",
        "spy_return_1d",
        "spy_return_5d",
        "spy_5d_return",
        "qqq_return_1d",
        "qqq_return_5d",
        "breadth",
        "breadth_score",
        "trend_strength",
    )
    for attr in attrs:
        row[attr] = _serialize(_extract(market_context, attr, None))

    return row


def build_options_feature_row(
    ticker: str,
    as_of_date: str,
    options_chain: pd.DataFrame,
    iv_analysis: Any = None,
    run_type: str = "nightly_scan",
    scan_mode: str = "standard",
) -> Dict[str, Any]:
    if options_chain is None:
        options_chain = pd.DataFrame()

    spot_price = 0.0
    if not options_chain.empty and "underlying_price" in options_chain.columns:
        try:
            spot_price = _safe_float(options_chain["underlying_price"].dropna().iloc[0], 0.0)
        except Exception:
            spot_price = 0.0

    surface = analyze_options_surface(
        ticker=ticker,
        spot_price=spot_price,
        chain=options_chain,
    )

    row = _common_fields(ticker, as_of_date, run_type, scan_mode)
    row.update(
        {
            "current_iv": _safe_float(_extract(iv_analysis, "current_iv", None), 0.0) if iv_analysis is not None else 0.0,
            "iv_rank": _safe_float(_extract(iv_analysis, "iv_rank", None), 0.0) if iv_analysis is not None else 0.0,
            "iv_percentile": _safe_float(_extract(iv_analysis, "iv_percentile", None), 0.0) if iv_analysis is not None else 0.0,
            "iv_regime": _safe_str(_extract(iv_analysis, "iv_regime", None), "") if iv_analysis is not None else "",
            "iv_hv_ratio": _safe_float(_extract(iv_analysis, "iv_hv_ratio", None), 0.0) if iv_analysis is not None else 0.0,
            "iv_rv_spread": _safe_float(_extract(iv_analysis, "iv_rv_spread", None), 0.0) if iv_analysis is not None else 0.0,
            "skew_legacy": _safe_float(_extract(iv_analysis, "skew", None), 0.0) if iv_analysis is not None else 0.0,
            "surface_atm_iv": _safe_float(surface.atm_iv, 0.0),
            "surface_near_atm_iv": _safe_float(surface.near_atm_iv, 0.0),
            "surface_mid_atm_iv": _safe_float(surface.mid_atm_iv, 0.0),
            "surface_term_slope": _safe_float(surface.term_slope, 0.0),
            "surface_term_ratio": _safe_float(surface.term_ratio, 0.0),
            "surface_call_iv_mean": _safe_float(surface.call_iv_mean, 0.0),
            "surface_put_iv_mean": _safe_float(surface.put_iv_mean, 0.0),
            "surface_skew_proxy": _safe_float(surface.skew_proxy, 0.0),
            "surface_put_call_oi_ratio": _safe_float(surface.put_call_oi_ratio, 0.0),
            "surface_put_call_volume_ratio": _safe_float(surface.put_call_volume_ratio, 0.0),
            "surface_avg_spread_pct": _safe_float(surface.avg_spread_pct, 0.0),
            "surface_median_spread_pct": _safe_float(surface.median_spread_pct, 0.0),
            "surface_liquid_contract_ratio": _safe_float(surface.liquid_contract_ratio, 0.0),
            "surface_valid_quote_ratio": _safe_float(surface.valid_quote_ratio, 0.0),
            "surface_valid_spread_count": _safe_int(surface.valid_spread_count, 0),
            "surface_spread_sample_size": _safe_int(surface.spread_sample_size, 0),
            "surface_top_oi_strike": _safe_float(surface.top_oi_strike, 0.0),
            "surface_top_oi_strike_distance_pct": _safe_float(surface.top_oi_strike_distance_pct, 0.0),
            "surface_top_volume_strike": _safe_float(surface.top_volume_strike, 0.0),
            "surface_top_volume_strike_distance_pct": _safe_float(surface.top_volume_strike_distance_pct, 0.0),
            "surface_front_expiry_concentration": _safe_float(surface.front_expiry_concentration, 0.0),
            "surface_total_contracts": _safe_int(surface.total_contracts, 0),
        }
    )
    return row


def build_candidate_feature_row(
    ticker: str,
    as_of_date: str,
    candidate: Any,
    run_type: str = "nightly_scan",
    scan_mode: str = "standard",
) -> Dict[str, Any]:
    row = _common_fields(ticker, as_of_date, run_type, scan_mode)
    if candidate is None:
        row.update(
            {
                "strategy": "",
                "direction": "",
                "confidence": 0.0,
                "candidate_score": 0.0,
                "composite_score": 0.0,
                "priority": 0,
                "signal": "",
                "regime": "",
                "sector": "",
                "price": 0.0,
                "notes": "",
                "surface_confidence_delta": 0.0,
                "surface_candidate_score_delta": 0.0,
                "surface_adjustment_notes": "",
                "surface_original_strategy": "",
                "surface_preferred_strategy": "",
                "surface_strategy_changed": False,
                "surface_bias_strength": "",
                "surface_bias_rationale": "",
                "surface_liquidity_quality": "",
                "surface_quote_availability": "",
                "surface_liquid_contract_ratio": 0.0,
                "surface_avg_spread_pct": 0.0,
                "surface_median_spread_pct": 0.0,
                "surface_valid_quote_ratio": 0.0,
                "surface_valid_spread_count": 0,
                "surface_spread_sample_size": 0,
                "surface_front_expiry_concentration": 0.0,
                "surface_top_oi_strike_distance_pct": 0.0,
                "surface_term_slope": 0.0,
                "surface_term_ratio": 0.0,
                "surface_put_call_oi_ratio": 0.0,
                "surface_put_call_volume_ratio": 0.0,
                "surface_total_contracts": 0,
                "surface_put_heavy": False,
                "surface_call_heavy": False,
                "surface_neutral_pin_risk": False,
                "surface_long_vega_friendly": False,
                "surface_short_premium_friendly": False,
            }
        )
        return row

    row.update(
        {
            "strategy": _extract_strategy(candidate),
            "direction": _safe_str(_extract(candidate, "direction", _extract(candidate, "bias", "")), ""),
            "confidence": _safe_float(_extract(candidate, "confidence", 0.0), 0.0),
            "candidate_score": _safe_float(_extract(candidate, "candidate_score", _extract(candidate, "cand_score", 0.0)), 0.0),
            "composite_score": _safe_float(_extract(candidate, "composite_score", 0.0), 0.0),
            "priority": _safe_int(_extract(candidate, "priority", 0), 0),
            "signal": _safe_str(_extract(candidate, "signal", ""), ""),
            "regime": _safe_str(_extract(candidate, "regime", ""), ""),
            "sector": _safe_str(_extract(candidate, "sector", ""), ""),
            "price": _safe_float(_extract(candidate, "price", _extract(candidate, "current_price", 0.0)), 0.0),
            "notes": _safe_str(_extract(candidate, "notes", ""), ""),
            "surface_confidence_delta": _safe_float(_extract(candidate, "surface_confidence_delta", 0.0), 0.0),
            "surface_candidate_score_delta": _safe_float(_extract(candidate, "surface_candidate_score_delta", 0.0), 0.0),
            "surface_adjustment_notes": _safe_str(_extract(candidate, "surface_adjustment_notes", ""), ""),
            "surface_original_strategy": _safe_str(_extract(candidate, "surface_original_strategy", ""), ""),
            "surface_preferred_strategy": _safe_str(_extract(candidate, "surface_preferred_strategy", ""), ""),
            "surface_strategy_changed": bool(_extract(candidate, "surface_strategy_changed", False)),
            "surface_bias_strength": _safe_str(_extract(candidate, "surface_bias_strength", ""), ""),
            "surface_bias_rationale": _safe_str(_extract(candidate, "surface_bias_rationale", ""), ""),
            "surface_liquidity_quality": _safe_str(_extract(candidate, "surface_liquidity_quality", ""), ""),
            "surface_liquid_contract_ratio": _safe_float(_extract(candidate, "surface_liquid_contract_ratio", 0.0), 0.0),
            "surface_avg_spread_pct": _safe_float(_extract(candidate, "surface_avg_spread_pct", 0.0), 0.0),
            "surface_median_spread_pct": _safe_float(_extract(candidate, "surface_median_spread_pct", 0.0), 0.0),
            "surface_valid_quote_ratio": _safe_float(_extract(candidate, "surface_valid_quote_ratio", 0.0), 0.0),
            "surface_valid_spread_count": _safe_int(_extract(candidate, "surface_valid_spread_count", 0), 0),
            "surface_spread_sample_size": _safe_int(_extract(candidate, "surface_spread_sample_size", 0), 0),
            "surface_front_expiry_concentration": _safe_float(_extract(candidate, "surface_front_expiry_concentration", 0.0), 0.0),
            "surface_top_oi_strike_distance_pct": _safe_float(_extract(candidate, "surface_top_oi_strike_distance_pct", 0.0), 0.0),
            "surface_term_slope": _safe_float(_extract(candidate, "surface_term_slope", 0.0), 0.0),
            "surface_term_ratio": _safe_float(_extract(candidate, "surface_term_ratio", 0.0), 0.0),
            "surface_put_call_oi_ratio": _safe_float(_extract(candidate, "surface_put_call_oi_ratio", 0.0), 0.0),
            "surface_put_call_volume_ratio": _safe_float(_extract(candidate, "surface_put_call_volume_ratio", 0.0), 0.0),
            "surface_total_contracts": _safe_int(_extract(candidate, "surface_total_contracts", 0), 0),
            "surface_put_heavy": bool(_extract(candidate, "surface_put_heavy", False)),
            "surface_call_heavy": bool(_extract(candidate, "surface_call_heavy", False)),
            "surface_neutral_pin_risk": bool(_extract(candidate, "surface_neutral_pin_risk", False)),
            "surface_long_vega_friendly": bool(_extract(candidate, "surface_long_vega_friendly", False)),
            "surface_short_premium_friendly": bool(_extract(candidate, "surface_short_premium_friendly", False)),
        }
    )
    return row


def build_recommendation_feature_row(
    ticker: str,
    as_of_date: str,
    recommendation: Any,
    run_type: str = "nightly_scan",
    scan_mode: str = "standard",
) -> Dict[str, Any]:
    row = _common_fields(ticker, as_of_date, run_type, scan_mode)
    if recommendation is None:
        row.update(
            {
                "strategy": "",
                "direction": "",
                "confidence": 0.0,
                "candidate_score": 0.0,
                "composite_score": 0.0,
                "priority": 0,
                "price": 0.0,
                "notes": "",
                "sector": "",
                "surface_confidence_delta": 0.0,
                "surface_candidate_score_delta": 0.0,
                "surface_adjustment_notes": "",
                "surface_original_strategy": "",
                "surface_preferred_strategy": "",
                "surface_strategy_changed": False,
                "surface_bias_strength": "",
                "surface_bias_rationale": "",
                "surface_liquidity_quality": "",
                "surface_liquid_contract_ratio": 0.0,
                "surface_avg_spread_pct": 0.0,
                "surface_median_spread_pct": 0.0,
                "surface_valid_quote_ratio": 0.0,
                "surface_valid_spread_count": 0,
                "surface_spread_sample_size": 0,
                "surface_front_expiry_concentration": 0.0,
                "surface_top_oi_strike_distance_pct": 0.0,
                "surface_term_slope": 0.0,
                "surface_term_ratio": 0.0,
                "surface_put_call_oi_ratio": 0.0,
                "surface_put_call_volume_ratio": 0.0,
                "surface_total_contracts": 0,
                "surface_put_heavy": False,
                "surface_call_heavy": False,
                "surface_neutral_pin_risk": False,
                "surface_long_vega_friendly": False,
                "surface_short_premium_friendly": False,
            }
        )
        return row

    row.update(
        {
            "strategy": _extract_strategy(recommendation),
            "direction": _safe_str(_extract(recommendation, "direction", _extract(recommendation, "bias", "")), ""),
            "confidence": _safe_float(_extract(recommendation, "confidence", 0.0), 0.0),
            "candidate_score": _safe_float(_extract(recommendation, "candidate_score", _extract(recommendation, "cand_score", 0.0)), 0.0),
            "composite_score": _safe_float(_extract(recommendation, "composite_score", 0.0), 0.0),
            "priority": _safe_int(_extract(recommendation, "priority", 0), 0),
            "price": _safe_float(_extract(recommendation, "price", _extract(recommendation, "current_price", 0.0)), 0.0),
            "notes": _safe_str(_extract(recommendation, "notes", ""), ""),
            "sector": _safe_str(_extract(recommendation, "sector", ""), ""),
            "surface_confidence_delta": _safe_float(_extract(recommendation, "surface_confidence_delta", 0.0), 0.0),
            "surface_candidate_score_delta": _safe_float(_extract(recommendation, "surface_candidate_score_delta", 0.0), 0.0),
            "surface_adjustment_notes": _safe_str(_extract(recommendation, "surface_adjustment_notes", ""), ""),
            "surface_original_strategy": _safe_str(_extract(recommendation, "surface_original_strategy", ""), ""),
            "surface_preferred_strategy": _safe_str(_extract(recommendation, "surface_preferred_strategy", ""), ""),
            "surface_strategy_changed": bool(_extract(recommendation, "surface_strategy_changed", False)),
            "surface_bias_strength": _safe_str(_extract(recommendation, "surface_bias_strength", ""), ""),
            "surface_bias_rationale": _safe_str(_extract(recommendation, "surface_bias_rationale", ""), ""),
            "surface_liquidity_quality": _safe_str(_extract(recommendation, "surface_liquidity_quality", ""), ""),
            "surface_quote_availability": _safe_str(_extract(recommendation, "surface_quote_availability", ""), ""),
            "surface_liquid_contract_ratio": _safe_float(_extract(recommendation, "surface_liquid_contract_ratio", 0.0), 0.0),
            "surface_avg_spread_pct": _safe_float(_extract(recommendation, "surface_avg_spread_pct", 0.0), 0.0),
            "surface_median_spread_pct": _safe_float(_extract(recommendation, "surface_median_spread_pct", 0.0), 0.0),
            "surface_valid_quote_ratio": _safe_float(_extract(recommendation, "surface_valid_quote_ratio", 0.0), 0.0),
            "surface_valid_spread_count": _safe_int(_extract(recommendation, "surface_valid_spread_count", 0), 0),
            "surface_spread_sample_size": _safe_int(_extract(recommendation, "surface_spread_sample_size", 0), 0),
            "surface_front_expiry_concentration": _safe_float(_extract(recommendation, "surface_front_expiry_concentration", 0.0), 0.0),
            "surface_top_oi_strike_distance_pct": _safe_float(_extract(recommendation, "surface_top_oi_strike_distance_pct", 0.0), 0.0),
            "surface_term_slope": _safe_float(_extract(recommendation, "surface_term_slope", 0.0), 0.0),
            "surface_term_ratio": _safe_float(_extract(recommendation, "surface_term_ratio", 0.0), 0.0),
            "surface_put_call_oi_ratio": _safe_float(_extract(recommendation, "surface_put_call_oi_ratio", 0.0), 0.0),
            "surface_put_call_volume_ratio": _safe_float(_extract(recommendation, "surface_put_call_volume_ratio", 0.0), 0.0),
            "surface_total_contracts": _safe_int(_extract(recommendation, "surface_total_contracts", 0), 0),
            "surface_put_heavy": bool(_extract(recommendation, "surface_put_heavy", False)),
            "surface_call_heavy": bool(_extract(recommendation, "surface_call_heavy", False)),
            "surface_neutral_pin_risk": bool(_extract(recommendation, "surface_neutral_pin_risk", False)),
            "surface_long_vega_friendly": bool(_extract(recommendation, "surface_long_vega_friendly", False)),
            "surface_short_premium_friendly": bool(_extract(recommendation, "surface_short_premium_friendly", False)),
        }
    )
    return row
