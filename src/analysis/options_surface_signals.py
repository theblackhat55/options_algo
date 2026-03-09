from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        return str(value)
    except Exception:
        return default


@dataclass
class SurfaceAdjustment:
    ticker: str
    liquidity_ok: bool
    liquidity_quality: str
    crowded_front: bool
    put_heavy: bool
    call_heavy: bool
    neutral_pin_risk: bool
    long_vega_friendly: bool
    short_premium_friendly: bool
    confidence_delta: float
    candidate_score_delta: float
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_surface_adjustment(
    ticker: str,
    direction: str,
    strategy: str,
    surface_row: Dict[str, Any] | None,
) -> SurfaceAdjustment:
    row = surface_row or {}

    liq_ratio = _safe_float(row.get("surface_liquid_contract_ratio", 0.0), 0.0)
    avg_spread = _safe_float(row.get("surface_avg_spread_pct", 0.0), 0.0)
    median_spread = _safe_float(row.get("surface_median_spread_pct", 0.0), 0.0)
    term_slope = _safe_float(row.get("surface_term_slope", 0.0), 0.0)
    term_ratio = _safe_float(row.get("surface_term_ratio", 0.0), 0.0)
    pc_oi = _safe_float(row.get("surface_put_call_oi_ratio", 0.0), 0.0)
    pc_vol = _safe_float(row.get("surface_put_call_volume_ratio", 0.0), 0.0)
    skew_proxy = _safe_float(row.get("surface_skew_proxy", 0.0), 0.0)
    top_oi_dist = abs(_safe_float(row.get("surface_top_oi_strike_distance_pct", 999.0), 999.0))
    front_conc = _safe_float(row.get("surface_front_expiry_concentration", 0.0), 0.0)
    total_contracts = _safe_float(row.get("surface_total_contracts", 0.0), 0.0)

    direction = _safe_str(direction).upper()
    strategy_u = _safe_str(strategy).upper()

    long_vega_keywords = ("LONG_CALL", "LONG_PUT", "LONG_STRADDLE", "LONG_STRANGLE", "LONG_BUTTERFLY", "BUTTERFLY")
    short_premium_keywords = ("IRON_CONDOR", "CREDIT", "SHORT", "VERTICAL", "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD")

    # Liquidity quality bucket: use one bucket instead of stacking many penalties
    if total_contracts <= 0:
        liquidity_quality = "UNKNOWN"
    elif liq_ratio >= 0.45 and avg_spread <= 0.08 and median_spread <= 0.06:
        liquidity_quality = "STRONG"
    elif liq_ratio >= 0.25 and avg_spread <= 0.15 and median_spread <= 0.10:
        liquidity_quality = "OK"
    elif liq_ratio >= 0.12 and avg_spread <= 0.25 and median_spread <= 0.18:
        liquidity_quality = "WEAK"
    else:
        liquidity_quality = "POOR"

    liquidity_ok = liquidity_quality in {"STRONG", "OK"}
    crowded_front = front_conc >= 0.60
    put_heavy = pc_oi >= 1.25 or pc_vol >= 1.25
    call_heavy = (0.0 < pc_oi <= 0.80) or (0.0 < pc_vol <= 0.80)
    neutral_pin_risk = top_oi_dist <= 0.02 and total_contracts > 0
    long_vega_friendly = term_slope > 0.0 or term_ratio > 1.0
    short_premium_friendly = term_slope < 0.0 or (0.0 < term_ratio < 1.0)

    confidence_delta = 0.0
    candidate_score_delta = 0.0
    notes = []

    # Liquidity bucket penalties/boosts: only one main bucket applied
    if liquidity_quality == "STRONG":
        candidate_score_delta += 0.02
        notes.append("strong_liquidity")
    elif liquidity_quality == "OK":
        candidate_score_delta += 0.01
        notes.append("acceptable_liquidity")
    elif liquidity_quality == "WEAK":
        confidence_delta -= 0.02
        candidate_score_delta -= 0.02
        notes.append("weak_liquidity")
    elif liquidity_quality == "POOR":
        confidence_delta -= 0.04
        candidate_score_delta -= 0.04
        notes.append("poor_liquidity")

    # Directional positioning: gentle adjustments
    if direction == "BULLISH":
        if call_heavy:
            confidence_delta += 0.02
            notes.append("bullish_call_positioning")
        elif put_heavy:
            confidence_delta -= 0.02
            notes.append("bearish_put_positioning_against_bull")
    elif direction == "BEARISH":
        if put_heavy:
            confidence_delta += 0.02
            notes.append("bearish_put_positioning")
        elif call_heavy:
            confidence_delta -= 0.02
            notes.append("bullish_call_positioning_against_bear")

    # Strategy-aware term structure
    if any(k in strategy_u for k in long_vega_keywords):
        if long_vega_friendly:
            confidence_delta += 0.02
            candidate_score_delta += 0.01
            notes.append("term_structure_supports_long_vega")
        elif short_premium_friendly:
            confidence_delta -= 0.02
            notes.append("term_structure_unfavorable_long_vega")

    if any(k in strategy_u for k in short_premium_keywords):
        if short_premium_friendly:
            confidence_delta += 0.02
            candidate_score_delta += 0.01
            notes.append("term_structure_supports_short_premium")
        elif long_vega_friendly:
            confidence_delta -= 0.02
            notes.append("term_structure_unfavorable_short_premium")

    # Neutral pinning / concentration: make more selective
    if neutral_pin_risk and crowded_front and direction == "NEUTRAL" and liquidity_ok:
        confidence_delta += 0.02
        candidate_score_delta += 0.01
        notes.append("neutral_pin_support")
    elif neutral_pin_risk and direction in {"BULLISH", "BEARISH"}:
        confidence_delta -= 0.01
        notes.append("pin_risk_against_directional")

    if crowded_front and not liquidity_ok:
        candidate_score_delta -= 0.01
        notes.append("crowded_front_with_weak_liquidity")

    # Skew proxy
    if skew_proxy > 0.05 and direction == "BEARISH":
        confidence_delta += 0.01
        notes.append("put_skew_supports_bearish")
    elif skew_proxy > 0.05 and direction == "BULLISH":
        confidence_delta -= 0.01
        notes.append("put_skew_against_bullish")

    confidence_delta = max(-0.08, min(0.08, confidence_delta))
    candidate_score_delta = max(-0.08, min(0.08, candidate_score_delta))

    return SurfaceAdjustment(
        ticker=ticker,
        liquidity_ok=liquidity_ok,
        liquidity_quality=liquidity_quality,
        crowded_front=crowded_front,
        put_heavy=put_heavy,
        call_heavy=call_heavy,
        neutral_pin_risk=neutral_pin_risk,
        long_vega_friendly=long_vega_friendly,
        short_premium_friendly=short_premium_friendly,
        confidence_delta=confidence_delta,
        candidate_score_delta=candidate_score_delta,
        notes=";".join(notes),
    )
