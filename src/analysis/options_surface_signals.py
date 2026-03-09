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

    liquidity_ok = liq_ratio >= 0.25 and avg_spread <= 0.15 and median_spread <= 0.10
    crowded_front = front_conc >= 0.60
    put_heavy = pc_oi >= 1.25 or pc_vol >= 1.25
    call_heavy = (0.0 < pc_oi <= 0.80) or (0.0 < pc_vol <= 0.80)
    neutral_pin_risk = top_oi_dist <= 0.02 and total_contracts > 0
    long_vega_friendly = term_slope > 0.0 or term_ratio > 1.0
    short_premium_friendly = term_slope < 0.0 or (0.0 < term_ratio < 1.0)

    confidence_delta = 0.0
    candidate_score_delta = 0.0
    notes = []

    # Liquidity rules
    if total_contracts > 0:
        if liq_ratio < 0.25:
            confidence_delta -= 0.07
            candidate_score_delta -= 0.05
            notes.append("poor_liquidity_ratio")
        if avg_spread > 0.15:
            confidence_delta -= 0.05
            candidate_score_delta -= 0.03
            notes.append("wide_avg_spread")
        if median_spread > 0.10:
            candidate_score_delta -= 0.03
            notes.append("wide_median_spread")

    # Directional positioning
    if direction == "BULLISH":
        if call_heavy:
            confidence_delta += 0.03
            notes.append("bullish_call_positioning")
        elif put_heavy:
            confidence_delta -= 0.03
            notes.append("bearish_put_positioning_against_bull")
    elif direction == "BEARISH":
        if put_heavy:
            confidence_delta += 0.03
            notes.append("bearish_put_positioning")
        elif call_heavy:
            confidence_delta -= 0.03
            notes.append("bullish_call_positioning_against_bear")

    # Strategy-aware term structure
    long_vega_keywords = ("LONG_CALL", "LONG_PUT", "LONG_STRADDLE", "LONG_STRANGLE", "BUTTERFLY")
    short_premium_keywords = ("IRON_CONDOR", "CREDIT", "SHORT", "VERTICAL")

    if any(k in strategy_u for k in long_vega_keywords):
        if long_vega_friendly:
            confidence_delta += 0.02
            candidate_score_delta += 0.02
            notes.append("term_structure_supports_long_vega")
        elif short_premium_friendly:
            confidence_delta -= 0.02
            notes.append("term_structure_unfavorable_long_vega")

    if any(k in strategy_u for k in short_premium_keywords):
        if short_premium_friendly:
            confidence_delta += 0.02
            candidate_score_delta += 0.02
            notes.append("term_structure_supports_short_premium")
        elif long_vega_friendly:
            confidence_delta -= 0.02
            notes.append("term_structure_unfavorable_short_premium")

    # Neutral pinning / crowding
    if neutral_pin_risk and direction == "NEUTRAL":
        confidence_delta += 0.02
        notes.append("neutral_pin_support")
    elif neutral_pin_risk and direction in {"BULLISH", "BEARISH"}:
        confidence_delta -= 0.02
        notes.append("pin_risk_against_directional")

    if crowded_front:
        candidate_score_delta -= 0.01
        notes.append("crowded_front_expiry")

    # Skew proxy mild context
    if skew_proxy > 0.05 and direction == "BEARISH":
        confidence_delta += 0.01
        notes.append("put_skew_supports_bearish")
    elif skew_proxy > 0.05 and direction == "BULLISH":
        confidence_delta -= 0.01
        notes.append("put_skew_against_bullish")

    # Clamp
    confidence_delta = max(-0.12, min(0.12, confidence_delta))
    candidate_score_delta = max(-0.10, min(0.10, candidate_score_delta))

    return SurfaceAdjustment(
        ticker=ticker,
        liquidity_ok=liquidity_ok,
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
