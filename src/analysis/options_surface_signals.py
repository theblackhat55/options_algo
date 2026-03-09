from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    try:
        s = str(value)
        return s if s else default
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class SurfaceAdjustment:
    confidence_delta: float = 0.0
    candidate_score_delta: float = 0.0
    notes: list[str] = field(default_factory=list)

    liquidity_quality: str = "UNKNOWN"
    quote_availability: str = "UNKNOWN"

    put_heavy: bool = False
    call_heavy: bool = False
    neutral_pin_risk: bool = False
    long_vega_friendly: bool = False
    short_premium_friendly: bool = False

    valid_quote_ratio: float = 0.0
    valid_spread_count: int = 0
    spread_sample_size: int = 0
    liquid_contract_ratio: float = 0.0
    avg_spread_pct: float = 0.0
    median_spread_pct: float = 0.0
    front_expiry_concentration: float = 0.0
    top_oi_strike_distance_pct: float = 0.0
    term_slope: float = 0.0
    term_ratio: float = 0.0
    put_call_oi_ratio: float = 0.0
    put_call_volume_ratio: float = 0.0
    total_contracts: int = 0


def analyze_surface_adjustment(
    ticker: str,
    direction: str,
    strategy: str,
    surface_row: dict[str, Any] | None,
) -> SurfaceAdjustment:
    row = surface_row or {}

    liquid_contract_ratio = _clamp(_safe_float(row.get("surface_liquid_contract_ratio", 0.0), 0.0), 0.0, 1.0)
    avg_spread_pct = _safe_float(row.get("surface_avg_spread_pct", 0.0), 0.0)
    median_spread_pct = _safe_float(row.get("surface_median_spread_pct", 0.0), 0.0)

    valid_quote_ratio = _clamp(_safe_float(row.get("surface_valid_quote_ratio", 0.0), 0.0), 0.0, 1.0)
    valid_spread_count = max(0, _safe_int(row.get("surface_valid_spread_count", 0), 0))
    spread_sample_size = max(0, _safe_int(row.get("surface_spread_sample_size", 0), 0))

    front_expiry_concentration = _clamp(_safe_float(row.get("surface_front_expiry_concentration", 0.0), 0.0), 0.0, 1.0)
    top_oi_strike_distance_pct = max(0.0, _safe_float(row.get("surface_top_oi_strike_distance_pct", 0.0), 0.0))
    term_slope = _safe_float(row.get("surface_term_slope", 0.0), 0.0)
    term_ratio = _safe_float(row.get("surface_term_ratio", 0.0), 0.0)
    put_call_oi_ratio = max(0.0, _safe_float(row.get("surface_put_call_oi_ratio", 0.0), 0.0))
    put_call_volume_ratio = max(0.0, _safe_float(row.get("surface_put_call_volume_ratio", 0.0), 0.0))
    total_contracts = max(0, _safe_int(row.get("surface_total_contracts", 0), 0))

    adj = SurfaceAdjustment(
        liquidity_quality="UNKNOWN",
        quote_availability="UNKNOWN",
        valid_quote_ratio=valid_quote_ratio,
        valid_spread_count=valid_spread_count,
        spread_sample_size=spread_sample_size,
        liquid_contract_ratio=liquid_contract_ratio,
        avg_spread_pct=avg_spread_pct,
        median_spread_pct=median_spread_pct,
        front_expiry_concentration=front_expiry_concentration,
        top_oi_strike_distance_pct=top_oi_strike_distance_pct,
        term_slope=term_slope,
        term_ratio=term_ratio,
        put_call_oi_ratio=put_call_oi_ratio,
        put_call_volume_ratio=put_call_volume_ratio,
        total_contracts=total_contracts,
    )

    direction_u = _safe_str(direction, "NEUTRAL").upper()
    strategy_u = _safe_str(strategy, "WATCHLIST").upper()

    adj.put_heavy = put_call_oi_ratio >= 1.25 or put_call_volume_ratio >= 1.25
    adj.call_heavy = (
        (put_call_oi_ratio > 0 and put_call_oi_ratio <= 0.80)
        or (put_call_volume_ratio > 0 and put_call_volume_ratio <= 0.80)
    )
    adj.neutral_pin_risk = (
        direction_u == "NEUTRAL"
        and front_expiry_concentration >= 0.20
        and top_oi_strike_distance_pct <= 0.02
    )
    adj.long_vega_friendly = term_slope >= 0.02 or term_ratio >= 1.05
    adj.short_premium_friendly = term_slope <= -0.02 or (0 < term_ratio <= 0.97)

    quote_unavailable = (
        spread_sample_size > 0
        and valid_quote_ratio == 0.0
        and valid_spread_count == 0
    )

    sparse_quotes = (
        spread_sample_size > 0
        and valid_quote_ratio > 0.0
        and valid_quote_ratio < 0.10
    )

    if spread_sample_size <= 0:
        adj.quote_availability = "UNKNOWN"
    elif quote_unavailable:
        adj.quote_availability = "NONE"
    elif sparse_quotes:
        adj.quote_availability = "SPARSE"
    else:
        adj.quote_availability = "AVAILABLE"

    if total_contracts <= 0:
        adj.liquidity_quality = "UNKNOWN"
    elif quote_unavailable:
        adj.liquidity_quality = "UNKNOWN"
        adj.confidence_delta -= 0.01
        adj.candidate_score_delta -= 0.01
        adj.notes.append("quote_unavailable")
    elif sparse_quotes:
        adj.liquidity_quality = "WEAK"
        adj.confidence_delta -= 0.02
        adj.candidate_score_delta -= 0.02
        adj.notes.append("sparse_quotes")
    elif liquid_contract_ratio >= 0.35 and avg_spread_pct <= 0.08 and median_spread_pct <= 0.06:
        adj.liquidity_quality = "OK"
    elif liquid_contract_ratio >= 0.15 and avg_spread_pct <= 0.18:
        adj.liquidity_quality = "WEAK"
        adj.confidence_delta -= 0.01
        adj.candidate_score_delta -= 0.01
        adj.notes.append("wide_spreads")
    else:
        adj.liquidity_quality = "POOR"
        adj.confidence_delta -= 0.04
        adj.candidate_score_delta -= 0.04
        adj.notes.append("poor_liquidity")

    if adj.neutral_pin_risk and adj.liquidity_quality in {"OK", "WEAK"}:
        adj.candidate_score_delta += 0.01
        adj.notes.append("neutral_pin_risk")

    if direction_u == "BULLISH" and adj.call_heavy:
        adj.candidate_score_delta += 0.01
        adj.notes.append("call_flow_support")
    elif direction_u == "BEARISH" and adj.put_heavy:
        adj.candidate_score_delta += 0.01
        adj.notes.append("put_flow_support")

    if strategy_u in {"CALENDAR", "DIAGONAL", "DEBIT_SPREAD"} and adj.long_vega_friendly:
        adj.candidate_score_delta += 0.01
        adj.notes.append("term_structure_supportive")

    if strategy_u in {"IRON_CONDOR", "CREDIT_SPREAD"} and adj.short_premium_friendly:
        adj.candidate_score_delta += 0.01
        adj.notes.append("term_structure_short_premium_friendly")

    adj.confidence_delta = _clamp(adj.confidence_delta, -0.08, 0.08)
    adj.candidate_score_delta = _clamp(adj.candidate_score_delta, -0.08, 0.08)

    deduped: list[str] = []
    seen: set[str] = set()
    for note in adj.notes:
        if note and note not in seen:
            seen.add(note)
            deduped.append(note)
    adj.notes = deduped

    return adj
