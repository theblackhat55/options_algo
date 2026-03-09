from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return default
        return out
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        out = str(value)
        return out if out else default
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _extract(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class SurfaceAdjustment:
    confidence_delta: float = 0.0
    candidate_score_delta: float = 0.0
    adjustment_notes: List[str] = field(default_factory=list)

    liquidity_quality: str = ""
    quote_availability: str = ""
    quote_source: str = ""

    put_heavy: bool = False
    call_heavy: bool = False
    neutral_pin_risk: bool = False
    long_vega_friendly: bool = False
    short_premium_friendly: bool = False

    bias_strength: float = 0.0
    bias_rationale: str = ""
    preferred_strategy: str = ""
    strategy_changed: bool = False


def analyze_surface_adjustment(
    ticker: str = "",
    direction: str = "",
    strategy: str = "",
    surface_row: Any = None,
    row: Any = None,
) -> SurfaceAdjustment:
    """
    Backward-compatible API.

    Existing callers use:
        analyze_surface_adjustment(
            ticker=...,
            direction=...,
            strategy=...,
            surface_row=...,
        )

    Simpler callers may pass:
        analyze_surface_adjustment(row=obj)
        analyze_surface_adjustment(surface_row=obj)
    """
    src = surface_row if surface_row is not None else row
    out = SurfaceAdjustment()

    if src is None:
        out.quote_availability = "UNKNOWN"
        out.liquidity_quality = "UNKNOWN"
        return out

    liquid_contract_ratio = _clamp(_safe_float(_extract(src, "surface_liquid_contract_ratio", 0.0), 0.0), 0.0, 1.0)
    valid_quote_ratio = _clamp(_safe_float(_extract(src, "surface_valid_quote_ratio", 0.0), 0.0), 0.0, 1.0)
    avg_spread_pct = _safe_float(_extract(src, "surface_avg_spread_pct", 0.0), 0.0)
    median_spread_pct = _safe_float(_extract(src, "surface_median_spread_pct", 0.0), 0.0)
    spread_sample_size = _safe_int(_extract(src, "surface_spread_sample_size", 0), 0)
    valid_spread_count = _safe_int(_extract(src, "surface_valid_spread_count", 0), 0)
    total_contracts = _safe_int(_extract(src, "surface_total_contracts", 0), 0)

    put_call_oi_ratio = _safe_float(_extract(src, "surface_put_call_oi_ratio", 1.0), 1.0)
    put_call_volume_ratio = _safe_float(_extract(src, "surface_put_call_volume_ratio", 1.0), 1.0)
    front_expiry_concentration = _clamp(_safe_float(_extract(src, "surface_front_expiry_concentration", 0.0), 0.0), 0.0, 1.0)

    out.quote_source = _safe_str(_extract(src, "surface_quote_source", ""), "")

    # quote availability
    if spread_sample_size <= 0 and total_contracts <= 0:
        out.quote_availability = "UNKNOWN"
    elif spread_sample_size > 0 and valid_spread_count == 0 and valid_quote_ratio == 0:
        out.quote_availability = "NONE"
    elif valid_quote_ratio < 0.25:
        out.quote_availability = "SPARSE"
    else:
        out.quote_availability = "AVAILABLE"

    # liquidity quality
    if out.quote_availability == "UNKNOWN":
        out.liquidity_quality = "UNKNOWN"
    elif out.quote_availability == "NONE":
        out.liquidity_quality = "UNKNOWN"
        out.confidence_delta -= 0.01
        out.candidate_score_delta -= 0.01
        out.adjustment_notes.append("quote_unavailable")
    elif liquid_contract_ratio >= 0.35 and valid_quote_ratio >= 0.35 and median_spread_pct > 0 and median_spread_pct <= 0.08:
        out.liquidity_quality = "OK"
    elif out.quote_availability == "SPARSE":
        out.liquidity_quality = "WEAK"
        out.confidence_delta -= 0.02
        out.candidate_score_delta -= 0.02
        out.adjustment_notes.append("sparse_quotes")
    elif liquid_contract_ratio >= 0.15 and valid_quote_ratio >= 0.15:
        out.liquidity_quality = "WEAK"
        out.confidence_delta -= 0.02
        out.candidate_score_delta -= 0.02
        out.adjustment_notes.append("weak_liquidity")
    else:
        out.liquidity_quality = "POOR"
        out.confidence_delta -= 0.04
        out.candidate_score_delta -= 0.04
        out.adjustment_notes.append("poor_liquidity")

    # directional tags
    if put_call_oi_ratio >= 1.25 or put_call_volume_ratio >= 1.25:
        out.put_heavy = True
    elif put_call_oi_ratio <= 0.80 or put_call_volume_ratio <= 0.80:
        out.call_heavy = True

    if front_expiry_concentration >= 0.20:
        out.neutral_pin_risk = True

    if median_spread_pct > 0 and median_spread_pct <= 0.05:
        out.short_premium_friendly = True
    if valid_quote_ratio >= 0.25 and total_contracts >= 100:
        out.long_vega_friendly = True

    # soft bias scoring
    strength = 0.0
    rationale: List[str] = []

    if out.put_heavy:
        strength -= 0.10
        rationale.append("put_heavy")
    if out.call_heavy:
        strength += 0.10
        rationale.append("call_heavy")
    if out.neutral_pin_risk:
        rationale.append("neutral_pin_risk")

    out.bias_strength = _clamp(strength, -0.20, 0.20)
    out.bias_rationale = ";".join(rationale)

    preferred = ""
    strategy_norm = _safe_str(strategy, "").upper()
    if out.short_premium_friendly and out.liquidity_quality in {"OK", "WEAK"}:
        preferred = "SHORT_PREMIUM"
    elif out.long_vega_friendly and out.quote_availability in {"SPARSE", "AVAILABLE"}:
        preferred = "LONG_VEGA"
    out.preferred_strategy = preferred

    if preferred and strategy_norm and preferred != strategy_norm:
        out.strategy_changed = True

    out.confidence_delta = _clamp(out.confidence_delta, -0.10, 0.10)
    out.candidate_score_delta = _clamp(out.candidate_score_delta, -0.10, 0.10)

    deduped: List[str] = []
    seen = set()
    for note in out.adjustment_notes:
        n = _safe_str(note, "").strip()
        if n and n not in seen:
            seen.add(n)
            deduped.append(n)
    out.adjustment_notes = deduped

    return out


__all__ = ["SurfaceAdjustment", "analyze_surface_adjustment"]
