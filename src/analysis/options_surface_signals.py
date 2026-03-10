from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

def _preferred_strategy_from_surface(
    liquidity_quality: str | None,
    current_strategy: str | None,
    adjustment_notes: list[str] | None = None,
) -> tuple[str | None, bool]:
    """
    Decide whether surface conditions justify changing strategy.

    Routing rules:
    - POOR / WEAK / UNKNOWN / missing => never promote
    - FAIR / OK + strong_neutral_pin_setup or strong_short_premium_setup
      => IRON_CONDOR
    - FAIR / OK + long_vega_friendly
      => CALENDAR
    - otherwise keep current strategy unchanged

    Priority:
    1. IRON_CONDOR signals
    2. CALENDAR signals
    3. keep existing strategy
    """
    quality = (liquidity_quality or "UNKNOWN").upper()
    strategy = current_strategy
    notes = {str(n).strip() for n in (adjustment_notes or []) if str(n).strip()}

    condor_supportive = bool(
        {"strong_neutral_pin_setup", "strong_short_premium_setup"} & notes
    )
    calendar_supportive = "long_vega_friendly" in notes

    if quality in {"POOR", "WEAK", "UNKNOWN"}:
        return strategy, False

    if quality in {"FAIR", "OK"}:
        if condor_supportive:
            if strategy != "IRON_CONDOR":
                return "IRON_CONDOR", True
            return strategy, False

        if calendar_supportive:
            if strategy != "CALENDAR":
                return "CALENDAR", True
            return strategy, False

    return strategy, False



def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _extract(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
    except Exception:
        pass
    try:
        return getattr(obj, key, default)
    except Exception:
        return default


def _dedupe_notes(notes: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for note in notes:
        n = _safe_str(note, "")
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


@dataclass
class SurfaceAdjustment:
    confidence_delta: float = 0.0
    candidate_score_delta: float = 0.0
    preferred_strategy: str = ""
    strategy_changed: bool = False

    bias_strength: str = "NONE"
    bias_rationale: str = ""

    liquidity_quality: str = "UNKNOWN"
    quote_availability: str = "NONE"
    quote_source: str = ""

    put_heavy: bool = False
    call_heavy: bool = False
    neutral_pin_risk: bool = False
    long_vega_friendly: bool = False
    short_premium_friendly: bool = False

    adjustment_notes: list[str] = field(default_factory=list)


def analyze_surface_adjustment(
    ticker: str | None = None,
    direction: str | None = None,
    strategy: str | None = None,
    surface_row: Any | None = None,
    row: Any | None = None,
) -> SurfaceAdjustment:
    """
    Backward-compatible surface adjustment analyzer.

    Accepts either `surface_row=` or legacy `row=` and returns a SurfaceAdjustment
    with quote availability, liquidity quality, strategy preferences, bias flags,
    and additive score deltas.
    """
    surface = surface_row if surface_row is not None else row
    if surface is None:
        return SurfaceAdjustment(
            confidence_delta=-0.01,
            candidate_score_delta=-0.01,
            preferred_strategy=_safe_str(strategy, ""),
            strategy_changed=False,
            liquidity_quality="UNKNOWN",
            quote_availability="NONE",
            quote_source="",
            adjustment_notes=["quote_unavailable"],
        )

    direction_s = _safe_str(direction, "").upper()
    strategy_s = _safe_str(strategy, "")
    preferred_strategy = strategy_s

    valid_quote_ratio = _safe_float(_extract(surface, "surface_valid_quote_ratio", 0.0), 0.0)
    median_spread_pct = _safe_float(_extract(surface, "surface_median_spread_pct", 0.0), 0.0)
    avg_spread_pct = _safe_float(_extract(surface, "surface_avg_spread_pct", 0.0), 0.0)
    liquid_contract_ratio = _safe_float(_extract(surface, "surface_liquid_contract_ratio", 0.0), 0.0)
    quote_source = _safe_str(_extract(surface, "surface_quote_source", ""), "")
    valid_spread_count = _safe_int(_extract(surface, "surface_valid_spread_count", 0), 0)
    spread_sample_size = _safe_int(_extract(surface, "surface_spread_sample_size", 0), 0)

    put_call_oi_ratio = _safe_float(_extract(surface, "surface_put_call_oi_ratio", 0.0), 0.0)
    put_call_volume_ratio = _safe_float(_extract(surface, "surface_put_call_volume_ratio", 0.0), 0.0)
    top_oi_strike_distance_pct = _safe_float(_extract(surface, "surface_top_oi_strike_distance_pct", 0.0), 0.0)
    term_slope = _safe_float(_extract(surface, "surface_term_slope", 0.0), 0.0)
    term_ratio = _safe_float(_extract(surface, "surface_term_ratio", 0.0), 0.0)

    adjustment_notes: list[str] = []
    confidence_delta = 0.0
    candidate_score_delta = 0.0

    # Quote availability
    if valid_quote_ratio <= 0.0 or valid_spread_count <= 0:
        quote_availability = "NONE"
    elif valid_quote_ratio < 0.10:
        quote_availability = "SPARSE"
    elif valid_quote_ratio < 0.25:
        quote_availability = "PARTIAL"
    else:
        quote_availability = "AVAILABLE"

    # Liquidity quality / penalty mapping
    if quote_availability == "NONE":
        liquidity_quality = "UNKNOWN"
        confidence_delta = -0.01
        candidate_score_delta = -0.01
        adjustment_notes.append("quote_unavailable")

    elif median_spread_pct > 0.18 and liquid_contract_ratio < 0.10:
        liquidity_quality = "POOR"
        confidence_delta = -0.04
        candidate_score_delta = -0.04
        adjustment_notes.append("poor_liquidity")

    elif (
        valid_quote_ratio >= 0.10
        and median_spread_pct <= 0.06
        and liquid_contract_ratio >= 0.60
    ):
        liquidity_quality = "OK"
        confidence_delta = 0.01
        candidate_score_delta = 0.01
        adjustment_notes.append("good_liquidity")

    elif (
        valid_quote_ratio > 0.0
        and median_spread_pct <= 0.08
        and liquid_contract_ratio >= 0.50
    ):
        liquidity_quality = "FAIR"
        confidence_delta = 0.00
        candidate_score_delta = 0.00
        adjustment_notes.append("fair_liquidity")

    else:
        liquidity_quality = "WEAK"
        confidence_delta = -0.02
        candidate_score_delta = -0.02
        adjustment_notes.append("sparse_quotes")

    # Bias flags
    put_heavy = bool(put_call_oi_ratio >= 1.20 or put_call_volume_ratio >= 1.20)
    call_heavy = bool(
        (put_call_oi_ratio > 0.0 and put_call_oi_ratio <= 0.80)
        or (put_call_volume_ratio > 0.0 and put_call_volume_ratio <= 0.80)
    )
    neutral_pin_risk = bool(top_oi_strike_distance_pct <= 0.02 if top_oi_strike_distance_pct > 0 else False)
    long_vega_friendly = bool(term_slope > 0.0 or term_ratio > 1.05)
    short_premium_friendly = bool(term_slope < 0.0 or (term_ratio > 0.0 and term_ratio < 0.97))

    # Bias interpretation
    bias_strength = "NONE"
    bias_rationale = ""

    if put_heavy and not call_heavy:
        bias_strength = "MODERATE"
        bias_rationale = "put_skew_or_put_demand"
        adjustment_notes.append("put_heavy_surface")
        if direction_s == "BULLISH":
            confidence_delta -= 0.01
            candidate_score_delta -= 0.01
        elif direction_s == "BEARISH":
            confidence_delta += 0.01
            candidate_score_delta += 0.01

    elif call_heavy and not put_heavy:
        bias_strength = "MODERATE"
        bias_rationale = "call_skew_or_call_demand"
        adjustment_notes.append("call_heavy_surface")
        if direction_s == "BEARISH":
            confidence_delta -= 0.01
            candidate_score_delta -= 0.01
        elif direction_s == "BULLISH":
            confidence_delta += 0.01
            candidate_score_delta += 0.01

    if neutral_pin_risk:
        adjustment_notes.append("pin_risk")

    # Strategy-routing notes
    balanced_surface = not put_heavy and not call_heavy

    if liquidity_quality in {"FAIR", "OK"} and neutral_pin_risk:
        adjustment_notes.append("strong_neutral_pin_setup")

    if liquidity_quality in {"FAIR", "OK"} and short_premium_friendly and balanced_surface:
        adjustment_notes.append("strong_short_premium_setup")

    if (
        liquidity_quality in {"FAIR", "OK"}
        and long_vega_friendly
        and not neutral_pin_risk
        and not (short_premium_friendly and balanced_surface)
    ):
        adjustment_notes.append("long_vega_friendly")

    # Explicit strategy routing
    preferred_strategy, strategy_changed = _preferred_strategy_from_surface(
        liquidity_quality=liquidity_quality,
        current_strategy=strategy_s,
        adjustment_notes=adjustment_notes,
    )

    return SurfaceAdjustment(
        confidence_delta=_clamp(confidence_delta, -0.10, 0.10),
        candidate_score_delta=_clamp(candidate_score_delta, -0.10, 0.10),
        preferred_strategy=preferred_strategy,
        strategy_changed=strategy_changed,
        bias_strength=bias_strength,
        bias_rationale=bias_rationale,
        liquidity_quality=liquidity_quality,
        quote_availability=quote_availability,
        quote_source=quote_source,
        put_heavy=put_heavy,
        call_heavy=call_heavy,
        neutral_pin_risk=neutral_pin_risk,
        long_vega_friendly=long_vega_friendly,
        short_premium_friendly=short_premium_friendly,
        adjustment_notes=_dedupe_notes(adjustment_notes),
    )


__all__ = [
    "SurfaceAdjustment",
    "analyze_surface_adjustment",
]

