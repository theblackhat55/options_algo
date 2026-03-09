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
class StrategyBiasDecision:
    ticker: str
    original_strategy: str
    preferred_strategy: str
    strategy_changed: bool
    confidence_delta: float
    candidate_score_delta: float
    bias_strength: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def choose_strategy_bias(
    ticker: str,
    direction: str,
    strategy: str,
    surface_row: Dict[str, Any] | None,
) -> StrategyBiasDecision:
    row = surface_row or {}
    direction = _safe_str(direction).upper()
    original = _safe_str(strategy).upper() or "WATCHLIST"

    liq_ratio = _safe_float(row.get("surface_liquid_contract_ratio", 0.0), 0.0)
    avg_spread = _safe_float(row.get("surface_avg_spread_pct", 0.0), 0.0)
    median_spread = _safe_float(row.get("surface_median_spread_pct", 0.0), 0.0)
    term_slope = _safe_float(row.get("surface_term_slope", 0.0), 0.0)
    term_ratio = _safe_float(row.get("surface_term_ratio", 0.0), 0.0)
    pc_oi = _safe_float(row.get("surface_put_call_oi_ratio", 0.0), 0.0)
    pc_vol = _safe_float(row.get("surface_put_call_volume_ratio", 0.0), 0.0)
    top_oi_dist = abs(_safe_float(row.get("surface_top_oi_strike_distance_pct", 999.0), 999.0))
    front_conc = _safe_float(row.get("surface_front_expiry_concentration", 0.0), 0.0)
    total_contracts = _safe_float(row.get("surface_total_contracts", 0.0), 0.0)

    liquid_ok = liq_ratio >= 0.25 and avg_spread <= 0.15 and median_spread <= 0.10
    liquid_strong = liq_ratio >= 0.45 and avg_spread <= 0.08 and median_spread <= 0.06
    long_vega_friendly = term_slope > 0.0 or term_ratio > 1.0
    short_premium_friendly = term_slope < 0.0 or (0.0 < term_ratio < 1.0)
    put_heavy = pc_oi >= 1.25 or pc_vol >= 1.25
    call_heavy = (0.0 < pc_oi <= 0.80) or (0.0 < pc_vol <= 0.80)
    pin_risk = top_oi_dist <= 0.02 and total_contracts > 0
    crowded_front = front_conc >= 0.60

    preferred = original
    conf_delta = 0.0
    score_delta = 0.0
    reasons = []
    bias_strength = "NONE"

    # Only allow actual overrides when evidence is fairly strong.
    if liquid_ok and crowded_front and pin_risk and direction == "NEUTRAL":
        preferred = "IRON_CONDOR"
        conf_delta += 0.02
        score_delta += 0.02
        reasons.append("strong_neutral_pin_setup")
        bias_strength = "STRONG"

    elif liquid_strong and short_premium_friendly:
        if direction == "NEUTRAL":
            preferred = "IRON_CONDOR"
        elif direction == "BULLISH":
            preferred = "BULL_PUT_SPREAD"
        elif direction == "BEARISH":
            preferred = "BEAR_CALL_SPREAD"
        if preferred != original:
            conf_delta += 0.02
            score_delta += 0.03
            reasons.append("strong_short_premium_setup")
            bias_strength = "STRONG"

    elif liquid_strong and long_vega_friendly:
        if direction == "BULLISH":
            preferred = "LONG_CALL"
        elif direction == "BEARISH":
            preferred = "LONG_PUT"
        elif direction == "NEUTRAL":
            preferred = "LONG_BUTTERFLY"
        if preferred != original:
            conf_delta += 0.01
            score_delta += 0.02
            reasons.append("strong_long_vega_setup")
            bias_strength = "MEDIUM"

    # If no strong override, allow only gentle reinforcement of an already-compatible strategy.
    if preferred == original:
        if direction == "BULLISH" and call_heavy:
            conf_delta += 0.01
            score_delta += 0.01
            reasons.append("call_heavy_supports_bullish")
            bias_strength = "LIGHT" if bias_strength == "NONE" else bias_strength
        elif direction == "BEARISH" and put_heavy:
            conf_delta += 0.01
            score_delta += 0.01
            reasons.append("put_heavy_supports_bearish")
            bias_strength = "LIGHT" if bias_strength == "NONE" else bias_strength

    # Poor liquidity should discourage changing strategy, not force a generic one.
    if total_contracts > 0 and not liquid_ok and preferred != original:
        preferred = original
        conf_delta = min(conf_delta, 0.0)
        score_delta = min(score_delta, 0.0)
        reasons.append("poor_liquidity_blocked_override")
        bias_strength = "LIGHT"

    conf_delta = max(-0.05, min(0.05, conf_delta))
    score_delta = max(-0.05, min(0.05, score_delta))

    return StrategyBiasDecision(
        ticker=ticker,
        original_strategy=original,
        preferred_strategy=preferred or original,
        strategy_changed=(preferred or original) != original,
        confidence_delta=conf_delta,
        candidate_score_delta=score_delta,
        bias_strength=bias_strength,
        rationale=";".join(reasons),
    )
