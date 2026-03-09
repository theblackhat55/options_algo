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
    preferred_strategy: str
    confidence_delta: float
    candidate_score_delta: float
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
    strategy_u = _safe_str(strategy).upper()

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

    liquid = liq_ratio >= 0.25 and avg_spread <= 0.15 and median_spread <= 0.10
    long_vega_friendly = term_slope > 0.0 or term_ratio > 1.0
    short_premium_friendly = term_slope < 0.0 or (0.0 < term_ratio < 1.0)
    put_heavy = pc_oi >= 1.25 or pc_vol >= 1.25
    call_heavy = (0.0 < pc_oi <= 0.80) or (0.0 < pc_vol <= 0.80)
    pin_risk = top_oi_dist <= 0.02 and total_contracts > 0
    crowded_front = front_conc >= 0.60

    preferred = strategy_u
    conf_delta = 0.0
    score_delta = 0.0
    reasons = []

    # Neutral pin / crowding preference
    if pin_risk and direction == "NEUTRAL":
        preferred = "IRON_CONDOR"
        conf_delta += 0.03
        score_delta += 0.03
        reasons.append("pin_risk_prefers_neutral_structure")

    # Strong short-premium setup
    elif liquid and short_premium_friendly and crowded_front:
        if direction == "NEUTRAL":
            preferred = "IRON_CONDOR"
        elif direction == "BULLISH":
            preferred = "BULL_PUT_SPREAD"
        elif direction == "BEARISH":
            preferred = "BEAR_CALL_SPREAD"
        conf_delta += 0.03
        score_delta += 0.04
        reasons.append("liquid_short_premium_setup")

    # Strong long-vega setup
    elif liquid and long_vega_friendly:
        if direction == "BULLISH":
            preferred = "LONG_CALL"
        elif direction == "BEARISH":
            preferred = "LONG_PUT"
        elif direction == "NEUTRAL":
            preferred = "LONG_BUTTERFLY"
        conf_delta += 0.02
        score_delta += 0.03
        reasons.append("liquid_long_vega_setup")

    # Positioning reinforcement
    if direction == "BULLISH" and call_heavy:
        if preferred in {"WATCHLIST", "", strategy_u}:
            preferred = "BULL_PUT_SPREAD" if liquid else strategy_u or "WATCHLIST"
        conf_delta += 0.01
        score_delta += 0.01
        reasons.append("call_heavy_supports_bullish")

    if direction == "BEARISH" and put_heavy:
        if preferred in {"WATCHLIST", "", strategy_u}:
            preferred = "BEAR_CALL_SPREAD" if liquid else strategy_u or "WATCHLIST"
        conf_delta += 0.01
        score_delta += 0.01
        reasons.append("put_heavy_supports_bearish")

    # Poor liquidity discourages aggressive long-vega changes
    if total_contracts > 0 and not liquid:
        if preferred in {"LONG_CALL", "LONG_PUT", "LONG_BUTTERFLY"}:
            preferred = strategy_u or "WATCHLIST"
        conf_delta -= 0.03
        score_delta -= 0.03
        reasons.append("poor_liquidity_caps_strategy_bias")

    conf_delta = max(-0.08, min(0.08, conf_delta))
    score_delta = max(-0.08, min(0.08, score_delta))

    return StrategyBiasDecision(
        ticker=ticker,
        preferred_strategy=preferred or (strategy_u or "WATCHLIST"),
        confidence_delta=conf_delta,
        candidate_score_delta=score_delta,
        rationale=";".join(reasons),
    )
