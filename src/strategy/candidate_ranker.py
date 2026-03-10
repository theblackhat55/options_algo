from __future__ import annotations

from typing import Any

from src.strategy.candidate_types import CandidateEvaluation, CandidateSpec


def score_candidate(
    candidate: CandidateSpec,
    recommendation,
    trade_payload: dict[str, Any],
    regime_detail: dict[str, Any] | None = None,
) -> CandidateEvaluation:
    """
    Heuristic scoring for Phase 2 candidate selection.

    This is intentionally simple and safe:
      - uses existing recommendation confidence as anchor
      - penalizes wide spreads / poor rr / high theta
      - rewards better ATR fit and better liquidity
    """
    regime_detail = regime_detail or {}

    base_conf = float(getattr(recommendation, "confidence", 0.5))

    liquidity_score = _liquidity_score(trade_payload)
    spread_penalty = _spread_penalty(trade_payload)
    rr_score = _rr_score(trade_payload)
    theta_penalty = _theta_penalty(trade_payload)
    atr_fit_score = _atr_fit_score(trade_payload, regime_detail)

    candidate_score = (
        base_conf
        + liquidity_score
        + rr_score
        + atr_fit_score
        - spread_penalty
        - theta_penalty
    )

    candidate_score = round(candidate_score, 4)

    return CandidateEvaluation(
        ticker=candidate.ticker,
        strategy=candidate.strategy,
        direction=candidate.direction,
        target_dte=candidate.target_dte,
        params=candidate.params,
        candidate_score=candidate_score,
        base_confidence=round(base_conf, 4),
        liquidity_score=round(liquidity_score, 4),
        spread_penalty=round(spread_penalty, 4),
        rr_score=round(rr_score, 4),
        theta_penalty=round(theta_penalty, 4),
        atr_fit_score=round(atr_fit_score, 4),
        trade_payload=trade_payload,
    )


def select_best_candidate(evaluations: list[CandidateEvaluation]) -> CandidateEvaluation | None:
    if not evaluations:
        return None
    best = sorted(evaluations, key=lambda x: x.candidate_score, reverse=True)[0]
    best.selected = True
    return best


def evaluation_to_feature_row(ev: CandidateEvaluation, as_of_date: str, scan_mode: str = "dry_run") -> dict[str, Any]:
    trade = ev.trade_payload or {}
    return {
        "ticker": ev.ticker,
        "as_of_date": as_of_date,
        "scan_mode": scan_mode,
        "strategy": ev.strategy,
        "direction": ev.direction,
        "target_dte": ev.target_dte,
        "params_json": str(ev.params),
        "candidate_score": ev.candidate_score,
        "base_confidence": ev.base_confidence,
        "liquidity_score": ev.liquidity_score,
        "spread_penalty": ev.spread_penalty,
        "rr_score": ev.rr_score,
        "theta_penalty": ev.theta_penalty,
        "atr_fit_score": ev.atr_fit_score,
        "selected": ev.selected,
        "rejected_reason": ev.rejected_reason,
        "trade_dry_run": trade.get("dry_run", False),
        "prob_profit": trade.get("prob_profit"),
        "risk_reward_ratio": trade.get("risk_reward_ratio"),
        "ev": trade.get("ev"),
        "max_risk": trade.get("max_risk"),
        "net_credit": trade.get("net_credit"),
        "premium": trade.get("premium"),
    }


def _liquidity_score(trade_payload: dict[str, Any]) -> float:
    trade = trade_payload or {}
    if trade.get("dry_run"):
        return 0.02

    score = 0.0
    if trade.get("net_credit", 0) and trade.get("max_risk", 0):
        score += 0.03
    if trade.get("premium", 0):
        score += 0.03
    return score


def _spread_penalty(trade_payload: dict[str, Any]) -> float:
    trade = trade_payload or {}
    rr = float(trade.get("risk_reward_ratio", 0) or 0)
    if rr <= 0:
        return 0.0
    if rr > 6:
        return 0.12
    if rr > 4:
        return 0.08
    if rr > 3:
        return 0.04
    return 0.01


def _rr_score(trade_payload: dict[str, Any]) -> float:
    trade = trade_payload or {}
    rr = float(trade.get("risk_reward_ratio", 0) or 0)
    if rr <= 0:
        return 0.0
    if rr <= 1.5:
        return 0.10
    if rr <= 2.5:
        return 0.06
    if rr <= 3.5:
        return 0.03
    return 0.0


def _theta_penalty(trade_payload: dict[str, Any]) -> float:
    trade = trade_payload or {}
    theta_rate = float(trade.get("theta_rate", 0) or 0)
    if theta_rate <= 0:
        return 0.0
    if theta_rate > 0.05:
        return 0.10
    if theta_rate > 0.03:
        return 0.05
    if theta_rate > 0.02:
        return 0.02
    return 0.0


def _atr_fit_score(trade_payload: dict[str, Any], regime_detail: dict[str, Any]) -> float:
    atr = float(regime_detail.get("atr", 0) or 0)
    if atr <= 0:
        return 0.0

    breakeven = trade_payload.get("breakeven")
    if breakeven is None:
        return 0.0

    price = float(regime_detail.get("price", 0) or 0)
    if price <= 0:
        return 0.0

    distance = abs(float(breakeven) - price)
    atr_units = distance / atr if atr > 0 else 0

    if atr_units <= 0.75:
        return 0.08
    if atr_units <= 1.5:
        return 0.05
    if atr_units <= 2.5:
        return 0.02
    return 0.0
