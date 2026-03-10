from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateSpec:
    ticker: str
    strategy: str
    direction: str
    target_dte: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateEvaluation:
    ticker: str
    strategy: str
    direction: str
    target_dte: int
    params: dict[str, Any]
    candidate_score: float
    base_confidence: float
    liquidity_score: float
    spread_penalty: float
    rr_score: float
    theta_penalty: float
    atr_fit_score: float
    selected: bool = False
    rejected_reason: str | None = None
    trade_payload: dict[str, Any] = field(default_factory=dict)
