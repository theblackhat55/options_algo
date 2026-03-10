from __future__ import annotations

from typing import Any

from src.strategy.candidate_types import CandidateSpec


def generate_candidates(rec) -> list[CandidateSpec]:
    """
    Generate a compact candidate grid from an existing StrategyRecommendation.

    Phase 2 goal:
      - keep current platform intact
      - add multiple candidate structures per ticker
      - choose best candidate later

    This generator intentionally stays modest in scope for safety.
    """
    ticker = rec.ticker
    strategy = rec.strategy.value
    direction = rec.direction

    candidates: list[CandidateSpec] = []

    if strategy == "BULL_PUT_SPREAD":
        for dte in (30, 45):
            for short_delta in (0.20, 0.25, 0.30):
                for width in (5.0, 10.0):
                    candidates.append(CandidateSpec(
                        ticker=ticker,
                        strategy="BULL_PUT_SPREAD",
                        direction=direction,
                        target_dte=dte,
                        params={
                            "short_delta": short_delta,
                            "spread_width": width,
                        },
                    ))

    elif strategy == "BEAR_CALL_SPREAD":
        for dte in (30, 45):
            for short_delta in (0.20, 0.25, 0.30):
                for width in (5.0, 10.0):
                    candidates.append(CandidateSpec(
                        ticker=ticker,
                        strategy="BEAR_CALL_SPREAD",
                        direction=direction,
                        target_dte=dte,
                        params={
                            "short_delta": short_delta,
                            "spread_width": width,
                        },
                    ))

    elif strategy == "BULL_CALL_SPREAD":
        for dte in (21, 30, 45):
            for long_delta in (0.55, 0.65):
                candidates.append(CandidateSpec(
                    ticker=ticker,
                    strategy="BULL_CALL_SPREAD",
                    direction=direction,
                    target_dte=dte,
                    params={
                        "long_delta": long_delta,
                    },
                ))

    elif strategy == "BEAR_PUT_SPREAD":
        for dte in (21, 30, 45):
            for long_delta in (0.55, 0.65):
                candidates.append(CandidateSpec(
                    ticker=ticker,
                    strategy="BEAR_PUT_SPREAD",
                    direction=direction,
                    target_dte=dte,
                    params={
                        "long_delta": long_delta,
                    },
                ))

    elif strategy == "IRON_CONDOR":
        for dte in (30, 45):
            for short_delta in (0.10, 0.16, 0.20):
                for width in (5.0, 10.0):
                    candidates.append(CandidateSpec(
                        ticker=ticker,
                        strategy="IRON_CONDOR",
                        direction=direction,
                        target_dte=dte,
                        params={
                            "short_delta": short_delta,
                            "spread_width": width,
                        },
                    ))

    elif strategy == "LONG_BUTTERFLY":
        for dte in (21, 30, 45):
            candidates.append(CandidateSpec(
                ticker=ticker,
                strategy="LONG_BUTTERFLY",
                direction=direction,
                target_dte=dte,
                params={},
            ))

    elif strategy == "LONG_CALL":
        for dte in (21, 35, 45):
            for delta in (0.55, 0.65, 0.75):
                candidates.append(CandidateSpec(
                    ticker=ticker,
                    strategy="LONG_CALL",
                    direction=direction,
                    target_dte=dte,
                    params={
                        "target_delta": delta,
                    },
                ))

    elif strategy == "LONG_PUT":
        for dte in (21, 35, 45):
            for delta in (0.55, 0.65, 0.75):
                candidates.append(CandidateSpec(
                    ticker=ticker,
                    strategy="LONG_PUT",
                    direction=direction,
                    target_dte=dte,
                    params={
                        "target_delta": delta,
                    },
                ))

    else:
        candidates.append(CandidateSpec(
            ticker=ticker,
            strategy=strategy,
            direction=direction,
            target_dte=rec.target_dte,
            params={},
        ))

    return candidates
