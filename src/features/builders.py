from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.features.schema import FEATURE_VERSION


def _common_fields(
    ticker: str | None,
    as_of_date: str,
    scan_mode: str,
    source_stock: str = "yfinance",
    source_options: str = "polygon",
    run_type: str = "nightly_scan",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_stock": source_stock,
        "source_options": source_options,
        "run_type": run_type,
        "scan_mode": scan_mode,
        "feature_version": FEATURE_VERSION,
    }


def build_stock_feature_row(
    ticker: str,
    as_of_date: str,
    regime,
    rs=None,
    sector: str | None = None,
    scan_mode: str = "dry_run",
) -> dict[str, Any]:
    row = _common_fields(ticker, as_of_date, scan_mode)
    row.update({
        "sector": sector,
        "price": getattr(regime, "price", None),
        "regime": getattr(getattr(regime, "regime", None), "value", None),
        "adx": getattr(regime, "adx", None),
        "rsi": getattr(regime, "rsi", None),
        "trend_strength": getattr(regime, "trend_strength", None),
        "direction_score": getattr(regime, "direction_score", None),
        "ema_alignment": getattr(regime, "ema_alignment", None),
        "bb_squeeze": getattr(regime, "bb_squeeze", None),
        "support": getattr(regime, "support", None),
        "resistance": getattr(regime, "resistance", None),
        "atr": getattr(regime, "atr", None),
        "atr_pct": getattr(regime, "atr_pct", None),
        "volume_trend": getattr(regime, "volume_trend", None),
        "roc_3d": getattr(regime, "roc_3d", None),
        "atr_move_5d": getattr(regime, "atr_move_5d", None),
        "rs_vs_spy": getattr(rs, "rs_vs_spy", None) if rs else None,
        "rs_rank": getattr(rs, "rs_rank", None) if rs else None,
        "rs_trend": getattr(rs, "rs_trend", None) if rs else None,
    })
    return row


def build_options_feature_row(
    ticker: str,
    as_of_date: str,
    iv,
    scan_mode: str = "dry_run",
) -> dict[str, Any]:
    row = _common_fields(ticker, as_of_date, scan_mode)
    row.update({
        "current_iv": getattr(iv, "current_iv", None),
        "iv_rank": getattr(iv, "iv_rank", None),
        "iv_percentile": getattr(iv, "iv_percentile", None),
        "hv_20": getattr(iv, "hv_20", None),
        "hv_60": getattr(iv, "hv_60", None),
        "iv_hv_ratio": getattr(iv, "iv_hv_ratio", None),
        "iv_regime": getattr(iv, "iv_regime", None),
        "premium_action": getattr(iv, "premium_action", None),
        "iv_trend": getattr(iv, "iv_trend", None),
        "iv_30d_avg": getattr(iv, "iv_30d_avg", None),
        "skew": getattr(iv, "skew", None),
        "iv_rv_spread": getattr(iv, "iv_rv_spread", None),
        "premium_rich": getattr(iv, "premium_rich", None),
    })
    return row


def build_market_feature_row(
    as_of_date: str,
    market_ctx,
    scan_mode: str = "dry_run",
) -> dict[str, Any]:
    row = _common_fields(None, as_of_date, scan_mode)
    row.update({
        "market_regime": getattr(market_ctx, "market_regime", None),
        "vix_level": getattr(market_ctx, "vix_level", None),
        "vix_regime": getattr(market_ctx, "vix_regime", None),
        "vix_5d_avg": getattr(market_ctx, "vix_5d_avg", None),
        "vix_spike": getattr(market_ctx, "vix_spike", None),
        "vix_tier": getattr(market_ctx, "vix_tier", None),
        "spy_trend": getattr(market_ctx, "spy_trend", None),
        "spy_return_5d": getattr(market_ctx, "spy_return_5d", None),
        "spy_return_20d": getattr(market_ctx, "spy_return_20d", None),
        "breadth_score": getattr(market_ctx, "breadth_score", None),
        "sector_leaders": ",".join(getattr(market_ctx, "sector_leaders", []) or []),
        "sector_laggards": ",".join(getattr(market_ctx, "sector_laggards", []) or []),
        "notes": getattr(market_ctx, "notes", None),
    })
    return row


def build_candidate_feature_row(
    ticker: str,
    as_of_date: str,
    recommendation,
    selected: bool = False,
    survived_event_filter: bool | None = None,
    scan_mode: str = "dry_run",
) -> dict[str, Any]:
    row = _common_fields(ticker, as_of_date, scan_mode)
    row.update({
        "strategy": getattr(getattr(recommendation, "strategy", None), "value", None),
        "direction": getattr(recommendation, "direction", None),
        "regime": getattr(recommendation, "regime", None),
        "iv_regime": getattr(recommendation, "iv_regime", None),
        "confidence": getattr(recommendation, "confidence", None),
        "target_dte": getattr(recommendation, "target_dte", None),
        "risk_reward": getattr(recommendation, "risk_reward", None),
        "rationale": getattr(recommendation, "rationale", None),
        "selected_flag": selected,
        "survived_event_filter": survived_event_filter,
    })
    return row


def build_recommendation_feature_row(
    as_of_date: str,
    pick: dict,
    scan_mode: str = "dry_run",
) -> dict[str, Any]:
    rec = pick.get("recommendation", {})
    trade = pick.get("trade", {})
    ctx = pick.get("context", {})
    row = _common_fields(rec.get("ticker"), as_of_date, scan_mode)
    row.update({
        "priority": pick.get("priority"),
        "composite_score": pick.get("composite_score"),
        "strategy": rec.get("strategy"),
        "direction": rec.get("direction"),
        "regime": rec.get("regime"),
        "iv_regime": rec.get("iv_regime"),
        "confidence": rec.get("confidence"),
        "target_dte": rec.get("target_dte"),
        "price": ctx.get("price"),
        "sector": ctx.get("sector"),
        "trade_dry_run": trade.get("dry_run", False),
    })
    return row
