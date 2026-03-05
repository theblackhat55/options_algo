"""
tests/test_morning_brief.py
============================
Phase 4C tests: LONG_CALL/LONG_PUT formatting in morning brief.

Tests:
  1. _format_trade renders LONG_CALL correctly (strike, breakeven, EV)
  2. _format_trade renders LONG_PUT correctly
  3. TA-signal one-liner appears in format_morning_brief output
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from src.pipeline.morning_brief import _format_trade


class TestFormatTradeLongCall:
    def test_long_call_format_contains_key_fields(self):
        """LONG_CALL trade dict → formatted string with strike, debit, breakeven, PoP, EV."""
        trade = {
            "spread_type": "LONG_CALL",
            "strike": 155.0,
            "expiration": "2025-03-21",
            "dte": 30,
            "premium": 5.50,
            "breakeven": 160.50,
            "prob_profit": 62.0,
            "ev": 2.30,
        }
        result = _format_trade(trade, "LONG_CALL")
        assert "155.0" in result
        assert "5.50" in result
        assert "160.50" in result
        assert "62%" in result
        assert "2.30" in result
        assert "Call" in result

    def test_long_call_format_fallback_fields(self):
        """Trade dict with short_strike fallback instead of strike."""
        trade = {
            "spread_type": "LONG_CALL",
            "short_strike": 160.0,  # fallback key
            "expiration": "2025-04-17",
            "dte": 42,
            "premium": 6.00,
            "breakeven": 166.00,
            "prob_profit": 55.0,
            "ev": 1.80,
        }
        result = _format_trade(trade, "LONG_CALL")
        assert "160.0" in result
        assert "6.00" in result


class TestFormatTradeLongPut:
    def test_long_put_format_contains_key_fields(self):
        """LONG_PUT trade dict → formatted string with put-specific fields."""
        trade = {
            "spread_type": "LONG_PUT",
            "strike": 140.0,
            "expiration": "2025-03-21",
            "dte": 28,
            "premium": 4.20,
            "breakeven": 135.80,
            "prob_profit": 58.0,
            "ev": 1.95,
        }
        result = _format_trade(trade, "LONG_PUT")
        assert "140.0" in result
        assert "4.20" in result
        assert "135.80" in result
        assert "58%" in result
        assert "Put" in result


class TestMorningBriefTASignalLine:
    """
    Verify that format_morning_brief() renders TA signal one-liner when ta_signals present.
    """

    def _make_signal(self, strategy: str = "LONG_CALL", ta_signals: dict = None) -> dict:
        trade_dict = {
            "spread_type": strategy,
            "strike": 155.0,
            "expiration": "2025-03-21",
            "dte": 30,
            "premium": 5.50,
            "breakeven": 160.50,
            "prob_profit": 62,
            "ev": 2.30,
            "dry_run": False,
        }
        rec_inner = {
            "ticker": "AAPL",
            "strategy": strategy,
            "direction": "BULLISH",
            "regime": "STRONG_UPTREND",
            "confidence": 0.75,
            "rationale": "Test signal",
            "ta_signals": ta_signals or {},
            "is_long_option": strategy in ("LONG_CALL", "LONG_PUT"),
        }
        ctx = {
            "price": 152.0,
            "sector": "Technology",
            "iv_detail": {"iv_rank": 25, "iv_percentile": 30, "current_iv": 28,
                          "hv_20": 22, "iv_hv_ratio": 1.27, "iv_trend": "stable",
                          "premium_action": "normal", "iv_rv_spread": 6, "premium_rich": False},
            "regime_detail": {"adx": 30, "rsi": 55, "trend_strength": 0.7,
                               "direction_score": 0.6, "ema_alignment": True,
                               "bb_squeeze": False, "support": 148.0, "resistance": 162.0,
                               "atr": 2.5, "atr_pct": 1.6, "volume_trend": "up",
                               "roc_3d": 1.2, "atr_move_5d": 3.1},
            "rs_detail": {"rs_vs_spy": 1.5, "rs_rank": 80, "rs_trend": "strong"},
            "options_flow": {},
            "ta_signals": ta_signals or {},
        }
        pick = {
            "priority": 1,
            "composite_score": 0.042,
            "recommendation": rec_inner,
            "trade": trade_dict,
            "context": ctx,
        }
        return {
            "scan_date": "2025-01-20",
            "generated_at": "2025-01-20T16:30:00",
            "universe_size": 100,
            "qualified": 40,
            "after_event_filter": 35,
            "market_context": {
                "vix_level": 18.0,
                "vix_tier": "LOW",
                "vix_spike": False,
                "spy_return_5d": 1.2,
                "breadth_score": 0.65,
                "market_regime": "BULL",
                "sector_leaders": [],
                "sector_laggards": [],
            },
            "top_picks": [pick],
            "regime_distribution": {},
        }

    def test_ta_breakout_one_liner_appears(self):
        """When breakout_above=True in ta_signals, brief includes TA line."""
        from src.pipeline.morning_brief import format_morning_brief
        signal = self._make_signal(
            ta_signals={"breakout_above": True, "bullish_divergence": False}
        )
        brief = format_morning_brief(signal)
        assert "breakout" in brief.lower() or "TA:" in brief

    def test_long_option_flag_line_appears(self):
        """LONG_CALL strategy → brief includes profit-target / stop-loss / time-stop line."""
        from src.pipeline.morning_brief import format_morning_brief
        signal = self._make_signal(strategy="LONG_CALL", ta_signals={})
        brief = format_morning_brief(signal)
        assert "profit-target" in brief.lower() or "Long Opt:" in brief

    def test_credit_spread_no_long_option_flag(self):
        """BULL_PUT_SPREAD → no Long Opt: line in brief."""
        from src.pipeline.morning_brief import format_morning_brief
        signal = self._make_signal(strategy="BULL_PUT_SPREAD", ta_signals={})
        # Adjust trade dict for credit spread
        signal["top_picks"][0]["recommendation"]["strategy"] = "BULL_PUT_SPREAD"
        signal["top_picks"][0]["recommendation"]["is_long_option"] = False
        signal["top_picks"][0]["trade"] = {
            "spread_type": "BULL_PUT",
            "short_strike": 100.0,
            "long_strike": 95.0,
            "expiration": "2025-03-21",
            "dte": 30,
            "net_credit": 1.50,
            "max_risk": 3.50,
            "prob_profit": 70,
            "dry_run": False,
        }
        brief = format_morning_brief(signal)
        assert "Long Opt:" not in brief
