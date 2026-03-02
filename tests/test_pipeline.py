"""
tests/test_pipeline.py
======================
Tests for the nightly scan pipeline (dry-run mode) and morning brief formatter.
Uses a small synthetic universe to avoid network calls.
"""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipeline.morning_brief import format_morning_brief
from src.analysis.options_analytics import (
    bs_greeks, prob_otm, prob_between, credit_spread_ev,
    annualised_return_on_risk, implied_move,
)


# ─── Options Analytics Tests ─────────────────────────────────────────────────

class TestBSGreeks:
    def test_call_price_positive(self):
        g = bs_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.25, option_type="call")
        assert g.price > 0

    def test_put_price_positive(self):
        g = bs_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.25, option_type="put")
        assert g.price > 0

    def test_call_delta_between_0_and_1(self):
        g = bs_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.25, option_type="call")
        assert 0 < g.delta < 1

    def test_put_delta_between_minus1_and_0(self):
        g = bs_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.25, option_type="put")
        assert -1 < g.delta < 0

    def test_atm_call_delta_near_half(self):
        g = bs_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.25, option_type="call")
        assert 0.45 < g.delta < 0.65   # ATM delta ≈ 0.5–0.6

    def test_deep_otm_call_has_low_delta(self):
        g = bs_greeks(S=100, K=150, T=0.1, r=0.05, sigma=0.20, option_type="call")
        assert g.delta < 0.10

    def test_deep_itm_call_has_high_delta(self):
        g = bs_greeks(S=150, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
        assert g.delta > 0.85

    def test_put_call_parity_approximately(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        c = bs_greeks(S, K, T, r, sigma, "call").price
        p = bs_greeks(S, K, T, r, sigma, "put").price
        # Put-call parity: C - P = S - K*e^(-r*T)
        import math
        parity_rhs = S - K * math.exp(-r * T)
        assert abs((c - p) - parity_rhs) < 0.10

    def test_zero_time_returns_zero_price(self):
        g = bs_greeks(S=100, K=110, T=0, r=0.05, sigma=0.25, option_type="call")
        assert g.price == 0.0

    def test_gamma_positive(self):
        g = bs_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.25, option_type="call")
        assert g.gamma > 0

    def test_theta_negative_for_long_call(self):
        g = bs_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.25, option_type="call")
        assert g.theta < 0


class TestProbabilities:
    def test_prob_otm_call_otm_near_1(self):
        # Deep OTM call: stock at 100, strike at 150, 10 days
        p = prob_otm(S=100, K=150, T=10/365, sigma=0.25, option_type="call")
        assert p > 0.90

    def test_prob_otm_put_itm_near_0(self):
        # Put deep OTM: stock at 200, strike at 100
        p = prob_otm(S=200, K=100, T=30/365, sigma=0.30, option_type="put")
        assert p > 0.90

    def test_prob_between_returns_float(self):
        p = prob_between(S=100, K_low=90, K_high=110, T=0.5, sigma=0.25)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_prob_between_wide_range_near_1(self):
        p = prob_between(S=100, K_low=50, K_high=200, T=0.1, sigma=0.20)
        assert p > 0.90


class TestSpreadEV:
    def test_positive_ev_when_high_pop(self):
        ev = credit_spread_ev(net_credit=1.0, max_risk=4.0, prob_profit=0.80)
        assert ev >= 0    # 0.80 × 1.0 − 0.20 × 4.0 = 0.0  (break-even)

    def test_negative_ev_when_low_pop(self):
        ev = credit_spread_ev(net_credit=1.0, max_risk=4.0, prob_profit=0.50)
        assert ev < 0

    def test_annualised_ror_positive(self):
        ror = annualised_return_on_risk(net_credit=1.0, max_risk=4.0, dte=45)
        assert ror > 0

    def test_implied_move_positive(self):
        move = implied_move(current_price=100, iv=0.25, dte=30)
        assert move > 0

    def test_implied_move_scales_with_iv(self):
        m1 = implied_move(100, 0.20, 30)
        m2 = implied_move(100, 0.40, 30)
        assert m2 > m1


# ─── Morning Brief Tests ──────────────────────────────────────────────────────

class TestMorningBrief:
    def _make_signal(self, n_picks: int = 3) -> dict:
        picks = []
        for i in range(n_picks):
            picks.append({
                "priority": i + 1,
                "composite_score": 0.5 - i * 0.05,
                "recommendation": {
                    "ticker": f"TICK{i}",
                    "strategy": "BULL_PUT_SPREAD",
                    "direction": "BULLISH",
                    "regime": "UPTREND",
                    "iv_regime": "HIGH",
                    "confidence": 0.75,
                    "rationale": "Test rationale",
                    "target_dte": 45,
                },
                "trade": {
                    "spread_type": "BULL_PUT",
                    "expiration": "2025-05-16",
                    "dte": 45,
                    "short_strike": 190.0,
                    "long_strike": 185.0,
                    "net_credit": 1.20,
                    "max_risk": 3.80,
                    "max_reward": 1.20,
                    "risk_reward_ratio": 3.17,
                    "breakeven": 188.80,
                    "prob_profit": 75.0,
                    "prob_touch": 40.0,
                    "short_delta": 0.25,
                    "width": 5.0,
                    "annual_ror": 28.5,
                    "ev": 0.18,
                },
                "context": {
                    "price": 200.0,
                    "sector": "Technology",
                    "regime_detail": {
                        "adx": 32.0, "rsi": 58.0, "trend_strength": 0.64,
                        "direction_score": 0.45, "ema_alignment": "bullish",
                        "bb_squeeze": False, "support": 192.0, "resistance": 208.0,
                        "atr": 3.2, "atr_pct": 1.6, "volume_trend": "neutral",
                    },
                    "iv_detail": {
                        "iv_rank": 78.0, "iv_percentile": 75.0,
                        "current_iv": 32.0, "hv_20": 25.0,
                        "iv_hv_ratio": 1.28, "iv_trend": "FLAT",
                        "premium_action": "SELL",
                    },
                    "rs_detail": {
                        "rs_vs_spy": 0.12, "rs_rank": 72.0, "rs_trend": "IMPROVING",
                    },
                },
            })

        return {
            "scan_date": "2025-03-01",
            "generated_at": "2025-03-01T23:00:00Z",
            "elapsed_seconds": 95.3,
            "universe_size": 120,
            "qualified": 98,
            "regimes_classified": 95,
            "recommendations": 42,
            "after_event_filter": 35,
            "trades_constructed": n_picks,
            "top_picks": picks,
            "regime_distribution": {
                "UPTREND": 25, "RANGE_BOUND": 30,
                "DOWNTREND": 18, "STRONG_UPTREND": 12,
            },
            "market_context": {
                "market_regime": "BULL",
                "vix_level": 16.5,
                "vix_regime": "NORMAL",
                "spy_trend": "UPTREND",
                "spy_return_5d": 0.8,
                "spy_return_20d": 2.1,
                "breadth_score": 0.68,
                "sector_leaders": ["Technology", "Financials"],
                "sector_laggards": ["Utilities", "Real Estate"],
                "notes": "",
            },
        }

    def test_returns_string(self):
        signal = self._make_signal(3)
        msg = format_morning_brief(signal)
        assert isinstance(msg, str)

    def test_contains_scan_date(self):
        signal = self._make_signal(3)
        msg = format_morning_brief(signal)
        assert "2025-03-01" in msg

    def test_contains_all_tickers(self):
        signal = self._make_signal(3)
        msg = format_morning_brief(signal)
        for i in range(3):
            assert f"TICK{i}" in msg

    def test_contains_market_regime(self):
        signal = self._make_signal(2)
        msg = format_morning_brief(signal)
        assert "BULL" in msg

    def test_contains_credit_info(self):
        signal = self._make_signal(1)
        msg = format_morning_brief(signal)
        assert "1.20" in msg   # net credit

    def test_no_picks_message(self):
        signal = self._make_signal(0)
        msg = format_morning_brief(signal)
        assert "No high-confidence picks" in msg

    def test_none_signal_returns_error(self):
        with patch("src.pipeline.morning_brief._load_latest_signal", return_value=None):
            msg = format_morning_brief(None)
        assert "No signal" in msg


# ─── Dry-Run Pipeline Smoke Test ─────────────────────────────────────────────

class TestPipelineDryRun:
    """
    Smoke test: run the pipeline with a tiny universe and mocked data.
    Verifies the pipeline completes without errors.
    """

    def _make_df(self, n: int = 150, price: float = 150.0) -> pd.DataFrame:
        np.random.seed(0)
        prices = price + np.cumsum(np.random.normal(0, 1.5, n))
        prices = np.clip(prices, 20, None)
        dates = pd.bdate_range(end="2025-01-01", periods=n)
        return pd.DataFrame({
            "open":   prices * 0.999,
            "high":   prices * 1.005,
            "low":    prices * 0.995,
            "close":  prices,
            "volume": np.random.randint(2_000_000, 8_000_000, n),
        }, index=dates)

    @patch("src.pipeline.nightly_scan.update_universe")
    @patch("src.pipeline.nightly_scan.filter_safe_tickers")
    def test_dry_run_completes(self, mock_safety, mock_update):
        from src.pipeline.nightly_scan import run_nightly_scan

        # Mock data for 5 tickers
        tickers = ["AAPL", "MSFT", "SPY", "XLK", "JPM"]
        mock_data = {t: self._make_df() for t in tickers}
        mock_update.return_value = mock_data

        # Mock safety check — all safe
        from src.risk.event_filter import EventFilterResult
        mock_safety.return_value = {
            t: EventFilterResult(safe=True, ticker=t, reason=None, earnings_in_days=None)
            for t in tickers
        }

        signal = run_nightly_scan(
            universe_override=tickers,
            dry_run=True,
        )

        assert isinstance(signal, dict)
        assert "scan_date" in signal
        assert "top_picks" in signal
        assert "regime_distribution" in signal
        assert signal["universe_size"] == len(tickers)
        assert signal["qualified"] <= len(tickers)

    @patch("src.pipeline.nightly_scan.update_universe")
    @patch("src.pipeline.nightly_scan.filter_safe_tickers")
    def test_signal_has_correct_structure(self, mock_safety, mock_update):
        from src.pipeline.nightly_scan import run_nightly_scan
        from src.risk.event_filter import EventFilterResult

        tickers = ["SPY", "QQQ"]
        mock_data = {t: self._make_df(200) for t in tickers}
        mock_update.return_value = mock_data
        mock_safety.return_value = {
            t: EventFilterResult(safe=True, ticker=t, reason=None, earnings_in_days=None)
            for t in tickers
        }

        signal = run_nightly_scan(universe_override=tickers, dry_run=True)

        required_keys = [
            "scan_date", "generated_at", "elapsed_seconds",
            "universe_size", "qualified", "top_picks",
            "regime_distribution", "market_context",
        ]
        for key in required_keys:
            assert key in signal, f"Missing key: {key}"
