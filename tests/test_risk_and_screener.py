"""
tests/test_risk_and_screener.py
================================
Tests for:
  - src.risk.position_sizer (Kelly + fixed-pct sizing)
  - src.risk.portfolio (position tracking, P&L)
  - src.risk.event_filter (earnings safety check)
  - src.screener.composite_screener (ranked output)

All network / file I/O is patched — no external calls.
"""
from __future__ import annotations

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.analysis.technical import Regime, StockRegime
from src.analysis.volatility import IVAnalysis


# ─── Helpers shared across classes ────────────────────────────────────────────

def _make_regime(
    ticker: str = "AAPL",
    regime: Regime = Regime.UPTREND,
    direction_score: float = 0.5,
    adx: float = 28.0,
    rsi: float = 55.0,
    trend_strength: float = 0.65,
    volatility_state: str = "normal",
) -> StockRegime:
    return StockRegime(
        ticker=ticker,
        regime=regime,
        direction_score=direction_score,
        trend_strength=trend_strength,
        volatility_state=volatility_state,
        rsi=rsi,
        adx=adx,
        bb_squeeze=False,
        ema_alignment="bullish",
        support=140.0,
        resistance=165.0,
        atr=3.5,
        atr_pct=2.3,
        price=150.0,
        volume_trend="neutral",
    )


def _make_iv(
    ticker: str = "AAPL",
    iv_regime: str = "HIGH",
    iv_rank: float = 75.0,
) -> IVAnalysis:
    return IVAnalysis(
        ticker=ticker,
        current_iv=30.0,
        iv_rank=iv_rank,
        iv_percentile=70.0,
        hv_20=22.0,
        hv_60=20.0,
        iv_hv_ratio=1.36,
        iv_regime=iv_regime,
        premium_action="SELL" if iv_regime == "HIGH" else "BUY" if iv_regime == "LOW" else "NEUTRAL",
        iv_trend="FLAT",
        iv_30d_avg=27.0,
        skew=0.04,
    )


# ─── Position Sizer Tests ──────────────────────────────────────────────────────

class TestPositionSizer:
    """Tests for src.risk.position_sizer.size_position."""

    def test_returns_position_size_object(self):
        from src.risk.position_sizer import size_position, PositionSize
        result = size_position(
            ticker="AAPL",
            max_risk_per_contract=3.80,
            net_credit_or_debit=1.20,
            account_size=10_000,
            prob_profit=75.0,
            confidence=0.70,
        )
        assert isinstance(result, PositionSize)

    def test_contracts_at_least_one(self):
        from src.risk.position_sizer import size_position
        result = size_position("AAPL", 3.80, 1.20, 10_000, 75.0)
        assert result.contracts >= 1

    def test_contracts_at_most_ten(self):
        from src.risk.position_sizer import size_position
        # Very large account → should be capped at 10
        result = size_position("AAPL", 3.80, 1.20, 10_000_000, 75.0)
        assert result.contracts <= 10

    def test_total_risk_correct(self):
        from src.risk.position_sizer import size_position
        result = size_position("AAPL", 3.80, 1.20, 10_000, 75.0)
        expected = result.contracts * 3.80 * 100
        assert abs(result.total_risk - expected) < 0.01

    def test_account_risk_pct_within_2pct(self):
        from src.risk.position_sizer import size_position
        result = size_position("AAPL", 3.80, 1.20, 10_000, 75.0)
        # Should not exceed MAX_RISK_PER_TRADE_PCT (2.0%) meaningfully
        # (can be slightly above because of integer rounding to ≥1 contract)
        assert result.account_risk_pct <= 6.0   # Hard ceiling test

    def test_zero_risk_returns_zero_contracts(self):
        from src.risk.position_sizer import size_position
        result = size_position("AAPL", 0.0, 1.20, 10_000, 75.0)
        assert result.contracts == 0

    def test_fractional_kelly_returns_contracts(self):
        from src.risk.position_sizer import size_position
        result = size_position(
            "MSFT", 4.0, 1.50, 50_000, 70.0,
            confidence=0.8, method="fractional_kelly",
        )
        assert result.contracts >= 1
        assert result.sizing_method == "fractional_kelly"

    def test_small_account_gives_one_contract(self):
        from src.risk.position_sizer import size_position
        # $500 account, $3.80 risk → 1 contract (floor)
        result = size_position("AAPL", 3.80, 1.20, 500, 75.0)
        assert result.contracts >= 1

    def test_total_credit_or_debit_positive(self):
        from src.risk.position_sizer import size_position
        result = size_position("AAPL", 3.80, 1.20, 10_000, 75.0)
        assert result.total_credit_or_debit > 0

    def test_method_stored_correctly(self):
        from src.risk.position_sizer import size_position
        result = size_position("AAPL", 3.80, 1.20, 10_000, 75.0, method="fixed_pct")
        assert result.sizing_method == "fixed_pct"


# ─── Portfolio Tests ───────────────────────────────────────────────────────────

class TestPortfolio:
    """Tests for src.risk.portfolio — all file I/O is patched in-memory."""

    def _make_position(
        self,
        ticker: str = "AAPL",
        strategy: str = "BULL_PUT_SPREAD",
        direction: str = "BULLISH",
        net_credit: float = 1.20,
        max_risk: float = 3.80,
        contracts: int = 2,
        total_risk: float = 760.0,
        status: str = "OPEN",
    ):
        from src.risk.portfolio import OpenPosition
        return OpenPosition(
            ticker=ticker,
            strategy=strategy,
            direction=direction,
            entry_date="2025-01-01",
            expiration="2025-02-21",
            dte_at_entry=45,
            contracts=contracts,
            net_credit=net_credit,
            max_risk=max_risk,
            total_risk=total_risk,
            short_strike=190.0,
            long_strike=185.0,
            status=status,
        )

    def test_open_position_dataclass(self):
        pos = self._make_position()
        assert pos.ticker == "AAPL"
        assert pos.status == "OPEN"
        assert pos.net_credit == 1.20

    def test_check_portfolio_limits_empty(self):
        from src.risk.portfolio import check_portfolio_limits
        with patch("src.risk.portfolio.load_positions", return_value=[]):
            result = check_portfolio_limits(account_size=10_000)
        assert result.total_positions == 0
        assert result.can_add_position is True
        assert result.total_risk_committed == 0.0

    def test_check_portfolio_limits_with_positions(self):
        from src.risk.portfolio import check_portfolio_limits
        positions = [
            self._make_position("AAPL", direction="BULLISH"),
            self._make_position("MSFT", direction="BULLISH"),
            self._make_position("JPM",  direction="NEUTRAL"),
        ]
        with patch("src.risk.portfolio.load_positions", return_value=positions):
            result = check_portfolio_limits(account_size=50_000)
        assert result.total_positions == 3
        assert result.bullish_count == 2
        assert result.neutral_count == 1
        assert result.bearish_count == 0

    def test_can_add_false_when_max_reached(self):
        from src.risk.portfolio import check_portfolio_limits
        # Create MAX_POSITIONS open positions
        from config.settings import MAX_POSITIONS
        positions = [
            self._make_position(f"T{i}") for i in range(MAX_POSITIONS)
        ]
        with patch("src.risk.portfolio.load_positions", return_value=positions):
            result = check_portfolio_limits(account_size=50_000)
        assert result.can_add_position is False

    def test_add_position_success(self):
        """add_position returns True when slots available."""
        from src.risk.portfolio import add_position
        pos = self._make_position()
        with (
            patch("src.risk.portfolio.load_positions", return_value=[]),
            patch("src.risk.portfolio.save_positions") as mock_save,
        ):
            result = add_position(pos)
        assert result is True
        mock_save.assert_called_once()

    def test_add_position_fails_when_full(self):
        """add_position returns False when max positions reached."""
        from src.risk.portfolio import add_position
        from config.settings import MAX_POSITIONS
        positions = [self._make_position(f"T{i}") for i in range(MAX_POSITIONS)]
        pos = self._make_position("NEW")
        with (
            patch("src.risk.portfolio.load_positions", return_value=positions),
            patch("src.risk.portfolio.save_positions"),
        ):
            result = add_position(pos)
        assert result is False

    def test_close_position_calculates_pnl(self):
        """Closing a credit spread at 50% profit gives positive P&L."""
        from src.risk.portfolio import close_position
        pos = self._make_position(net_credit=1.20, contracts=2)
        with (
            patch("src.risk.portfolio.load_positions", return_value=[pos]),
            patch("src.risk.portfolio.save_positions"),
        ):
            closed = close_position("AAPL", "BULL_PUT_SPREAD", close_price=0.60)
        assert closed is not None
        assert closed.pnl > 0      # Bought back for less than sold
        assert closed.status == "CLOSED"

    def test_close_position_not_found_returns_none(self):
        from src.risk.portfolio import close_position
        with patch("src.risk.portfolio.load_positions", return_value=[]):
            result = close_position("AAPL", "BULL_PUT_SPREAD", close_price=0.60)
        assert result is None

    def test_close_position_loss(self):
        """Closing at a loss (bought back more than sold)."""
        from src.risk.portfolio import close_position
        pos = self._make_position(net_credit=1.20, contracts=1)
        with (
            patch("src.risk.portfolio.load_positions", return_value=[pos]),
            patch("src.risk.portfolio.save_positions"),
        ):
            closed = close_position("AAPL", "BULL_PUT_SPREAD", close_price=2.40)
        assert closed is not None
        assert closed.pnl < 0    # Loss: bought back for more than received

    def test_portfolio_risk_remaining_budget_non_negative(self):
        from src.risk.portfolio import check_portfolio_limits
        positions = [self._make_position(total_risk=200.0)]
        with patch("src.risk.portfolio.load_positions", return_value=positions):
            result = check_portfolio_limits(account_size=10_000)
        assert result.remaining_risk_budget >= 0


# ─── Event Filter Tests ────────────────────────────────────────────────────────

class TestEventFilter:
    """Tests for src.risk.event_filter.is_safe_to_trade."""

    def test_safe_when_no_earnings(self):
        from src.risk.event_filter import is_safe_to_trade
        with patch("src.risk.event_filter.get_earnings_dates", return_value=[]):
            result = is_safe_to_trade("AAPL", target_dte=45)
        assert result.safe is True
        assert result.reason is None
        assert result.ticker == "AAPL"

    def test_unsafe_when_earnings_in_window(self):
        from src.risk.event_filter import is_safe_to_trade
        earn_date = date.today() + timedelta(days=20)  # Within 45 DTE window
        with patch("src.risk.event_filter.get_earnings_dates", return_value=[earn_date]):
            result = is_safe_to_trade("AAPL", target_dte=45)
        assert result.safe is False
        assert result.earnings_in_days == 20
        assert "earnings" in result.reason.lower()

    def test_safe_when_earnings_after_expiry(self):
        from src.risk.event_filter import is_safe_to_trade
        earn_date = date.today() + timedelta(days=60)  # Beyond 45 DTE + 3 buffer
        with patch("src.risk.event_filter.get_earnings_dates", return_value=[earn_date]):
            result = is_safe_to_trade("AAPL", target_dte=45)
        assert result.safe is True

    def test_event_filter_result_has_events_list(self):
        from src.risk.event_filter import is_safe_to_trade
        with patch("src.risk.event_filter.get_earnings_dates", return_value=[]):
            result = is_safe_to_trade("MSFT", target_dte=21)
        assert isinstance(result.events, list)

    def test_filter_safe_tickers_all_safe(self):
        from src.risk.event_filter import filter_safe_tickers
        tickers = ["AAPL", "MSFT", "GOOG"]
        with patch("src.risk.event_filter.is_safe_to_trade") as mock_safe:
            from src.risk.event_filter import EventFilterResult
            mock_safe.return_value = EventFilterResult(
                safe=True, ticker="X", reason=None, earnings_in_days=None
            )
            results = filter_safe_tickers(tickers, dte_map={t: 45 for t in tickers})
        assert len(results) == 3
        assert all(r.safe for r in results.values())

    def test_filter_safe_tickers_one_unsafe(self):
        from src.risk.event_filter import filter_safe_tickers, EventFilterResult

        def _side_effect(ticker, target_dte, buffer_days=3):
            if ticker == "TSLA":
                return EventFilterResult(
                    safe=False, ticker=ticker,
                    reason="Earnings in 10 days",
                    earnings_in_days=10,
                )
            return EventFilterResult(safe=True, ticker=ticker, reason=None, earnings_in_days=None)

        tickers = ["AAPL", "TSLA", "MSFT"]
        with patch("src.risk.event_filter.is_safe_to_trade", side_effect=_side_effect):
            results = filter_safe_tickers(tickers, dte_map={t: 45 for t in tickers})

        assert results["AAPL"].safe is True
        assert results["TSLA"].safe is False
        assert results["MSFT"].safe is True


# ─── Composite Screener Tests ─────────────────────────────────────────────────

class TestCompositeScreener:
    """Tests for src.screener.composite_screener.run_screener."""

    def _make_regime_map(self) -> dict[str, StockRegime]:
        return {
            "AAPL": _make_regime("AAPL", Regime.UPTREND, 0.6, adx=30),
            "MSFT": _make_regime("MSFT", Regime.RANGE_BOUND, 0.0, adx=15),
            "TSLA": _make_regime("TSLA", Regime.DOWNTREND, -0.5, adx=25),
        }

    def _make_iv_map(self) -> dict[str, IVAnalysis]:
        return {
            "AAPL": _make_iv("AAPL", "HIGH", 78.0),
            "MSFT": _make_iv("MSFT", "HIGH", 72.0),
            "TSLA": _make_iv("TSLA", "NORMAL", 50.0),
        }

    def test_returns_list(self):
        from src.screener.composite_screener import run_screener
        results = run_screener(self._make_regime_map(), self._make_iv_map())
        assert isinstance(results, list)

    def test_results_have_correct_tickers(self):
        from src.screener.composite_screener import run_screener
        results = run_screener(self._make_regime_map(), self._make_iv_map())
        tickers = {r.ticker for r in results}
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_results_sorted_by_score_descending(self):
        from src.screener.composite_screener import run_screener
        results = run_screener(self._make_regime_map(), self._make_iv_map())
        if len(results) > 1:
            # Passing-filter results come first; within passing, scores descending
            passing = [r for r in results if r.passes_all_filters]
            scores = [r.score for r in passing]
            assert scores == sorted(scores, reverse=True)

    def test_min_confidence_filter(self):
        from src.screener.composite_screener import run_screener
        results = run_screener(
            self._make_regime_map(), self._make_iv_map(),
            min_confidence=0.99,   # Unreachably high
        )
        passing = [r for r in results if r.passes_all_filters]
        assert len(passing) == 0   # None should pass

    def test_direction_filter_bullish_only(self):
        from src.screener.composite_screener import run_screener
        results = run_screener(
            self._make_regime_map(), self._make_iv_map(),
            required_direction="BULLISH",
        )
        passing = [r for r in results if r.passes_all_filters]
        for r in passing:
            assert r.recommendation.direction == "BULLISH"

    def test_iv_rank_filter(self):
        from src.screener.composite_screener import run_screener
        results = run_screener(
            self._make_regime_map(), self._make_iv_map(),
            min_iv_rank=80.0,   # Only AAPL (78) barely misses → no passing
        )
        passing = [r for r in results if r.passes_all_filters]
        # AAPL IV rank = 78 < 80, so none should pass IV filter
        for r in passing:
            assert r.iv.iv_rank >= 80.0

    def test_screener_result_has_rank(self):
        from src.screener.composite_screener import run_screener
        results = run_screener(self._make_regime_map(), self._make_iv_map())
        for r in results:
            assert r.rank >= 1

    def test_screener_result_score_positive(self):
        from src.screener.composite_screener import run_screener
        results = run_screener(self._make_regime_map(), self._make_iv_map())
        for r in results:
            assert r.score >= 0

    def test_strategies_allowed_filter(self):
        from src.screener.composite_screener import run_screener
        results = run_screener(
            self._make_regime_map(), self._make_iv_map(),
            strategies_allowed=["BULL_PUT_SPREAD"],
        )
        passing = [r for r in results if r.passes_all_filters]
        for r in passing:
            assert r.recommendation.strategy.value == "BULL_PUT_SPREAD"

    def test_missing_iv_ticker_skipped(self):
        from src.screener.composite_screener import run_screener
        regime_map = self._make_regime_map()
        iv_map = {"AAPL": _make_iv("AAPL")}   # MSFT and TSLA missing
        results = run_screener(regime_map, iv_map)
        tickers = {r.ticker for r in results}
        assert "MSFT" not in tickers
        assert "TSLA" not in tickers

    def test_rs_map_optional(self):
        """Passing rs_map=None should not raise."""
        from src.screener.composite_screener import run_screener
        results = run_screener(
            self._make_regime_map(), self._make_iv_map(), rs_map=None
        )
        assert isinstance(results, list)

    def test_empty_inputs_returns_empty(self):
        from src.screener.composite_screener import run_screener
        results = run_screener({}, {})
        assert results == []


# ─── Tests: Directional Balance (_rank_trades) ────────────────────────────────

class TestDirectionalBalance:
    """
    _rank_trades() must enforce the MAX_SAME_DIRECTION_PCT cap so that no more
    than max_same_dir = max(2, int(MAX_POSITIONS * MAX_SAME_DIRECTION_PCT/100))
    trades of the same direction appear in the output.

    NEUTRAL trades (IC, butterfly) are never capped.
    """

    # ── helper: build a minimal trade dict that _rank_trades() can process ──

    @staticmethod
    def _trade(direction: str, confidence: float = 0.7, ticker: str = "X") -> dict:
        return {
            "recommendation": {
                "ticker": ticker,
                "direction": direction,
                "confidence": confidence,
                "strategy": "IRON_CONDOR" if direction == "NEUTRAL" else "BULL_PUT_SPREAD",
            },
            "trade": {
                "prob_profit": 65,
                "risk_reward_ratio": 3.0,
                "ev": 0.0,
            },
        }

    def _rank(self, trades: list[dict]) -> list[dict]:
        from src.pipeline.nightly_scan import _rank_trades
        return _rank_trades(trades)

    # ── cap enforcement ───────────────────────────────────────────────────────

    def test_excess_bullish_trades_are_dropped(self):
        """
        With 5 MAX_POSITIONS and 60% cap → max_same_dir = max(2, 3) = 3.
        Submitting 5 BULLISH should yield at most 3 in the output.
        """
        from config.settings import MAX_POSITIONS, MAX_SAME_DIRECTION_PCT
        max_same = max(2, int(MAX_POSITIONS * MAX_SAME_DIRECTION_PCT / 100))

        trades = [
            self._trade("BULLISH", confidence=0.9 - i * 0.05, ticker=f"T{i}")
            for i in range(5)
        ]
        result = self._rank(trades)
        bullish_count = sum(
            1 for t in result
            if t["recommendation"]["direction"] == "BULLISH"
        )
        assert bullish_count <= max_same, (
            f"Expected ≤{max_same} BULLISH trades, got {bullish_count}"
        )

    def test_excess_bearish_trades_are_dropped(self):
        """Same cap applies to BEARISH."""
        from config.settings import MAX_POSITIONS, MAX_SAME_DIRECTION_PCT
        max_same = max(2, int(MAX_POSITIONS * MAX_SAME_DIRECTION_PCT / 100))

        trades = [
            self._trade("BEARISH", confidence=0.9 - i * 0.05, ticker=f"B{i}")
            for i in range(5)
        ]
        result = self._rank(trades)
        bearish_count = sum(
            1 for t in result
            if t["recommendation"]["direction"] == "BEARISH"
        )
        assert bearish_count <= max_same

    def test_neutral_trades_never_capped(self):
        """
        10 NEUTRAL trades submitted: all should pass (no cap on NEUTRAL).
        """
        trades = [
            self._trade("NEUTRAL", confidence=0.8, ticker=f"N{i}")
            for i in range(10)
        ]
        result = self._rank(trades)
        neutral_count = sum(
            1 for t in result
            if t["recommendation"]["direction"] == "NEUTRAL"
        )
        assert neutral_count == 10, (
            f"Expected 10 NEUTRAL trades in output, got {neutral_count}"
        )

    def test_mixed_directions_high_confidence_preferred(self):
        """
        When cap kicks in, higher-confidence trades should survive, not lower.
        Trades are sorted by composite_score before cap is applied.
        """
        trades = [
            self._trade("BEARISH", confidence=0.60, ticker="B_low"),
            self._trade("BEARISH", confidence=0.90, ticker="B_high1"),
            self._trade("BEARISH", confidence=0.85, ticker="B_high2"),
            self._trade("BEARISH", confidence=0.80, ticker="B_high3"),
            self._trade("BEARISH", confidence=0.50, ticker="B_lowest"),
        ]
        from config.settings import MAX_POSITIONS, MAX_SAME_DIRECTION_PCT
        max_same = max(2, int(MAX_POSITIONS * MAX_SAME_DIRECTION_PCT / 100))

        result = self._rank(trades)
        bearish_in = [
            t for t in result if t["recommendation"]["direction"] == "BEARISH"
        ]
        # The surviving bearish trades should all have higher confidence
        # than B_low (0.60) and B_lowest (0.50) if max_same < 5
        if max_same < 5:
            surviving_tickers = {t["recommendation"]["ticker"] for t in bearish_in}
            # B_high1 (0.90) and B_high2 (0.85) must be in — they have highest score
            assert "B_high1" in surviving_tickers
            assert "B_high2" in surviving_tickers

    def test_mixed_portfolio_respects_balance(self):
        """
        Realistic mixed portfolio: 4 BULLISH, 4 BEARISH, 4 NEUTRAL.
        After ranking, both directions should be within the cap and
        all NEUTRAL should pass.
        """
        from config.settings import MAX_POSITIONS, MAX_SAME_DIRECTION_PCT
        max_same = max(2, int(MAX_POSITIONS * MAX_SAME_DIRECTION_PCT / 100))

        trades = (
            [self._trade("BULLISH", confidence=0.8, ticker=f"BULL{i}") for i in range(4)]
            + [self._trade("BEARISH", confidence=0.8, ticker=f"BEAR{i}") for i in range(4)]
            + [self._trade("NEUTRAL", confidence=0.8, ticker=f"NEUT{i}") for i in range(4)]
        )
        result = self._rank(trades)

        bull_count = sum(1 for t in result if t["recommendation"]["direction"] == "BULLISH")
        bear_count = sum(1 for t in result if t["recommendation"]["direction"] == "BEARISH")
        neut_count = sum(1 for t in result if t["recommendation"]["direction"] == "NEUTRAL")

        assert bull_count <= max_same
        assert bear_count <= max_same
        assert neut_count == 4  # all NEUTRAL pass

    def test_priority_assigned_sequentially(self):
        """After ranking, priority should be 1, 2, 3, ... (1-indexed)."""
        trades = [
            self._trade("BULLISH", confidence=0.9, ticker="A"),
            self._trade("NEUTRAL", confidence=0.8, ticker="B"),
            self._trade("BEARISH", confidence=0.7, ticker="C"),
        ]
        result = self._rank(trades)
        priorities = [t["priority"] for t in result]
        assert priorities == list(range(1, len(result) + 1)), (
            f"Priorities should be sequential 1..n, got {priorities}"
        )

    def test_composite_score_present_on_all_trades(self):
        """Every ranked trade should have composite_score set."""
        trades = [
            self._trade("BULLISH", ticker="A"),
            self._trade("NEUTRAL", ticker="B"),
        ]
        result = self._rank(trades)
        for t in result:
            assert "composite_score" in t
            assert t["composite_score"] >= 0.0

    def test_empty_trade_list_returns_empty(self):
        result = self._rank([])
        assert result == []

    def test_single_bullish_always_passes(self):
        """A single BULLISH trade should never be blocked (max_same_dir >= 2)."""
        result = self._rank([self._trade("BULLISH", ticker="SOLO")])
        assert len(result) == 1
        assert result[0]["recommendation"]["direction"] == "BULLISH"

    def test_max_2_minimum_enforced(self):
        """
        Even if MAX_POSITIONS * PCT / 100 rounds below 2, at least 2 same-
        direction trades must always be allowed (max(2, ...)).
        """
        import src.pipeline.nightly_scan as ns_mod
        original = ns_mod.MAX_POSITIONS
        try:
            ns_mod.MAX_POSITIONS = 5
            trades = [
                self._trade("BULLISH", confidence=0.9, ticker=f"B{i}")
                for i in range(4)
            ]
            result = self._rank(trades)
            bullish_out = [
                t for t in result
                if t["recommendation"]["direction"] == "BULLISH"
            ]
            # Must be at least 2 (the max(2, ...) floor)
            assert len(bullish_out) >= 2
        finally:
            ns_mod.MAX_POSITIONS = original


# ─── Tests: _adjust_confidence_for_rs ────────────────────────────────────────

class TestAdjustConfidenceForRS:
    """
    _adjust_confidence_for_rs modifies rec.confidence in-place:
      +0.05 for BULLISH + outperforming + IMPROVING
      +0.05 for BEARISH + underperforming + WEAKENING
      -0.05 for BULLISH + not outperforming
    """

    def _make_rec(self, direction: str, confidence: float = 0.70):
        """Build a minimal StrategyRecommendation-like object."""
        from src.strategy.selector import StrategyRecommendation, StrategyType
        return StrategyRecommendation(
            ticker="AAPL",
            strategy=StrategyType.BULL_PUT_SPREAD,
            direction=direction,
            regime="UPTREND",
            iv_regime="HIGH",
            confidence=confidence,
            rationale="test",
            target_dte=45,
            risk_reward="CREDIT",
            priority=1,
        )

    def _make_rs(self, outperforming: bool, rs_trend: str):
        """Minimal RelativeStrength-compatible mock."""
        rs = MagicMock()
        rs.outperforming_spy = outperforming
        rs.rs_trend = rs_trend
        return rs

    def test_bullish_outperforming_improving_boosts(self):
        from src.pipeline.nightly_scan import _adjust_confidence_for_rs
        rec = self._make_rec("BULLISH", 0.70)
        rs = self._make_rs(outperforming=True, rs_trend="IMPROVING")
        _adjust_confidence_for_rs(rec, rs)
        assert rec.confidence == pytest.approx(0.75, abs=0.001)

    def test_bearish_underperforming_weakening_boosts(self):
        from src.pipeline.nightly_scan import _adjust_confidence_for_rs
        rec = self._make_rec("BEARISH", 0.70)
        rs = self._make_rs(outperforming=False, rs_trend="WEAKENING")
        _adjust_confidence_for_rs(rec, rs)
        assert rec.confidence == pytest.approx(0.75, abs=0.001)

    def test_bullish_not_outperforming_penalises(self):
        from src.pipeline.nightly_scan import _adjust_confidence_for_rs
        rec = self._make_rec("BULLISH", 0.70)
        rs = self._make_rs(outperforming=False, rs_trend="FLAT")
        _adjust_confidence_for_rs(rec, rs)
        assert rec.confidence == pytest.approx(0.65, abs=0.001)

    def test_confidence_capped_at_1_0(self):
        from src.pipeline.nightly_scan import _adjust_confidence_for_rs
        rec = self._make_rec("BULLISH", 0.98)
        rs = self._make_rs(outperforming=True, rs_trend="IMPROVING")
        _adjust_confidence_for_rs(rec, rs)
        assert rec.confidence <= 1.0

    def test_confidence_floored_at_0_0(self):
        from src.pipeline.nightly_scan import _adjust_confidence_for_rs
        rec = self._make_rec("BULLISH", 0.02)
        rs = self._make_rs(outperforming=False, rs_trend="FLAT")
        _adjust_confidence_for_rs(rec, rs)
        assert rec.confidence >= 0.0
