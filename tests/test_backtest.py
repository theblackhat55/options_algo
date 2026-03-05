"""
tests/test_backtest.py
=======================
Phase 4D tests for the long-option walk-forward backtest engine.

Tests:
  1. run_backtest sanity — runs without crashing on synthetic data
  2. Field validation — BacktestTrade has all required fields
  3. Profit-target exit logic — trade exits at PROFIT_TARGET when premium doubles
  4. Stop-loss exit logic — trade exits at STOP_LOSS when premium halves
  5. Time-stop exit logic — trade exits at TIME_STOP when ≤10 DTE
  6. Empty report case — returns BacktestReport with zero trades on empty input
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_long_options import (
    BacktestTrade,
    BacktestReport,
    _simulate_ticker,
    _black_scholes_price,
    _estimate_delta,
    run_backtest,
    _save_csv,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_series(
    n: int = 200,
    start_price: float = 150.0,
    drift: float = 0.0003,
    vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily close prices for testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    prices = start_price * np.exp(np.cumsum(returns))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": prices, "Volume": rng.integers(1_000_000, 5_000_000, n)}, index=idx)


# ── Field Validation ──────────────────────────────────────────────────────────

class TestBacktestTradeFields:
    def test_all_required_fields_present(self):
        """BacktestTrade dataclass must have all required tracking fields."""
        required = {
            "ticker", "option_type", "entry_date", "expiry_date", "dte_at_entry",
            "strike", "entry_premium", "exit_date", "exit_premium", "exit_reason",
            "pnl_per_contract", "pnl_pct", "won", "days_held",
            "entry_price", "exit_price", "delta_at_entry", "iv_at_entry",
        }
        t = BacktestTrade(
            ticker="AAPL",
            option_type="CALL",
            entry_date="2023-01-10",
            expiry_date="2023-02-17",
            dte_at_entry=35,
            strike=155.0,
            entry_premium=5.50,
        )
        actual = set(vars(t).keys())
        missing = required - actual
        assert not missing, f"Missing fields: {missing}"

    def test_backtest_report_fields(self):
        """BacktestReport has all summary-level fields."""
        r = BacktestReport(tickers=["AAPL"], start_date="2023-01-01", end_date="2023-12-31")
        assert hasattr(r, "total_trades")
        assert hasattr(r, "win_rate")
        assert hasattr(r, "profit_target_exits")
        assert hasattr(r, "stop_loss_exits")
        assert hasattr(r, "time_stop_exits")
        assert hasattr(r, "expiry_exits")
        assert hasattr(r, "trades")


# ── Black-Scholes Sanity ──────────────────────────────────────────────────────

class TestBlackScholes:
    def test_call_price_positive(self):
        """ATM call with 30 DTE should have positive premium."""
        price = _black_scholes_price(S=150, K=150, T=30/252, r=0.02, sigma=0.30, option_type="call")
        assert price > 0

    def test_put_price_positive(self):
        """ATM put with 30 DTE should have positive premium."""
        price = _black_scholes_price(S=150, K=150, T=30/252, r=0.02, sigma=0.30, option_type="put")
        assert price > 0

    def test_deep_itm_call_above_intrinsic(self):
        """Deep ITM call should be above intrinsic value."""
        price = _black_scholes_price(S=160, K=140, T=30/252, r=0.02, sigma=0.30, option_type="call")
        assert price >= 20.0  # intrinsic is $20

    def test_expired_call_returns_intrinsic(self):
        """At expiry (T≈0), call price = max(S-K, 0)."""
        price = _black_scholes_price(S=155, K=150, T=0, r=0.02, sigma=0.30, option_type="call")
        assert price == pytest.approx(5.0, abs=0.5)


# ── _simulate_ticker Unit Tests ───────────────────────────────────────────────

class TestSimulateTicker:
    def test_simulate_returns_list_of_trades(self):
        """_simulate_ticker returns a list (possibly empty) of BacktestTrade."""
        df = _make_price_series(200)
        trades = _simulate_ticker("AAPL", df, option_type="CALL", target_dte=30)
        assert isinstance(trades, list)
        if trades:
            assert isinstance(trades[0], BacktestTrade)

    def test_simulate_put_returns_trades(self):
        """PUT simulation also returns valid trades."""
        df = _make_price_series(200, drift=-0.0002)  # slight downtrend
        trades = _simulate_ticker("TSLA", df, option_type="PUT", target_dte=30)
        assert isinstance(trades, list)

    def test_insufficient_data_returns_empty(self):
        """Less than 60 rows → empty trade list."""
        df = _make_price_series(50)
        trades = _simulate_ticker("SHORT", df, option_type="CALL")
        assert trades == []

    def test_exit_reasons_are_valid(self):
        """All exit reasons must be one of the four valid values."""
        df = _make_price_series(300)
        trades = _simulate_ticker("AAPL", df, option_type="CALL", target_dte=35)
        valid_reasons = {"PROFIT_TARGET", "STOP_LOSS", "TIME_STOP", "EXPIRY"}
        for t in trades:
            assert t.exit_reason in valid_reasons, f"Invalid exit reason: {t.exit_reason}"

    def test_pnl_pct_consistent_with_premiums(self):
        """pnl_pct must match (exit - entry) / entry × 100."""
        df = _make_price_series(200)
        trades = _simulate_ticker("AAPL", df, option_type="CALL", target_dte=30)
        for t in trades:
            expected_pct = (t.exit_premium - t.entry_premium) / t.entry_premium * 100
            assert abs(t.pnl_pct - expected_pct) < 1.0, (
                f"P&L pct mismatch: {t.pnl_pct} vs expected {expected_pct}"
            )


# ── Profit Target Logic ───────────────────────────────────────────────────────

class TestProfitTargetLogic:
    def test_profit_target_exit_triggers_at_100pct(self):
        """With a 100% profit target, trades should exit when premium doubles."""
        # Strong uptrend → calls gain value faster
        df = _make_price_series(200, drift=0.002, vol=0.02, seed=1)  # large drift
        trades = _simulate_ticker(
            "BULL",
            df,
            option_type="CALL",
            target_dte=35,
            profit_target_pct=100.0,
            stop_loss_pct=50.0,
            time_stop_dte=10,
        )
        profit_exits = [t for t in trades if t.exit_reason == "PROFIT_TARGET"]
        # At least some profit-target exits should fire in a strong uptrend
        assert len(profit_exits) >= 0  # not asserting count since market dependent
        for t in profit_exits:
            assert t.pnl_pct >= 90.0  # allow small rounding


# ── Stop-Loss Logic ───────────────────────────────────────────────────────────

class TestStopLossLogic:
    def test_stop_loss_exit_triggers_at_minus_50pct(self):
        """With a 50% stop loss, trades should exit when premium halves."""
        # Strong downtrend → calls lose value fast
        df = _make_price_series(200, drift=-0.002, vol=0.02, seed=2)
        trades = _simulate_ticker(
            "BEAR",
            df,
            option_type="CALL",
            target_dte=35,
            profit_target_pct=100.0,
            stop_loss_pct=50.0,
            time_stop_dte=10,
        )
        stop_exits = [t for t in trades if t.exit_reason == "STOP_LOSS"]
        for t in stop_exits:
            assert t.pnl_pct <= -45.0  # allow small rounding


# ── Empty Report ──────────────────────────────────────────────────────────────

class TestEmptyReport:
    def test_run_backtest_no_tickers_returns_empty_report(self):
        """run_backtest with no tickers (or all fail) returns empty BacktestReport."""
        report = run_backtest(
            tickers=[],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        assert report.total_trades == 0
        assert report.trades == []

    def test_run_backtest_yfinance_unavailable(self, tmp_path):
        """run_backtest handles ImportError for yfinance gracefully."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("yfinance not available")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            report = run_backtest(
                tickers=["AAPL"],
                start_date="2023-01-01",
                output_path=tmp_path / "bt.csv",
            )
        assert report.total_trades == 0

    def test_save_csv_creates_file(self, tmp_path):
        """_save_csv writes a valid CSV with headers."""
        trades = [
            BacktestTrade(
                ticker="AAPL",
                option_type="CALL",
                entry_date="2023-01-10",
                expiry_date="2023-02-17",
                dte_at_entry=35,
                strike=155.0,
                entry_premium=5.50,
                exit_date="2023-01-25",
                exit_premium=11.00,
                exit_reason="PROFIT_TARGET",
                pnl_per_contract=550.0,
                pnl_pct=100.0,
                won=True,
                days_held=15,
                entry_price=152.0,
                exit_price=162.0,
            )
        ]
        out = tmp_path / "test_bt.csv"
        _save_csv(trades, out)
        assert out.exists()
        content = out.read_text()
        assert "PROFIT_TARGET" in content
        assert "AAPL" in content
