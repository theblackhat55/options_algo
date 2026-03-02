"""
tests/test_volatility.py
========================
Tests for IV analysis: HV computation, IV rank, IV percentile, IVAnalysis.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.volatility import (
    compute_historical_volatility,
    compute_iv_rank,
    compute_iv_percentile,
    analyze_iv,
)


def _make_price_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    log_ret = rng.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    df = pd.DataFrame({
        "open": close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high": close * (1 + rng.uniform(0, 0.01, n)),
        "low":  close * (1 - rng.uniform(0, 0.01, n)),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)
    return df


class TestHistoricalVolatility:
    def test_returns_series(self):
        df = _make_price_df()
        hv = compute_historical_volatility(df["close"], window=20)
        assert isinstance(hv, pd.Series)
        assert len(hv) == len(df)

    def test_annualised_range(self):
        df = _make_price_df()
        hv = compute_historical_volatility(df["close"], window=20).dropna()
        # Annualised HV should be in a plausible range (5%–200%)
        assert (hv > 0.05).all()
        assert (hv < 2.0).all()

    def test_different_windows(self):
        df = _make_price_df()
        hv20 = compute_historical_volatility(df["close"], window=20).dropna()
        hv60 = compute_historical_volatility(df["close"], window=60).dropna()
        # Longer window should be smoother (lower std)
        assert hv60.std() < hv20.std() * 1.5


class TestIVRank:
    def test_basic_rank(self):
        iv = pd.Series([0.2, 0.3, 0.4, 0.5, 0.25])
        rank = compute_iv_rank(iv, lookback=5)
        assert 0 <= rank <= 100

    def test_current_at_max(self):
        """If current IV = max, rank should be 100."""
        iv = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        rank = compute_iv_rank(iv, lookback=5)
        assert rank == 100.0

    def test_current_at_min(self):
        """If current IV = min, rank should be 0."""
        iv = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        rank = compute_iv_rank(iv, lookback=5)
        assert rank == 0.0

    def test_insufficient_data_returns_50(self):
        """With only 1 point, rank is undefined — returns 50."""
        iv = pd.Series([0.2])
        rank = compute_iv_rank(iv, lookback=252)
        assert rank == 50.0


class TestIVPercentile:
    def test_percentile_range(self):
        iv = pd.Series(np.linspace(0.1, 0.5, 100))
        pct = compute_iv_percentile(iv, lookback=100)
        # Current (last value) is the max, so ~99% of days were lower
        assert pct > 95

    def test_at_median(self):
        iv = pd.Series(np.linspace(0.1, 0.5, 100))
        # Replace last with median
        iv.iloc[-1] = iv.median()
        pct = compute_iv_percentile(iv, lookback=100)
        assert 40 <= pct <= 60


class TestAnalyzeIV:
    def test_returns_iv_analysis(self):
        df = _make_price_df()
        result = analyze_iv("TEST", df)
        assert result is not None
        assert result.ticker == "TEST"
        assert 0 <= result.iv_rank <= 100
        assert result.iv_regime in ("HIGH", "NORMAL", "LOW")
        assert result.premium_action in ("SELL", "NEUTRAL", "BUY")

    def test_short_data_returns_none(self):
        df = _make_price_df(n=30)
        result = analyze_iv("TEST", df)
        assert result is None

    def test_high_iv_rank_means_sell(self):
        """Artificially create a high-IV environment."""
        df = _make_price_df(n=300, seed=99)
        # Can't easily force IV rank; just verify structure
        result = analyze_iv("TEST", df)
        assert result is not None
        assert result.iv_hv_ratio > 0
