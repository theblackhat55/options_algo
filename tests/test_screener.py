"""
tests/test_screener.py
======================
Tests for technical regime classification and screener logic.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.technical import classify_regime, classify_universe, Regime, get_regime_summary


def _make_trending_up(n: int = 200) -> pd.DataFrame:
    """Strong uptrend: steady positive returns."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0.003, 0.008, n))
    vol = rng.uniform(0.5, 1.5, n)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.008,
        "low":  close * 0.995,
        "close": close,
        "volume": (1_000_000 * vol).astype(float),
    }, index=dates)


def _make_downtrend(n: int = 200) -> pd.DataFrame:
    """Steady downtrend."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    rng = np.random.default_rng(2)
    close = 100 * np.cumprod(1 + rng.normal(-0.003, 0.008, n))
    vol = rng.uniform(0.5, 1.5, n)
    return pd.DataFrame({
        "open": close * 1.001,
        "high": close * 1.005,
        "low":  close * 0.992,
        "close": close,
        "volume": (1_000_000 * vol).astype(float),
    }, index=dates)


def _make_range_bound(n: int = 200) -> pd.DataFrame:
    """Mean-reverting range."""
    dates = pd.bdate_range("2023-01-01", periods=n)
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    # Keep it roughly mean-reverting
    for i in range(1, n):
        close[i] = close[i] * 0.5 + 100 * 0.5
    close = close + rng.normal(0, 0.3, n)
    vol = rng.uniform(0.5, 1.5, n)
    return pd.DataFrame({
        "open": close * 0.9995,
        "high": close * 1.003,
        "low":  close * 0.997,
        "close": close,
        "volume": (1_000_000 * vol).astype(float),
    }, index=dates)


class TestClassifyRegime:
    def test_returns_stock_regime(self):
        df = _make_trending_up()
        result = classify_regime("AAPL", df)
        assert result is not None
        assert result.ticker == "AAPL"

    def test_uptrend_detected(self):
        df = _make_trending_up()
        result = classify_regime("AAPL", df)
        assert result is not None
        assert result.regime in (
            Regime.UPTREND, Regime.STRONG_UPTREND, Regime.REVERSAL_DOWN
        )
        assert result.direction_score > 0

    def test_downtrend_detected(self):
        df = _make_downtrend()
        result = classify_regime("TSLA", df)
        assert result is not None
        # Should be bearish
        assert result.direction_score < 0.3

    def test_insufficient_data_returns_none(self):
        df = _make_trending_up(n=30)
        result = classify_regime("AAPL", df)
        assert result is None

    def test_regime_fields(self):
        df = _make_trending_up()
        result = classify_regime("AAPL", df)
        assert result is not None
        assert 0 <= result.adx <= 100
        assert 0 <= result.rsi <= 100
        assert 0 <= result.trend_strength <= 1
        assert -1 <= result.direction_score <= 1
        assert result.atr > 0
        assert result.support < result.price
        assert result.resistance > result.support

    def test_volume_trend(self):
        df = _make_trending_up()
        result = classify_regime("AAPL", df)
        assert result is not None
        assert result.volume_trend in ("rising", "falling", "neutral")


class TestClassifyUniverse:
    def test_classifies_multiple(self):
        data = {
            "AAPL": _make_trending_up(),
            "TSLA": _make_downtrend(),
            "JPM": _make_range_bound(),
        }
        results = classify_universe(data)
        assert len(results) == 3

    def test_handles_short_data(self):
        data = {
            "AAPL": _make_trending_up(),
            "SHORT": _make_trending_up(n=10),
        }
        results = classify_universe(data)
        tickers = [r.ticker for r in results]
        assert "AAPL" in tickers
        assert "SHORT" not in tickers


class TestGetRegimeSummary:
    def test_counts_regimes(self):
        data = {
            "A": _make_trending_up(),
            "B": _make_downtrend(),
            "C": _make_range_bound(),
        }
        results = classify_universe(data)
        summary = get_regime_summary(results)
        assert isinstance(summary, dict)
        assert sum(summary.values()) == len(results)
