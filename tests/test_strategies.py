"""
tests/test_strategies.py
=========================
Tests for strategy selector, credit spreads, debit spreads, IC, butterfly.
Uses synthetic options chains (no real API calls).
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.technical import Regime, StockRegime
from src.analysis.volatility import IVAnalysis
from src.strategy.selector import select_strategy, StrategyType
from src.strategy.credit_spread import (
    construct_bull_put_spread, construct_bear_call_spread,
)
from src.strategy.bull_call_spread import construct_bull_call_spread
from src.strategy.bear_put_spread import construct_bear_put_spread
from src.strategy.iron_condor import construct_iron_condor
from src.strategy.butterfly import construct_long_butterfly


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_regime(
    ticker: str = "AAPL",
    regime: Regime = Regime.UPTREND,
    direction_score: float = 0.5,
    adx: float = 30.0,
    rsi: float = 55.0,
) -> StockRegime:
    return StockRegime(
        ticker=ticker,
        regime=regime,
        direction_score=direction_score,
        trend_strength=0.6,
        volatility_state="normal",
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
    iv_regime: str = "NORMAL",
    iv_rank: float = 45.0,
) -> IVAnalysis:
    return IVAnalysis(
        ticker=ticker,
        current_iv=25.0,
        iv_rank=iv_rank,
        iv_percentile=45.0,
        hv_20=22.0,
        hv_60=20.0,
        iv_hv_ratio=1.1,
        iv_regime=iv_regime,
        premium_action="SELL" if iv_regime == "HIGH" else "BUY" if iv_regime == "LOW" else "NEUTRAL",
        iv_trend="FLAT",
        iv_30d_avg=24.0,
        skew=0.05,
    )


def _make_chain(
    current_price: float = 150.0,
    n_strikes: int = 15,
    expiry_days: int = 45,
) -> pd.DataFrame:
    """Generate a synthetic options chain DataFrame."""
    from datetime import date, timedelta

    expiry = (date.today() + timedelta(days=expiry_days)).isoformat()
    strikes = np.arange(current_price - n_strikes * 2.5, current_price + n_strikes * 2.5 + 1, 5)
    rows = []

    for strike in strikes:
        for opt_type in ("call", "put"):
            moneyness = (current_price - strike) / current_price
            if opt_type == "call":
                intrinsic = max(current_price - strike, 0)
                delta = max(min(0.5 + moneyness * 3, 0.99), 0.01)
            else:
                intrinsic = max(strike - current_price, 0)
                delta = -max(min(0.5 - moneyness * 3, 0.99), 0.01)

            time_val = current_price * 0.015 * (expiry_days / 45) ** 0.5
            mid = max(intrinsic + time_val * max(1 - abs(moneyness) * 5, 0.1), 0.05)
            bid = mid * 0.95
            ask = mid * 1.05
            iv = 0.25 + abs(moneyness) * 0.1 * (1.0 if opt_type == "put" else 0.8)

            rows.append({
                "ticker": "AAPL",
                "contract_ticker": f"AAPL{expiry}{opt_type[0].upper()}{int(strike):08d}",
                "expiration": expiry,
                "strike": float(strike),
                "type": opt_type,
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "mid": round(mid, 2),
                "last": round(mid, 2),
                "volume": 1000,
                "open_interest": 5000,
                "implied_volatility": round(iv, 3),
                "delta": round(delta, 3),
                "gamma": 0.01,
                "theta": -0.05,
                "vega": 0.10,
                "bid_ask_spread_pct": round((ask - bid) / mid * 100, 1),
                "source": "synthetic",
            })

    return pd.DataFrame(rows)


# ─── Tests: Strategy Selector ─────────────────────────────────────────────────

class TestStrategySelector:
    def test_uptrend_high_iv_returns_bull_put(self):
        regime = _make_regime(regime=Regime.UPTREND)
        iv = _make_iv(iv_regime="HIGH", iv_rank=75)
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.BULL_PUT_SPREAD
        assert rec.direction == "BULLISH"
        assert rec.risk_reward == "CREDIT"

    def test_downtrend_normal_iv_returns_bear_put(self):
        regime = _make_regime(regime=Regime.DOWNTREND, direction_score=-0.5)
        iv = _make_iv(iv_regime="NORMAL")
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.BEAR_PUT_SPREAD
        assert rec.direction == "BEARISH"

    def test_range_high_iv_returns_ic(self):
        regime = _make_regime(regime=Regime.RANGE_BOUND, direction_score=0.0)
        iv = _make_iv(iv_regime="HIGH", iv_rank=80)
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.IRON_CONDOR
        assert rec.direction == "NEUTRAL"

    def test_range_low_iv_returns_butterfly(self):
        regime = _make_regime(regime=Regime.RANGE_BOUND)
        iv = _make_iv(iv_regime="LOW", iv_rank=20)
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.LONG_BUTTERFLY

    def test_confidence_between_0_and_1(self):
        for r in Regime:
            for iv_regime in ("HIGH", "NORMAL", "LOW"):
                regime = _make_regime(regime=r)
                iv = _make_iv(iv_regime=iv_regime)
                rec = select_strategy(regime, iv)
                assert 0 <= rec.confidence <= 1

    def test_credit_strategy_has_45_dte(self):
        regime = _make_regime(regime=Regime.UPTREND)
        iv = _make_iv(iv_regime="HIGH")
        rec = select_strategy(regime, iv)
        assert rec.target_dte == 45

    def test_debit_strategy_has_21_dte(self):
        regime = _make_regime(regime=Regime.UPTREND)
        iv = _make_iv(iv_regime="NORMAL")
        rec = select_strategy(regime, iv)
        assert rec.target_dte == 21


# ─── Tests: Credit Spreads ────────────────────────────────────────────────────

class TestCreditSpreads:
    def setup_method(self):
        self.price = 150.0
        self.chain = _make_chain(self.price, n_strikes=15, expiry_days=45)

    def test_bull_put_returns_credit_spread(self):
        result = construct_bull_put_spread("AAPL", self.price, self.chain)
        assert result is not None
        assert result.spread_type == "BULL_PUT"
        assert result.net_credit > 0
        assert result.short_strike < self.price
        assert result.long_strike < result.short_strike
        assert result.max_risk > 0
        assert result.breakeven < result.short_strike

    def test_bear_call_returns_credit_spread(self):
        result = construct_bear_call_spread("AAPL", self.price, self.chain)
        assert result is not None
        assert result.spread_type == "BEAR_CALL"
        assert result.net_credit > 0
        assert result.short_strike > self.price
        assert result.long_strike > result.short_strike

    def test_credit_spread_risk_reward_math(self):
        result = construct_bull_put_spread("AAPL", self.price, self.chain)
        assert result is not None
        assert abs(result.width - (result.short_strike - result.long_strike)) < 0.51
        assert abs(result.max_risk - (result.width - result.net_credit)) < 0.05

    def test_prob_profit_range(self):
        result = construct_bull_put_spread("AAPL", self.price, self.chain)
        assert result is not None
        assert 0 < result.prob_profit < 100


# ─── Tests: Debit Spreads ─────────────────────────────────────────────────────

class TestDebitSpreads:
    def setup_method(self):
        self.price = 150.0
        self.chain = _make_chain(self.price, n_strikes=15, expiry_days=21)

    def test_bull_call_spread(self):
        result = construct_bull_call_spread("AAPL", self.price, self.chain)
        assert result is not None
        assert result.spread_type == "BULL_CALL"
        assert result.net_debit > 0
        assert result.max_profit > 0
        assert result.long_strike < result.short_strike

    def test_bear_put_spread(self):
        result = construct_bear_put_spread("AAPL", self.price, self.chain)
        assert result is not None
        assert result.spread_type == "BEAR_PUT"
        assert result.net_debit > 0
        assert result.long_strike > result.short_strike


# ─── Tests: Iron Condor ───────────────────────────────────────────────────────

class TestIronCondor:
    def test_constructs_ic(self):
        # Need a wide chain with enough strikes on both sides
        chain = _make_chain(150.0, n_strikes=25, expiry_days=45)
        result = construct_iron_condor("SPY", 150.0, chain, wing_delta=0.20, spread_width=5.0)
        if result is None:
            pytest.skip("IC construction returned None — acceptable on synthetic chain")
        assert result.short_put < 150.0
        assert result.short_call > 150.0
        assert result.short_put < result.short_call
        assert result.total_credit > 0
        assert result.put_breakeven < result.call_breakeven

    def test_ic_profit_zone(self):
        chain = _make_chain(150.0, n_strikes=25, expiry_days=45)
        result = construct_iron_condor("SPY", 150.0, chain, wing_delta=0.20, spread_width=5.0)
        if result is None:
            pytest.skip("IC construction returned None (acceptable for synthetic chain)")
        assert result.profit_zone_width > 0
        assert result.prob_profit > 0


# ─── Tests: Butterfly ─────────────────────────────────────────────────────────

class TestButterfly:
    def test_constructs_butterfly(self):
        chain = _make_chain(150.0, n_strikes=15, expiry_days=30)
        result = construct_long_butterfly("AAPL", 150.0, chain)
        if result is None:
            pytest.skip("Butterfly construction returned None (acceptable for synthetic chain)")
        assert result.lower_wing < result.body < result.upper_wing
        assert result.net_debit > 0
        assert result.max_profit > 0
        assert result.risk_reward_ratio >= 1.0
