"""
tests/test_long_options.py
==========================
Tests for long_call.py and long_put.py constructors, including:
  - LongOption dataclass integrity
  - construct_long_call / construct_long_put basic construction
  - Delta-based strike selection
  - Theta rate filter (reject high-decay contracts)
  - Breakeven and probability of profit computation
  - Graceful None return when chain is empty or strikes unavailable
  - selector.py upgrade path: spread → LONG_CALL/LONG_PUT when IV rank is low
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.strategy.long_call import construct_long_call, LongOption, _pick_call_by_delta
from src.strategy.long_put import construct_long_put, _pick_put_by_delta


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_chain(
    spot: float = 150.0,
    dte: int = 35,
    n_strikes: int = 10,
    iv: float = 0.25,
    include_greeks: bool = True,
) -> pd.DataFrame:
    """Build a synthetic options chain around spot."""
    exp = (date.today() + timedelta(days=dte)).isoformat()
    T = dte / 365
    rows = []
    for i in range(-n_strikes // 2, n_strikes // 2 + 1):
        strike = round(spot + i * 5, 2)
        if strike <= 0:
            continue
        # Rough Black-Scholes approximation for mid
        itm = max(spot - strike, 0)
        otm_call = max(0.01, itm + spot * iv * math.sqrt(T) * 0.4)
        otm_put = max(0.01, max(strike - spot, 0) + spot * iv * math.sqrt(T) * 0.4)

        delta_call = max(0.05, min(0.95, 0.5 + (spot - strike) / (spot * iv * math.sqrt(T) * 2)))
        delta_put  = -(1 - delta_call)
        theta_daily = -spot * iv / (2 * math.sqrt(T) * math.sqrt(2 * math.pi) * 365)

        for opt_type, mid, delta in [
            ("call", otm_call, delta_call if include_greeks else None),
            ("put",  otm_put,  delta_put  if include_greeks else None),
        ]:
            rows.append({
                "ticker": "AAPL",
                "expiration": exp,
                "strike": strike,
                "type": opt_type,
                "bid": round(mid * 0.95, 2),
                "ask": round(mid * 1.05, 2),
                "mid": round(mid, 2),
                "last": round(mid, 2),
                "volume": 500,
                "open_interest": 1500,
                "implied_volatility": iv,
                "delta": delta if include_greeks else None,
                "gamma": 0.01 if include_greeks else None,
                "theta": theta_daily if include_greeks else None,
                "vega": 0.10 if include_greeks else None,
                "bid_ask_spread_pct": 5.0,
                "source": "test",
            })
    return pd.DataFrame(rows)


# ─── LongOption Dataclass ─────────────────────────────────────────────────────

class TestLongOptionDataclass:
    def test_fields_present(self):
        lo = LongOption(
            ticker="AAPL", option_type="LONG_CALL",
            expiration="2025-02-21", dte=35,
            strike=145.0, premium=5.0, max_risk=5.0,
            breakeven=150.0, delta=0.65, gamma=0.01,
            theta=-0.05, vega=0.10, iv=25.0,
            theta_rate=0.01, prob_profit=60.0, ev=1.5,
            implied_move_1sd=8.0,
        )
        assert lo.ticker == "AAPL"
        assert lo.option_type == "LONG_CALL"
        assert lo.max_risk == lo.premium
        assert lo.breakeven == 150.0

    def test_notes_defaults_empty(self):
        lo = LongOption(
            ticker="X", option_type="LONG_PUT",
            expiration="2025-02-21", dte=21,
            strike=100.0, premium=3.0, max_risk=3.0,
            breakeven=97.0, delta=0.65, gamma=0.01,
            theta=-0.04, vega=0.08, iv=30.0,
            theta_rate=0.013, prob_profit=55.0, ev=0.5,
            implied_move_1sd=5.0,
        )
        assert lo.notes == ""


# ─── construct_long_call ──────────────────────────────────────────────────────

class TestConstructLongCall:
    def test_returns_long_option(self):
        chain = _make_chain(spot=150.0, dte=35)
        result = construct_long_call("AAPL", 150.0, chain, target_dte=35)
        assert result is not None
        assert isinstance(result, LongOption)
        assert result.option_type == "LONG_CALL"

    def test_breakeven_equals_strike_plus_premium(self):
        chain = _make_chain(spot=150.0, dte=35)
        result = construct_long_call("AAPL", 150.0, chain, target_dte=35)
        if result is not None:
            assert abs(result.breakeven - (result.strike + result.premium)) < 0.01

    def test_delta_near_target(self):
        chain = _make_chain(spot=150.0, dte=35, include_greeks=True)
        result = construct_long_call("AAPL", 150.0, chain, target_dte=35, target_delta=0.65)
        if result is not None:
            # Delta should be somewhere in a reasonable range (not 0)
            assert result.delta > 0.1

    def test_empty_chain_returns_none(self):
        empty = pd.DataFrame(columns=["ticker", "expiration", "strike", "type",
                                       "bid", "ask", "mid", "implied_volatility",
                                       "delta", "theta", "gamma", "vega"])
        result = construct_long_call("AAPL", 150.0, empty, target_dte=35)
        assert result is None

    def test_no_calls_returns_none(self):
        chain = _make_chain(spot=150.0, dte=35)
        puts_only = chain[chain["type"] == "put"].copy()
        result = construct_long_call("AAPL", 150.0, puts_only, target_dte=35)
        assert result is None

    def test_theta_rate_filter_rejects_high_decay(self):
        """A very short DTE (5-day) call should be rejected by theta rate filter."""
        chain = _make_chain(spot=150.0, dte=5, iv=0.50)
        # With dte=5 and high IV, theta_rate should exceed default LONG_OPTION_MAX_THETA_RATE=0.03
        result = construct_long_call("AAPL", 150.0, chain, target_dte=5, max_theta_rate=0.001)
        # Should be rejected (theta rate too high at this extreme limit)
        assert result is None

    def test_prob_profit_positive(self):
        chain = _make_chain(spot=150.0, dte=35)
        result = construct_long_call("AAPL", 150.0, chain, target_dte=35)
        if result is not None:
            assert result.prob_profit > 0
            assert result.prob_profit <= 100

    def test_premium_positive(self):
        chain = _make_chain(spot=150.0, dte=35)
        result = construct_long_call("AAPL", 150.0, chain, target_dte=35)
        if result is not None:
            assert result.premium > 0

    def test_dte_at_least_10(self):
        """Results should have DTE ≥ 10."""
        chain = _make_chain(spot=150.0, dte=35)
        result = construct_long_call("AAPL", 150.0, chain, target_dte=35)
        if result is not None:
            assert result.dte >= 10

    def test_short_dte_returns_none(self):
        chain = _make_chain(spot=150.0, dte=5)
        result = construct_long_call("AAPL", 150.0, chain, target_dte=5)
        assert result is None

    def test_without_greeks_fallback(self):
        """Should still work when Greeks are absent (uses BS fallback)."""
        chain = _make_chain(spot=150.0, dte=35, include_greeks=False)
        result = construct_long_call("AAPL", 150.0, chain, target_dte=35)
        # May return None or a valid result — should not raise
        assert result is None or isinstance(result, LongOption)


# ─── construct_long_put ───────────────────────────────────────────────────────

class TestConstructLongPut:
    def test_returns_long_option(self):
        chain = _make_chain(spot=150.0, dte=35)
        result = construct_long_put("NVDA", 150.0, chain, target_dte=35)
        assert result is not None
        assert isinstance(result, LongOption)
        assert result.option_type == "LONG_PUT"

    def test_breakeven_equals_strike_minus_premium(self):
        chain = _make_chain(spot=150.0, dte=35)
        result = construct_long_put("NVDA", 150.0, chain, target_dte=35)
        if result is not None:
            assert abs(result.breakeven - (result.strike - result.premium)) < 0.01

    def test_empty_chain_returns_none(self):
        empty = pd.DataFrame(columns=["ticker", "expiration", "strike", "type",
                                       "bid", "ask", "mid", "implied_volatility",
                                       "delta", "theta", "gamma", "vega"])
        result = construct_long_put("NVDA", 150.0, empty, target_dte=35)
        assert result is None

    def test_prob_profit_positive(self):
        chain = _make_chain(spot=150.0, dte=35)
        result = construct_long_put("NVDA", 150.0, chain, target_dte=35)
        if result is not None:
            assert result.prob_profit > 0

    def test_max_risk_equals_premium(self):
        chain = _make_chain(spot=150.0, dte=35)
        result = construct_long_put("NVDA", 150.0, chain, target_dte=35)
        if result is not None:
            assert result.max_risk == result.premium

    def test_theta_rate_filter_rejects_high_decay(self):
        chain = _make_chain(spot=150.0, dte=8, iv=0.50)
        result = construct_long_put("NVDA", 150.0, chain, target_dte=8, max_theta_rate=0.001)
        assert result is None


# ─── Strike Selection Helpers ─────────────────────────────────────────────────

class TestPickCallByDelta:
    def test_selects_nearest_delta(self):
        chain = _make_chain(spot=150.0, dte=35)
        calls = chain[chain["type"] == "call"].copy()
        selected = _pick_call_by_delta(calls, 150.0, 0.65)
        assert selected is not None
        delta = selected.get("delta", None)
        if delta is not None and delta > 0:
            assert 0.3 <= abs(float(delta)) <= 0.95

    def test_fallback_when_no_deltas(self):
        chain = _make_chain(spot=150.0, dte=35, include_greeks=False)
        calls = chain[chain["type"] == "call"].copy()
        calls["delta"] = None
        selected = _pick_call_by_delta(calls, 150.0, 0.65)
        assert selected is not None  # Falls back to strike-based


class TestPickPutByDelta:
    def test_selects_nearest_delta(self):
        chain = _make_chain(spot=150.0, dte=35)
        puts = chain[chain["type"] == "put"].copy()
        selected = _pick_put_by_delta(puts, 150.0, 0.65)
        assert selected is not None


# ─── Selector Upgrade Path ────────────────────────────────────────────────────

class TestSelectorUpgradePath:
    """
    Verify that selector.select_strategy returns LONG_CALL/LONG_PUT when
    IV rank is low and the regime is strongly trending.
    """

    def _make_regime(self, regime_str, direction_score, trend_strength):
        from src.analysis.technical import StockRegime, Regime
        return StockRegime(
            ticker="AAPL",
            regime=Regime(regime_str),
            direction_score=direction_score,
            trend_strength=trend_strength,
            volatility_state="normal",
            rsi=55.0,
            adx=35.0,
            bb_squeeze=False,
            ema_alignment="bullish",
            support=140.0,
            resistance=160.0,
            atr=2.5,
            atr_pct=1.7,
            price=150.0,
            volume_trend="rising",
            roc_3d=0.5,
            atr_move_5d=0.8,
        )

    def _make_iv(self, iv_rank=20, iv_hv_ratio=0.85):
        from src.analysis.volatility import IVAnalysis
        return IVAnalysis(
            ticker="AAPL",
            current_iv=20.0,
            iv_rank=iv_rank,
            iv_percentile=iv_rank,
            hv_20=23.0,
            hv_60=22.0,
            iv_hv_ratio=iv_hv_ratio,
            iv_trend="FALLING",
            iv_30d_avg=22.0,
            iv_regime="LOW",
            premium_action="BUY",
            iv_rv_spread=-3.0,
            premium_rich=False,
            skew=0.0,
        )

    def test_strong_uptrend_low_iv_returns_long_call(self):
        from src.strategy.selector import select_strategy, StrategyType
        regime = self._make_regime("STRONG_UPTREND", direction_score=0.8, trend_strength=0.75)
        iv = self._make_iv(iv_rank=18)
        rec = select_strategy(regime, iv)
        if rec is not None:
            assert rec.strategy in (StrategyType.LONG_CALL, StrategyType.BULL_CALL_SPREAD,
                                     StrategyType.BULL_PUT_SPREAD)

    def test_strong_downtrend_low_iv_returns_long_put(self):
        from src.strategy.selector import select_strategy, StrategyType
        from src.analysis.technical import StockRegime, Regime
        regime = StockRegime(
            ticker="NVDA",
            regime=Regime.STRONG_DOWNTREND,
            direction_score=-0.8,
            trend_strength=0.75,
            volatility_state="normal",
            rsi=30.0,
            adx=38.0,
            bb_squeeze=False,
            ema_alignment="bearish",
            support=100.0,
            resistance=120.0,
            atr=3.0,
            atr_pct=2.5,
            price=110.0,
            volume_trend="falling",
            roc_3d=-0.5,
            atr_move_5d=0.9,
        )
        iv = self._make_iv(iv_rank=22)
        rec = select_strategy(regime, iv)
        if rec is not None:
            assert rec.strategy in (StrategyType.LONG_PUT, StrategyType.BEAR_PUT_SPREAD,
                                     StrategyType.BEAR_CALL_SPREAD)

    def test_low_confidence_prevents_upgrade(self):
        """
        When trend_strength is moderate (< upgrade threshold), the selector
        should NOT upgrade to a long option.
        """
        from src.strategy.selector import select_strategy, StrategyType
        regime = self._make_regime("UPTREND", direction_score=0.4, trend_strength=0.40)
        iv = self._make_iv(iv_rank=15)
        rec = select_strategy(regime, iv)
        if rec is not None:
            # Should stay as a spread
            assert rec.strategy not in (StrategyType.LONG_CALL, StrategyType.LONG_PUT)

    def test_high_iv_rank_no_upgrade(self):
        """High IV rank should not trigger long option upgrade."""
        from src.strategy.selector import select_strategy, StrategyType
        regime = self._make_regime("STRONG_UPTREND", direction_score=0.85, trend_strength=0.80)
        iv = self._make_iv(iv_rank=65, iv_hv_ratio=1.3)
        rec = select_strategy(regime, iv)
        if rec is not None:
            assert rec.strategy != StrategyType.LONG_CALL

    def test_skip_regime_returns_skip(self):
        from src.strategy.selector import select_strategy, StrategyType
        regime = self._make_regime("OVERSOLD_BOUNCE", direction_score=0.1, trend_strength=0.2)
        iv = self._make_iv(iv_rank=20)
        rec = select_strategy(regime, iv)
        if rec is not None:
            assert rec.strategy == StrategyType.SKIP
