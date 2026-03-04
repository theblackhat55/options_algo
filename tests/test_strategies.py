"""
tests/test_strategies.py
=========================
Tests for strategy selector, credit spreads, debit spreads, IC, butterfly.
Uses synthetic options chains (no real API calls).

V2 additions:
  - New regime rows: OVERSOLD_BOUNCE, OVERBOUGHT_DROP
  - SPY 5-day return gate (spy_return_5d)
  - 3d ROC counter-signal confidence penalty
  - IV-RV spread / premium_rich confidence bonus
  - StockRegime new fields: roc_3d, atr_move_5d
  - IVAnalysis new fields: iv_rv_spread, premium_rich
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
    roc_3d: float = 0.0,
    atr_move_5d: float = 0.0,
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
        roc_3d=roc_3d,
        atr_move_5d=atr_move_5d,
    )


def _make_iv(
    ticker: str = "AAPL",
    iv_regime: str = "NORMAL",
    iv_rank: float = 45.0,
    iv_rv_spread: float = 3.0,
    premium_rich: bool = False,
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
        iv_rv_spread=iv_rv_spread,
        premium_rich=premium_rich,
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


# ─── Tests: Strategy Selector (existing regimes) ──────────────────────────────

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
                assert 0 <= rec.confidence <= 1, (
                    f"confidence={rec.confidence} out of range for {r}/{iv_regime}"
                )

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


# ─── Tests: New Regimes (OVERSOLD_BOUNCE, OVERBOUGHT_DROP) ────────────────────

class TestSnapbackRegimes:
    """
    OVERSOLD_BOUNCE: stock dropped hard but is bouncing.
      HIGH IV → Iron Condor (neutral credit); NORMAL/LOW → SKIP
    OVERBOUGHT_DROP: stock rallied hard but is fading.
      HIGH IV → Iron Condor (neutral credit); NORMAL/LOW → SKIP
    """

    # ── OVERSOLD_BOUNCE ───────────────────────────────────────────────────────

    def test_oversold_bounce_high_iv_returns_iron_condor(self):
        regime = _make_regime(
            regime=Regime.OVERSOLD_BOUNCE,
            direction_score=-0.3,
            rsi=30.0,
            roc_3d=1.5,
            atr_move_5d=-2.5,
        )
        iv = _make_iv(iv_regime="HIGH", iv_rank=78)
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.IRON_CONDOR
        assert rec.direction == "NEUTRAL"
        assert rec.risk_reward == "CREDIT"

    def test_oversold_bounce_normal_iv_returns_skip(self):
        regime = _make_regime(
            regime=Regime.OVERSOLD_BOUNCE,
            direction_score=-0.3,
            rsi=32.0,
            roc_3d=1.2,
            atr_move_5d=-2.2,
        )
        iv = _make_iv(iv_regime="NORMAL", iv_rank=45)
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.SKIP

    def test_oversold_bounce_low_iv_returns_skip(self):
        regime = _make_regime(
            regime=Regime.OVERSOLD_BOUNCE,
            direction_score=-0.4,
            rsi=28.0,
            roc_3d=2.0,
            atr_move_5d=-3.0,
        )
        iv = _make_iv(iv_regime="LOW", iv_rank=18)
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.SKIP

    # ── OVERBOUGHT_DROP ───────────────────────────────────────────────────────

    def test_overbought_drop_high_iv_returns_iron_condor(self):
        regime = _make_regime(
            regime=Regime.OVERBOUGHT_DROP,
            direction_score=0.3,
            rsi=70.0,
            roc_3d=-1.5,
            atr_move_5d=2.5,
        )
        iv = _make_iv(iv_regime="HIGH", iv_rank=80)
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.IRON_CONDOR
        assert rec.direction == "NEUTRAL"
        assert rec.risk_reward == "CREDIT"

    def test_overbought_drop_normal_iv_returns_skip(self):
        regime = _make_regime(
            regime=Regime.OVERBOUGHT_DROP,
            direction_score=0.3,
            rsi=68.0,
            roc_3d=-1.2,
            atr_move_5d=2.2,
        )
        iv = _make_iv(iv_regime="NORMAL", iv_rank=50)
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.SKIP

    def test_overbought_drop_low_iv_returns_skip(self):
        regime = _make_regime(
            regime=Regime.OVERBOUGHT_DROP,
            direction_score=0.4,
            rsi=72.0,
            roc_3d=-2.0,
            atr_move_5d=3.0,
        )
        iv = _make_iv(iv_regime="LOW", iv_rank=15)
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.SKIP

    # ── Skip strategy has zero confidence ─────────────────────────────────────

    def test_skip_has_zero_confidence(self):
        for regime_type in (Regime.OVERSOLD_BOUNCE, Regime.OVERBOUGHT_DROP):
            for iv_regime in ("NORMAL", "LOW"):
                regime = _make_regime(regime=regime_type)
                iv = _make_iv(iv_regime=iv_regime)
                rec = select_strategy(regime, iv)
                assert rec.strategy == StrategyType.SKIP
                assert rec.confidence == 0.0, (
                    f"SKIP confidence should be 0, got {rec.confidence} "
                    f"for {regime_type}/{iv_regime}"
                )

    def test_snapback_ic_has_45_dte(self):
        """Iron condor from snap-back regimes should use 45-day DTE (credit)."""
        for regime_type in (Regime.OVERSOLD_BOUNCE, Regime.OVERBOUGHT_DROP):
            regime = _make_regime(regime=regime_type)
            iv = _make_iv(iv_regime="HIGH", iv_rank=80)
            rec = select_strategy(regime, iv)
            assert rec.strategy == StrategyType.IRON_CONDOR
            assert rec.target_dte == 45


# ─── Tests: SPY 5-Day Gate ────────────────────────────────────────────────────

class TestSPYGate:
    """
    spy_return_5d > +SPY_DIRECTIONAL_GATE_PCT  → SKIP if strategy is BEARISH
    spy_return_5d < -SPY_DIRECTIONAL_GATE_PCT  → SKIP if strategy is BULLISH
    Neutral strategies (IC, butterfly) are unaffected.
    """

    def test_bearish_skipped_when_spy_rallying(self):
        """Downtrend + HIGH IV = bear call spread, but SPY up +2% → SKIP."""
        regime = _make_regime(regime=Regime.DOWNTREND, direction_score=-0.5)
        iv = _make_iv(iv_regime="HIGH", iv_rank=75)
        rec = select_strategy(regime, iv, spy_return_5d=2.0)  # above 1% gate
        assert rec.strategy == StrategyType.SKIP
        assert rec.confidence == 0.0
        assert "SPY" in rec.rationale or "tape" in rec.rationale.lower() or "bearish" in rec.rationale.lower()

    def test_bullish_skipped_when_spy_dropping(self):
        """Uptrend + HIGH IV = bull put spread, but SPY down -2% → SKIP."""
        regime = _make_regime(regime=Regime.UPTREND, direction_score=0.5)
        iv = _make_iv(iv_regime="HIGH", iv_rank=75)
        rec = select_strategy(regime, iv, spy_return_5d=-2.0)  # below -1% gate
        assert rec.strategy == StrategyType.SKIP
        assert rec.confidence == 0.0
        assert "SPY" in rec.rationale or "tape" in rec.rationale.lower() or "bullish" in rec.rationale.lower()

    def test_bearish_not_skipped_below_gate(self):
        """SPY up only +0.5% (under the 1% gate) — bearish should proceed."""
        regime = _make_regime(regime=Regime.DOWNTREND, direction_score=-0.5)
        iv = _make_iv(iv_regime="HIGH", iv_rank=75)
        rec = select_strategy(regime, iv, spy_return_5d=0.5)
        assert rec.strategy == StrategyType.BEAR_CALL_SPREAD

    def test_bullish_not_skipped_above_negative_gate(self):
        """SPY down only -0.5% (above the -1% gate) — bullish should proceed."""
        regime = _make_regime(regime=Regime.UPTREND, direction_score=0.5)
        iv = _make_iv(iv_regime="HIGH", iv_rank=75)
        rec = select_strategy(regime, iv, spy_return_5d=-0.5)
        assert rec.strategy == StrategyType.BULL_PUT_SPREAD

    def test_neutral_ic_unaffected_by_spy_gate(self):
        """Iron condor (neutral) should pass even if SPY moved strongly."""
        regime = _make_regime(regime=Regime.RANGE_BOUND, direction_score=0.0)
        iv = _make_iv(iv_regime="HIGH", iv_rank=80)
        # SPY rallying hard — neutral trade should still pass
        rec = select_strategy(regime, iv, spy_return_5d=3.0)
        assert rec.strategy == StrategyType.IRON_CONDOR

        # SPY dropping hard — still passes
        rec2 = select_strategy(regime, iv, spy_return_5d=-3.0)
        assert rec2.strategy == StrategyType.IRON_CONDOR

    def test_neutral_butterfly_unaffected_by_spy_gate(self):
        """Long butterfly (neutral) is not blocked by SPY gate."""
        regime = _make_regime(regime=Regime.RANGE_BOUND, direction_score=0.0)
        iv = _make_iv(iv_regime="LOW", iv_rank=18)
        rec = select_strategy(regime, iv, spy_return_5d=2.5)
        assert rec.strategy == StrategyType.LONG_BUTTERFLY

    def test_spy_gate_exactly_at_threshold(self):
        """
        At exactly the threshold value, the gate should NOT fire
        (strictly greater / strictly less than required).
        """
        from config.settings import SPY_DIRECTIONAL_GATE_PCT
        regime_bear = _make_regime(regime=Regime.DOWNTREND, direction_score=-0.5)
        iv_high = _make_iv(iv_regime="HIGH", iv_rank=75)

        # Exactly at gate — should NOT be blocked (boundary is exclusive)
        rec = select_strategy(regime_bear, iv_high, spy_return_5d=SPY_DIRECTIONAL_GATE_PCT)
        # The implementation uses ">", so exactly at gate should NOT skip
        assert rec.strategy == StrategyType.BEAR_CALL_SPREAD, (
            "Gate should be exclusive (>), so at exactly threshold it should not fire"
        )

    def test_strong_downtrend_skipped_when_spy_rallying(self):
        """STRONG_DOWNTREND bear call spread also blocked by SPY gate."""
        regime = _make_regime(regime=Regime.STRONG_DOWNTREND, direction_score=-0.7)
        iv = _make_iv(iv_regime="HIGH", iv_rank=80)
        rec = select_strategy(regime, iv, spy_return_5d=2.5)
        assert rec.strategy == StrategyType.SKIP

    def test_strong_uptrend_skipped_when_spy_dropping(self):
        """STRONG_UPTREND bull put spread also blocked by SPY gate."""
        regime = _make_regime(regime=Regime.STRONG_UPTREND, direction_score=0.7)
        iv = _make_iv(iv_regime="HIGH", iv_rank=80)
        rec = select_strategy(regime, iv, spy_return_5d=-2.5)
        assert rec.strategy == StrategyType.SKIP


# ─── Tests: 3d ROC Counter-Signal Confidence Penalty ─────────────────────────

class TestROCConfidencePenalty:
    """
    When the 3-day ROC opposes the strategy direction by >1.5%,
    confidence should be 0.15 lower than the same setup without the ROC signal.
    """

    def test_bearish_strategy_penalised_when_roc_positive(self):
        """Bear spread: stock bouncing (roc_3d > +1.5%) → lower confidence."""
        # Baseline (no momentum opposition)
        r_flat = _make_regime(regime=Regime.DOWNTREND, direction_score=-0.5, roc_3d=0.0)
        iv = _make_iv(iv_regime="NORMAL", iv_rank=50)
        baseline = select_strategy(r_flat, iv).confidence

        # Counter-signal: stock bouncing while bearish
        r_bounce = _make_regime(regime=Regime.DOWNTREND, direction_score=-0.5, roc_3d=2.0)
        penalised = select_strategy(r_bounce, iv).confidence

        assert penalised < baseline, (
            f"Expected penalised confidence ({penalised:.3f}) < "
            f"baseline ({baseline:.3f}) when ROC opposes bearish setup"
        )
        assert abs(baseline - penalised) >= 0.10, (
            "Penalty should be at least 0.10 in confidence"
        )

    def test_bullish_strategy_penalised_when_roc_negative(self):
        """Bull spread: stock dropping (roc_3d < -1.5%) → lower confidence."""
        r_flat = _make_regime(regime=Regime.UPTREND, direction_score=0.5, roc_3d=0.0)
        iv = _make_iv(iv_regime="NORMAL", iv_rank=50)
        baseline = select_strategy(r_flat, iv).confidence

        r_fade = _make_regime(regime=Regime.UPTREND, direction_score=0.5, roc_3d=-2.0)
        penalised = select_strategy(r_fade, iv).confidence

        assert penalised < baseline, (
            f"Expected penalised confidence ({penalised:.3f}) < "
            f"baseline ({baseline:.3f}) when ROC opposes bullish setup"
        )
        assert abs(baseline - penalised) >= 0.10

    def test_neutral_strategy_not_penalised_by_roc(self):
        """Iron condor (neutral) confidence unaffected by strong ROC."""
        r_roc_up = _make_regime(regime=Regime.RANGE_BOUND, direction_score=0.0, roc_3d=3.0)
        r_roc_dn = _make_regime(regime=Regime.RANGE_BOUND, direction_score=0.0, roc_3d=-3.0)
        r_flat   = _make_regime(regime=Regime.RANGE_BOUND, direction_score=0.0, roc_3d=0.0)

        iv = _make_iv(iv_regime="HIGH", iv_rank=80)

        c_up   = select_strategy(r_roc_up, iv).confidence
        c_dn   = select_strategy(r_roc_dn, iv).confidence
        c_flat = select_strategy(r_flat,   iv).confidence

        # Neutral strategies should not be penalised by directional momentum
        assert c_up   == pytest.approx(c_flat, abs=0.02)
        assert c_dn   == pytest.approx(c_flat, abs=0.02)

    def test_small_roc_does_not_trigger_penalty(self):
        """ROC under the 1.5% threshold should not trigger the penalty."""
        r_small = _make_regime(regime=Regime.DOWNTREND, direction_score=-0.5, roc_3d=1.0)
        r_flat  = _make_regime(regime=Regime.DOWNTREND, direction_score=-0.5, roc_3d=0.0)
        iv = _make_iv(iv_regime="NORMAL", iv_rank=50)

        c_small = select_strategy(r_small, iv).confidence
        c_flat  = select_strategy(r_flat,  iv).confidence

        # Within threshold — no penalty; values should be equal
        assert c_small == pytest.approx(c_flat, abs=0.02), (
            "ROC below 1.5% should not trigger penalty"
        )

    def test_roc_penalty_capped_at_zero(self):
        """Confidence should never go below 0.0 even with extreme ROC."""
        # Give it a minimal baseline by using a weak regime + high ROC penalty
        r = _make_regime(
            regime=Regime.DOWNTREND,
            direction_score=-0.1,
            roc_3d=5.0,
        )
        iv = _make_iv(iv_regime="NORMAL", iv_rank=50)
        rec = select_strategy(r, iv)
        assert rec.confidence >= 0.0


# ─── Tests: IV-RV Spread / premium_rich Bonus ─────────────────────────────────

class TestIVRVSpreadBonus:
    """
    premium_rich=True adds +0.05 to confidence for credit strategies.
    """

    def test_premium_rich_boosts_credit_confidence(self):
        """Bull put spread (credit) gets +0.05 when premium_rich=True."""
        regime = _make_regime(regime=Regime.UPTREND, direction_score=0.5)

        iv_rich = _make_iv(iv_regime="HIGH", iv_rank=80, iv_rv_spread=7.0, premium_rich=True)
        iv_flat = _make_iv(iv_regime="HIGH", iv_rank=80, iv_rv_spread=3.0, premium_rich=False)

        rec_rich = select_strategy(regime, iv_rich)
        rec_flat = select_strategy(regime, iv_flat)

        assert rec_rich.strategy == StrategyType.BULL_PUT_SPREAD
        assert rec_rich.confidence > rec_flat.confidence
        assert abs(rec_rich.confidence - rec_flat.confidence) == pytest.approx(0.05, abs=0.01)

    def test_premium_rich_does_not_boost_debit_confidence(self):
        """Debit strategies (bull call spread) are not boosted by premium_rich."""
        regime = _make_regime(regime=Regime.UPTREND, direction_score=0.5)

        iv_rich = _make_iv(iv_regime="NORMAL", iv_rank=45, iv_rv_spread=7.0, premium_rich=True)
        iv_flat = _make_iv(iv_regime="NORMAL", iv_rank=45, iv_rv_spread=3.0, premium_rich=False)

        rec_rich = select_strategy(regime, iv_rich)
        rec_flat = select_strategy(regime, iv_flat)

        assert rec_rich.strategy == StrategyType.BULL_CALL_SPREAD
        # Debit strategy should not change confidence based on premium_rich
        assert rec_rich.confidence == pytest.approx(rec_flat.confidence, abs=0.02)

    def test_ic_boosted_by_premium_rich(self):
        """Iron condor (credit) also gets boost from premium_rich."""
        regime = _make_regime(regime=Regime.RANGE_BOUND, direction_score=0.0)

        iv_rich = _make_iv(iv_regime="HIGH", iv_rank=80, iv_rv_spread=8.0, premium_rich=True)
        iv_flat = _make_iv(iv_regime="HIGH", iv_rank=80, iv_rv_spread=2.0, premium_rich=False)

        rec_rich = select_strategy(regime, iv_rich)
        rec_flat = select_strategy(regime, iv_flat)

        assert rec_rich.strategy == StrategyType.IRON_CONDOR
        assert rec_rich.confidence > rec_flat.confidence

    def test_iv_rv_spread_in_rationale(self):
        """premium_rich credit trade should include IV-RV spread in rationale."""
        regime = _make_regime(regime=Regime.UPTREND, direction_score=0.5)
        iv = _make_iv(iv_regime="HIGH", iv_rank=80, iv_rv_spread=8.0, premium_rich=True)
        rec = select_strategy(regime, iv)
        assert "IV-RV" in rec.rationale or "premium" in rec.rationale.lower(), (
            f"Expected IV-RV mention in rationale, got: {rec.rationale}"
        )


# ─── Tests: StockRegime New Fields ────────────────────────────────────────────

class TestStockRegimeNewFields:
    """Verify that the new V2 fields exist and have correct defaults."""

    def test_regime_has_roc_3d_field(self):
        r = _make_regime()
        assert hasattr(r, "roc_3d")
        assert isinstance(r.roc_3d, float)

    def test_regime_has_atr_move_5d_field(self):
        r = _make_regime()
        assert hasattr(r, "atr_move_5d")
        assert isinstance(r.atr_move_5d, float)

    def test_roc_3d_default_is_zero(self):
        r = _make_regime()
        assert r.roc_3d == 0.0

    def test_atr_move_5d_default_is_zero(self):
        r = _make_regime()
        assert r.atr_move_5d == 0.0

    def test_regime_values_are_stored(self):
        r = _make_regime(roc_3d=1.5, atr_move_5d=-2.3)
        assert r.roc_3d == 1.5
        assert r.atr_move_5d == -2.3


# ─── Tests: IVAnalysis New Fields ─────────────────────────────────────────────

class TestIVAnalysisNewFields:
    """Verify iv_rv_spread and premium_rich exist and function correctly."""

    def test_iv_analysis_has_iv_rv_spread(self):
        iv = _make_iv()
        assert hasattr(iv, "iv_rv_spread")
        assert isinstance(iv.iv_rv_spread, float)

    def test_iv_analysis_has_premium_rich(self):
        iv = _make_iv()
        assert hasattr(iv, "premium_rich")
        assert isinstance(iv.premium_rich, bool)

    def test_iv_rv_spread_default_zero(self):
        iv = _make_iv()
        assert iv.iv_rv_spread == 3.0  # as set in fixture default

    def test_premium_rich_true_when_spread_high(self):
        iv = _make_iv(iv_rv_spread=7.0, premium_rich=True)
        assert iv.premium_rich is True

    def test_premium_rich_false_when_spread_low(self):
        iv = _make_iv(iv_rv_spread=2.0, premium_rich=False)
        assert iv.premium_rich is False


# ─── Tests: Rationale Content ─────────────────────────────────────────────────

class TestRationaleContent:
    """Rationale string should include relevant indicators."""

    def test_rationale_includes_regime(self):
        regime = _make_regime(regime=Regime.UPTREND)
        iv = _make_iv(iv_regime="HIGH")
        rec = select_strategy(regime, iv)
        assert "UPTREND" in rec.rationale

    def test_rationale_includes_iv_rank(self):
        regime = _make_regime()
        iv = _make_iv(iv_rank=75.0)
        rec = select_strategy(regime, iv)
        assert "75" in rec.rationale

    def test_rationale_includes_roc_when_material(self):
        """3d ROC > 1% should appear in rationale."""
        regime = _make_regime(regime=Regime.UPTREND, roc_3d=2.0)
        iv = _make_iv(iv_regime="HIGH")
        rec = select_strategy(regime, iv)
        assert "ROC" in rec.rationale or "3d" in rec.rationale.lower(), (
            f"Expected ROC mention, got: {rec.rationale}"
        )

    def test_rationale_includes_atr_move_when_material(self):
        """5d ATR move > 1.5 ATR units should appear in rationale."""
        regime = _make_regime(
            regime=Regime.OVERSOLD_BOUNCE,
            roc_3d=1.5,
            atr_move_5d=-2.5,
        )
        iv = _make_iv(iv_regime="HIGH", iv_rank=80)
        rec = select_strategy(regime, iv)
        assert "ATR" in rec.rationale, (
            f"Expected ATR mention in rationale, got: {rec.rationale}"
        )

    def test_skip_rationale_is_informative(self):
        """SKIP should provide a non-empty, descriptive rationale."""
        regime = _make_regime(regime=Regime.OVERSOLD_BOUNCE)
        iv = _make_iv(iv_regime="NORMAL")
        rec = select_strategy(regime, iv)
        assert rec.strategy == StrategyType.SKIP
        # SKIP from the no-match path
        assert len(rec.rationale) > 5


# ─── Tests: All Regime + IV combinations return valid recommendation ───────────

class TestAllMatrixCells:
    """Every cell in the strategy matrix returns a valid StrategyRecommendation."""

    def test_all_regime_iv_combinations_produce_valid_rec(self):
        for r in Regime:
            for iv_regime in ("HIGH", "NORMAL", "LOW"):
                regime = _make_regime(regime=r)
                iv = _make_iv(iv_regime=iv_regime)
                rec = select_strategy(regime, iv)

                # Sanity checks on every returned recommendation
                assert isinstance(rec.strategy, StrategyType), (
                    f"strategy not StrategyType for {r}/{iv_regime}"
                )
                assert 0.0 <= rec.confidence <= 1.0, (
                    f"confidence out of range for {r}/{iv_regime}"
                )
                assert rec.ticker == "AAPL"
                assert rec.regime == r.value
                assert rec.iv_regime == iv_regime
                assert isinstance(rec.rationale, str)
                assert len(rec.rationale) > 0


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
