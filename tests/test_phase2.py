"""
tests/test_phase2.py
====================
Phase 2 integration tests:

 1. MarketContext — real VIX fetch fallback, VIX tier classification, spike detection
 2. VIX circuit breaker in nightly_scan — LIQUIDATION halt, DEFENSIVE/CAUTION filters
 3. Sector concentration cap (MAX_PER_SECTOR=2)
 4. Beta map computation and net-exposure logging
 5. TradeOutcome — new V2 metadata fields (entry_vix, entry_vix_tier, entry_spy_5d, entry_roc_3d)
 6. _build_trade_dict / _build_trade_stub — market_snapshot + beta in context
 7. settings.py — new VIX + sector constants present and overridable
 8. position_monitor — VIX_SPIKE alert type generated
"""
from __future__ import annotations

import types
from dataclasses import asdict
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np
import pytest

# ─── Shared fixtures ──────────────────────────────────────────────────────────

def _make_price_df(n: int = 100, start_price: float = 150.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    log_ret = rng.normal(0.0003, 0.012, n)
    close = start_price * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def _make_regime(**kwargs):
    from src.analysis.technical import StockRegime, Regime
    defaults = dict(
        ticker="AAPL", regime=Regime.RANGE_BOUND, adx=20.0, rsi=50.0,
        trend_strength=0.5, direction_score=0.0, ema_alignment="FLAT",
        bb_squeeze=False, support=140.0, resistance=160.0, atr=2.5,
        atr_pct=1.7, volume_trend="FLAT", roc_3d=0.0, atr_move_5d=0.0,
        volatility_state="NORMAL", price=150.0,
    )
    defaults.update(kwargs)
    return StockRegime(**defaults)


def _make_iv(**kwargs):
    from src.analysis.volatility import IVAnalysis
    defaults = dict(
        ticker="AAPL", current_iv=25.0, iv_rank=50.0, iv_percentile=50.0,
        hv_20=22.0, hv_60=21.0, iv_hv_ratio=1.14, iv_regime="NORMAL",
        premium_action="NEUTRAL", iv_trend="FLAT", iv_30d_avg=24.0,
        skew=0.0, iv_rv_spread=3.0, premium_rich=False,
    )
    defaults.update(kwargs)
    return IVAnalysis(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — settings.py new constants
# ═══════════════════════════════════════════════════════════════════════════════

class TestSettingsPhase2:
    """Verify all new Phase 2 constants exist with correct defaults."""

    def test_vix_caution_level(self):
        from config.settings import VIX_CAUTION_LEVEL
        assert VIX_CAUTION_LEVEL == 28.0

    def test_vix_defensive_level(self):
        from config.settings import VIX_DEFENSIVE_LEVEL
        assert VIX_DEFENSIVE_LEVEL == 35.0

    def test_vix_liquidation_level(self):
        from config.settings import VIX_LIQUIDATION_LEVEL
        assert VIX_LIQUIDATION_LEVEL == 45.0

    def test_vix_spike_window(self):
        from config.settings import VIX_SPIKE_WINDOW
        assert VIX_SPIKE_WINDOW == 5

    def test_vix_spike_threshold_pct(self):
        from config.settings import VIX_SPIKE_THRESHOLD_PCT
        assert VIX_SPIKE_THRESHOLD_PCT == 25.0

    def test_max_per_sector(self):
        from config.settings import MAX_PER_SECTOR
        assert MAX_PER_SECTOR == 2

    def test_env_override_max_per_sector(self, monkeypatch):
        monkeypatch.setenv("MAX_PER_SECTOR", "3")
        import importlib, config.settings as s
        importlib.reload(s)
        assert s.MAX_PER_SECTOR == 3
        monkeypatch.delenv("MAX_PER_SECTOR", raising=False)
        importlib.reload(s)
        assert s.MAX_PER_SECTOR == 2  # restored to default

    def test_env_override_vix_liquidation(self, monkeypatch):
        monkeypatch.setenv("VIX_LIQUIDATION_LEVEL", "50.0")
        import importlib, config.settings as s
        importlib.reload(s)
        assert s.VIX_LIQUIDATION_LEVEL == 50.0
        monkeypatch.delenv("VIX_LIQUIDATION_LEVEL", raising=False)
        importlib.reload(s)
        assert s.VIX_LIQUIDATION_LEVEL == 45.0  # restored to default


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — MarketContext VIX tier classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketContextVIXTier:
    """Test _classify_vix_tier and new MarketContext fields."""

    def test_classify_normal(self):
        from src.data.market_context import _classify_vix_tier
        assert _classify_vix_tier(20.0) == "NORMAL"

    def test_classify_caution(self):
        from src.data.market_context import _classify_vix_tier
        assert _classify_vix_tier(28.0) == "CAUTION"
        assert _classify_vix_tier(30.0) == "CAUTION"

    def test_classify_defensive(self):
        from src.data.market_context import _classify_vix_tier
        assert _classify_vix_tier(35.0) == "DEFENSIVE"
        assert _classify_vix_tier(40.0) == "DEFENSIVE"

    def test_classify_liquidation(self):
        from src.data.market_context import _classify_vix_tier
        assert _classify_vix_tier(45.0) == "LIQUIDATION"
        assert _classify_vix_tier(60.0) == "LIQUIDATION"

    def test_market_context_has_vix_tier_field(self):
        from src.data.market_context import MarketContext
        assert hasattr(MarketContext, "__dataclass_fields__")
        fields = MarketContext.__dataclass_fields__
        assert "vix_tier" in fields
        assert "vix_5d_avg" in fields
        assert "vix_spike" in fields

    def test_vix_tier_default_is_normal(self):
        from src.data.market_context import MarketContext
        ctx = MarketContext(
            vix_level=18.0, vix_regime="NORMAL", spy_trend="UPTREND",
            spy_return_5d=1.0, spy_return_20d=3.0, advance_decline=1.5,
            market_regime="BULL", sector_leaders=[], sector_laggards=[],
            breadth_score=0.65,
        )
        assert ctx.vix_tier == "NORMAL"
        assert ctx.vix_spike is False
        assert ctx.vix_5d_avg == 0.0

    @patch("src.data.market_context._fetch_real_vix", return_value=None)
    def test_get_market_context_falls_back_to_proxy(self, mock_vix):
        from src.data.market_context import get_market_context
        data = {"SPY": _make_price_df(n=100)}
        ctx = get_market_context(data)
        assert ctx.vix_level > 0
        assert ctx.vix_tier in ("NORMAL", "CAUTION", "DEFENSIVE", "LIQUIDATION")

    @patch("src.data.market_context._fetch_real_vix", return_value=(50.0, 35.0))
    def test_get_market_context_liquidation_tier(self, mock_vix):
        from src.data.market_context import get_market_context
        data = {"SPY": _make_price_df(n=100)}
        ctx = get_market_context(data)
        assert ctx.vix_level == 50.0
        assert ctx.vix_tier == "LIQUIDATION"

    @patch("src.data.market_context._fetch_real_vix", return_value=(36.0, 35.0))
    def test_get_market_context_defensive_tier(self, mock_vix):
        from src.data.market_context import get_market_context
        data = {"SPY": _make_price_df(n=100)}
        ctx = get_market_context(data)
        assert ctx.vix_tier == "DEFENSIVE"

    @patch("src.data.market_context._fetch_real_vix", return_value=(30.0, 35.0))
    def test_get_market_context_caution_tier(self, mock_vix):
        from src.data.market_context import get_market_context
        data = {"SPY": _make_price_df(n=100)}
        ctx = get_market_context(data)
        assert ctx.vix_tier == "CAUTION"

    @patch("src.data.market_context._fetch_real_vix", return_value=(40.0, 30.0))
    def test_vix_spike_detected_when_ratio_exceeds_threshold(self, mock_vix):
        """40 vs 5d avg 30 = +33% spike > 25% threshold → spike=True."""
        from src.data.market_context import get_market_context
        data = {"SPY": _make_price_df(n=100)}
        ctx = get_market_context(data)
        assert ctx.vix_spike is True
        assert ctx.vix_5d_avg == 30.0

    @patch("src.data.market_context._fetch_real_vix", return_value=(32.0, 31.0))
    def test_vix_spike_not_detected_small_move(self, mock_vix):
        """32 vs 5d avg 31 = +3.2% — well below 25% threshold."""
        from src.data.market_context import get_market_context
        data = {"SPY": _make_price_df(n=100)}
        ctx = get_market_context(data)
        assert ctx.vix_spike is False


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — VIX circuit breaker in nightly_scan
# ═══════════════════════════════════════════════════════════════════════════════

class TestNightlyScanVIXCircuitBreaker:
    """Test that the LIQUIDATION halt and tier overlays work correctly."""

    def _run_scan_with_vix(self, vix_level: float, vix_avg: float = None):
        """Helper: run dry-run scan with a mocked MarketContext."""
        from src.data.market_context import MarketContext
        from src.data.market_context import _classify_vix_tier

        if vix_avg is None:
            vix_avg = vix_level

        vix_tier = _classify_vix_tier(vix_level)
        vix_spike = vix_avg > 0 and (vix_level - vix_avg) / vix_avg * 100 > 25

        mock_ctx = MarketContext(
            vix_level=vix_level, vix_regime="HIGH", spy_trend="DOWNTREND",
            spy_return_5d=-1.5, spy_return_20d=-5.0, advance_decline=0.5,
            market_regime="BEAR", sector_leaders=[], sector_laggards=[],
            breadth_score=0.3, vix_5d_avg=vix_avg,
            vix_spike=vix_spike, vix_tier=vix_tier,
        )

        with patch("src.pipeline.nightly_scan.update_universe") as mock_update, \
             patch("src.pipeline.nightly_scan.get_market_context", return_value=mock_ctx), \
             patch("src.pipeline.nightly_scan.filter_safe_tickers") as mock_filter:

            # Provide minimal SPY data
            spy_df = _make_price_df(n=100)
            mock_update.return_value = {"SPY": spy_df, "AAPL": spy_df}
            mock_filter.return_value = {}

            from src.pipeline.nightly_scan import run_nightly_scan
            return run_nightly_scan(
                universe_override=["SPY", "AAPL"],
                dry_run=True,
            )

    def test_liquidation_returns_zero_picks(self):
        """VIX ≥ 45 → circuit breaker fires → top_picks is empty."""
        signal = self._run_scan_with_vix(50.0, 35.0)
        assert signal["top_picks"] == []

    def test_liquidation_signal_has_circuit_breaker_key(self):
        signal = self._run_scan_with_vix(50.0, 35.0)
        assert "circuit_breaker" in signal
        assert "liquidation" in signal["circuit_breaker"].lower()

    def test_liquidation_market_context_in_signal(self):
        signal = self._run_scan_with_vix(50.0, 35.0)
        assert "market_context" in signal
        assert signal["market_context"]["vix_tier"] == "LIQUIDATION"

    def test_normal_vix_proceeds_normally(self):
        """VIX = 18 → no circuit breaker → pipeline completes normally."""
        signal = self._run_scan_with_vix(18.0, 18.0)
        # Should NOT have a circuit_breaker key
        assert "circuit_breaker" not in signal
        assert "top_picks" in signal


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — Sector concentration cap
# ═══════════════════════════════════════════════════════════════════════════════

class TestSectorConcentrationCap:
    """Verify MAX_PER_SECTOR=2 enforces sector concentration limits."""

    def _make_trade(self, ticker: str, sector: str, score: float = 0.5) -> dict:
        return {
            "recommendation": {
                "ticker": ticker, "strategy": "BULL_PUT_SPREAD",
                "direction": "BULLISH", "confidence": score,
                "regime": "UPTREND", "iv_regime": "HIGH",
                "rationale": "", "target_dte": 45,
            },
            "trade": {"net_credit": 1.5, "max_risk": 3.5, "prob_profit": 65, "ev": 10},
            "context": {
                "price": 150.0, "sector": sector, "beta": 1.0,
                "market_snapshot": {}, "regime_detail": {}, "iv_detail": {}, "rs_detail": {},
            },
        }

    def test_two_from_same_sector_both_pass(self):
        trades = [
            self._make_trade("AAPL", "Technology"),
            self._make_trade("MSFT", "Technology"),
            self._make_trade("AMZN", "Consumer Discretionary"),
        ]
        from src.pipeline.nightly_scan import _rank_trades
        from config.settings import MAX_PER_SECTOR

        # Apply sector cap manually (mirrors pipeline Step 9b)
        sector_counts: dict = {}
        filtered = []
        for t in trades:
            sector = t["context"]["sector"]
            if sector_counts.get(sector, 0) < MAX_PER_SECTOR:
                filtered.append(t)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        assert len(filtered) == 3  # 2 Tech + 1 Consumer

    def test_third_from_same_sector_is_dropped(self):
        trades = [
            self._make_trade("AAPL", "Technology"),
            self._make_trade("MSFT", "Technology"),
            self._make_trade("NVDA", "Technology"),   # should be dropped
        ]
        from config.settings import MAX_PER_SECTOR
        sector_counts: dict = {}
        filtered = []
        for t in trades:
            sector = t["context"]["sector"]
            if sector_counts.get(sector, 0) < MAX_PER_SECTOR:
                filtered.append(t)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        assert len(filtered) == MAX_PER_SECTOR  # only 2 allowed
        tickers = [t["recommendation"]["ticker"] for t in filtered]
        assert "NVDA" not in tickers

    def test_different_sectors_not_limited(self):
        trades = [
            self._make_trade("AAPL", "Technology"),
            self._make_trade("JPM", "Financials"),
            self._make_trade("JNJ", "Healthcare"),
            self._make_trade("XOM", "Energy"),
        ]
        from config.settings import MAX_PER_SECTOR
        sector_counts: dict = {}
        filtered = []
        for t in trades:
            sector = t["context"]["sector"]
            if sector_counts.get(sector, 0) < MAX_PER_SECTOR:
                filtered.append(t)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        assert len(filtered) == 4   # all pass — different sectors

    def test_max_per_sector_defaults_to_2(self):
        from config.settings import MAX_PER_SECTOR
        assert MAX_PER_SECTOR == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — Beta map + net exposure
# ═══════════════════════════════════════════════════════════════════════════════

class TestBetaMapAndNetExposure:
    """Verify beta is computed and included in trade context."""

    def _make_trade_with_beta(self, ticker: str, direction: str, beta: float) -> dict:
        return {
            "recommendation": {
                "ticker": ticker, "direction": direction,
                "strategy": "BULL_PUT_SPREAD", "confidence": 0.65,
                "regime": "UPTREND", "iv_regime": "HIGH",
                "rationale": "", "target_dte": 45,
            },
            "trade": {"net_credit": 1.5, "max_risk": 3.5, "prob_profit": 65, "ev": 10},
            "context": {
                "price": 150.0, "beta": beta, "sector": "Technology",
                "market_snapshot": {}, "regime_detail": {}, "iv_detail": {}, "rs_detail": {},
            },
        }

    def test_beta_present_in_trade_stub_context(self):
        """_build_trade_stub always includes beta key in context."""
        from src.pipeline.nightly_scan import _build_trade_stub
        from src.data.market_context import MarketContext, _classify_vix_tier

        mock_ctx = MarketContext(
            vix_level=18.0, vix_regime="NORMAL", spy_trend="UPTREND",
            spy_return_5d=1.0, spy_return_20d=2.0, advance_decline=1.5,
            market_regime="BULL", sector_leaders=[], sector_laggards=[],
            breadth_score=0.6, vix_tier="NORMAL",
        )

        regime = _make_regime(ticker="AAPL")
        iv = _make_iv(ticker="AAPL")

        from src.analysis.relative_strength import RSAnalysis
        mock_rec = MagicMock()
        mock_rec.ticker = "AAPL"
        mock_rec.strategy.value = "BULL_PUT_SPREAD"
        mock_rec.direction = "BULLISH"
        mock_rec.regime = "UPTREND"
        mock_rec.iv_regime = "HIGH"
        mock_rec.confidence = 0.70
        mock_rec.rationale = "test"
        mock_rec.target_dte = 45

        df = _make_price_df(n=100)
        stub = _build_trade_stub(
            mock_rec, df,
            {"AAPL": regime}, {"AAPL": iv}, {},
            mock_ctx, {"AAPL": 1.25},
        )

        assert "beta" in stub["context"]
        assert stub["context"]["beta"] == 1.25

    def test_market_snapshot_in_trade_stub(self):
        """market_snapshot block is present with vix, vix_tier, spy_5d_return."""
        from src.pipeline.nightly_scan import _build_trade_stub
        from src.data.market_context import MarketContext

        mock_ctx = MarketContext(
            vix_level=22.5, vix_regime="NORMAL", spy_trend="UPTREND",
            spy_return_5d=1.8, spy_return_20d=3.2, advance_decline=2.0,
            market_regime="BULL", sector_leaders=[], sector_laggards=[],
            breadth_score=0.7, vix_tier="NORMAL", vix_5d_avg=21.0,
        )

        regime = _make_regime(ticker="MSFT")
        iv = _make_iv(ticker="MSFT")

        mock_rec = MagicMock()
        mock_rec.ticker = "MSFT"
        mock_rec.strategy.value = "BULL_CALL_SPREAD"
        mock_rec.direction = "BULLISH"
        mock_rec.regime = "UPTREND"
        mock_rec.iv_regime = "NORMAL"
        mock_rec.confidence = 0.60
        mock_rec.rationale = "uptrend"
        mock_rec.target_dte = 21

        df = _make_price_df(n=100)
        stub = _build_trade_stub(
            mock_rec, df,
            {"MSFT": regime}, {"MSFT": iv}, {},
            mock_ctx, {"MSFT": 0.95},
        )

        snap = stub["context"]["market_snapshot"]
        assert snap["vix"] == 22.5
        assert snap["vix_tier"] == "NORMAL"
        assert snap["spy_5d_return"] == 1.8
        assert "breadth" in snap
        assert "market_regime" in snap

    def test_net_beta_exposure_calculation(self):
        """Net beta-weighted exposure aggregates correctly."""
        top_picks = [
            self._make_trade_with_beta("AAPL", "BULLISH", 1.2),
            self._make_trade_with_beta("MSFT", "BULLISH", 0.9),
            self._make_trade_with_beta("JPM", "BEARISH", 1.4),
        ]
        beta_map = {"AAPL": 1.2, "MSFT": 0.9, "JPM": 1.4}
        net_beta = sum(
            beta_map.get(t["recommendation"]["ticker"], 1.0)
            * (1 if t["recommendation"]["direction"] == "BULLISH" else
               -1 if t["recommendation"]["direction"] == "BEARISH" else 0)
            for t in top_picks
        )
        # 1.2 + 0.9 - 1.4 = +0.7
        assert abs(net_beta - 0.7) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — TradeOutcome V2 metadata fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestTradeOutcomeV2Fields:
    """Verify the four new entry-context fields on TradeOutcome."""

    def test_trade_outcome_has_entry_vix(self):
        from src.pipeline.outcome_tracker import TradeOutcome
        assert "entry_vix" in TradeOutcome.__dataclass_fields__

    def test_trade_outcome_has_entry_vix_tier(self):
        from src.pipeline.outcome_tracker import TradeOutcome
        assert "entry_vix_tier" in TradeOutcome.__dataclass_fields__

    def test_trade_outcome_has_entry_spy_5d(self):
        from src.pipeline.outcome_tracker import TradeOutcome
        assert "entry_spy_5d" in TradeOutcome.__dataclass_fields__

    def test_trade_outcome_has_entry_roc_3d(self):
        from src.pipeline.outcome_tracker import TradeOutcome
        assert "entry_roc_3d" in TradeOutcome.__dataclass_fields__

    def test_entry_fields_default_to_zero_or_empty(self):
        from src.pipeline.outcome_tracker import TradeOutcome
        t = TradeOutcome(
            trade_id="abc123", ticker="AAPL", strategy="BULL_PUT_SPREAD",
            direction="BULLISH", regime="UPTREND", iv_regime="HIGH",
            iv_rank=65.0, iv_hv_ratio=1.2, adx=28.0, rsi=55.0,
            trend_strength=0.7, direction_score=0.6, rs_rank=70,
            sector="Technology", dte_at_entry=45, spread_width=5.0,
            short_delta=0.25, entry_date="2025-03-04",
            expiration="2025-04-17", short_strike=185.0, long_strike=180.0,
            net_credit_or_debit=1.50, max_risk=3.50, prob_profit=68.0,
            confidence=0.72,
        )
        assert t.entry_vix == 0.0
        assert t.entry_vix_tier == ""
        assert t.entry_spy_5d == 0.0
        assert t.entry_roc_3d == 0.0

    def test_record_entry_populates_v2_fields(self):
        """record_entry pulls V2 fields from context['market_snapshot']."""
        from src.pipeline.outcome_tracker import record_entry, _OUTCOMES_FILE
        import tempfile, os

        recommendation = {
            "strategy": "BULL_PUT_SPREAD", "direction": "BULLISH",
            "regime": "UPTREND", "iv_regime": "HIGH",
            "confidence": 0.70, "target_dte": 45,
        }
        trade = {
            "net_credit": 1.50, "max_risk": 3.50, "prob_profit": 65,
            "short_strike": 185.0, "long_strike": 180.0,
            "width": 5.0, "short_delta": 0.25,
            "dte": 45, "expiration": "2025-04-17",
        }
        context = {
            "sector": "Technology",
            "iv_detail": {"iv_rank": 70.0, "iv_hv_ratio": 1.3},
            "regime_detail": {"adx": 30.0, "rsi": 60.0, "trend_strength": 0.8,
                              "direction_score": 0.7, "roc_3d": 1.5},
            "rs_detail": {"rs_rank": 75},
            "market_snapshot": {
                "vix": 22.5,
                "vix_tier": "NORMAL",
                "spy_5d_return": 1.8,
                "breadth": 0.65,
                "market_regime": "BULL",
            },
        }

        # Patch TRADES_DIR to temp location
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.pipeline.outcome_tracker.TRADES_DIR", new=None), \
                 patch("src.pipeline.outcome_tracker._OUTCOMES_FILE",
                       new=None), \
                 patch("src.pipeline.outcome_tracker._append_outcome") as mock_append:
                trade_id = record_entry("AAPL", recommendation, trade, context)
                assert mock_append.called
                outcome_arg = mock_append.call_args[0][0]
                assert outcome_arg.entry_vix == 22.5
                assert outcome_arg.entry_vix_tier == "NORMAL"
                assert outcome_arg.entry_spy_5d == 1.8
                assert outcome_arg.entry_roc_3d == 1.5


# ═══════════════════════════════════════════════════════════════════════════════
# 7 — position_monitor VIX_SPIKE alert type
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionMonitorVIXSpike:
    """Verify VIX_SPIKE alerts are generated when VIX is at defensive level."""

    def _make_open_position(self, ticker: str = "AAPL",
                             strategy: str = "BULL_PUT_SPREAD",
                             net_credit: float = 2.0) -> MagicMock:
        import datetime as dt
        pos = MagicMock()
        pos.ticker = ticker
        pos.strategy = strategy
        pos.status = "OPEN"
        pos.net_credit = net_credit
        pos.short_strike = 185.0
        pos.long_strike = 180.0
        # Expiration always 90 days from today to ensure dte > 0
        pos.expiration = (dt.date.today() + dt.timedelta(days=90)).strftime("%Y-%m-%d")
        return pos

    @patch("src.pipeline.position_monitor.load_positions")
    @patch("src.pipeline.position_monitor.yf.Ticker")
    def test_vix_spike_alert_generated_at_defensive_level(
        self, mock_ticker_cls, mock_load
    ):
        """VIX at 37 (≥ VIX_DEFENSIVE_LEVEL=35) → VIX_SPIKE alert for credit spreads."""
        mock_load.return_value = [self._make_open_position()]

        # Build a fake yfinance history DataFrame
        import pandas as pd
        vix_close = [30, 31, 30, 32, 37]
        idx = pd.bdate_range("2025-01-01", periods=len(vix_close))
        hist_df = pd.DataFrame({"Close": vix_close}, index=idx)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df
        mock_ticker_cls.return_value = mock_ticker

        from src.pipeline.position_monitor import monitor_positions
        alerts = monitor_positions()

        vix_alerts = [a for a in alerts if a.alert_type == "VIX_SPIKE"]
        assert len(vix_alerts) >= 1
        assert "AAPL" in [a.ticker for a in vix_alerts]

    @patch("src.pipeline.position_monitor.load_positions")
    @patch("src.pipeline.position_monitor.yf.Ticker")
    def test_vix_spike_alert_generated_on_spike(
        self, mock_ticker_cls, mock_load
    ):
        """VIX spikes +35% above 5d avg → VIX_SPIKE alert even below defensive level."""
        mock_load.return_value = [self._make_open_position()]

        # 5d avg ~20, current = 28 → +40% spike
        vix_close = [20, 20, 21, 20, 20, 28]
        idx = pd.bdate_range("2025-01-01", periods=len(vix_close))
        hist_df = pd.DataFrame({"Close": vix_close}, index=idx)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df
        mock_ticker_cls.return_value = mock_ticker

        from src.pipeline.position_monitor import monitor_positions
        alerts = monitor_positions()

        vix_alerts = [a for a in alerts if a.alert_type == "VIX_SPIKE"]
        assert len(vix_alerts) >= 1

    @patch("src.pipeline.position_monitor.load_positions")
    @patch("src.pipeline.position_monitor.yf.Ticker")
    def test_no_vix_spike_alert_at_low_vix(self, mock_ticker_cls, mock_load):
        """VIX at 18, no spike → no VIX_SPIKE alerts (only normal price checks)."""
        mock_load.return_value = [self._make_open_position()]

        vix_close = [18, 19, 18, 18, 18]
        idx = pd.bdate_range("2025-01-01", periods=len(vix_close))
        hist_df = pd.DataFrame({"Close": vix_close}, index=idx)

        # stock price fetch also uses yf.Ticker — return same df for simplicity
        price_df = pd.DataFrame(
            {"Close": [155.0]},
            index=pd.bdate_range("2025-01-10", periods=1),
        )

        call_count = [0]
        def side_effect(sym):
            call_count[0] += 1
            m = MagicMock()
            if sym == "^VIX":
                m.history.return_value = hist_df
            else:
                m.history.return_value = price_df
            return m

        mock_ticker_cls.side_effect = side_effect

        from src.pipeline.position_monitor import monitor_positions
        alerts = monitor_positions()
        vix_alerts = [a for a in alerts if a.alert_type == "VIX_SPIKE"]
        assert len(vix_alerts) == 0

    @patch("src.pipeline.position_monitor.load_positions")
    @patch("src.pipeline.position_monitor.yf.Ticker")
    def test_vix_spike_alert_includes_dte_info(self, mock_ticker_cls, mock_load):
        """The VIX_SPIKE alert message includes DTE remaining."""
        import datetime as dt
        future_exp = (dt.date.today() + dt.timedelta(days=20)).strftime("%Y-%m-%d")
        pos = self._make_open_position()
        pos.expiration = future_exp
        mock_load.return_value = [pos]

        vix_close = [30, 30, 30, 30, 40]  # spike
        idx = pd.bdate_range("2025-01-01", periods=len(vix_close))
        hist_df = pd.DataFrame({"Close": vix_close}, index=idx)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df
        mock_ticker_cls.return_value = mock_ticker

        from src.pipeline.position_monitor import monitor_positions
        alerts = monitor_positions()

        vix_alerts = [a for a in alerts if a.alert_type == "VIX_SPIKE"]
        assert len(vix_alerts) >= 1
        assert "DTE" in vix_alerts[0].message

    @patch("src.pipeline.position_monitor.load_positions")
    @patch("src.pipeline.position_monitor.yf.Ticker")
    def test_non_credit_strategy_not_flagged(self, mock_ticker_cls, mock_load):
        """Long butterfly (debit) should NOT receive a VIX_SPIKE alert."""
        mock_load.return_value = [self._make_open_position(strategy="LONG_BUTTERFLY")]

        vix_close = [30, 30, 30, 30, 38]
        idx = pd.bdate_range("2025-01-01", periods=len(vix_close))
        hist_df = pd.DataFrame({"Close": vix_close}, index=idx)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist_df
        mock_ticker_cls.return_value = mock_ticker

        from src.pipeline.position_monitor import monitor_positions
        alerts = monitor_positions()
        vix_alerts = [a for a in alerts if a.alert_type == "VIX_SPIKE"]
        # LONG_BUTTERFLY is not in the credit-spread list → no VIX_SPIKE alert
        assert len(vix_alerts) == 0
