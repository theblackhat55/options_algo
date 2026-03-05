"""
tests/test_dashboard.py
========================
Phase 4B tests for dashboard page rendering.

Tests:
  1. Long-option trade renders in today's picks (LONG_CALL details visible)
  2. TA signals expander renders without error
  3. IV Snapshot page renders gracefully when no snapshot directory exists
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── Streamlit mock must be patched BEFORE importing dashboard modules ─────────
# We mock streamlit so tests run headlessly without a browser.
st_mock = MagicMock()
st_mock.cache_data = lambda ttl=None: (lambda f: f)  # passthrough decorator
st_mock.set_page_config = MagicMock()
st_mock.sidebar = MagicMock()
st_mock.sidebar.title = MagicMock()
st_mock.sidebar.markdown = MagicMock()
st_mock.sidebar.selectbox = MagicMock(return_value="Today's Picks")
st_mock.sidebar.caption = MagicMock()
# st.columns(n) must return a list of n MagicMocks so tuple unpacking works
def _columns_mock(n, *args, **kwargs):
    if isinstance(n, (list, tuple)):
        n = len(n)
    return [MagicMock() for _ in range(int(n))]
st_mock.columns = _columns_mock
st_mock.tabs = lambda names: [MagicMock() for _ in names]
# st.expander as context manager
_exp_cm = MagicMock()
_exp_cm.__enter__ = lambda s: s
_exp_cm.__exit__ = MagicMock(return_value=False)
st_mock.expander = MagicMock(return_value=_exp_cm)
sys.modules["streamlit"] = st_mock

# Also stub plotly so it doesn't need a display
plotly_mock = MagicMock()
sys.modules["plotly"] = plotly_mock
sys.modules["plotly.express"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["plotly.subplots"] = MagicMock()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_long_call_pick() -> dict:
    """Minimal today's-picks signal with a LONG_CALL recommendation."""
    return {
        "scan_date": "2025-01-20",
        "generated_at": "2025-01-20T16:30:00",
        "universe_size": 100,
        "qualified": 40,
        "after_event_filter": 35,
        "elapsed_seconds": 12,
        "market_context": {
            "vix_level": 18,
            "spy_return_5d": 1.2,
            "breadth_score": 0.65,
            "spy_trend": "UP",
            "market_regime": "BULL",
        },
        "top_picks": [
            {
                "priority": 1,
                "composite_score": 0.0462,
                "recommendation": {
                    "ticker": "AAPL",
                    "strategy": "LONG_CALL",
                    "direction": "BULLISH",
                    "regime": "STRONG_UPTREND",
                    "confidence": 0.78,
                    "rationale": "Strong uptrend, low IV, TA breakout confirmed",
                    "is_long_option": True,
                    "ta_signals": {
                        "breakout_above": True,
                        "bullish_divergence": True,
                        "squeeze_direction": "up",
                        "near_support": False,
                        "near_resistance": False,
                        "above_anchored_vwap": True,
                        "below_anchored_vwap": False,
                        "pattern_score": 0.65,
                    },
                },
                "trade": {
                    "spread_type": "LONG_CALL",
                    "strike": 155.0,
                    "expiration": "2025-03-21",
                    "dte": 30,
                    "premium": 5.50,
                    "breakeven": 160.50,
                    "prob_profit": 62,
                    "ev": 2.30,
                    "theta_rate": 0.018,
                    "dry_run": False,
                },
                "context": {
                    "price": 152.0,
                    "sector": "Technology",
                    "iv_detail": {
                        "iv_rank": 22,
                        "iv_percentile": 28,
                        "current_iv": 25,
                        "hv_20": 20,
                        "iv_hv_ratio": 1.25,
                        "iv_trend": "falling",
                        "premium_action": "BUY",
                        "iv_rv_spread": 5,
                        "premium_rich": False,
                    },
                    "regime_detail": {
                        "adx": 32,
                        "rsi": 58,
                        "trend_strength": 0.75,
                        "direction_score": 0.65,
                        "ema_alignment": "BULLISH",
                        "bb_squeeze": False,
                        "support": 148.0,
                        "resistance": 162.0,
                        "atr": 2.5,
                        "atr_pct": 1.6,
                        "volume_trend": "up",
                        "roc_3d": 1.2,
                        "atr_move_5d": 3.1,
                    },
                    "rs_detail": {"rs_vs_spy": 1.5, "rs_rank": 82, "rs_trend": "strong"},
                    "options_flow": {},
                    "ta_signals": {
                        "breakout_above": True,
                        "bullish_divergence": True,
                        "squeeze_direction": "up",
                        "pattern_score": 0.65,
                    },
                    "entry_theta_rate": 0.018,
                },
            }
        ],
        "regime_distribution": {"STRONG_UPTREND": 5, "UPTREND": 10, "RANGE_BOUND": 15},
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLongOptionRenderingInPicks:
    """page_todays_picks should render LONG_CALL details without errors."""

    def test_long_call_pick_renders_without_exception(self):
        """page_todays_picks() runs to completion with a LONG_CALL pick."""
        from dashboard.app import page_todays_picks

        signal = _make_long_call_pick()
        with patch("dashboard.app.load_latest_signal", return_value=signal):
            # Should not raise
            try:
                page_todays_picks()
            except Exception as exc:
                pytest.fail(f"page_todays_picks() raised: {exc}")

    def test_format_trade_long_call_output(self):
        """_format_trade returns formatted string with LONG_CALL specific fields."""
        from src.pipeline.morning_brief import _format_trade
        trade = {
            "spread_type": "LONG_CALL",
            "strike": 155.0,
            "expiration": "2025-03-21",
            "dte": 30,
            "premium": 5.50,
            "breakeven": 160.50,
            "prob_profit": 62,
            "ev": 2.30,
        }
        result = _format_trade(trade, "LONG_CALL")
        assert "Call" in result
        assert "155" in result
        assert "5.50" in result


class TestTASignalsExpander:
    """TA signals expander renders when ta_signals dict is present."""

    def test_ta_signals_expander_renders(self):
        """page_todays_picks() with ta_signals dict does not crash."""
        from dashboard.app import page_todays_picks
        signal = _make_long_call_pick()
        # Confirm ta_signals in context
        pick_ctx = signal["top_picks"][0]["context"]
        assert "breakout_above" in pick_ctx["ta_signals"]

        with patch("dashboard.app.load_latest_signal", return_value=signal):
            try:
                page_todays_picks()
            except Exception as exc:
                pytest.fail(f"TA signals expander raised: {exc}")

    def test_empty_ta_signals_no_crash(self):
        """Empty ta_signals dict: expander simply doesn't render (no crash)."""
        from dashboard.app import page_todays_picks
        signal = _make_long_call_pick()
        signal["top_picks"][0]["context"]["ta_signals"] = {}
        signal["top_picks"][0]["recommendation"]["ta_signals"] = {}

        with patch("dashboard.app.load_latest_signal", return_value=signal):
            try:
                page_todays_picks()
            except Exception as exc:
                pytest.fail(f"Empty TA signals raised: {exc}")


class TestIVSnapshotPageNoData:
    """IV Snapshots page handles missing directory gracefully."""

    def test_iv_snapshot_page_no_directory(self, tmp_path):
        """When IV_SNAPSHOT_DIR doesn't exist, show warning message (no crash)."""
        from dashboard.app import page_iv_snapshots
        import config.settings as _settings

        # Patch IV_SNAPSHOT_DIR to a non-existent path
        fake_dir = tmp_path / "nonexistent_snapshots"
        # Do not create it — page should detect missing dir and show warning

        original = _settings.IV_SNAPSHOT_DIR
        _settings.IV_SNAPSHOT_DIR = fake_dir
        try:
            page_iv_snapshots()  # should not raise
        except (SystemExit, StopIteration):
            pass  # st.stop() may bubble up in test context
        except Exception as exc:
            pytest.fail(f"page_iv_snapshots raised unexpectedly: {exc}")
        finally:
            _settings.IV_SNAPSHOT_DIR = original
