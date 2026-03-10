from src.analysis.options_surface_signals import analyze_surface_adjustment


def test_calendar_note_added_for_fair_long_vega_without_balanced_short_premium():
    row = {
        "surface_valid_quote_ratio": 0.12,
        "surface_median_spread_pct": 0.05,
        "surface_avg_spread_pct": 0.05,
        "surface_liquid_contract_ratio": 0.70,
        "surface_quote_source": "IBKR",
        "surface_valid_spread_count": 8,
        "surface_spread_sample_size": 50,
        "surface_put_call_oi_ratio": 0.60,   # call-heavy, not balanced
        "surface_put_call_volume_ratio": 0.60,
        "surface_top_oi_strike_distance_pct": 0.05,
        "surface_term_slope": 0.10,
        "surface_term_ratio": 1.10,
    }
    adj = analyze_surface_adjustment(
        ticker="NVDA",
        direction="NEUTRAL",
        strategy="WATCHLIST",
        surface_row=row,
    )
    assert adj.liquidity_quality in {"FAIR", "OK"}
    assert adj.long_vega_friendly is True
    assert "long_vega_friendly" in adj.adjustment_notes
    assert adj.preferred_strategy == "CALENDAR"
    assert adj.strategy_changed is True


def test_short_premium_setup_requires_balanced_surface():
    row = {
        "surface_valid_quote_ratio": 0.12,
        "surface_median_spread_pct": 0.05,
        "surface_avg_spread_pct": 0.05,
        "surface_liquid_contract_ratio": 0.70,
        "surface_quote_source": "IBKR",
        "surface_valid_spread_count": 8,
        "surface_spread_sample_size": 50,
        "surface_put_call_oi_ratio": 0.60,   # call-heavy, not balanced
        "surface_put_call_volume_ratio": 0.60,
        "surface_top_oi_strike_distance_pct": 0.05,
        "surface_term_slope": -0.10,
        "surface_term_ratio": 0.95,
    }
    adj = analyze_surface_adjustment(
        ticker="AAPL",
        direction="NEUTRAL",
        strategy="WATCHLIST",
        surface_row=row,
    )
    assert adj.short_premium_friendly is True
    assert "strong_short_premium_setup" not in adj.adjustment_notes
    assert adj.preferred_strategy == "WATCHLIST"
    assert adj.strategy_changed is False


def test_balanced_short_premium_can_still_route_to_iron_condor():
    row = {
        "surface_valid_quote_ratio": 0.12,
        "surface_median_spread_pct": 0.05,
        "surface_avg_spread_pct": 0.05,
        "surface_liquid_contract_ratio": 0.70,
        "surface_quote_source": "IBKR",
        "surface_valid_spread_count": 8,
        "surface_spread_sample_size": 50,
        "surface_put_call_oi_ratio": 1.00,   # balanced
        "surface_put_call_volume_ratio": 1.00,
        "surface_top_oi_strike_distance_pct": 0.05,
        "surface_term_slope": -0.10,
        "surface_term_ratio": 0.95,
    }
    adj = analyze_surface_adjustment(
        ticker="SPY",
        direction="NEUTRAL",
        strategy="WATCHLIST",
        surface_row=row,
    )
    assert adj.short_premium_friendly is True
    assert "strong_short_premium_setup" in adj.adjustment_notes
    assert adj.preferred_strategy == "IRON_CONDOR"
    assert adj.strategy_changed is True
