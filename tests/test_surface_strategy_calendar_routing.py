from src.analysis.options_surface_signals import _preferred_strategy_from_surface


def test_fair_promotes_to_calendar_when_long_vega_friendly():
    strategy, changed = _preferred_strategy_from_surface(
        liquidity_quality="FAIR",
        current_strategy="WATCHLIST",
        adjustment_notes=["fair_liquidity", "long_vega_friendly"],
    )
    assert strategy == "CALENDAR"
    assert changed is True


def test_ok_promotes_to_calendar_when_long_vega_friendly():
    strategy, changed = _preferred_strategy_from_surface(
        liquidity_quality="OK",
        current_strategy="WATCHLIST",
        adjustment_notes=["good_liquidity", "long_vega_friendly"],
    )
    assert strategy == "CALENDAR"
    assert changed is True


def test_condor_priority_beats_calendar_when_both_signals_present():
    strategy, changed = _preferred_strategy_from_surface(
        liquidity_quality="FAIR",
        current_strategy="WATCHLIST",
        adjustment_notes=[
            "fair_liquidity",
            "long_vega_friendly",
            "strong_neutral_pin_setup",
        ],
    )
    assert strategy == "IRON_CONDOR"
    assert changed is True


def test_weak_does_not_promote_to_calendar():
    strategy, changed = _preferred_strategy_from_surface(
        liquidity_quality="WEAK",
        current_strategy="WATCHLIST",
        adjustment_notes=["long_vega_friendly"],
    )
    assert strategy == "WATCHLIST"
    assert changed is False


def test_existing_calendar_stays_without_change_flag():
    strategy, changed = _preferred_strategy_from_surface(
        liquidity_quality="FAIR",
        current_strategy="CALENDAR",
        adjustment_notes=["fair_liquidity", "long_vega_friendly"],
    )
    assert strategy == "CALENDAR"
    assert changed is False
