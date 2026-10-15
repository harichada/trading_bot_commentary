"""Registry test: ensure all twenty-four families are available and well-formed."""
import pytest


def test_registry_has_twenty_four_unique_families_with_bounded_grids():
    from strategies.families import FAMILIES, family_names, get_family
    from strategies.families.common import HOLD_KINDS, grid_combos
    assert family_names() == sorted(FAMILIES)
    assert set(FAMILIES) == {"dt_momentum", "orb", "vwap_reversion", "gap_play", "mean_rev_intraday",
                             "news_catalyst", "post_news_drift", "pullback_trend", "mr_20d", "vol_breakout",
                             "orb_inplay", "pead_smallcap", "rs_leader",
                             # Batch-1 slower families (swing2)
                             "overnight_hold", "mr_short_term", "multi_day_drop", "pead_wide", "rs_rotation",
                             # Batch-2 factor families (Task C)
                             "mom_12_1", "low_vol_trend",
                             # RSI-extreme families (2026-10-05)
                             "rsi_dip", "rsi_fade",
                             # Adaptive RSI Turn engine (2026-10-05)
                             "rsi_turn",
                             # prior-high / prior-low fade with RSI filter (2026-10-08, MU style)
                             "rsi_peak_fade"}
    assert len(FAMILIES) == 24
    for name, fam in FAMILIES.items():
        assert fam.name == name and callable(fam.fn) and 1 <= len(grid_combos(fam.grid)) <= 27
        assert fam.hold_kind in HOLD_KINDS and set(fam.sides) <= {"long", "short"}
    assert FAMILIES["overnight_hold"].hold_kind == "overnight"
    assert get_family("orb").name == "orb"
    with pytest.raises(KeyError):
        get_family("nope")


def test_get_family_error_message_lists_valid_names():
    from strategies.families import get_family
    try:
        get_family("bad_family_name")
        assert False, "should have raised KeyError"
    except KeyError as e:
        msg = str(e)
        assert "unknown family" in msg and "valid:" in msg
