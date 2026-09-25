"""Registry test: ensure all ten families are available and well-formed."""
import pytest


def test_registry_has_ten_unique_families_with_bounded_grids():
    from strategies.families import FAMILIES, family_names, get_family
    from strategies.families.common import grid_combos
    assert family_names() == sorted(FAMILIES)
    assert set(FAMILIES) == {"dt_momentum", "orb", "vwap_reversion", "gap_play", "mean_rev_intraday",
                             "news_catalyst", "post_news_drift", "pullback_trend", "mr_20d", "vol_breakout"}
    for name, fam in FAMILIES.items():
        assert fam.name == name and callable(fam.fn) and 1 <= len(grid_combos(fam.grid)) <= 27
        assert fam.hold_kind in ("intraday", "multiday") and set(fam.sides) <= {"long", "short"}
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
