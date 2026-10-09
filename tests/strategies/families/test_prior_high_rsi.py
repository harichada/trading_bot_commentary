"""prior_high_rsi family (2026-10-08): fade a retest of the session's prior swing high with RSI
above a threshold (short); mirror at a prior swing low with RSI below 100 - threshold (long).
Encodes the user's discretionary MU trades; see research/reports/mu_style/."""
import numpy as np
import pandas as pd
import pytest

P = {"near_pct": 0.3, "rsi_short": 60, "stop_pct": 0.3, "target_r": 1.5}


def _bars_and_ind(n=40, day="2025-01-02"):
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="5min", name="ts")
    bars = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx)
    ind = pd.DataFrame(index=idx)
    ind["bar_of_day"] = np.arange(n)
    ind["rsi"] = 50.0
    ind["vwap"] = 100.0
    return bars, ind


def _entries(bars, ind, p=P):
    from strategies.families.prior_high_rsi import prior_high_rsi_entries
    return prior_high_rsi_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)


def test_short_on_failed_retest_of_prior_swing_high_with_high_rsi():
    bars, ind = _bars_and_ind()
    bars.loc[bars.index[10], "high"] = 102.0                       # the session's prior swing high
    i = 20
    bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]   # retest, fails to exceed
    bars.loc[bars.index[i + 1], "open"] = 101.8
    ind.loc[ind.index[i], "rsi"] = 63.0
    es = _entries(bars, ind)
    assert len(es) == 1
    e = es[0]
    assert e.side == "short" and e.entry_ts == bars.index[i + 1] and e.tag == "prior_high_fade"
    assert e.stop_px == pytest.approx(102.0 * 1.003)
    assert e.target_px == pytest.approx(101.8 - 1.5 * (102.0 * 1.003 - 101.8))
    assert e.hold_kind == "intraday" and e.time_stop == 24


def test_no_short_when_rsi_below_threshold_or_level_broken_or_too_far():
    bars, ind = _bars_and_ind()
    bars.loc[bars.index[10], "high"] = 102.0
    i = 20
    bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
    bars.loc[bars.index[i + 1], "open"] = 101.8
    ind.loc[ind.index[i], "rsi"] = 58.0                            # RSI too low
    assert _entries(bars, ind) == []
    ind.loc[ind.index[i], "rsi"] = 63.0
    bars.loc[bars.index[i], "high"] = 102.4                        # broke the level -> not a failed retest
    assert _entries(bars, ind) == []
    bars.loc[bars.index[i], ["high", "close"]] = [101.3, 101.2]    # 0.78% below the level -> too far
    bars.loc[bars.index[i + 1], "open"] = 101.2
    assert _entries(bars, ind) == []


def test_long_mirror_at_prior_swing_low_with_low_rsi():
    bars, ind = _bars_and_ind()
    bars.loc[bars.index[10], "low"] = 98.0
    i = 20
    bars.loc[bars.index[i], ["open", "low", "close"]] = [98.4, 98.1, 98.2]
    bars.loc[bars.index[i + 1], "open"] = 98.2
    ind.loc[ind.index[i], "rsi"] = 37.0                            # <= 100 - 60
    es = _entries(bars, ind)
    assert len(es) == 1 and es[0].side == "long" and es[0].tag == "prior_low_fade"
    assert es[0].stop_px == pytest.approx(98.0 * 0.997)
    assert es[0].target_px == pytest.approx(98.2 + 1.5 * (98.2 - 98.0 * 0.997))


def test_swing_level_ignores_prior_session_and_the_current_bar():
    bars, ind = _bars_and_ind(n=80)
    second = bars.index[40:]
    ind.loc[second, "bar_of_day"] = np.arange(40)                  # bars 40.. are a new session
    bars.index = list(bars.index[:40]) + list(pd.date_range("2025-01-03 09:30", periods=40, freq="5min"))
    bars.index.name = "ts"
    ind.index = bars.index
    bars.loc[bars.index[10], "high"] = 102.0                       # yesterday's high must not count
    i = 60                                                          # bar 20 of session two
    bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
    bars.loc[bars.index[i + 1], "open"] = 101.8
    ind.loc[ind.index[i], "rsi"] = 63.0
    assert _entries(bars, ind) == []
    bars.loc[bars.index[50], "high"] = 102.0                       # same-session prior high -> fires
    assert len(_entries(bars, ind)) == 1
    bars.loc[bars.index[50], "high"] = 100.5
    bars.loc[bars.index[i], "high"] = 102.0                        # the current bar's own high is not a "prior" high
    bars.loc[bars.index[i], "close"] = 101.9
    assert _entries(bars, ind) == []


def test_only_fires_inside_the_bar_of_day_window():
    bars, ind = _bars_and_ind(n=70)
    bars.loc[bars.index[55], "high"] = 102.0
    i = 65                                                          # 14:55 ET, past the window
    bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
    bars.loc[bars.index[i + 1], "open"] = 101.8
    ind.loc[ind.index[i], "rsi"] = 63.0
    assert _entries(bars, ind) == []


def test_family_descriptor_and_registry():
    from strategies.families.prior_high_rsi import FAMILIES_PRIOR_HIGH_RSI
    from strategies.families.common import grid_combos
    from strategies.families import get_family
    (fam,) = FAMILIES_PRIOR_HIGH_RSI
    assert fam.name == "prior_high_rsi" and fam.hold_kind == "intraday" and set(fam.sides) == {"long", "short"}
    assert set(fam.grid) == set(P) and 1 <= len(grid_combos(fam.grid)) <= 27
    assert get_family("prior_high_rsi") is fam


def test_a_high_set_in_the_last_three_bars_is_not_a_prior_high():
    """The level must be at least EXCLUDE_LAST bars old: a bar one tick under the previous bar's
    high in a steady climb is a continuation, not a retest."""
    bars, ind = _bars_and_ind()
    i = 20
    bars.loc[bars.index[i - 1], "high"] = 102.0                   # set one bar ago
    bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
    bars.loc[bars.index[i + 1], "open"] = 101.8
    ind.loc[ind.index[i], "rsi"] = 63.0
    assert _entries(bars, ind) == []
    bars.loc[bars.index[i - 1], "high"] = 100.5
    bars.loc[bars.index[i - 3], "high"] = 102.0                   # three bars ago -> old enough
    assert len(_entries(bars, ind)) == 1


def test_level_expires_after_the_lookback_window_and_needs_min_history():
    bars, ind = _bars_and_ind(n=70)
    bars.loc[bars.index[10], "high"] = 102.0
    i = 40                                                          # 30 bars later: outside the 24-bar window
    bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
    bars.loc[bars.index[i + 1], "open"] = 101.8
    ind.loc[ind.index[i], "rsi"] = 63.0
    assert _entries(bars, ind) == []
    bars, ind = _bars_and_ind()
    bars.loc[bars.index[1], "high"] = 102.0
    i = 4                                                           # only 4 session bars of history
    bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
    bars.loc[bars.index[i + 1], "open"] = 101.8
    ind.loc[ind.index[i], "rsi"] = 63.0
    assert _entries(bars, ind) == []


def test_long_side_rejects_rsi_just_above_mirror_threshold_broken_level_and_too_far():
    bars, ind = _bars_and_ind()
    bars.loc[bars.index[10], "low"] = 98.0
    i = 20
    bars.loc[bars.index[i], ["open", "low", "close"]] = [98.4, 98.1, 98.2]
    bars.loc[bars.index[i + 1], "open"] = 98.2
    ind.loc[ind.index[i], "rsi"] = 41.0                            # > 100 - 60
    assert _entries(bars, ind) == []
    ind.loc[ind.index[i], "rsi"] = 40.0                            # == 100 - 60 fires
    assert len(_entries(bars, ind)) == 1
    bars.loc[bars.index[i], "low"] = 97.6                          # broke the level
    assert _entries(bars, ind) == []
    bars.loc[bars.index[i], ["low", "close"]] = [98.5, 98.6]       # 0.61% above the level -> too far
    bars.loc[bars.index[i + 1], "open"] = 98.6
    assert _entries(bars, ind) == []


def test_one_entry_per_session_and_side():
    bars, ind = _bars_and_ind(n=50)
    bars.loc[bars.index[10], "high"] = 102.0
    for i in (20, 30):
        bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
        bars.loc[bars.index[i + 1], "open"] = 101.8
        ind.loc[ind.index[i], "rsi"] = 63.0
    es = _entries(bars, ind)
    assert len(es) == 1 and es[0].entry_ts == bars.index[21]


def test_grid_is_exactly_twenty_four_combos():
    from strategies.families.prior_high_rsi import FAMILIES_PRIOR_HIGH_RSI
    from strategies.families.common import grid_combos
    assert len(grid_combos(FAMILIES_PRIOR_HIGH_RSI[0].grid)) == 24


def test_deep_in_session_uses_the_rolling_window_and_never_yesterdays_high():
    """bod >= 26 takes the rolling-window branch: a high 10 bars back fires, yesterday's does not."""
    bars, ind = _bars_and_ind(n=140)
    ind.loc[ind.index[70:], "bar_of_day"] = np.arange(70)
    bars.index = list(bars.index[:70]) + list(pd.date_range("2025-01-03 09:30", periods=70, freq="5min"))
    bars.index.name = "ts"
    ind.index = bars.index
    bars.loc[bars.index[30], "high"] = 102.0                       # yesterday
    i = 110                                                         # bar 40 of session two
    bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
    bars.loc[bars.index[i + 1], "open"] = 101.8
    ind.loc[ind.index[i], "rsi"] = 63.0
    assert _entries(bars, ind) == []
    bars.loc[bars.index[i - 10], "high"] = 102.0                   # same session, 10 bars back
    es = _entries(bars, ind)
    assert len(es) == 1 and es[0].side == "short" and es[0].entry_ts == bars.index[i + 1]


def test_nan_rsi_never_fires_and_both_sides_can_fire_in_one_session():
    bars, ind = _bars_and_ind(n=50)
    bars.loc[bars.index[10], "high"] = 102.0
    bars.loc[bars.index[10], "low"] = 98.0
    bars.loc[bars.index[20], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
    bars.loc[bars.index[21], "open"] = 101.8
    bars.loc[bars.index[30], ["open", "low", "close"]] = [98.4, 98.1, 98.2]
    bars.loc[bars.index[31], "open"] = 98.2
    ind.loc[ind.index[20], "rsi"] = np.nan
    ind.loc[ind.index[30], "rsi"] = 37.0
    es = _entries(bars, ind)
    assert [e.side for e in es] == ["long"]
    ind.loc[ind.index[20], "rsi"] = 63.0
    assert sorted(e.side for e in _entries(bars, ind)) == ["long", "short"]


def test_next_open_through_the_stop_yields_no_entry():
    bars, ind = _bars_and_ind()
    bars.loc[bars.index[10], "high"] = 102.0
    i = 20
    bars.loc[bars.index[i], ["open", "high", "close"]] = [101.6, 101.9, 101.8]
    bars.loc[bars.index[i + 1], "open"] = 102.5                    # gaps above the 102.306 stop
    ind.loc[ind.index[i], "rsi"] = 63.0
    assert _entries(bars, ind) == []
