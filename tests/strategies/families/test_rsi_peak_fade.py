"""rsi_peak_fade family (2026-10-15): the user's short rule, "short when RSI is at its max and
about to fall". See the module docstring for the rule; these tests pin each clause."""
import numpy as np
import pandas as pd
import pytest

from strategies.families.common import grid_combos
from strategies.families.registry import FAMILIES
from strategies.families.rsi_peak_fade import (FAMILIES_RSI_PEAK_FADE, MAX_BOD, TIME_STOP_BARS, _turn_masks,
                                               rsi_peak_fade_entries)

P = {"peak_min": 60, "ext_min": 0, "target_r": 1.5}
BASE_RSI = 45.0     # below RSI_MID so a post-pattern bar never sits on the short-side boundary


def _bars_and_ind(n=40, days=("2025-01-02",), ext_pct=5.0):
    idx = pd.DatetimeIndex([], name="ts")
    for day in days:
        idx = idx.append(pd.date_range(f"{day} 09:30", periods=n, freq="5min", name="ts"))
    bars = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx)
    ind = pd.DataFrame(index=idx)
    ind["bar_of_day"] = np.tile(np.arange(n), len(days))
    ind["rsi"] = BASE_RSI
    ind["vwap"] = 100.0
    ind["d_close"] = 100.0 * (1 + ext_pct / 100)       # lagged daily context: prior close vs 20-day EMA
    ind["d_ema20"] = 100.0
    return bars, ind


def _short_setup(bars, ind, i=20, rsis=(68.0, 66.0, 62.0)):
    """RSI prints `rsis` ending at bar i (the first one is the peak); price sits just under the
    bounce high of 102.0 set on the peak bar; the next open is 101.8."""
    k = i - len(rsis) + 1
    for j, r in enumerate(rsis):
        ind.loc[ind.index[k + j], "rsi"] = r
    bars.loc[bars.index[k], ["open", "high", "close"]] = [101.5, 102.0, 101.9]
    for j in range(k + 1, i + 1):
        bars.loc[bars.index[j], ["open", "high", "close"]] = [101.8, 101.95, 101.8]
    bars.loc[bars.index[i + 1], "open"] = 101.8


def _long_setup(bars, ind, i=20, rsis=(32.0, 34.0, 38.0)):
    k = i - len(rsis) + 1
    for j, r in enumerate(rsis):
        ind.loc[ind.index[k + j], "rsi"] = r
    bars.loc[bars.index[k], ["open", "low", "close"]] = [98.5, 98.0, 98.1]
    for j in range(k + 1, i + 1):
        bars.loc[bars.index[j], ["open", "low", "close"]] = [98.2, 98.05, 98.2]
    bars.loc[bars.index[i + 1], "open"] = 98.2


def _entries(bars, ind, p=P):
    return rsi_peak_fade_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)


def test_short_when_rsi_has_dropped_from_its_peak_and_price_still_near_the_high():
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind)
    es = _entries(bars, ind)
    assert len(es) == 1
    e = es[0]
    assert e.side == "short" and e.entry_ts == bars.index[21] and e.tag == "rsi_peak_fade"
    assert e.entry_px == pytest.approx(101.8)
    assert e.stop_px == pytest.approx(102.1054)                  # the 0.3% floor binds (102.051 from the pad)
    assert e.target_px == pytest.approx(101.8 - 1.5 * (102.1054 - 101.8))
    assert e.hold_kind == "intraday" and e.time_stop == TIME_STOP_BARS


def test_short_fires_on_the_stall_after_the_drop_not_only_on_straight_declines():
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind, rsis=(78.7, 68.3, 72.3))             # the user's 09-29 10:11 shape
    assert len(_entries(bars, ind)) == 1
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind, rsis=(62.0, 58.0))                   # one bar after the peak is enough
    assert len(_entries(bars, ind)) == 1


@pytest.mark.parametrize("rsis, why", [
    ((58.0, 55.0, 54.0), "peak below peak_min"),
    ((68.0, 67.0, 66.0), "only 2 points off the peak"),
    ((68.0, 66.0, 70.5, 68.0), "a higher print became the peak and the drop from it is only 2.5"),
    ((68.0, 66.0, 48.0), "already below 50: that is a chase"),
    ((68.0, 66.0, 67.0, 66.0, 61.0), "peak 4 bars back: too old"),
])
def test_no_short(rsis, why):
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind, rsis=rsis)
    assert _entries(bars, ind) == [], why


def test_a_higher_print_after_the_peak_becomes_the_peak():
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind, rsis=(68.0, 66.0, 70.5, 64.0))
    bars.loc[bars.index[19], "high"] = 102.2                     # the bar of the 70.5 print made the high
    bars.loc[bars.index[20], ["open", "close"]] = [102.0, 102.0]
    bars.loc[bars.index[21], "open"] = 102.0
    es = _entries(bars, ind)
    assert len(es) == 1
    assert es[0].stop_px == pytest.approx(max(102.2 * 1.0005, 102.0 * 1.003))


def test_no_short_when_price_is_far_from_the_bounce_high():
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind)
    bars.loc[bars.index[20], ["open", "high", "close"]] = [101.0, 101.1, 101.0]   # ~1% off the high
    bars.loc[bars.index[21], "open"] = 101.0
    assert _entries(bars, ind) == []


def test_peak_must_be_the_highest_rsi_of_the_prior_half_hour():
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind)                                      # peak at bar 18
    ind.loc[ind.index[12], "rsi"] = 70.0                         # 6 bars before the peak: inside the window
    assert _entries(bars, ind) == []
    ind.loc[ind.index[12], "rsi"] = BASE_RSI
    ind.loc[ind.index[11], "rsi"] = 70.0                         # 7 bars before: outside
    assert len(_entries(bars, ind)) == 1
    ind.loc[ind.index[12], "rsi"] = 68.0                         # a tie before the peak is allowed
    assert len(_entries(bars, ind)) == 1


def test_lag_three_peak_with_the_bounce_high_made_after_the_peak_bar():
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind, rsis=(68.0, 66.0, 66.0, 64.0))      # qualifies only at lag 3
    bars.loc[bars.index[18], "high"] = 102.3                     # price kept rising one bar after the RSI peak
    bars.loc[bars.index[20], ["open", "close"]] = [101.9, 102.0]
    bars.loc[bars.index[21], "open"] = 102.0
    es = _entries(bars, ind)
    assert len(es) == 1 and es[0].entry_ts == bars.index[21]
    assert es[0].stop_px == pytest.approx(102.35115)             # pad above the later high binds, not the 0.3% floor
    bars.loc[bars.index[20], ["open", "close"]] = [101.8, 101.8]  # 0.49% under that later high: too far
    bars.loc[bars.index[21], "open"] = 101.8
    assert _entries(bars, ind) == []


def test_rsi_warmup_nans_before_the_peak_do_not_qualify_it():
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind, i=12, rsis=(65.0, 60.0, 55.0))
    ind.loc[ind.index[:10], "rsi"] = np.nan                      # RSI warm-up: the half hour before the peak is unknown
    ind.loc[ind.index[13:], "rsi"] = np.nan
    assert _entries(bars, ind) == []
    bars, ind = _bars_and_ind(n=70)
    _short_setup(bars, ind, i=2)                                 # peak at bar 0: no prior bars exist, still fine
    assert len(_entries(bars, ind)) == 1


def test_extension_context_is_optional_and_gates_shorts_when_set():
    p = dict(P, ext_min=3)
    bars, ind = _bars_and_ind(ext_pct=5.0)
    _short_setup(bars, ind)
    assert len(_entries(bars, ind, p)) == 1
    bars, ind = _bars_and_ind(ext_pct=1.0)                       # not extended enough
    _short_setup(bars, ind)
    assert _entries(bars, ind, p) == []
    assert len(_entries(bars, ind)) == 1                         # ext_min 0: no context needed
    ind["d_close"] = np.nan                                      # no daily data at all
    assert _entries(bars, ind, p) == []
    assert len(_entries(bars, ind)) == 1


def test_long_mirror_on_the_stall_after_an_rsi_trough():
    bars, ind = _bars_and_ind(ext_pct=-5.0)
    _long_setup(bars, ind)
    es = _entries(bars, ind)
    assert len(es) == 1 and es[0].side == "long" and es[0].tag == "rsi_trough_fade"
    assert es[0].entry_ts == bars.index[21]
    assert es[0].stop_px == pytest.approx(97.9054)               # the 0.3% floor binds (97.951 from the pad)
    assert es[0].target_px == pytest.approx(98.2 + 1.5 * (98.2 - 97.9054))
    assert len(_entries(bars, ind, dict(P, ext_min=3))) == 1    # -5% below the EMA satisfies ext_min 3
    ind["d_close"] = 99.0                                        # only -1% below: context blocks it
    assert _entries(bars, ind, dict(P, ext_min=3)) == []


@pytest.mark.parametrize("rsis, why", [
    ((42.0, 45.0, 46.0), "trough above 100 - peak_min"),
    ((32.0, 33.0, 34.0), "only 2 points off the trough"),
    ((32.0, 34.0, 52.0), "already above 50: that is a chase"),
])
def test_no_long(rsis, why):
    bars, ind = _bars_and_ind(ext_pct=-5.0)
    _long_setup(bars, ind, rsis=rsis)
    assert _entries(bars, ind) == [], why


def test_peak_in_the_previous_session_does_not_count():
    bars, ind = _bars_and_ind(n=10, days=("2025-01-02", "2025-01-03"))
    _short_setup(bars, ind, i=11)                                # peak at day-1 bar 9, prints at day-2 bars 0-1
    assert _entries(bars, ind) == []
    bars, ind = _bars_and_ind(n=10, days=("2025-01-02", "2025-01-03"))
    _short_setup(bars, ind, i=14)                                # same frame, pattern inside day 2: fine
    assert [e.entry_ts for e in _entries(bars, ind)] == [bars.index[15]]


def test_entry_window_edges():
    bars, ind = _bars_and_ind(n=70)
    _short_setup(bars, ind, i=2)                                 # peak at bar 0: earliest allowed
    assert len(_entries(bars, ind)) == 1
    bars, ind = _bars_and_ind(n=70)
    _short_setup(bars, ind, i=MAX_BOD)                           # last trigger bar: enters at the next open
    assert [e.entry_ts for e in _entries(bars, ind)] == [bars.index[MAX_BOD + 1]]
    bars, ind = _bars_and_ind(n=70)
    _short_setup(bars, ind, i=MAX_BOD + 1)                       # one bar later: no new entry
    assert _entries(bars, ind) == []


def test_a_next_open_above_the_bounce_high_still_gets_a_stop_above_it():
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind)
    bars.loc[bars.index[21], "open"] = 102.5                     # gaps above the 102.0 bounce high
    es = _entries(bars, ind)
    assert len(es) == 1 and es[0].entry_px == pytest.approx(102.5)
    assert es[0].stop_px == pytest.approx(102.5 * 1.003)         # the 0.3% floor re-prices off the actual entry


def test_one_entry_per_session_per_side_keeps_both_sides():
    bars, ind = _bars_and_ind()
    _short_setup(bars, ind, i=30)
    assert [e.entry_ts for e in _entries(bars, ind)] == [bars.index[31]]   # the later setup fires on its own
    _short_setup(bars, ind, i=20)
    assert [e.entry_ts for e in _entries(bars, ind)] == [bars.index[21]]   # ... but the first one wins
    ind["d_close"] = np.nan                                      # ext_min 0: context irrelevant
    _long_setup(bars, ind, i=10)
    es = _entries(bars, ind)
    assert [(e.side, e.entry_ts) for e in es] == [("long", bars.index[11]), ("short", bars.index[21])]


def test_short_and_long_lags_are_tracked_separately():
    rsi = np.array([BASE_RSI] * 17 + [50.0, 40.0, 65.0, 50.0])  # bar 20: short peak at lag 1, long trough at lag 2
    bod = np.arange(len(rsi))
    short, long, lag_s, lag_l = _turn_masks(rsi, bod, 60.0)
    assert short[20] and long[20] and lag_s[20] == 1 and lag_l[20] == 2


def test_descriptor_and_registry():
    fam = FAMILIES_RSI_PEAK_FADE[0]
    assert fam.name == "rsi_peak_fade" and fam.hold_kind == "intraday" and fam.sides == ("long", "short")
    assert len(grid_combos(fam.grid)) == 18
    assert FAMILIES["rsi_peak_fade"] is fam
    assert "prior_high_rsi" not in FAMILIES
