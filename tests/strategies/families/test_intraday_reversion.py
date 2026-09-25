import numpy as np
import pandas as pd
import pytest


def _bars_and_ind(n=40, day="2025-01-02"):
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="5min", name="ts")
    bars = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx)
    ind = pd.DataFrame(index=idx)
    ind["bar_of_day"] = np.arange(n)
    for col, val in {"vwap": 100.0, "vwap_sd": 0.5, "rsi": 50.0, "atr": 1.0, "session_open": 100.0, "gap_pct": 0.0,
                     "bb_lower": 98.0, "volume_ratio": 1.0}.items():
        ind[col] = val
    return bars, ind


def test_vwap_reversion_long_targets_vwap_and_requires_1r():
    from strategies.families.intraday_reversion import vwap_reversion_entries
    bars, ind = _bars_and_ind()
    i = 10
    bars.loc[bars.index[i], ["open", "close"]] = [98.6, 98.9]      # green bar, 2.2 sd below vwap
    bars.loc[bars.index[i + 1], "open"] = 98.9
    ind.loc[ind.index[i], "rsi"] = 30.0
    p = {"k": 2.0, "time_stop_bars": 12, "atr_stop": 1.0}
    es = vwap_reversion_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)
    assert len(es) == 1 and es[0].side == "long" and es[0].target_px == 100.0 and es[0].stop_px == pytest.approx(97.9)
    assert es[0].time_stop == 12
    ind.loc[ind.index[i], "atr"] = 2.0                                # target only 0.55R away -> skip
    assert vwap_reversion_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p) == []


def test_vwap_reversion_short_mirror():
    from strategies.families.intraday_reversion import vwap_reversion_entries
    bars, ind = _bars_and_ind()
    i = 10
    bars.loc[bars.index[i], ["open", "close"]] = [101.4, 101.1]      # red bar, 2.2 sd above
    bars.loc[bars.index[i + 1], "open"] = 101.1
    ind.loc[ind.index[i], "rsi"] = 70.0
    es = vwap_reversion_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {"k": 2.0, "time_stop_bars": 6, "atr_stop": 1.0})
    assert len(es) == 1 and es[0].side == "short" and es[0].target_px == 100.0 and es[0].stop_px == pytest.approx(102.1)


def test_gap_play_go_and_fade():
    from strategies.families.intraday_reversion import gap_play_entries
    bars, ind = _bars_and_ind()
    ind["gap_pct"] = 3.5
    ind["session_open"] = 103.5
    bars["open"] = 103.5; bars["close"] = 103.5
    bars.loc[bars.index[2], "close"] = 104.0                          # bar 2 = end of 15-min window, holding
    p = {"gap_pct": 3.0, "confirm_minutes": 15, "atr_stop": 1.0}
    es = gap_play_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)
    assert len(es) == 1 and es[0].side == "long" and es[0].tag == "gap_go" and es[0].entry_ts == bars.index[3]
    assert es[0].stop_px == pytest.approx(102.5) and es[0].target_px == pytest.approx(105.5)
    bars.loc[bars.index[2], "close"] = 103.0                          # failed to hold -> fade short
    es2 = gap_play_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)
    assert len(es2) == 1 and es2[0].side == "short" and es2[0].tag == "gap_fade"
    assert gap_play_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {**p, "gap_pct": 5.0}) == []


def test_mean_rev_intraday_is_two_session_multiday():
    from strategies.families.intraday_reversion import mean_rev_intraday_entries
    bars, ind = _bars_and_ind()
    i = 10
    bars.loc[bars.index[i], ["open", "close"]] = [97.5, 97.8]        # green, below bb_lower 98
    bars.loc[bars.index[i + 1], "open"] = 97.8
    ind.loc[ind.index[i], ["rsi", "volume_ratio"]] = [22.0, 1.5]
    es = mean_rev_intraday_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {"rsi_max": 25, "atr_stop": 2.0, "target_r": 1.5})
    assert len(es) == 1 and es[0].hold_kind == "multiday" and es[0].time_stop == 2
    assert es[0].stop_px == pytest.approx(95.8) and es[0].target_px == pytest.approx(97.8 + 3.0)
    ind.loc[ind.index[i], "rsi"] = 28.0
    assert mean_rev_intraday_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {"rsi_max": 25, "atr_stop": 2.0, "target_r": 1.5}) == []


def test_family_descriptors():
    from strategies.families.intraday_reversion import FAMILIES_INTRADAY_REVERSION
    from strategies.families.common import grid_combos
    assert {f.name for f in FAMILIES_INTRADAY_REVERSION} == {"vwap_reversion", "gap_play", "mean_rev_intraday"}
    assert all(len(grid_combos(f.grid)) <= 27 for f in FAMILIES_INTRADAY_REVERSION)
