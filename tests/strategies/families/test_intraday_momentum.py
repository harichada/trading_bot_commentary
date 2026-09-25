import numpy as np
import pandas as pd
import pytest


def _bars_and_ind(n=40, day="2025-01-02"):
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="5min", name="ts")
    bars = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx)
    ind = pd.DataFrame(index=idx)
    ind["bar_of_day"] = np.arange(n)
    for col, val in {"rs_vs_spy_pct": 0.0, "volume_ratio": 1.0, "rsi": 50.0, "high_20": 101.0, "low_20": 99.0,
                     "adx": 15.0, "sma_20": 100.0, "macd": 0.0, "macd_signal": 0.0, "atr": 1.0, "rvol": 1.0,
                     "or15_high": 100.5, "or15_low": 99.5, "or30_high": 100.5, "or30_low": 99.5,
                     "or15_range_avg20": 2.0, "or30_range_avg20": 2.0}.items():
        ind[col] = val
    return bars, ind


def test_dt_momentum_breakout_long_entry_and_levels():
    from strategies.families.intraday_momentum import dt_momentum_entries
    bars, ind = _bars_and_ind()
    i = 10
    ind.loc[ind.index[i], ["rs_vs_spy_pct", "volume_ratio", "adx"]] = [0.8, 2.0, 25.0]
    bars.loc[bars.index[i], "close"] = 101.5           # > high_20 (101) and <= 1.03x
    params = {"rs_floor": 0.5, "atr_stop": 1.5, "target_r": 2.0}
    es = dt_momentum_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), params)
    assert len(es) == 1
    e = es[0]
    assert e.side == "long" and e.tag == "breakout" and e.entry_ts == bars.index[i + 1]
    assert e.stop_px == pytest.approx(100.0 - 1.5) and e.target_px == pytest.approx(100.0 + 3.0)
    assert e.time_stop == 66 - (i + 1) and e.hold_kind == "intraday"


def test_dt_momentum_rsi70_veto_and_rs_floor():
    from strategies.families.intraday_momentum import dt_momentum_entries
    bars, ind = _bars_and_ind()
    i = 10
    ind.loc[ind.index[i], ["rs_vs_spy_pct", "volume_ratio", "adx", "rsi"]] = [0.8, 2.0, 25.0, 72.0]
    bars.loc[bars.index[i], "close"] = 101.5
    params = {"rs_floor": 0.5, "atr_stop": 1.5, "target_r": 2.0}
    assert dt_momentum_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), params) == []
    ind.loc[ind.index[i], "rsi"] = 60.0
    assert dt_momentum_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {**params, "rs_floor": 1.0}) == []


def test_dt_momentum_short_breakdown_and_one_per_session():
    from strategies.families.intraday_momentum import dt_momentum_entries
    bars, ind = _bars_and_ind()
    for i in (10, 20):
        ind.loc[ind.index[i], ["rs_vs_spy_pct", "volume_ratio", "adx"]] = [-0.8, 2.0, 25.0]
        bars.loc[bars.index[i], "close"] = 98.5      # < low_20 (99) and >= 0.97x
    es = dt_momentum_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {"rs_floor": 0.5, "atr_stop": 1.0, "target_r": 1.5})
    assert len(es) == 1 and es[0].side == "short" and es[0].tag == "breakdown"
    assert es[0].stop_px == pytest.approx(101.0) and es[0].target_px == pytest.approx(98.5)


def test_orb_long_and_short_with_contraction_and_rvol():
    from strategies.families.intraday_momentum import orb_entries
    bars, ind = _bars_and_ind()
    i = 5
    ind.loc[ind.index[i], "rvol"] = 1.8
    bars.loc[bars.index[i], "close"] = 100.8                # > or15_high 100.5; range 1.0 vs avg 2.0 -> 50% contraction
    es = orb_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {"range_minutes": 15, "rvol_min": 1.5, "atr_stop": 1.0})
    assert len(es) == 1 and es[0].side == "long" and es[0].tag == "orb_long"
    assert es[0].stop_px == pytest.approx(max(100.0 - 1.0, 99.5 - 0.5)) and es[0].target_px == pytest.approx(100.0 + 2 * 1.0)
    ind.loc[ind.index[i], "or15_range_avg20"] = 1.1           # only 9% contraction -> no entry
    assert orb_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {"range_minutes": 15, "rvol_min": 1.5, "atr_stop": 1.0}) == []
    ind.loc[ind.index[i], "or15_range_avg20"] = 2.0
    bars.loc[bars.index[i], "close"] = 99.2                 # < or15_low
    es2 = orb_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {"range_minutes": 15, "rvol_min": 1.5, "atr_stop": 1.0})
    assert len(es2) == 1 and es2[0].side == "short"


def test_orb_window_limits():
    from strategies.families.intraday_momentum import orb_entries
    bars, ind = _bars_and_ind()
    for i in (2, 20):                                       # before range formed / after 10:30
        ind.loc[ind.index[i], "rvol"] = 3.0
        bars.loc[bars.index[i], "close"] = 100.8
    assert orb_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {"range_minutes": 15, "rvol_min": 1.5, "atr_stop": 1.0}) == []


def test_family_descriptors():
    from strategies.families.intraday_momentum import FAMILIES_INTRADAY_MOMENTUM
    from strategies.families.common import grid_combos
    names = {f.name for f in FAMILIES_INTRADAY_MOMENTUM}
    assert names == {"dt_momentum", "orb"}
    for f in FAMILIES_INTRADAY_MOMENTUM:
        assert len(grid_combos(f.grid)) <= 27 and f.hold_kind == "intraday"
