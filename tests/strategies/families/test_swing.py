import numpy as np
import pandas as pd
import pytest


def _bars(days, n=78):
    return pd.concat([pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000},
                                   index=pd.date_range(f"{d} 09:30", periods=n, freq="5min", name="ts")) for d in days])


def _ind(bars):
    ind = pd.DataFrame(index=bars.index)
    ind["bar_of_day"] = bars.groupby(bars.index.normalize()).cumcount()
    defaults = {"d_close": 100.0, "d_low": 99.0, "d_high": 101.0, "d_sma50": 95.0, "d_ema20": 99.5, "d_ema10": 99.8,
                "d_rsi14": 45.0, "d_ret1": 0.5, "d_atr14": 2.0, "d_rsi3": 50.0, "d_bb_mid": 100.0,
                "d_bb20_upper": 104.0, "d_bb20_lower": 96.0, "d_bb25_upper": 105.0, "d_bb25_lower": 95.0,
                "d_atr20": 2.0, "d_atr100": 2.0, "d_high20": 103.0, "d_low20": 97.0, "d_high55": 106.0, "d_low55": 94.0}
    for k, v in defaults.items():
        ind[k] = v
    return ind


DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-02", periods=30)]


def test_pullback_trend_entry_at_session_open_and_spacing():
    from strategies.families.swing import pullback_trend_entries
    bars = _bars(DAYS)
    ind = _ind(bars)
    p = {"pullback_to": "ema20", "rsi_band": "35-50", "atr_stop": 2.0}
    es = pullback_trend_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)     # defaults satisfy the rule every day
    assert es[0].entry_ts == pd.Timestamp(DAYS[1] + " 09:30") and es[0].side == "long" and es[0].time_stop == 10
    assert es[0].stop_px == pytest.approx(96.0) and es[0].target_px == pytest.approx(108.0)
    dates = [e.entry_ts.normalize() for e in es]
    assert all((dates[i + 1] - dates[i]).days >= 14 for i in range(len(dates) - 1))     # 10 sessions apart
    ind["d_rsi14"] = 60.0
    assert pullback_trend_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p) == []
    ind["d_rsi14"] = 45.0
    ind["d_ema20"] = 98.0                                                              # low 99 never touched EMA
    assert pullback_trend_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p) == []


def test_mr_20d_long_short_and_r_requirement():
    from strategies.families.swing import mr_20d_entries
    bars = _bars(DAYS[:5])
    ind = _ind(bars)
    # d_bb_mid is set away from the session open (100) so the reward (target-entry) clears the
    # required risk (entry-stop = 1.5*atr14 = 3.0); the fixture's session opens are pinned at 100
    # regardless of d_close, so leaving d_bb_mid at its 100 default gives a zero-reward trade that
    # the documented "target - entry >= entry - stop" rule correctly rejects (see v1 Task 12
    # interface text). Edited from the v1 reference test, which left d_bb_mid at 100 here.
    ind["d_close"] = 95.5; ind["d_rsi3"] = 8.0; ind["d_bb_mid"] = 105.0   # below 2.0 band (96), extreme rsi3
    p = {"band_sigma": 2.0, "rsi3_extreme": 10, "time_stop_days": 5}
    es = mr_20d_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)
    assert es and es[0].side == "long" and es[0].target_px == 105.0 and es[0].stop_px == pytest.approx(97.0)
    assert mr_20d_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {**p, "band_sigma": 2.5}) == []   # 95.5 not below 95
    ind["d_close"] = 104.5; ind["d_rsi3"] = 93.0; ind["d_bb_mid"] = 95.0
    es2 = mr_20d_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)
    assert es2 and es2[0].side == "short" and es2[0].target_px == 95.0
    ind["d_bb_mid"] = 99.0                                     # target 1.0 away vs risk 3.0 -> skip
    assert mr_20d_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p) == []


def test_vol_breakout_requires_compression():
    from strategies.families.swing import vol_breakout_entries
    bars = _bars(DAYS[:5])
    ind = _ind(bars)
    ind["d_close"] = 103.5; ind["d_atr20"] = 1.2               # > d_high20 103; compression 0.6
    p = {"lookback": 20, "compression": 0.75, "atr_stop": 1.5}
    es = vol_breakout_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)
    assert es and es[0].side == "long" and es[0].time_stop == 15 and es[0].stop_px == pytest.approx(97.0)
    assert vol_breakout_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {**p, "compression": 0.5}) == []
    assert vol_breakout_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {**p, "lookback": 55}) == []   # 103.5 < 106
    ind["d_close"] = 96.5
    assert vol_breakout_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)[0].side == "short"


def test_family_descriptors():
    from strategies.families.swing import FAMILIES_SWING
    from strategies.families.common import grid_combos
    assert {f.name for f in FAMILIES_SWING} == {"pullback_trend", "mr_20d", "vol_breakout"}
    assert all(len(grid_combos(f.grid)) <= 27 and f.hold_kind == "multiday" for f in FAMILIES_SWING)


def test_mr_20d_carries_the_batch2_entry_style_axis():
    """Batch-2 Task B. `mr_20d` is the only `swing.py` family in the Batch-2 list; 2 band_sigma x
    3 rsi3_extreme x 2 time_stop_days x 2 entry_style = 24, exactly the cap, so no axis is dropped.
    `pullback_trend` and `vol_breakout` are not in the run and keep their Batch-1 grids."""
    from strategies.families.common import grid_combos, median_params
    from strategies.families.models import ENTRY_STYLES
    from strategies.families.swing import FAMILIES_SWING, mr_20d_entries
    by_name = {f.name: f for f in FAMILIES_SWING}
    assert by_name["mr_20d"].grid["entry_style"] == list(ENTRY_STYLES)
    assert len(grid_combos(by_name["mr_20d"].grid)) == 24
    assert median_params(by_name["mr_20d"].grid)["entry_style"] == "market"
    assert all("entry_style" not in by_name[n].grid for n in ("pullback_trend", "vol_breakout"))

    bars = _bars(DAYS[:5])
    ind = _ind(bars)
    ind["d_close"] = 95.5; ind["d_rsi3"] = 8.0; ind["d_bb_mid"] = 105.0
    p = {"band_sigma": 2.0, "rsi3_extreme": 10, "time_stop_days": 5}
    mkt = mr_20d_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), p)
    lim = mr_20d_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), {**p, "entry_style": "limit_open"})
    assert mkt and all(e.entry_style == "market" for e in mkt)
    assert lim and all(e.entry_style == "limit_open" for e in lim)
    assert [e.entry_ts for e in lim] == [e.entry_ts for e in mkt]
