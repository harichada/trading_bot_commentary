"""Batch-2 Task C factor families: 12-1 momentum and low-volatility trend. Two calendar weeks of
flat 5-minute sessions with the lagged daily/context columns pinned (mirrors
tests/strategies/families/test_swing2.py's fixture style), plus two days fabricated as
`d_month_start` so a family either fires on those bars or not at all -- the real
`_month_start`/`daily_features` boundary logic is exercised separately in test_indicators.py.
"""
import numpy as np
import pandas as pd
import pytest

DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-06", periods=10)]   # Mon 6th .. Fri 17th
MONTH_STARTS = (DAYS[0], DAYS[5])


def _bars(days, n=78):
    return pd.concat([pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000},
                                   index=pd.date_range(f"{d} 09:30", periods=n, freq="5min", name="ts")) for d in days])


DEFAULTS = {"d_close": 100.0, "d_atr14": 2.0, "d_sma200": 90.0}
CTX_DEFAULTS = {"ctx_mom_rank_12_1": 0.95, "ctx_vol_rank_60": 0.1, "ctx_rs_120": 5.0}


def _ind(bars, with_ctx=True, **over):
    ind = pd.DataFrame(index=bars.index)
    ind["bar_of_day"] = bars.groupby(bars.index.normalize()).cumcount()
    cols = {**DEFAULTS, **(CTX_DEFAULTS if with_ctx else {}), **over}
    for k, v in cols.items():
        ind[k] = v
    days = bars.index.normalize()
    ind["d_month_start"] = np.where(np.isin(days, pd.to_datetime(list(MONTH_STARTS))), 1.0, 0.0)
    return ind


def _run(fn, bars, ind, params):
    return fn(bars, ind, pd.DataFrame(), pd.DataFrame(), params)


# ---- mom_12_1 ----------------------------------------------------------------------------------

def test_mom_12_1_long_signals_at_month_start_and_fills_at_bar_one():
    from strategies.families.factor import mom_12_1_entries
    bars, params = _bars(DAYS), {"rank_min": 0.8, "entry_style": "market"}
    es = _run(mom_12_1_entries, bars, _ind(bars), params)
    assert len(es) == len(MONTH_STARTS)
    e = es[0]
    assert e.side == "long" and e.tag == "mom_12_1_long" and e.hold_kind == "multiday"
    assert e.entry_ts == pd.Timestamp(DAYS[0] + " 09:35") and e.entry_px == pytest.approx(100.0)
    assert e.stop_px == pytest.approx(94.0) and e.target_px == pytest.approx(220.0) and e.time_stop == 20
    assert e.entry_style == "market"


def test_mom_12_1_needs_month_start_and_bar_zero():
    from strategies.families.factor import mom_12_1_entries
    bars = _bars(DAYS)
    ind = _ind(bars)
    ind["d_month_start"] = 0.0
    assert _run(mom_12_1_entries, bars, ind, {"rank_min": 0.8}) == []


def test_mom_12_1_long_filters_rank_and_trend():
    from strategies.families.factor import mom_12_1_entries
    bars, params = _bars(DAYS), {"rank_min": 0.9}
    assert _run(mom_12_1_entries, bars, _ind(bars, ctx_mom_rank_12_1=0.85), params) == []
    assert _run(mom_12_1_entries, bars, _ind(bars, d_close=80.0), params) == []   # close below sma200


def test_mom_12_1_short_side_with_reward_capped():
    """Bottom-decile rank in a downtrend shorts; the nominal 20R target would go negative below
    the entry once risk exceeds 5% of price, so it is capped (mirrors swing2.py choice 3)."""
    from strategies.families.factor import mom_12_1_entries
    bars = _bars(DAYS)
    ind = _ind(bars, ctx_mom_rank_12_1=0.03, d_close=80.0, d_sma200=90.0)
    es = _run(mom_12_1_entries, bars, ind, {"rank_min": 0.8})
    assert len(es) == len(MONTH_STARTS) and es[0].side == "short" and es[0].tag == "mom_12_1_short"
    assert es[0].stop_px == pytest.approx(106.0) and es[0].target_px == pytest.approx(10.0)   # capped 90% of px


def test_mom_12_1_needs_context():
    from strategies.families.factor import mom_12_1_entries
    bars = _bars(DAYS)
    assert _run(mom_12_1_entries, bars, _ind(bars, with_ctx=False), {"rank_min": 0.8}) == []


def test_mom_12_1_entry_style_axis_reaches_make_entry():
    from strategies.families.common import grid_combos
    from strategies.families.factor import FAMILIES_FACTOR, mom_12_1_entries
    grid = {f.name: f.grid for f in FAMILIES_FACTOR}["mom_12_1"]
    assert set(grid) == {"rank_min", "entry_style"}
    assert set(grid["entry_style"]) == {"market", "limit_open"}
    assert len(grid_combos(grid)) == 4

    bars = _bars(DAYS)
    es = _run(mom_12_1_entries, bars, _ind(bars), {"rank_min": 0.8, "entry_style": "limit_open"})
    assert es and all(e.entry_style == "limit_open" for e in es)


# ---- low_vol_trend ------------------------------------------------------------------------------

def test_low_vol_trend_long_signals_at_month_start():
    from strategies.families.factor import low_vol_trend_entries
    bars, params = _bars(DAYS), {"vol_max": 0.2}
    es = _run(low_vol_trend_entries, bars, _ind(bars), params)
    assert len(es) == len(MONTH_STARTS)
    e = es[0]
    assert e.side == "long" and e.tag == "low_vol_trend" and e.hold_kind == "multiday"
    assert e.entry_ts == pd.Timestamp(DAYS[0] + " 09:35")
    assert e.stop_px == pytest.approx(95.0) and e.target_px == pytest.approx(200.0) and e.time_stop == 20


def test_low_vol_trend_filters_vol_rank_and_relative_strength_and_trend():
    from strategies.families.factor import low_vol_trend_entries
    bars, params = _bars(DAYS), {"vol_max": 0.2}
    assert _run(low_vol_trend_entries, bars, _ind(bars, ctx_vol_rank_60=0.5), params) == []
    assert _run(low_vol_trend_entries, bars, _ind(bars, ctx_rs_120=-1.0), params) == []
    assert _run(low_vol_trend_entries, bars, _ind(bars, d_close=80.0), params) == []


def test_low_vol_trend_needs_context_and_month_start():
    from strategies.families.factor import low_vol_trend_entries
    bars = _bars(DAYS)
    assert _run(low_vol_trend_entries, bars, _ind(bars, with_ctx=False), {"vol_max": 0.2}) == []
    ind = _ind(bars)
    ind["d_month_start"] = 0.0
    assert _run(low_vol_trend_entries, bars, ind, {"vol_max": 0.2}) == []


def test_low_vol_trend_has_no_entry_style_axis():
    """Per the plan's fallback: low_vol_trend's grid is vol_max only (the brief lists no
    entry_style axis for it)."""
    from strategies.families.factor import FAMILIES_FACTOR
    grid = {f.name: f.grid for f in FAMILIES_FACTOR}["low_vol_trend"]
    assert set(grid) == {"vol_max"}


# ---- descriptors ---------------------------------------------------------------------------

def test_family_descriptors_and_bounded_grids():
    from strategies.families.common import grid_combos
    from strategies.families.factor import FAMILIES_FACTOR
    names = {f.name for f in FAMILIES_FACTOR}
    assert names == {"mom_12_1", "low_vol_trend"}
    by_name = {f.name: f for f in FAMILIES_FACTOR}
    assert all(f.hold_kind == "multiday" for f in FAMILIES_FACTOR)
    assert by_name["mom_12_1"].sides == ("long", "short")
    assert by_name["low_vol_trend"].sides == ("long",)
    for f in FAMILIES_FACTOR:
        assert 1 <= len(grid_combos(f.grid)) <= 24 and callable(f.fn)


def test_every_family_tolerates_an_indicator_frame_with_no_columns():
    from strategies.families.common import median_params
    from strategies.families.factor import FAMILIES_FACTOR
    bars = _bars(DAYS[:3])
    empty = pd.DataFrame(index=bars.index)
    for f in FAMILIES_FACTOR:
        assert f.fn(bars, empty, pd.DataFrame(), pd.DataFrame(), median_params(f.grid)) == []


def test_families_run_against_a_real_indicator_frame_with_context():
    """Integration smoke: run each family over a real `indicator_frame` (lagged d_* columns, a
    real `d_month_start` computed from calendar dates, the per-session ctx_* join) to prove the
    column names and dtypes line up end to end -- mirrors test_swing2.py's own such test."""
    from strategies.families.common import median_params
    from strategies.families.factor import FAMILIES_FACTOR
    from strategies.families.indicators import indicator_frame
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=280)]
    frames, base = [], 100.0
    for k, d in enumerate(days):
        base *= 1.006 if k % 8 < 4 else 0.997   # net rising, so d_close clears d_sma200
        idx = pd.date_range(f"{d} 09:30", periods=78, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": base, "high": base * 1.004, "low": base * 0.996,
                                    "close": base * 1.001, "volume": 2000}, index=idx))
    bars = pd.concat(frames)
    daily = bars.groupby(bars.index.normalize()).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    daily.index.name = "date"
    dates = pd.DatetimeIndex(daily.index, name="date")
    context = pd.DataFrame({"mom_rank_12_1": 0.95, "vol_rank_60": 0.1, "rs_120": 5.0}, index=dates)
    ind = indicator_frame(bars, daily, context=context)
    assert ind["d_month_start"].sum() > 1   # more than one calendar month in a 280-session fixture
    assert (ind[ind["bar_of_day"] == 0]["d_sma200"].dropna() > 0).any()
    proposed = {}
    for f in FAMILIES_FACTOR:
        es = f.fn(bars, ind, pd.DataFrame(), pd.DataFrame(), median_params(f.grid))
        assert isinstance(es, list)
        assert all(e.hold_kind == f.hold_kind and e.side in f.sides for e in es)
        proposed[f.name] = es
    assert all(proposed[name] for name in proposed), {k: len(v) for k, v in proposed.items()}
