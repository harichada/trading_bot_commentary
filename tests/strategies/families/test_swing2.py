"""Batch-1 slower families (swing2): overnight hold, 2-day mean reversion, multi-day drop,
wide PEAD, weekly relative-strength rotation. Two calendar weeks of flat 5-minute sessions with
the lagged daily/context columns pinned, so a family either fires every eligible bar or not at all.
"""
import numpy as np
import pandas as pd
import pytest

DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-06", periods=10)]   # Mon 6th .. Fri 17th
WEEK_STARTS = (DAYS[0], DAYS[5])


def _bars(days, n=78):
    return pd.concat([pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000},
                                   index=pd.date_range(f"{d} 09:30", periods=n, freq="5min", name="ts")) for d in days])


DEFAULTS = {
    "d_close": 100.0, "d_ret1": -1.5, "d_atr14": 2.0, "d_rsi2": 3.0,
    "d_sma200": 90.0, "d_sma50": 95.0, "d_sma5": 99.0, "d_down_streak": 3.0,
}
CTX_DEFAULTS = {"ctx_rs_20": 10.0, "ctx_rs_rank_60": 0.97, "ctx_earnings_prev": 1.0, "ctx_eps_surprise_pct": 8.0}


def _ind(bars, with_ctx=True, **over):
    ind = pd.DataFrame(index=bars.index)
    ind["bar_of_day"] = bars.groupby(bars.index.normalize()).cumcount()
    cols = {**DEFAULTS, **(CTX_DEFAULTS if with_ctx else {}), **over}
    for k, v in cols.items():
        ind[k] = v
    days = bars.index.normalize()
    ind["d_week_start"] = np.where(np.isin(days, pd.to_datetime(list(WEEK_STARTS))), 1.0, 0.0)
    return ind


def _run(fn, bars, ind, params):
    return fn(bars, ind, pd.DataFrame(), pd.DataFrame(), params)


# ---- overnight_hold ---------------------------------------------------------------------------

def test_overnight_hold_long_signals_at_1550_and_fills_at_1555():
    from strategies.families.swing2 import overnight_hold_entries
    bars, params = _bars(DAYS), {"move_pct": 1.0, "rs_min": 0.0, "stop_atr": 1.5}
    es = _run(overnight_hold_entries, bars, _ind(bars), params)
    assert len(es) == len(DAYS)
    e = es[0]
    assert e.side == "long" and e.tag == "overnight_long" and e.hold_kind == "overnight"
    assert e.entry_ts == pd.Timestamp(DAYS[0] + " 15:55") and e.entry_px == pytest.approx(100.0)
    assert e.stop_px == pytest.approx(97.0) and e.target_px == pytest.approx(160.0) and e.time_stop == 1


def test_overnight_hold_filters_move_and_relative_strength():
    from strategies.families.swing2 import overnight_hold_entries
    bars = _bars(DAYS)
    assert _run(overnight_hold_entries, bars, _ind(bars), {"move_pct": 2.0, "rs_min": 0.0, "stop_atr": 1.5}) == []
    assert _run(overnight_hold_entries, bars, _ind(bars, ctx_rs_20=1.0),
                {"move_pct": 1.0, "rs_min": 5.0, "stop_atr": 1.5}) == []


def test_overnight_hold_short_on_an_up_day_with_weak_relative_strength():
    from strategies.families.swing2 import overnight_hold_entries
    bars = _bars(DAYS)
    ind = _ind(bars, d_ret1=1.5, ctx_rs_20=-10.0)
    es = _run(overnight_hold_entries, bars, ind, {"move_pct": 1.0, "rs_min": 5.0, "stop_atr": 1.5})
    assert len(es) == len(DAYS) and es[0].side == "short" and es[0].tag == "overnight_short"
    assert es[0].stop_px == pytest.approx(103.0) and es[0].target_px == pytest.approx(100.0 - 20 * 3.0)


def test_short_reward_is_capped_so_a_twenty_r_target_stays_positive():
    """20R below a short entry goes negative once risk exceeds 5% of price, and make_entry drops
    a non-positive target -- which would silently delete the whole short side. The reward is
    capped at 90% of the entry price (still effectively unreachable)."""
    from strategies.families.swing2 import overnight_hold_entries
    bars = _bars(DAYS)
    ind = _ind(bars, d_ret1=1.5, ctx_rs_20=-10.0)
    es = _run(overnight_hold_entries, bars, ind, {"move_pct": 1.0, "rs_min": 5.0, "stop_atr": 3.0})
    assert len(es) == len(DAYS) and es[0].stop_px == pytest.approx(106.0)
    assert es[0].target_px == pytest.approx(10.0)


def test_overnight_hold_needs_context():
    from strategies.families.swing2 import overnight_hold_entries
    bars = _bars(DAYS)
    assert _run(overnight_hold_entries, bars, _ind(bars, with_ctx=False),
                {"move_pct": 1.0, "rs_min": 0.0, "stop_atr": 1.5}) == []


# ---- mr_short_term ---------------------------------------------------------------------------

def test_mr_short_term_long_fills_at_bar_one_and_spaces_by_the_time_stop():
    from strategies.families.swing2 import mr_short_term_entries
    bars = _bars(DAYS)
    params = {"rsi_extreme": 5.0, "time_stop_days": 2}
    es = _run(mr_short_term_entries, bars, _ind(bars, with_ctx=False), params)   # needs no context
    assert [e.entry_ts for e in es] == [pd.Timestamp(DAYS[i] + " 09:35") for i in (0, 2, 4, 6, 8)]
    e = es[0]
    assert e.side == "long" and e.tag == "mr_st_long" and e.hold_kind == "multiday" and e.time_stop == 2
    assert e.stop_px == pytest.approx(95.0) and e.target_px == pytest.approx(100.0 + 20 * 5.0)


def test_mr_short_term_needs_the_trend_filter_and_the_rsi_extreme():
    from strategies.families.swing2 import mr_short_term_entries
    bars, params = _bars(DAYS), {"rsi_extreme": 5.0, "time_stop_days": 3}
    assert _run(mr_short_term_entries, bars, _ind(bars, d_sma200=110.0), params) == []   # close below sma200
    assert _run(mr_short_term_entries, bars, _ind(bars, d_rsi2=30.0), params) == []      # not oversold


def test_mr_short_term_short_side():
    from strategies.families.swing2 import mr_short_term_entries
    bars = _bars(DAYS)
    ind = _ind(bars, d_rsi2=97.0, d_sma200=110.0)
    es = _run(mr_short_term_entries, bars, ind, {"rsi_extreme": 5.0, "time_stop_days": 5})
    assert es and es[0].side == "short" and es[0].tag == "mr_st_short"
    assert es[0].stop_px == pytest.approx(105.0) and es[0].target_px == pytest.approx(10.0)   # capped 20R


def test_mr_short_term_grid_is_one_mirrored_rsi_axis():
    """v-batch1-task1-r1 (Minor 3): the long side reads only the low threshold and the short side
    only the high one, so two independent axes spent 12 grid points on 6 distinct per-side
    configurations. One `rsi_extreme` axis, mirrored as (v, 100 - v) the way `mr_20d`'s
    `rsi3_extreme` already is, searches the same thing without inflating the search accounting."""
    from strategies.families.common import grid_combos
    from strategies.families.swing2 import FAMILIES_SWING2, mr_short_term_entries
    grid = {f.name: f.grid for f in FAMILIES_SWING2}["mr_short_term"]
    assert set(grid) == {"rsi_extreme", "time_stop_days"}
    assert grid["rsi_extreme"] == [5, 10] and len(grid_combos(grid)) == 6

    bars, params = _bars(DAYS), {"rsi_extreme": 10.0, "time_stop_days": 2}
    assert _run(mr_short_term_entries, bars, _ind(bars, with_ctx=False, d_rsi2=10.0), params)          # lo = 10
    assert _run(mr_short_term_entries, bars, _ind(bars, with_ctx=False, d_rsi2=10.5), params) == []
    hi = _run(mr_short_term_entries, bars, _ind(bars, with_ctx=False, d_rsi2=90.0, d_sma200=110.0), params)
    assert hi and hi[0].side == "short"                                                                # hi = 90
    assert _run(mr_short_term_entries, bars, _ind(bars, with_ctx=False, d_rsi2=89.5, d_sma200=110.0), params) == []


# ---- multi_day_drop --------------------------------------------------------------------------

def test_multi_day_drop_needs_three_down_closes_and_an_uptrend():
    from strategies.families.swing2 import multi_day_drop_entries
    bars, params = _bars(DAYS), {"time_stop_days": 3, "target_atr": 1.0}
    es = _run(multi_day_drop_entries, bars, _ind(bars, with_ctx=False), params)          # needs no context
    e = es[0]
    assert e.side == "long" and e.tag == "multi_day_drop" and e.entry_ts == pd.Timestamp(DAYS[0] + " 09:35")
    assert e.stop_px == pytest.approx(96.0) and e.target_px == pytest.approx(102.0) and e.time_stop == 3
    assert _run(multi_day_drop_entries, bars, _ind(bars, d_down_streak=2.0), params) == []
    assert _run(multi_day_drop_entries, bars, _ind(bars, d_sma200=110.0), params) == []


def test_multi_day_drop_target_scales_with_atr():
    from strategies.families.swing2 import multi_day_drop_entries
    bars = _bars(DAYS)
    es = _run(multi_day_drop_entries, bars, _ind(bars), {"time_stop_days": 5, "target_atr": 2.0})
    assert es[0].target_px == pytest.approx(104.0) and es[0].time_stop == 5


# ---- pead_wide -------------------------------------------------------------------------------

def test_pead_wide_long_signals_at_bar_five_with_no_adv_restriction():
    from strategies.families.swing2 import pead_wide_entries
    bars, params = _bars(DAYS), {"surprise_min": 5.0, "time_stop_days": 5}
    es = _run(pead_wide_entries, bars, _ind(bars), params)     # no ctx_adv_bucket column at all
    e = es[0]
    assert e.side == "long" and e.tag == "pead_wide_long" and e.entry_ts == pd.Timestamp(DAYS[0] + " 10:00")
    assert e.stop_px == pytest.approx(97.0) and e.target_px == pytest.approx(109.0) and e.time_stop == 5


def test_pead_wide_filters_surprise_and_earnings_flag_and_shorts():
    from strategies.families.swing2 import pead_wide_entries
    bars, params = _bars(DAYS), {"surprise_min": 10.0, "time_stop_days": 3}
    assert _run(pead_wide_entries, bars, _ind(bars), params) == []                     # surprise 8 < 10
    assert _run(pead_wide_entries, bars, _ind(bars, ctx_earnings_prev=0.0),
                {"surprise_min": 5.0, "time_stop_days": 3}) == []
    es = _run(pead_wide_entries, bars, _ind(bars, ctx_eps_surprise_pct=-12.0), params)
    assert es and es[0].side == "short" and es[0].tag == "pead_wide_short"
    assert es[0].stop_px == pytest.approx(103.0) and es[0].target_px == pytest.approx(91.0)


def test_pead_wide_needs_context():
    from strategies.families.swing2 import pead_wide_entries
    bars = _bars(DAYS)
    assert _run(pead_wide_entries, bars, _ind(bars, with_ctx=False), {"surprise_min": 5.0, "time_stop_days": 5}) == []


# ---- rs_rotation -----------------------------------------------------------------------------

def test_rs_rotation_fires_once_per_week():
    from strategies.families.swing2 import rs_rotation_entries
    bars = _bars(DAYS)
    es = _run(rs_rotation_entries, bars, _ind(bars), {"rank_min": 0.9, "time_stop_days": 5})
    assert [e.entry_ts for e in es] == [pd.Timestamp(d + " 09:35") for d in WEEK_STARTS]
    e = es[0]
    assert e.side == "long" and e.tag == "rs_rotation" and e.hold_kind == "multiday" and e.time_stop == 5
    assert e.stop_px == pytest.approx(95.0) and e.target_px == pytest.approx(100.0 + 20 * 5.0)
    # A 10-session hold spans both weeks, so the second week's signal is skipped.
    es10 = _run(rs_rotation_entries, bars, _ind(bars), {"rank_min": 0.9, "time_stop_days": 10})
    assert [e.entry_ts for e in es10] == [pd.Timestamp(WEEK_STARTS[0] + " 09:35")]


def test_rs_rotation_filters_rank_and_trend_and_needs_context():
    from strategies.families.swing2 import rs_rotation_entries
    bars, params = _bars(DAYS), {"rank_min": 0.95, "time_stop_days": 5}
    assert _run(rs_rotation_entries, bars, _ind(bars, ctx_rs_rank_60=0.92), params) == []
    assert _run(rs_rotation_entries, bars, _ind(bars, d_sma50=105.0), params) == []
    assert _run(rs_rotation_entries, bars, _ind(bars, with_ctx=False), params) == []


def test_rs_rotation_ignores_mid_week_sessions():
    from strategies.families.swing2 import rs_rotation_entries
    bars = _bars(DAYS)
    ind = _ind(bars)
    ind["d_week_start"] = 0.0
    assert _run(rs_rotation_entries, bars, ind, {"rank_min": 0.9, "time_stop_days": 5}) == []


# ---- descriptors -----------------------------------------------------------------------------

def test_family_descriptors_and_bounded_grids():
    from strategies.families.common import grid_combos
    from strategies.families.swing2 import FAMILIES_SWING2
    names = {f.name for f in FAMILIES_SWING2}
    assert names == {"overnight_hold", "mr_short_term", "multi_day_drop", "pead_wide", "rs_rotation"}
    by_name = {f.name: f for f in FAMILIES_SWING2}
    assert by_name["overnight_hold"].hold_kind == "overnight"
    assert all(by_name[n].hold_kind == "multiday" for n in names - {"overnight_hold"})
    assert by_name["multi_day_drop"].sides == ("long",) and by_name["rs_rotation"].sides == ("long",)
    for f in FAMILIES_SWING2:
        assert 1 <= len(grid_combos(f.grid)) <= 24 and callable(f.fn)


def test_every_family_tolerates_an_indicator_frame_with_no_columns():
    from strategies.families.swing2 import FAMILIES_SWING2
    from strategies.families.common import median_params
    bars = _bars(DAYS[:3])
    empty = pd.DataFrame(index=bars.index)
    for f in FAMILIES_SWING2:
        assert f.fn(bars, empty, pd.DataFrame(), pd.DataFrame(), median_params(f.grid)) == []


def test_families_run_against_a_real_indicator_frame_with_context():
    """Integration smoke: the pinned fixtures above hand each family a hand-built `ind`, so this
    runs them over a real `indicator_frame` (lagged d_* columns, the per-session ctx_* join and its
    dtypes) to prove the column names and dtypes line up end to end.

    v-batch1-task1-r1 (Minor 5): 210 sessions, not 60. `d_sma200` needs 200 daily closes before it
    forms, so a shorter fixture could only ever prove that `mr_short_term` and `multi_day_drop`
    propose *nothing* -- which is also what a misspelled column name would produce. The path is a
    repeating 4-up (+2.5%) / 4-down (-2.2%) cycle: net rising (so `d_close` clears `d_sma200` and
    `d_sma50`), |d_ret1| over 2% on every session (both `overnight_hold` grid points fire), a
    3+-session down streak every cycle (`multi_day_drop`) and a `d_rsi2` under 10 at the end of each
    down leg (`mr_short_term`)."""
    from strategies.families.common import median_params
    from strategies.families.indicators import indicator_frame
    from strategies.families.swing2 import FAMILIES_SWING2
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=210)]
    frames, base = [], 100.0
    for k, d in enumerate(days):
        base *= 1.025 if k % 8 < 4 else 0.978
        idx = pd.date_range(f"{d} 09:30", periods=78, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": base, "high": base * 1.006, "low": base * 0.994,
                                    "close": base * 1.002, "volume": 2000}, index=idx))
    bars = pd.concat(frames)
    daily = bars.groupby(bars.index.normalize()).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    daily.index.name = "date"
    dates = pd.DatetimeIndex(daily.index, name="date")
    context = pd.DataFrame({"rs_20": 12.0, "rs_rank_60": 0.97, "earnings_prev": True,
                            "eps_surprise_pct": 9.0}, index=dates)
    ind = indicator_frame(bars, daily, context=context)
    mondays = [d for d in dates if d.weekday() == 0]
    assert (ind.loc[mondays[1].strftime("%Y-%m-%d"), "d_week_start"] == 1.0).all()
    # The 200-session trend filter is reachable in this fixture, which is what lets the two
    # d_sma200 families below actually propose.
    session_opens = ind[ind["bar_of_day"] == 0]
    assert session_opens["d_sma200"].notna().any() and (session_opens["d_down_streak"] >= 3).any()
    proposed = {}
    for f in FAMILIES_SWING2:
        es = f.fn(bars, ind, pd.DataFrame(), pd.DataFrame(), median_params(f.grid))
        assert isinstance(es, list)
        assert all(e.hold_kind == f.hold_kind and e.side in f.sides for e in es)
        proposed[f.name] = es
    # Every family proposes on its own median grid point -- a column name or dtype that did not
    # line up would show up here as an empty list.
    assert all(proposed[name] for name in proposed), {k: len(v) for k, v in proposed.items()}
