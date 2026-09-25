import numpy as np
import pandas as pd
import pytest


def _bars(day: str, rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", periods=len(rows), freq="5min", name="ts")
    df = pd.DataFrame(rows, index=idx)
    df["volume"] = 1000
    return df


def _const_ind(bars: pd.DataFrame, **cols) -> pd.DataFrame:
    ind = pd.DataFrame(index=bars.index)
    ind["bar_of_day"] = bars.groupby(bars.index.normalize()).cumcount()
    for k, v in cols.items():
        ind[k] = v
    return ind


# ---- orb_inplay -------------------------------------------------------------------------------

def test_orb_inplay_long_entry_and_levels():
    from strategies.families.inplay import orb_inplay_entries
    bars = _bars("2025-01-02", [
        {"open": 10.00, "high": 10.20, "low": 9.90, "close": 10.50},   # bar 0: green, h0=10.20, l0=9.90
        {"open": 10.30, "high": 10.60, "low": 10.25, "close": 10.55},  # bar 1: breaks h0 -> long signal
        {"open": 10.55, "high": 10.65, "low": 10.50, "close": 10.60},  # bar 2: fill bar (i+1)
        {"open": 10.60, "high": 10.70, "low": 10.55, "close": 10.65},
        {"open": 10.65, "high": 10.75, "low": 10.60, "close": 10.70},
    ])
    ind = _const_ind(bars, ctx_rvol_rank5=5, ctx_atr14_pct=2.0)
    params = {"top_n": 10, "stop_atr": 0.10, "trail_r": 0}
    es = orb_inplay_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), params)
    assert len(es) == 1
    e = es[0]
    px = 10.55
    atr_dollars = 2.0 / 100 * px
    stop = max(px - 0.10 * atr_dollars, 9.90)
    risk = px - stop
    target = px + 20 * risk
    assert e.side == "long" and e.tag == "orb_inplay_long" and e.entry_ts == bars.index[2]
    assert e.entry_px == pytest.approx(px) and e.stop_px == pytest.approx(stop) and e.target_px == pytest.approx(target)
    assert e.time_stop == max(1, 77 - 2) and e.hold_kind == "intraday"


def test_orb_inplay_short_entry_with_trail_r_target():
    from strategies.families.inplay import orb_inplay_entries
    bars = _bars("2025-01-02", [
        {"open": 10.50, "high": 10.60, "low": 10.20, "close": 10.00},  # bar 0: red, h0=10.60, l0=10.20
        {"open": 10.15, "high": 10.20, "low": 10.10, "close": 10.12},  # bar 1: breaks l0 -> short signal
        {"open": 10.05, "high": 10.10, "low": 10.00, "close": 10.05},  # bar 2: fill bar (i+1)
        {"open": 10.05, "high": 10.10, "low": 10.00, "close": 10.05},
        {"open": 10.05, "high": 10.10, "low": 10.00, "close": 10.05},
    ])
    ind = _const_ind(bars, ctx_rvol_rank5=5, ctx_atr14_pct=2.0)
    params = {"top_n": 10, "stop_atr": 0.10, "trail_r": 5}
    es = orb_inplay_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), params)
    assert len(es) == 1
    e = es[0]
    px = 10.05
    atr_dollars = 2.0 / 100 * px
    stop = min(px + 0.10 * atr_dollars, 10.60)
    risk = stop - px
    target = px - 5 * risk
    assert e.side == "short" and e.tag == "orb_inplay_short" and e.entry_ts == bars.index[2]
    assert e.stop_px == pytest.approx(stop) and e.target_px == pytest.approx(target)
    assert e.time_stop == max(1, 77 - 2)


def test_orb_inplay_rank_filter_blocks_entry():
    from strategies.families.inplay import orb_inplay_entries
    bars = _bars("2025-01-02", [
        {"open": 10.00, "high": 10.20, "low": 9.90, "close": 10.50},
        {"open": 10.30, "high": 10.60, "low": 10.25, "close": 10.55},
        {"open": 10.55, "high": 10.65, "low": 10.50, "close": 10.60},
        {"open": 10.60, "high": 10.70, "low": 10.55, "close": 10.65},
        {"open": 10.65, "high": 10.75, "low": 10.60, "close": 10.70},
    ])
    ind = _const_ind(bars, ctx_rvol_rank5=15, ctx_atr14_pct=2.0)   # rank 15 > top_n 10
    params = {"top_n": 10, "stop_atr": 0.10, "trail_r": 0}
    assert orb_inplay_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), params) == []


# ---- pead_smallcap -----------------------------------------------------------------------------

def test_pead_smallcap_long_entry_and_levels():
    from strategies.families.inplay import pead_smallcap_entries
    rows = [{"open": 20.0, "high": 20.2, "low": 19.8, "close": 20.0} for _ in range(6)]
    rows.append({"open": 20.9, "high": 21.1, "low": 20.8, "close": 21.0})   # bar 6 (10:00): closed above open
    rows.append({"open": 21.05, "high": 21.2, "low": 21.0, "close": 21.1})  # bar 7: fill bar (i+1)
    rows.append({"open": 21.10, "high": 21.2, "low": 21.0, "close": 21.1})
    bars = _bars("2025-01-02", rows)
    ind = _const_ind(bars, ctx_earnings_prev=True, ctx_adv_bucket="small", ctx_gap_pct=5.0,
                      ctx_eps_surprise_pct=12.0, ctx_atr14_pct=3.0, session_open=20.0)
    params = {"s": 10, "hold_days": 10, "min_gap": 2}
    es = pead_smallcap_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), params)
    assert len(es) == 1
    e = es[0]
    px = 21.05
    atr_dollars = 3.0 / 100 * px
    stop = px - 1.5 * atr_dollars
    risk = px - stop
    target = px + 3 * risk
    assert e.side == "long" and e.tag == "pead_long" and e.entry_ts == bars.index[7]
    assert e.stop_px == pytest.approx(stop) and e.target_px == pytest.approx(target)
    assert e.time_stop == 10 and e.hold_kind == "multiday"


def test_pead_smallcap_requires_earnings_prev():
    from strategies.families.inplay import pead_smallcap_entries
    rows = [{"open": 20.0, "high": 20.2, "low": 19.8, "close": 20.0} for _ in range(6)]
    rows.append({"open": 20.9, "high": 21.1, "low": 20.8, "close": 21.0})
    rows.append({"open": 21.05, "high": 21.2, "low": 21.0, "close": 21.1})
    rows.append({"open": 21.10, "high": 21.2, "low": 21.0, "close": 21.1})
    bars = _bars("2025-01-02", rows)
    ind = _const_ind(bars, ctx_earnings_prev=False, ctx_adv_bucket="small", ctx_gap_pct=5.0,
                      ctx_eps_surprise_pct=12.0, ctx_atr14_pct=3.0, session_open=20.0)
    params = {"s": 10, "hold_days": 10, "min_gap": 2}
    assert pead_smallcap_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), params) == []


# ---- rs_leader -----------------------------------------------------------------------------------

DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-02", periods=5)]


def _rs_bars_and_ind(**overrides):
    bars = pd.concat([_bars(d, [{"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}] * 5) for d in DAYS])
    defaults = dict(ctx_rs_rank_60=0.95, ctx_pct_from_52w_high=-5.0, d_low=99.0, d_close=100.0,
                     d_ema20=99.5, d_ema10=99.8, d_ret1=0.5, d_atr14=2.0)
    defaults.update(overrides)
    ind = _const_ind(bars, **defaults)
    return bars, ind


def test_rs_leader_entry_and_levels():
    from strategies.families.inplay import rs_leader_entries
    bars, ind = _rs_bars_and_ind()
    params = {"ema": 20, "rs_floor": 0.90, "atr_stop": 2.0}
    es = rs_leader_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), params)
    assert es
    e = es[0]
    dist = 2.0 * 2.0
    assert e.entry_ts == pd.Timestamp(DAYS[1] + " 09:30") and e.side == "long" and e.tag == "rs_leader"
    assert e.stop_px == pytest.approx(100.0 - dist) and e.target_px == pytest.approx(100.0 + 3 * dist)
    assert e.time_stop == 20 and e.hold_kind == "multiday"


def test_rs_leader_requires_rs_floor():
    from strategies.families.inplay import rs_leader_entries
    bars, ind = _rs_bars_and_ind(ctx_rs_rank_60=0.5)   # below rs_floor
    params = {"ema": 20, "rs_floor": 0.90, "atr_stop": 2.0}
    assert rs_leader_entries(bars, ind, pd.DataFrame(), pd.DataFrame(), params) == []


# ---- registry ---------------------------------------------------------------------------------

def test_family_descriptors():
    from strategies.families.common import grid_combos
    from strategies.families.inplay import FAMILIES_INPLAY
    names = {f.name for f in FAMILIES_INPLAY}
    assert names == {"orb_inplay", "pead_smallcap", "rs_leader"}
    for f in FAMILIES_INPLAY:
        assert 1 <= len(grid_combos(f.grid)) <= 27
    by_name = {f.name: f for f in FAMILIES_INPLAY}
    assert by_name["orb_inplay"].hold_kind == "intraday" and set(by_name["orb_inplay"].sides) == {"long", "short"}
    assert by_name["pead_smallcap"].hold_kind == "multiday" and set(by_name["pead_smallcap"].sides) == {"long", "short"}
    assert by_name["rs_leader"].hold_kind == "multiday" and set(by_name["rs_leader"].sides) == {"long"}
