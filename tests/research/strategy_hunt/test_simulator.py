import numpy as np
import pandas as pd
import pytest

from strategies.families.models import Entry
from research.strategy_hunt.sim.costs import ZERO_COSTS, CostModel


def _day(day, opens, highs, lows, closes, start="09:30"):
    n = len(opens)
    idx = pd.date_range(f"{day} {start}", periods=n, freq="5min", name="ts")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": [10_000] * n}, index=idx)


def _flat_session(day, px=100.0):
    n = 78
    return _day(day, [px] * n, [px + 0.5] * n, [px - 0.5] * n, [px] * n)


def test_target_hit_before_stop_on_later_bar():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _day("2025-01-02", [100, 100, 101], [100.5, 100.5, 104.5], [99.5, 99.5, 100.5], [100, 100, 104])
    e = Entry("X", "long", bars.index[0], 100.0, 98.0, 104.0, 20, "intraday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "target" and t.exit_fill == 104.0 and t.exit_ts == bars.index[2]
    assert t.r_multiple == pytest.approx(2.0) and t.hold_bars == 2


def test_stop_checked_before_target_same_bar():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _day("2025-01-02", [100, 100], [100.5, 105], [99.5, 97], [100, 100])
    e = Entry("X", "long", bars.index[0], 100.0, 98.0, 104.0, 20, "intraday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "stop" and t.exit_fill == 98.0 and t.r_multiple == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _day("2025-01-02", [100, 95], [100.5, 96], [99.5, 94], [100, 95])
    e = Entry("X", "long", bars.index[0], 100.0, 98.0, 104.0, 20, "intraday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "stop" and t.exit_fill == 95.0 and t.r_multiple == pytest.approx(-2.5)


def test_short_stop_and_target_mirror():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _day("2025-01-02", [100, 100, 100], [100.5, 100.5, 100.5], [99.5, 99.5, 95.5], [100, 100, 96])
    e = Entry("X", "short", bars.index[0], 100.0, 102.0, 96.0, 20, "intraday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "target" and t.exit_fill == 96.0 and t.r_multiple == pytest.approx(2.0)
    bars2 = _day("2025-01-02", [100, 103], [100.5, 104], [99.5, 102.5], [100, 103])
    t2 = simulate([e], bars2, ZERO_COSTS, set(), "fam")[0]
    assert t2.exit_reason == "stop" and t2.exit_fill == 103.0     # gapped above stop -> open


def test_intraday_time_stop_then_eod():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _flat_session("2025-01-02")
    e = Entry("X", "long", bars.index[10], 100.0, 90.0, 120.0, 3, "intraday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "time_stop" and t.exit_ts == bars.index[13] and t.hold_bars == 3
    e2 = Entry("X", "long", bars.index[70], 100.0, 90.0, 120.0, 50, "intraday")
    t2 = simulate([e2], bars, ZERO_COSTS, set(), "fam")[0]
    assert t2.exit_reason == "eod" and t2.exit_ts == bars.index[77]


def test_multiday_time_stop_counts_sessions_and_overnight_gap_hits_stop():
    from research.strategy_hunt.sim.simulator import simulate
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03"), _flat_session("2025-01-06")])
    e = Entry("X", "long", bars.index[5], 100.0, 90.0, 120.0, 2, "multiday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "time_stop" and t.exit_ts == pd.Timestamp("2025-01-03 15:55")
    gap = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03", px=85.0)])
    t2 = simulate([Entry("X", "long", gap.index[5], 100.0, 90.0, 120.0, 5, "multiday")], gap, ZERO_COSTS, set(), "fam")[0]
    assert t2.exit_reason == "stop" and t2.exit_fill == 85.0 and t2.exit_ts == pd.Timestamp("2025-01-03 09:30")


def test_multiday_earnings_eve_flatten():
    from research.strategy_hunt.sim.simulator import simulate
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03"), _flat_session("2025-01-06")])
    e = Entry("X", "long", bars.index[5], 100.0, 90.0, 120.0, 10, "multiday")
    t = simulate([e], bars, ZERO_COSTS, {pd.Timestamp("2025-01-06")}, "fam")[0]
    assert t.exit_reason == "earnings_eve" and t.exit_ts == pd.Timestamp("2025-01-03 15:55")


def test_data_end_and_missing_entry_bar():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _day("2025-01-02", [100, 100], [100.5, 100.5], [99.5, 99.5], [100, 101])
    e = Entry("X", "long", bars.index[0], 100.0, 90.0, 120.0, 50, "multiday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "data_end" and t.exit_fill == 101.0
    ghost = Entry("X", "long", pd.Timestamp("2025-01-02 12:00"), 100.0, 90.0, 120.0, 5, "intraday")
    assert simulate([ghost], bars, ZERO_COSTS, set(), "fam") == []


def test_costs_reduce_r_and_are_reported():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _day("2025-01-02", [100, 100, 101], [100.5, 100.5, 104.5], [99.5, 99.5, 100.5], [100, 100, 104])
    cm = CostModel(half_spread_by_month=pd.Series([0.001], index=pd.PeriodIndex(["2025-01"], freq="M")), notional=6000.0)
    e = Entry("X", "long", bars.index[0], 100.0, 98.0, 104.0, 20, "intraday")
    t = simulate([e], bars, cm, set(), "fam", window_id=7, params_json='{"a":1}')[0]
    assert t.r_multiple < 2.0 and t.cost_pct > 0 and t.window_id == 7 and t.params_json == '{"a":1}'
    assert t.entry_fill > 100.0 and t.exit_fill < 104.0


def test_regime_lookup_tags_trade():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _day("2025-01-02", [100, 100, 101], [100.5, 100.5, 104.5], [99.5, 99.5, 100.5], [100, 100, 104])
    e = Entry("X", "long", bars.index[0], 100.0, 98.0, 104.0, 20, "intraday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam", regime_lookup=lambda ts: "risk_on")[0]
    assert t.regime == "risk_on"


def test_hold_sessions_and_short_borrow():
    from research.strategy_hunt.sim.simulator import simulate
    from research.strategy_hunt.sim.costs import CostModel
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03"), _flat_session("2025-01-06")])
    cm = CostModel(half_spread_by_month=pd.Series([0.0], index=pd.PeriodIndex(["2025-01"], freq="M")), notional=0.0,
                   sell_fee_pct=0.0, slippage_floor=0.0, borrow_rate_annual=0.252)
    e = Entry("X", "short", bars.index[5], 100.0, 110.0, 80.0, 3, "multiday")
    t = simulate([e], bars, cm, set(), "fam")[0]
    assert t.exit_reason == "time_stop" and t.hold_sessions == 3
    assert t.exit_fill == pytest.approx(100.0 * (1 + 0.252 * 3 / 252))       # 0.3% borrow over 3 sessions
    assert t.r_multiple == pytest.approx(-(t.exit_fill - 100.0) / 10.0)


def test_intraday_time_stop_ties_eod_labels_time_stop():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _flat_session("2025-01-02")
    e = Entry("X", "long", bars.index[10], 100.0, 90.0, 120.0, 67, "intraday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "time_stop" and t.exit_ts == bars.index[77]


def test_multiday_earnings_eve_beats_time_stop_on_tie():
    from research.strategy_hunt.sim.simulator import simulate
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03"), _flat_session("2025-01-06")])
    e = Entry("X", "long", bars.index[5], 100.0, 90.0, 120.0, 2, "multiday")
    t = simulate([e], bars, ZERO_COSTS, {pd.Timestamp("2025-01-06")}, "fam")[0]
    assert t.exit_reason == "earnings_eve" and t.exit_ts == pd.Timestamp("2025-01-03 15:55")


def test_session_table():
    from research.strategy_hunt.sim.simulator import build_session_table
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03")])
    tb = build_session_table(bars)
    assert list(tb.first_pos) == [0, 78] and list(tb.last_pos) == [77, 155]
    assert (tb.session_id[:78] == 0).all() and (tb.session_id[78:] == 1).all()
    assert list(tb.next_is_earnings({pd.Timestamp("2025-01-03")})) == [True, False]


def test_trade_records_the_entrys_time_stop():
    """Batch-2 Task F item 4: exactly `Entry.time_stop`, whatever the trade's realized hold was --
    the selection replay needs the horizon the family *chose*, not the one it happened to use."""
    from research.strategy_hunt.sim.simulator import simulate
    bars = _day("2025-01-02", [100, 100, 101], [100.5, 100.5, 104.5], [99.5, 99.5, 100.5], [100, 100, 104])
    e = Entry("X", "long", bars.index[0], 100.0, 98.0, 104.0, 20, "intraday")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.time_stop == 20 and isinstance(t.time_stop, int)
    assert t.hold_bars < t.time_stop      # it exited on target well before the horizon
