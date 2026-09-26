"""hold_kind == "overnight": the position is closed at the next session's first bar OPEN.

Batch-1 Task 1. The exit price for an overnight hold is the open of the next session's first
bar (reason "next_open"), not that bar's close: an overnight family that was marked at the
09:30 bar's close would be credited with five extra minutes of intraday drift it never held.
Because the exit is at that open, the stop/target scan covers only the bars strictly between
the fill bar and the exit bar -- for the canonical 15:50 signal / 15:55 fill there are none, so
the stop is protective-only and an overnight gap through it is realized at the open (which is
the same fill a gap-through-stop would have produced).
"""
import pandas as pd
import pytest

from research.strategy_hunt.sim.costs import ZERO_COSTS
from strategies.families.models import Entry


def _day(day, opens, highs, lows, closes, start="09:30"):
    n = len(opens)
    idx = pd.date_range(f"{day} {start}", periods=n, freq="5min", name="ts")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": [10_000] * n}, index=idx)


def _flat_session(day, px=100.0, open_px=None):
    n = 78
    opens = [px] * n
    if open_px is not None:
        opens = [open_px] + [px] * (n - 1)
    return _day(day, opens, [max(px, opens[0]) + 0.5] * n, [min(px, opens[0]) - 0.5] * n, [px] * n)


def test_overnight_exits_at_next_session_first_bar_open():
    from research.strategy_hunt.sim.simulator import simulate
    # 01-03 opens at 102 and closes every bar at 105: exiting at the open must fill at 102.
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03", px=105.0, open_px=102.0)])
    e = Entry("X", "long", bars.index[77], 100.0, 97.0, 160.0, 1, "overnight")   # fill at 15:55's open
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "next_open"
    assert t.exit_ts == pd.Timestamp("2025-01-03 09:30") and t.exit_fill == pytest.approx(102.0)
    assert t.r_multiple == pytest.approx(2.0 / 3.0) and t.hold_sessions == 2


def test_overnight_short_mirrors_and_gap_through_stop_fills_at_that_open():
    from research.strategy_hunt.sim.simulator import simulate
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03", px=90.0, open_px=90.0)])
    short = Entry("X", "short", bars.index[77], 100.0, 103.0, 40.0, 1, "overnight")
    t = simulate([short], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "next_open" and t.exit_fill == pytest.approx(90.0)
    assert t.r_multiple == pytest.approx(10.0 / 3.0)
    # A long gapped far below its protective stop is closed at that same open, not at the stop.
    long_ = Entry("X", "long", bars.index[77], 100.0, 97.0, 160.0, 1, "overnight")
    t2 = simulate([long_], bars, ZERO_COSTS, set(), "fam")[0]
    assert t2.exit_reason == "next_open" and t2.exit_fill == pytest.approx(90.0)
    assert t2.r_multiple == pytest.approx(-10.0 / 3.0)


def test_overnight_on_last_session_is_data_end():
    from research.strategy_hunt.sim.simulator import simulate
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03")])
    e = Entry("X", "long", bars.index[78 + 70], 100.0, 97.0, 160.0, 1, "overnight")   # no next session
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "data_end" and t.exit_ts == bars.index[-1]


def test_overnight_earnings_eve_still_flattens_at_the_session_close():
    from research.strategy_hunt.sim.simulator import simulate
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03", px=105.0, open_px=102.0)])
    e = Entry("X", "long", bars.index[5], 100.0, 97.0, 160.0, 1, "overnight")
    t = simulate([e], bars, ZERO_COSTS, {pd.Timestamp("2025-01-03")}, "fam")[0]
    assert t.exit_reason == "earnings_eve" and t.exit_ts == pd.Timestamp("2025-01-02 15:55")


def test_overnight_entry_on_the_last_bar_of_an_earnings_eve_is_skipped_entirely():
    """v-batch1-task1-r1 (Important 2): the canonical overnight fill IS the session's last bar, so
    when the next session is an earnings date the earnings-eve rule would flatten the position on
    the very bar it was opened on -- bars_held 0, R ~ 0, both legs' costs paid. That is a phantom
    trade, not a trade the family could have taken (live, the 15:50 earnings-eve flatten check and
    this entry contradict each other), so the entry is skipped and no trade is emitted. An entry
    earlier in the same session is a real hold and still exits `earnings_eve`."""
    from research.strategy_hunt.sim.simulator import simulate
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03", px=105.0, open_px=102.0)])
    earnings = {pd.Timestamp("2025-01-03")}
    last_bar = Entry("X", "long", bars.index[77], 100.0, 97.0, 160.0, 1, "overnight")     # 15:55 fill
    assert simulate([last_bar], bars, ZERO_COSTS, earnings, "fam") == []

    mid = Entry("X", "long", bars.index[5], 100.0, 90.0, 120.0, 3, "multiday")
    t = simulate([mid], bars, ZERO_COSTS, earnings, "fam")[0]
    assert t.exit_reason == "earnings_eve" and t.exit_ts == pd.Timestamp("2025-01-02 15:55")
    assert t.hold_bars > 0

    # Nothing is skipped when the next session is not an earnings date.
    assert len(simulate([last_bar], bars, ZERO_COSTS, set(), "fam")) == 1


def test_overnight_stop_on_an_intervening_bar_wins_over_next_open():
    from research.strategy_hunt.sim.simulator import simulate
    day1 = _day("2025-01-02", [100.0] * 4, [100.5] * 4, [99.5, 99.5, 96.0, 99.5], [100.0] * 4)
    bars = pd.concat([day1, _flat_session("2025-01-03", px=105.0, open_px=102.0)])
    e = Entry("X", "long", bars.index[0], 100.0, 97.0, 160.0, 1, "overnight")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "stop" and t.exit_fill == pytest.approx(97.0) and t.exit_ts == bars.index[2]


def test_multiday_and_intraday_exits_unchanged_by_the_overnight_branch():
    from research.strategy_hunt.sim.simulator import simulate
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03"), _flat_session("2025-01-06")])
    md = simulate([Entry("X", "long", bars.index[5], 100.0, 90.0, 120.0, 2, "multiday")], bars, ZERO_COSTS, set(), "fam")[0]
    assert md.exit_reason == "time_stop" and md.exit_ts == pd.Timestamp("2025-01-03 15:55")
    it = simulate([Entry("X", "long", bars.index[70], 100.0, 90.0, 120.0, 50, "intraday")], bars, ZERO_COSTS, set(), "fam")[0]
    assert it.exit_reason == "eod" and it.exit_ts == pd.Timestamp("2025-01-02 15:55")


def test_overnight_trades_flow_through_the_fit_row_writer():
    """`hold_kind="overnight"` must be rejected by nothing downstream: research/strategy_hunt
    has no hold_kind whitelist, and build_row (the fit-row builder every lane uses) must take
    an overnight trade as-is."""
    from research.strategy_hunt.fit.writer import build_row
    from research.strategy_hunt.sim.simulator import simulate
    bars = pd.concat([_flat_session("2025-01-02"), _flat_session("2025-01-03", px=105.0, open_px=102.0)])
    trades = simulate([Entry("X", "long", bars.index[77], 100.0, 97.0, 160.0, 1, "overnight")],
                      bars, ZERO_COSTS, set(), "overnight_hold")
    row = build_row("X", "overnight_hold", "long", trades, 1, {"move_pct": 1.0}, "2025-01-04", "2025-01-03")
    assert row["n"] == 1 and row["strategy"] == "overnight_hold" and row["verdict"]


def test_overnight_hold_family_entries_exit_at_the_next_open_end_to_end():
    """The family proposes 15:55 fills; every one of them must leave the simulator with a
    next_open exit (the last session's has no next open, so it ends at data_end)."""
    import numpy as np
    from research.strategy_hunt.sim.simulator import simulate
    from strategies.families.swing2 import overnight_hold_entries
    days = ["2025-01-02", "2025-01-03", "2025-01-06"]
    bars = pd.concat([_flat_session(d, px=100.0 + i, open_px=99.0 + i) for i, d in enumerate(days)])
    ind = pd.DataFrame(index=bars.index)
    ind["bar_of_day"] = bars.groupby(bars.index.normalize()).cumcount()
    ind["d_ret1"], ind["d_atr14"], ind["ctx_rs_20"] = -2.0, 2.0, 10.0
    entries = overnight_hold_entries(bars, ind, pd.DataFrame(), pd.DataFrame(),
                                     {"move_pct": 1.0, "rs_min": 0, "stop_atr": 1.5})
    assert [e.entry_ts for e in entries] == [pd.Timestamp(f"{d} 15:55") for d in days]
    trades = simulate(entries, bars, ZERO_COSTS, set(), "overnight_hold")
    assert [t.exit_reason for t in trades] == ["next_open", "next_open", "data_end"]
    assert trades[0].exit_fill == pytest.approx(100.0)      # 01-03's session open (99 + 1)
    assert all(t.hold_sessions == 2 for t in trades[:2]) and not np.isnan(trades[0].r_multiple)
