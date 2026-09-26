import numpy as np
import pandas as pd
import pytest


def _bars(days=("2025-01-02", "2025-01-03"), n=78):
    frames = []
    for d in days:
        idx = pd.date_range(f"{d} 09:30", periods=n, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx))
    return pd.concat(frames)


def test_grid_combos_is_sorted_and_complete():
    from strategies.families.common import grid_combos, median_params
    combos = grid_combos({"a": [3, 1, 2], "b": ["x", "y"]})
    assert len(combos) == 6 and combos[0] == {"a": 1, "b": "x"} and combos[-1] == {"a": 3, "b": "y"}
    assert median_params({"a": [1, 2, 3], "b": [10, 20]}) == {"a": 2, "b": 20}


def test_make_entry_uses_next_bar_open_and_respects_session_boundary():
    from strategies.families.common import make_entry
    b = _bars()
    e = make_entry(b, 10, "long", 99.0, 102.0, 5, "intraday", "t")
    assert e.entry_ts == b.index[11] and e.entry_px == 100.0 and e.tag == "t"
    assert make_entry(b, 77, "long", 99.0, 102.0, 5, "intraday", "t") is None       # next bar is next session
    assert make_entry(b, 77, "long", 99.0, 102.0, 5, "multiday", "t") is not None   # multiday may cross
    assert make_entry(b, 155, "long", 99.0, 102.0, 5, "multiday", "t") is None      # out of range


def test_one_per_session_and_spacing():
    from strategies.families.common import one_per_session, spaced_by_sessions
    from strategies.families.models import Entry
    def mk(ts, side="long"):
        # stop/target must bracket entry on the correct side (Entry.__post_init__ validation).
        return Entry("X", side, pd.Timestamp(ts), 100, 99, 102, 5, "multiday") if side == "long" \
            else Entry("X", side, pd.Timestamp(ts), 100, 101, 98, 5, "multiday")
    es = [mk("2025-01-02 10:00"), mk("2025-01-02 11:00"), mk("2025-01-02 11:00", "short"), mk("2025-01-03 10:00"),
          mk("2025-01-06 10:00"), mk("2025-01-07 10:00")]
    kept = one_per_session(es)
    assert [e.entry_ts for e in kept] == [pd.Timestamp("2025-01-02 10:00"), pd.Timestamp("2025-01-02 11:00"),
                                          pd.Timestamp("2025-01-03 10:00"), pd.Timestamp("2025-01-06 10:00"),
                                          pd.Timestamp("2025-01-07 10:00")]
    spaced = spaced_by_sessions(kept, 3)
    assert [e.entry_ts.date().isoformat() for e in spaced if e.side == "long"] == ["2025-01-02", "2025-01-07"]


def test_bars_to_1500():
    from strategies.families.common import bars_to_1500
    assert bars_to_1500(0) == 66 and bars_to_1500(66) == 1 and bars_to_1500(70) == 1


def test_candidate_positions_and_col():
    import numpy as np, pandas as pd
    from strategies.families.common import candidate_positions, col
    assert list(candidate_positions(np.array([True, False, True, True]))) == [0, 2]
    ind = pd.DataFrame({"a": [1.0, 2.0]})
    assert list(col(ind, "a")) == [1.0, 2.0] and np.isnan(col(ind, "zzz")).all()


def test_make_entry_accepts_overnight_and_rejects_unknown_hold_kinds():
    """Batch-1 Task 1: `overnight` is a first-class hold kind (fills like `multiday`, i.e. the
    entry may be the last bar of a session), and an unknown kind fails loudly at the family
    boundary instead of reaching the simulator."""
    from strategies.families.common import HOLD_KINDS, make_entry
    assert HOLD_KINDS == ("intraday", "multiday", "overnight")
    b = _bars()
    e = make_entry(b, 76, "long", 99.0, 102.0, 1, "overnight", "on")
    assert e is not None and e.hold_kind == "overnight" and e.entry_ts == b.index[77]
    assert make_entry(b, 77, "long", 99.0, 102.0, 1, "overnight", "on") is not None    # may cross sessions
    with pytest.raises(ValueError):
        make_entry(b, 10, "long", 99.0, 102.0, 1, "weekly", "t")
