import json
import numpy as np
import pandas as pd
import pytest

from strategies.families.common import Family, make_entry, one_per_session
from research.strategy_hunt.models import Window
from research.strategy_hunt.sim.costs import ZERO_COSTS


def _bars(start="2024-01-02", sessions=420, n=78):
    rng = np.random.default_rng(0)
    frames, px = [], 100.0
    for d in pd.bdate_range(start, periods=sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=n, freq="5min", name="ts")
        close = px + rng.normal(0, 0.05, n).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1000}, index=idx))
        px = float(close[-1])
    return pd.concat(frames)


def _always_long(bars5, ind, daily, news, params):
    out = []
    for i in np.flatnonzero(ind["bar_of_day"].to_numpy() == 10):
        if i + 1 < len(bars5):
            px = float(bars5["open"].iloc[i + 1]); w = params["width"]
            e = make_entry(bars5, int(i), "long", px - w, px + 2 * w, 30, "intraday", "toy")
            if e: out.append(e)
    return one_per_session(out)


TOY = Family("toy", _always_long, {"width": [0.5, 1.0, 2.0]}, "intraday", ("long",))


def test_precompute_one_key_per_combo_and_select_uses_in_sample_only():
    from research.strategy_hunt.walkforward.runner import precompute, select_params
    from strategies.families.indicators import indicator_frame
    from research.strategy_hunt.data.bars import to_daily
    b = _bars(sessions=300); daily = to_daily(b); ind = indicator_frame(b, daily)
    pre = precompute(TOY, "X", b, ind, daily, pd.DataFrame(), ZERO_COSTS, set())
    assert set(pre) == {json.dumps({"width": w}) for w in (0.5, 1.0, 2.0)}
    assert all(t.window_id == -1 for ts in pre.values() for t in ts)
    w = Window(0, pd.Timestamp("2024-02-01"), pd.Timestamp("2024-11-24"), pd.Timestamp("2024-12-01"), pd.Timestamp("2025-03-01"))
    chosen = select_params(pre, w, grid=TOY.grid)
    assert json.loads(chosen)["width"] in (0.5, 1.0, 2.0)
    # in-sample restricted: compute expected winner by hand
    def mean_is(k):
        ts = [t.r_multiple for t in pre[k] if w.is_start <= t.entry_ts < w.is_end]
        return np.mean(ts) if len(ts) >= 30 else -np.inf
    assert chosen == max(sorted(pre), key=mean_is)
    tiny = Window(0, pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-10"), pd.Timestamp("2024-12-01"), pd.Timestamp("2025-03-01"))
    assert json.loads(select_params(pre, tiny, grid=TOY.grid)) == {"width": 1.0}


def test_run_symbol_family_keeps_only_oos_entries_and_tags_windows():
    from research.strategy_hunt.walkforward.runner import run_symbol_family
    from research.strategy_hunt.walkforward.windows import build_windows
    from research.strategy_hunt.data.bars import to_daily
    b = _bars(start="2023-01-03", sessions=520); daily = to_daily(b)
    windows = build_windows(b.index[0], b.index[-1])
    trades, choices = run_symbol_family(TOY, "X", b, daily, pd.DataFrame(), ZERO_COSTS, set(), windows)
    assert trades and len(choices) == len(windows)
    for t in trades:
        w = windows[t.window_id]
        assert w.oos_start <= t.entry_ts < w.oos_end and t.symbol == "X" and t.family == "toy"
        assert json.loads(t.params_json)["width"] == choices[t.window_id]["params"]["width"]
    assert {t.window_id for t in trades} == set(range(len(windows)))
