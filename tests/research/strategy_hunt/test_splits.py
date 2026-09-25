import numpy as np
import pandas as pd
import pytest


def _daily(n=30, split_at=20, ratio=4.0):
    idx = pd.bdate_range("2025-01-02", periods=n, name="date")
    close = np.full(n, 400.0)
    close[split_at:] = 100.0
    open_ = close.copy()
    vol = np.full(n, 1_000_000.0)
    vol[split_at:] *= ratio
    return pd.DataFrame({"open": open_, "high": close + 1, "low": open_ - 1, "close": close, "volume": vol}, index=idx)


def test_detect_forward_and_reverse_split():
    from research.strategy_hunt.data.splits import detect_splits
    d = _daily()
    assert detect_splits(d) == [(d.index[20], 0.25)]
    rev = _daily(ratio=0.1)
    rev["close"] = np.where(np.arange(30) >= 20, 4000.0, 400.0); rev["open"] = rev["close"]
    assert detect_splits(rev) == [(rev.index[20], 10.0)]
    flat = _daily(); flat["close"] = 400.0; flat["open"] = 400.0
    assert detect_splits(flat) == []


def test_detect_ignores_crash_gap():
    from research.strategy_hunt.data.splits import detect_splits
    d = _daily(ratio=1.0)
    d.loc[d.index[20]:, ["open", "close"]] = 200.0          # -50%: matches ratio 2 but keep checking intraday move
    d.loc[d.index[20], "close"] = 140.0                      # crashed further intraday (-30%) -> not a split
    assert detect_splits(d) == []


def test_adjust_for_splits_prices_and_volume():
    from research.strategy_hunt.data.splits import adjust_for_splits
    d = _daily()
    adj = adjust_for_splits(d, [(d.index[20], 0.25)])
    assert adj["close"].iloc[0] == 100.0 and adj["volume"].iloc[0] == 4_000_000.0
    assert adj["close"].iloc[25] == 100.0 and adj["volume"].iloc[25] == 4_000_000.0
    assert d["close"].iloc[0] == 400.0                       # input untouched


def test_cache_union_and_load_adjusted(tmp_path):
    from research.strategy_hunt.data.splits import update_splits, load_splits, all_splits, load_adjusted
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path
    d = _daily()
    b5 = pd.DataFrame({"open": 400.0, "high": 401.0, "low": 399.0, "close": 400.0, "volume": 100},
                      index=pd.date_range("2025-01-02 09:30", periods=78 * 30, freq="5min", name="ts"))
    b5 = b5[b5.index.time <= pd.Timestamp("15:55").time()]
    _write(bars5_path(tmp_path, "X"), b5, "ts"); _write(daily_path(tmp_path, "X"), d, "date")
    n = update_splits(tmp_path, "X", lambda s: pd.Series([4.0], index=pd.DatetimeIndex([d.index[21]])))
    assert n == 1 and load_splits(tmp_path, "X") == [(d.index[21], 0.25)]
    assert all_splits(tmp_path, "X", d) == [(d.index[21], 0.25)]       # detected (index 20) merged into cached (21)
    b_adj, d_adj = load_adjusted(tmp_path, "X")
    assert d_adj["close"].iloc[0] == 100.0 and b_adj["close"].iloc[0] == 100.0 and b_adj["volume"].iloc[0] == 400
