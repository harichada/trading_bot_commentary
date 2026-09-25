import numpy as np
import pandas as pd
import pytest


def _one_min_day(day: str) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:25", f"{day} 16:04", freq="1min")
    n = len(idx)
    return pd.DataFrame({"open": np.arange(n, dtype=float) + 100, "high": np.arange(n, dtype=float) + 101,
                         "low": np.arange(n, dtype=float) + 99, "close": np.arange(n, dtype=float) + 100.5,
                         "volume": np.full(n, 10)}, index=pd.DatetimeIndex(idx, name="ts"))


def test_to_rth_5min_drops_outside_hours_and_aggregates():
    from research.strategy_hunt.data.bars import to_rth_5min
    df5 = to_rth_5min(_one_min_day("2025-01-02"))
    assert df5.index[0] == pd.Timestamp("2025-01-02 09:30") and df5.index[-1] == pd.Timestamp("2025-01-02 15:55")
    assert len(df5) == 78
    first = df5.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (105.0, 110.0, 104.0, 109.5, 50)


def test_to_daily_from_5min():
    from research.strategy_hunt.data.bars import to_rth_5min, to_daily
    df5 = pd.concat([to_rth_5min(_one_min_day("2025-01-02")), to_rth_5min(_one_min_day("2025-01-03"))])
    d = to_daily(df5)
    assert list(d.index) == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")] and d.index.name == "date"
    assert d.iloc[0]["open"] == df5.iloc[0]["open"] and d.iloc[0]["close"] == df5.iloc[77]["close"] and d.iloc[0]["volume"] == 78 * 50


def test_load_missing_returns_empty_with_columns(tmp_path):
    from research.strategy_hunt.data.bars import load_bars5, load_daily
    b = load_bars5(tmp_path, "ZZZZ")
    assert b.empty and list(b.columns) == ["open", "high", "low", "close", "volume"]
    assert load_daily(tmp_path, "ZZZZ").index.name == "date"


def test_export_symbol_appends_incrementally(tmp_path, monkeypatch):
    from research.strategy_hunt.data import bars as B
    day1, day2 = _one_min_day("2025-01-02").reset_index(), _one_min_day("2025-01-03").reset_index()
    calls = []

    def fake_read(engine, symbol, since):
        calls.append(since)
        return day1 if since is None else day2[day2["ts"] > since]

    monkeypatch.setattr(B, "_read_minute_bars", fake_read)
    assert B.export_symbol(object(), "NVDA", tmp_path) == 78
    assert B.last_ts(tmp_path, "NVDA") == pd.Timestamp("2025-01-02 15:55")
    assert B.export_symbol(object(), "NVDA", tmp_path) == 78 and calls[1] == pd.Timestamp("2025-01-02 15:55")
    b = B.load_bars5(tmp_path, "NVDA")
    assert len(b) == 156 and b.index.is_monotonic_increasing and not b.index.duplicated().any()
    assert len(B.load_daily(tmp_path, "NVDA")) == 2


def test_select_universe_applies_days_and_liquidity_floor(monkeypatch):
    from research.strategy_hunt.data import bars as B

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, q, params=None):
            s = str(q)
            class R:
                def __init__(self, v): self.v = v
                def scalar(self): return self.v
                def __iter__(self): return iter(self.v)
            if "RECURSIVE" in s:
                return R([("AAA",), ("BBB",), ("CCC",)])
            if "count(DISTINCT" in s:
                return R({"AAA": 300, "BBB": 300, "CCC": 100}[params["s"]])
            if "avg(" in s:
                return R({"AAA": 5e7, "BBB": 1e6, "CCC": 9e7}[params["s"]])
            raise AssertionError(s)

    class _Engine:
        def connect(self): return _Conn()

    assert B.select_universe(_Engine(), min_days=250, min_adv_dollars=2e7) == ["AAA"]
    assert B.select_universe(_Engine(), min_days=250, min_adv_dollars=0) == ["AAA", "BBB"]
