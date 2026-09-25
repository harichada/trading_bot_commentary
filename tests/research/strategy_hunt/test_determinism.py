"""Two identical runs of process_symbol over the same inputs must produce byte-identical trade output.

If Parquet metadata ever embeds a non-deterministic value (e.g. a write timestamp) the SHA-256
comparison below would start failing even though the trade rows themselves are unchanged; in that
case switch the assertion to `pd.read_parquet(path1).equals(pd.read_parquet(path2))` instead of
hashing the raw bytes. As of this writing the raw-byte comparison passes.
"""
import hashlib

import numpy as np
import pandas as pd

from strategies.families.common import Family, make_entry, one_per_session


def _bars(sessions=420, seed=3):
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    for d in pd.bdate_range("2023-01-03", periods=sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        close = px + rng.normal(0, 0.08, 78).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 2000}, index=idx))
        px = float(close[-1])
    return pd.concat(frames)


def _toy(bars5, ind, daily, news, params):
    out = []
    for i in np.flatnonzero(ind["bar_of_day"].to_numpy() == 20):
        if i + 1 < len(bars5):
            px = float(bars5["open"].iloc[i + 1])
            e = make_entry(bars5, int(i), "long", px - params["w"], px + params["w"], 20, "intraday", "toy")
            if e:
                out.append(e)
    return one_per_session(out)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_runs_produce_byte_identical_trades(tmp_path, monkeypatch):
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily
    from research.strategy_hunt.run import process_symbol, trades_path
    from strategies.families import registry

    monkeypatch.setitem(registry.FAMILIES, "toy", Family("toy", _toy, {"w": [0.5, 1.0, 1.5]}, "intraday", ("long",)))
    b = _bars()
    _write(bars5_path(tmp_path, "DET"), b, "ts")
    _write(daily_path(tmp_path, "DET"), to_daily(b), "date")
    assert process_symbol("DET", tmp_path, ["toy"], force=True)["status"] == "done"
    h1 = _sha(trades_path(tmp_path, "DET"))
    assert process_symbol("DET", tmp_path, ["toy"], force=True)["status"] == "done"
    assert _sha(trades_path(tmp_path, "DET")) == h1
