import json
import numpy as np
import pandas as pd
import pytest

from strategies.families.common import Family, make_entry, one_per_session


def _bars(start="2023-01-03", sessions=480, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=sessions)
    frames, px = [], 100.0
    for d in days:
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        close = px + rng.normal(0, 0.05, 78).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1000}, index=idx))
        px = float(close[-1])
    return pd.concat(frames)


def _toy(bars5, ind, daily, news, params):
    out = []
    for i in np.flatnonzero(ind["bar_of_day"].to_numpy() == 10):
        if i + 1 < len(bars5):
            px = float(bars5["open"].iloc[i + 1])
            e = make_entry(bars5, int(i), "long", px - params["w"], px + 2 * params["w"], 30, "intraday", "toy")
            if e:
                out.append(e)
    return one_per_session(out)


TOY = Family("toy", _toy, {"w": [0.5, 1.0]}, "intraday", ("long",))


@pytest.fixture
def root(tmp_path, monkeypatch):
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily
    from strategies.families import registry
    b = _bars()
    for sym in ("AAA", "SPY"):
        _write(bars5_path(tmp_path, sym), b, "ts")
        _write(daily_path(tmp_path, sym), to_daily(b), "date")
    monkeypatch.setitem(registry.FAMILIES, "toy", TOY)
    return tmp_path


def test_shuffle_bars_preserves_session_return_multiset():
    from research.strategy_hunt.run import shuffle_bars
    b = _bars(sessions=3)
    s = shuffle_bars(b, seed=1)
    assert s.index.equals(b.index) and not np.allclose(s["close"], b["close"])
    for day in b.index.normalize().unique():
        o, n = b.loc[str(day.date())], s.loc[str(day.date())]
        ro = np.sort(np.diff(np.log(np.concatenate([[o["open"].iloc[0]], o["close"].to_numpy()]))))
        rn = np.sort(np.diff(np.log(np.concatenate([[n["open"].iloc[0]], n["close"].to_numpy()]))))
        assert np.allclose(ro, rn)
        assert (n["high"] >= n["close"]).all() and (n["low"] <= n["close"]).all()
    assert shuffle_bars(b, seed=1).equals(s)


def test_process_symbol_writes_outputs_then_skips(root):
    from research.strategy_hunt.run import process_symbol, trades_path, checkpoint_path, choices_path
    rec = process_symbol("AAA", root, ["toy"])
    assert rec["status"] == "done" and rec["n_trades"] > 0
    t = pd.read_parquet(trades_path(root, "AAA"))
    assert set(t["family"]) == {"toy"} and {"r_multiple", "window_id", "params_json", "regime"} <= set(t.columns)
    assert json.loads(choices_path(root, "AAA").read_text())["toy"][0]["window_id"] == 0
    assert checkpoint_path(root, "AAA").exists()
    assert process_symbol("AAA", root, ["toy"])["status"] == "skipped"
    assert process_symbol("AAA", root, ["toy"], force=True)["status"] == "done"
    assert process_symbol("AAA", root, ["toy"], cost_mult=2.0)["status"] == "done"     # different config re-runs
    assert process_symbol("ZZZ", root, ["toy"])["status"] == "empty"


def test_run_symbols_serial_and_score_root(root):
    from research.strategy_hunt.run import run_symbols, score_root
    recs = run_symbols(["AAA"], root, ["toy"], workers=1)
    assert [r["symbol"] for r in recs] == ["AAA"] and recs[0]["status"] == "done"
    rows = score_root(root, ["toy"], "2025-09-25T00:00:00", coldstart=False)
    assert len(rows) == 1 and rows[0]["strategy"] == "toy" and rows[0]["side"] == "long" and rows[0]["source"] == "walkforward"
    assert rows[0]["n"] > 0 and rows[0]["verdict"] in ("PASS", "WATCH", "FAIL")


def test_cli_run_and_score_end_to_end(root, monkeypatch, capsys):
    from research.strategy_hunt import cli
    from research.strategy_hunt.fit.writer import load_snapshot
    assert cli.main(["run", "--root", str(root), "--symbols", "AAA", "--families", "toy", "--workers", "1"]) == 0
    assert cli.main(["score", "--root", str(root), "--families", "toy", "--no-coldstart"]) == 0
    snap = load_snapshot(root)
    assert "AAA" in snap["rows"] and "toy:long" in snap["rows"]["AAA"]
    out = capsys.readouterr().out
    assert "AAA" in out


def test_process_symbol_uses_split_adjusted_bars(root, monkeypatch):
    from research.strategy_hunt import run as R
    from research.strategy_hunt.data import splits as S
    seen = {}
    real = S.load_adjusted
    def spy_load(r, s):
        seen["called"] = True
        return real(r, s)
    monkeypatch.setattr(R, "load_adjusted", spy_load)
    assert R.process_symbol("AAA", root, ["toy"], force=True)["status"] == "done" and seen["called"]


def test_checkpoint_invalidated_by_calibration_change(root):
    from research.strategy_hunt.run import process_symbol
    from research.strategy_hunt.sim.calibrate import write_calibration
    assert process_symbol("AAA", root, ["toy"])["status"] in ("done", "skipped")
    assert process_symbol("AAA", root, ["toy"])["status"] == "skipped"
    write_calibration(root, {"n": 40, "median_frac": 0.0003, "p75_frac": 0.001, "by_bucket": {}, "usable": True})
    assert process_symbol("AAA", root, ["toy"])["status"] == "done"
