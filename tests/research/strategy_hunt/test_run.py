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


def test_cli_run_and_score_end_to_end(root, monkeypatch, capsys, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt import report as R
    from research.strategy_hunt import run as RUN
    from research.strategy_hunt.fit.writer import load_snapshot
    # v-strategy-fit-review2-2026-09-25 (Important 8): `run`'s manifest and
    # `score`'s report both land in REPORTS_DIR, which defaults to the real
    # repo path research/reports/strategy_hunt/ — a test must never write
    # there. run.py owns the constant; report.py binds its own module-level
    # copy at import time, so both need redirecting.
    reports = tmp_path / "reports"
    monkeypatch.setattr(RUN, "REPORTS_DIR", reports)
    monkeypatch.setattr(R, "REPORTS_DIR", reports)
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


# ──────────────────────
# v-playbook-desk-b1-2026-09-25 — regime-conditioned and pooled scoring
# ──────────────────────

def _trade(symbol, family, side, day, w, regime, r):
    from research.strategy_hunt.models import SimTrade
    ts = pd.Timestamp(f"{day} 10:00")
    return SimTrade(symbol, family, side, ts, ts + pd.Timedelta(minutes=30), 100.0, 99.0, 102.0, 100.0 + r,
                    100.0, 100.0 + r, r, r, 0.0, "target" if r > 0 else "stop", 6, 1, f"{family}_{side}",
                    regime, w, '{"w": 1.0}')


def _write_trades(root, symbol, trades):
    from research.strategy_hunt.models import TRADE_COLUMNS
    from research.strategy_hunt.run import choices_path, trades_path
    trades_path(root, symbol).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([t.to_dict() for t in trades], columns=list(TRADE_COLUMNS)).to_parquet(
        trades_path(root, symbol), index=False)
    choices_path(root, symbol).parent.mkdir(parents=True, exist_ok=True)
    choices_path(root, symbol).write_text(json.dumps(
        {"toy": [{"window_id": w, "params": {"w": 1.0}} for w in range(3)]}))


def _regime_trades(symbol):
    out = []
    for di, d in enumerate(pd.bdate_range("2025-01-02", periods=30)):
        for j in range(10):
            regime = ("unknown", "risk_on", "mixed")[di % 3]
            out.append(_trade(symbol, "toy", "long", d.strftime("%Y-%m-%d"), di // 10, regime,
                              1.0 if (di * 10 + j) % 3 else -1.0))
    return out


def test_score_regime_root_builds_symbol_regime_and_pooled_rows(tmp_path):
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import REGIMES
    for sym in ("AAA", "BBB"):
        _write_trades(tmp_path, sym, _regime_trades(sym))

    reg_rows, pool_rows = score_regime_root(tmp_path, ["toy"], "2025-09-25T00:00:00")

    assert {r["symbol"] for r in reg_rows} == {"AAA", "BBB"}
    assert {r["regime"] for r in reg_rows} <= set(REGIMES) and "unknown" not in {r["regime"] for r in reg_rows}
    assert all(r["strategy"] == "toy" and r["side"] == "long" for r in reg_rows)
    assert all(r["source"] == "walkforward" and r["computed_at"] == "2025-09-25T00:00:00" for r in reg_rows)
    assert all(json.loads(r["params_json"]) == {"w": 1.0} for r in reg_rows)
    # each regime row's window_end is that regime's own last exit date, not the family's
    assert len({r["window_end"] for r in reg_rows if r["symbol"] == "AAA"}) == 2

    assert pool_rows and all(r["symbol"] == "*" and r["source"] == "pooled" for r in pool_rows)
    by_regime = {r["regime"]: r for r in pool_rows}
    # pooled rows aggregate both symbols for the same (family, side, regime)
    per_symbol = {r["regime"]: r for r in reg_rows if r["symbol"] == "AAA"}
    for regime, prow in by_regime.items():
        assert prow["n"] == 2 * per_symbol[regime]["n"]


def test_score_regime_root_empty_root_and_score_root_unchanged(root):
    from research.strategy_hunt.run import run_symbols, score_regime_root, score_root
    assert score_regime_root(root, ["toy"], "2025-09-25T00:00:00") == ([], [])   # no trades yet
    # score_root keeps returning only the unconditional lane rows (the report and the
    # cold-start model consume that list).
    run_symbols(["AAA"], root, ["toy"], workers=1)
    rows = score_root(root, ["toy"], "2025-09-25T00:00:00", coldstart=False)
    assert len(rows) == 1 and "regime" not in rows[0]
