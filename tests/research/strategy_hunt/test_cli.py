"""CLI-level tests: universe-context wiring and cmd_score -> score_regime_root -> write_fit."""
import numpy as np
import pandas as pd
import pytest

from strategies.families.common import Family, make_entry, one_per_session


def _bars(start="2023-01-03", sessions=60, seed=0):
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    for d in pd.bdate_range(start, periods=sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        close = px + rng.normal(0, 0.05, 78).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3,
                                    "close": close, "volume": 1000}, index=idx))
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


TOY = Family("toy", _toy, {"w": [1.0]}, "intraday", ("long",))


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


def test_cmd_score_calls_score_regime_root_and_snapshot_has_regime_and_pooled_keys(root, monkeypatch, tmp_path):
    """The controller's Task 6 report flagged that cmd_score never called score_regime_root, so
    production snapshots kept empty regime_rows/pooled_rows forever (lookup_regime always falling
    back to lookup()). This proves the wiring: score_regime_root is actually invoked by `score`,
    and its two return values land in the persisted strategy_fit.json under those two keys -- even
    if this toy dataset is too small to produce any rows in them (no VIX means every trade's regime
    is "unknown", which regime_rows/pooled_rows deliberately drop; see score/regime_rows.py)."""
    import json as _json

    from research.strategy_hunt import cli
    from research.strategy_hunt import report as R
    from research.strategy_hunt import run as RUN
    from research.strategy_hunt.fit.writer import fit_dir

    reports = tmp_path / "reports"
    monkeypatch.setattr(RUN, "REPORTS_DIR", reports)
    monkeypatch.setattr(R, "REPORTS_DIR", reports)

    calls = []
    real_score_regime_root = cli.score_regime_root

    def _recording(root_, families, computed_at):
        calls.append((root_, families, computed_at))
        return real_score_regime_root(root_, families, computed_at)

    monkeypatch.setattr(cli, "score_regime_root", _recording)

    assert cli.main(["run", "--root", str(root), "--symbols", "AAA", "--families", "toy", "--workers", "1"]) == 0
    assert cli.main(["score", "--root", str(root), "--families", "toy", "--no-coldstart"]) == 0

    assert len(calls) == 1
    called_root, called_families, _ = calls[0]
    assert str(called_root) == str(root) and called_families == ["toy"]

    snap = _json.loads((fit_dir(root) / "strategy_fit.json").read_text())
    assert "regime_rows" in snap and "pooled_rows" in snap
    assert isinstance(snap["regime_rows"], dict) and isinstance(snap["pooled_rows"], dict)
