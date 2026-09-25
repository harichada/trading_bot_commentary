import json
import os
import time
from datetime import datetime, timedelta

import pytest


def _snap(tmp_path, computed_at, rows):
    p = tmp_path / "strategy_fit.json"
    p.write_text(json.dumps({"computed_at": computed_at, "floors": {"min_n": 150}, "rows": rows}))
    return p


def _row(sym, strat, side, verdict, score, params=None, source="walkforward", n=200):
    return {"symbol": sym, "strategy": strat, "side": side, "verdict": verdict, "score": score, "exp_r": score, "exp_r_lb": 0.01,
            "t_stat": 2.5, "pf": 1.4, "wr": 0.5, "stability": 0.7, "decay": 0.9, "n": n, "source": source,
            "params_json": json.dumps(params or {"k": 1}), "computed_at": "x"}


def test_lookup_allowed_lanes_and_params(tmp_path):
    from core.strategy_fit import StrategyFit
    now = datetime(2025, 9, 25, 10, 0, 0)
    p = _snap(tmp_path, "2025-09-25T01:00:00", {"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3),
                                                       "dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.5, {"rs_floor": 0.5}),
                                                       "mr_20d:short": _row("NVDA", "mr_20d", "short", "WATCH", 0.1)}})
    fit = StrategyFit(str(p), max_age_days=7, clock=lambda: now)
    assert fit.reload_if_changed() is True and fit.reload_if_changed() is False
    assert fit.lookup("NVDA", "orb", "long").verdict == "PASS" and fit.lookup("NVDA", "orb", "short") is None
    assert [r.strategy for r in fit.allowed_lanes("NVDA")] == ["dt_momentum", "orb"]
    assert fit.params_for("NVDA", "dt_momentum") == {"rs_floor": 0.5} and fit.params_for("TSLA", "orb") == {}
    s = fit.summary()
    assert s["counts"]["PASS"] == 2 and s["counts"]["WATCH"] == 1 and s["stale"] is False and 0.3 < s["age_days"] < 0.4


def test_stale_snapshot_returns_none(tmp_path):
    from core.strategy_fit import StrategyFit
    p = _snap(tmp_path, "2025-09-01T01:00:00", {"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}})
    fit = StrategyFit(str(p), max_age_days=7, clock=lambda: datetime(2025, 9, 25))
    fit.reload_if_changed()
    assert fit.is_stale() and fit.lookup("NVDA", "orb", "long") is None and fit.allowed_lanes("NVDA") == []


def test_missing_and_corrupt_never_raise(tmp_path):
    from core.strategy_fit import StrategyFit
    fit = StrategyFit(str(tmp_path / "nope.json"), clock=lambda: datetime(2025, 9, 25))
    assert fit.reload_if_changed() is False and fit.lookup("NVDA", "orb", "long") is None and fit.summary()["last_error"]
    p = tmp_path / "bad.json"; p.write_text("{not json")
    fit2 = StrategyFit(str(p), clock=lambda: datetime(2025, 9, 25))
    assert fit2.reload_if_changed() is False and fit2.summary()["counts"] == {} and "json" in fit2.summary()["last_error"].lower()


def test_reload_on_mtime_change_keeps_old_rows_on_error(tmp_path):
    from core.strategy_fit import StrategyFit
    now = datetime(2025, 9, 25, 10)
    p = _snap(tmp_path, "2025-09-25T01:00:00", {"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}})
    fit = StrategyFit(str(p), clock=lambda: now)
    fit.reload_if_changed()
    time.sleep(0.01)
    p.write_text("{broken")
    os.utime(p, None)
    assert fit.reload_if_changed() is False
    assert fit.lookup("NVDA", "orb", "long") is not None       # previous rows retained
