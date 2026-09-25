import json
from datetime import datetime

import pytest


def _fit(tmp_path, rows, computed_at="2025-09-25T01:00:00", now=datetime(2025, 9, 25, 10)):
    from core.strategy_fit import StrategyFit
    p = tmp_path / "fit.json"
    p.write_text(json.dumps({"computed_at": computed_at, "floors": {}, "rows": rows}))
    f = StrategyFit(str(p), max_age_days=7, clock=lambda: now)
    f.reload_if_changed()
    return f


def _row(sym, strat, side, verdict, score):
    return {"symbol": sym, "strategy": strat, "side": side, "verdict": verdict, "score": score, "exp_r": score, "exp_r_lb": 0.0,
            "t_stat": 2.0, "pf": 1.3, "wr": 0.5, "stability": 0.6, "decay": 0.8, "n": 160, "source": "walkforward", "params_json": "{}"}


def test_disabled_and_ungoverned(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate, family_for
    fit = _fit(tmp_path, {})
    d = evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", enabled=False, shadow=True)
    assert (d.action, d.reason, d.governed) == ("proceed", "gate_disabled", False)
    d2 = evaluate_fit_gate(fit, "breakout", "long", "NVDA", enabled=True, shadow=True)
    assert (d2.action, d2.reason, d2.governed) == ("proceed", "fit_not_governed", False)
    assert family_for("fit:vwap_reversion") == "vwap_reversion" and family_for("orb_contraction_rvol") == "orb" and family_for("x") is None


def test_pass_watch_fail_missing_in_shadow_and_enforce(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    fit = _fit(tmp_path, {"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.3),
                                   "orb:long": _row("NVDA", "orb", "long", "WATCH", 0.1),
                                   "mr_20d:long": _row("NVDA", "mr_20d", "long", "FAIL", -0.1)}})
    assert evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, True).reason == "fit_pass"
    for strat, reason in (("orb_contraction_rvol", "fit_watch_shadow"), ("fit:mr_20d", "fit_fail"), ("news", "fit_missing_row")):
        d_shadow = evaluate_fit_gate(fit, strat, "long", "NVDA", True, True)
        d_live = evaluate_fit_gate(fit, strat, "long", "NVDA", True, False)
        assert (d_shadow.action, d_shadow.reason) == ("shadow", reason) and (d_live.action, d_live.reason) == ("skip", reason)
        assert d_shadow.governed and d_live.governed


def test_stale_and_missing_snapshot(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    stale = _fit(tmp_path, {"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.3)}}, computed_at="2025-09-01T00:00:00")
    d = evaluate_fit_gate(stale, "day_trade_momentum", "long", "NVDA", True, False)
    assert (d.action, d.reason) == ("skip", "fit_stale") and d.age_days > 7
    from core.strategy_fit import StrategyFit
    missing = StrategyFit(str(tmp_path / "none.json"), clock=lambda: datetime(2025, 9, 25))
    missing.reload_if_changed()
    assert evaluate_fit_gate(missing, "day_trade_momentum", "long", "NVDA", True, True).reason == "fit_missing"


def test_priority_between_competing_pass_lanes(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    fit = _fit(tmp_path, {"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.3),
                                   "orb:long": _row("NVDA", "orb", "long", "PASS", 0.5)}})
    d = evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, False, competing_families=["orb"])
    assert (d.action, d.reason) == ("skip", "fit_lower_priority")
    d2 = evaluate_fit_gate(fit, "orb_contraction_rvol", "long", "NVDA", True, False, competing_families=["dt_momentum"])
    assert d2.reason == "fit_pass"
    assert set(d2.audit_fields()) >= {"family", "verdict", "score", "exp_r", "stability", "source", "age_days"}
