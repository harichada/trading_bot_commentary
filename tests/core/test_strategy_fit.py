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


def test_rows_for_returns_all_rows_regardless_of_verdict(tmp_path):
    from core.strategy_fit import StrategyFit
    now = datetime(2025, 9, 25, 10, 0, 0)
    p = _snap(tmp_path, "2025-09-25T01:00:00", {"NVDA": {
        "orb:long": _row("NVDA", "orb", "long", "PASS", 0.3),
        "dt_momentum:long": _row("NVDA", "dt_momentum", "long", "FAIL", 0.1),
        "mr_20d:short": _row("NVDA", "mr_20d", "short", "WATCH", 0.2),
    }})
    fit = StrategyFit(str(p), max_age_days=7, clock=lambda: now)
    fit.reload_if_changed()
    rows = fit.rows_for("NVDA")
    assert {r.strategy for r in rows} == {"orb", "dt_momentum", "mr_20d"}
    assert {r.verdict for r in rows} == {"PASS", "FAIL", "WATCH"}
    assert fit.rows_for("TSLA") == []


def test_rows_for_empty_when_stale_or_unloaded(tmp_path):
    from core.strategy_fit import StrategyFit
    p = _snap(tmp_path, "2025-09-01T01:00:00", {"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}})
    stale_fit = StrategyFit(str(p), max_age_days=7, clock=lambda: datetime(2025, 9, 25))
    stale_fit.reload_if_changed()
    assert stale_fit.rows_for("NVDA") == []

    unloaded_fit = StrategyFit(str(tmp_path / "nope.json"), clock=lambda: datetime(2025, 9, 25))
    assert unloaded_fit.rows_for("NVDA") == []


def test_non_string_computed_at_never_raises(tmp_path):
    from core.strategy_fit import StrategyFit
    p = _snap(tmp_path, 20250925, {"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}})
    fit = StrategyFit(str(p), max_age_days=7, clock=lambda: datetime(2025, 9, 25))
    assert fit.reload_if_changed() is True
    assert fit.age_days() is None
    assert fit.is_stale() is True
    assert fit.lookup("NVDA", "orb", "long") is None
    s = fit.summary()
    assert s["age_days"] is None and s["stale"] is True


def test_invalid_utf8_file_never_raises(tmp_path):
    from core.strategy_fit import StrategyFit
    p = tmp_path / "binary.json"
    p.write_bytes(b"\xff\xfe\x00\x01not valid utf-8")
    fit = StrategyFit(str(p), clock=lambda: datetime(2025, 9, 25))
    assert fit.reload_if_changed() is False
    assert fit.summary()["last_error"]


# ──────────────────────
# v-playbook-desk-b1-2026-09-25 — regime-conditioned and pooled rows
# ──────────────────────

def _regime_snap(tmp_path, computed_at, rows, regime_rows=None, pooled_rows=None):
    payload = {"computed_at": computed_at, "floors": {"min_n": 150}, "rows": rows}
    if regime_rows is not None:
        payload["regime_rows"] = regime_rows
    if pooled_rows is not None:
        payload["pooled_rows"] = pooled_rows
    p = tmp_path / "strategy_fit.json"
    p.write_text(json.dumps(payload))
    return p


def _rrow(sym, strat, side, verdict, score, regime, source="walkforward", n=200):
    row = _row(sym, strat, side, verdict, score, source=source, n=n)
    row["regime"] = regime
    return row


def _fit(path, now=datetime(2025, 9, 25, 10)):
    from core.strategy_fit import StrategyFit
    f = StrategyFit(str(path), max_age_days=7, clock=lambda: now)
    f.reload_if_changed()
    return f


def test_lookup_regime_precedence_symbol_then_pooled_then_unconditional(tmp_path):
    p = _regime_snap(
        tmp_path, "2025-09-25T01:00:00",
        rows={"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}},
        regime_rows={"NVDA": {"orb:long:risk_on": _rrow("NVDA", "orb", "long", "FAIL", -0.1, "risk_on")}},
        pooled_rows={"orb:long:risk_off": _rrow("*", "orb", "long", "PASS", 0.4, "risk_off", source="pooled"),
                     "orb:long:risk_on": _rrow("*", "orb", "long", "PASS", 0.9, "risk_on", source="pooled")},
    )
    fit = _fit(p)

    # 1. the symbol's own regime row wins over the pooled one
    own = fit.lookup_regime("NVDA", "orb", "long", "risk_on")
    assert own.verdict == "FAIL" and own.regime == "risk_on" and own.pooled is False and own.symbol == "NVDA"
    # 2. no symbol regime row -> the pooled regime row, flagged pooled
    pooled = fit.lookup_regime("NVDA", "orb", "long", "risk_off")
    assert pooled.verdict == "PASS" and pooled.symbol == "*" and pooled.pooled is True and pooled.source == "pooled"
    # 3. neither -> the unconditional row
    fallback = fit.lookup_regime("NVDA", "orb", "long", "mixed")
    assert fallback.verdict == "PASS" and fallback.regime == "" and fallback.pooled is False
    # 4. nothing at all -> None
    assert fit.lookup_regime("TSLA", "orb", "long", "mixed") is None
    assert fit.lookup_regime("NVDA", "mr_20d", "long", "risk_on") is None


def test_lookup_regime_none_when_stale_or_snapshot_lacks_keys(tmp_path):
    stale = _fit(_regime_snap(tmp_path, "2025-09-01T01:00:00",
                              rows={"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}},
                              regime_rows={"NVDA": {"orb:long:risk_on": _rrow("NVDA", "orb", "long", "PASS", 0.3,
                                                                              "risk_on")}}))
    assert stale.lookup_regime("NVDA", "orb", "long", "risk_on") is None

    # An older snapshot with neither key must degrade to the unconditional row, not raise.
    older_dir = tmp_path / "older"
    older_dir.mkdir()
    old = _fit(_regime_snap(older_dir, "2025-09-25T01:00:00",
                            rows={"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}}))
    assert old.lookup_regime("NVDA", "orb", "long", "risk_on").verdict == "PASS"
    s = old.summary()
    assert s["n_regime_rows"] == 0 and s["n_pooled_pass"] == 0


def test_summary_counts_regime_and_pooled_rows(tmp_path):
    p = _regime_snap(
        tmp_path, "2025-09-25T01:00:00",
        rows={"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}},
        regime_rows={"NVDA": {"orb:long:risk_on": _rrow("NVDA", "orb", "long", "PASS", 0.3, "risk_on"),
                              "orb:long:mixed": _rrow("NVDA", "orb", "long", "FAIL", 0.0, "mixed")},
                     "TSLA": {"orb:short:risk_off": _rrow("TSLA", "orb", "short", "WATCH", 0.1, "risk_off")}},
        pooled_rows={"orb:long:risk_on": _rrow("*", "orb", "long", "PASS", 0.5, "risk_on", source="pooled"),
                     "orb:long:mixed": _rrow("*", "orb", "long", "WATCH", 0.2, "mixed", source="pooled")},
    )
    s = _fit(p).summary()
    assert s["n_regime_rows"] == 3 and s["n_pooled_pass"] == 1
    # the unconditional counts are unchanged by the new lanes
    assert s["counts"] == {"PASS": 1} and s["n_symbols"] == 1


def test_fit_row_defaults_and_pooled_flag_from_source(tmp_path):
    from core.strategy_fit import FitRow
    plain = FitRow.from_dict({"symbol": "NVDA", "strategy": "orb", "side": "long", "verdict": "PASS"})
    assert plain.regime == "" and plain.pooled is False
    pooled = FitRow.from_dict({"symbol": "*", "strategy": "orb", "side": "long", "verdict": "PASS",
                               "source": "pooled", "regime": "risk_on"})
    assert pooled.pooled is True and pooled.regime == "risk_on"
    assert pooled.to_json_dict()["regime"] == "risk_on" and pooled.to_json_dict()["pooled"] is True
    # a non-dict / garbage regime never raises
    assert FitRow.from_dict({"regime": 7, "pooled": "yes"}).regime == ""


def test_fit_row_from_dict_ignores_unknown_metric_keys():
    """v-playbook-desk-b1-5 task 9: pooled rows will start carrying t_stat_session,
    recovery, and (later) excess_r. FitRow.from_dict must tolerate keys it doesn't
    know yet -- core never imports research, so it can't validate against the
    research-side row shape and just ignores what it doesn't recognize."""
    from core.strategy_fit import FitRow
    row = _row("NVDA", "orb", "long", "PASS", 0.3)
    row.update({"t_stat_session": 4.2, "recovery": 5.1, "excess_r": 0.02})
    parsed = FitRow.from_dict(row)
    assert parsed.symbol == "NVDA" and parsed.verdict == "PASS" and parsed.score == 0.3


def test_malformed_regime_and_pooled_sections_never_raise(tmp_path):
    p = tmp_path / "bad_sections.json"
    p.write_text(json.dumps({"computed_at": "2025-09-25T01:00:00", "floors": {},
                             "rows": {"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}},
                             "regime_rows": {"NVDA": ["not", "a", "dict"], "TSLA": {"orb:long:risk_on": 5}},
                             "pooled_rows": ["nope"]}))
    fit = _fit(p)
    assert fit.lookup("NVDA", "orb", "long").verdict == "PASS"
    assert fit.lookup_regime("NVDA", "orb", "long", "risk_on").verdict == "PASS"   # falls back
    assert fit.summary()["n_regime_rows"] == 0 and fit.summary()["last_error"] is None


def test_snapshot_with_a_baseline_pooled_rows_section_loads_and_is_ignored(tmp_path):
    """v-playbook-desk-b1-5 task 10: the research writer adds a `baseline_pooled_rows` section
    (the shuffled control's pooled rows). The live reader must load such a snapshot without
    error and must never treat a baseline row as a pooled lane -- it only reads `pooled_rows`."""
    p = tmp_path / "with_baseline.json"
    baseline = _row("*", "orb", "long", "PASS", 0.9)
    baseline.update({"source": "baseline", "regime": "risk_on", "excess_r": None, "recovery": None})
    p.write_text(json.dumps({"computed_at": "2025-09-25T01:00:00", "floors": {},
                             "rows": {"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.3)}},
                             "regime_rows": {}, "pooled_rows": {},
                             "baseline_pooled_rows": {"orb:long:risk_on": baseline}}))
    fit = _fit(p)
    assert fit.summary()["last_error"] is None
    assert fit.lookup("NVDA", "orb", "long").verdict == "PASS"
    assert fit.summary()["n_pooled_pass"] == 0 and fit.summary()["n_regime_rows"] == 0
    assert fit.lookup_regime("NVDA", "orb", "long", "risk_on").symbol == "NVDA"
