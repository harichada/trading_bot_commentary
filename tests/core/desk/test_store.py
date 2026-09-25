"""v-playbook-desk-b2-2026-09-25: desk model serialization and the plan store.

The live loop reads these files, so `from_dict` must tolerate anything on disk
(unknown keys, wrong types, nulls) and `load_plan` must never raise.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

import pytest

from core.desk.models import DailyPlan, InPlayRow, PlanInputs
from core.desk.store import load_plan, plan_path, save_plan

NOW = datetime(2026, 9, 25, 9, 0)


def _row(**kw) -> InPlayRow:
    base = dict(symbol="AAPL", score=0.75, rvol_open15=3.0, gap_pct=-2.5,
                catalyst=1.0, rs_20=4.0, reasons=("rvol", "gap"))
    base.update(kw)
    return InPlayRow(**base)


def _plan(**kw) -> DailyPlan:
    base = dict(
        date="2026-09-25", regime="mixed", spy_change_pct=0.4, vix=17.0,
        vix_change_pct=3.0, breadth_score=0.55, calendar_flags=("high:14:00:FOMC",),
        long_ok=True, short_ok=False, risk_budget_pct=1.2, max_trades=4,
        sit_out=False, sit_out_reason="", in_play=(_row(),), notes=("regime mixed",),
        built_at=NOW.isoformat(), refreshed_at="",
    )
    base.update(kw)
    return DailyPlan(**base)


# ────────────────────────────────────────────────────────────────────
# models
# ────────────────────────────────────────────────────────────────────

def test_models_are_frozen():
    with pytest.raises(Exception):
        _row().symbol = "MSFT"          # type: ignore[misc]
    with pytest.raises(Exception):
        _plan().max_trades = 99         # type: ignore[misc]


def test_in_play_row_round_trip():
    row = _row()
    assert InPlayRow.from_dict(row.to_dict()) == row


def test_in_play_row_non_finite_floats_become_null():
    d = _row(rvol_open15=float("nan"), rs_20=float("inf"), score=float("-inf")).to_dict()
    assert d["rvol_open15"] is None and d["rs_20"] is None and d["score"] is None
    json.dumps(d)  # must be plain JSON, not NaN/Infinity literals
    back = InPlayRow.from_dict(d)
    assert math.isnan(back.rvol_open15) and math.isnan(back.rs_20)


def test_in_play_row_from_dict_ignores_unknown_keys_and_coerces():
    row = InPlayRow.from_dict({
        "symbol": "msft", "score": "0.5", "rvol_open15": None, "gap_pct": "bad",
        "catalyst": 1, "rs_20": 2, "reasons": ["a", 3], "surprise": {"x": 1},
    })
    assert row.symbol == "MSFT" and row.score == 0.5
    assert math.isnan(row.rvol_open15) and math.isnan(row.gap_pct)
    assert row.catalyst == 1.0 and row.rs_20 == 2.0
    assert row.reasons == ("a", "3")


def test_in_play_row_from_dict_on_garbage_returns_defaults():
    row = InPlayRow.from_dict("not a dict")  # type: ignore[arg-type]
    assert row.symbol == "" and math.isnan(row.score) and row.reasons == ()


def test_daily_plan_round_trip():
    plan = _plan()
    back = DailyPlan.from_dict(plan.to_dict())
    assert back == plan
    assert isinstance(back.in_play, tuple) and isinstance(back.in_play[0], InPlayRow)


def test_daily_plan_breadth_none_survives_round_trip():
    plan = _plan(breadth_score=None)
    assert plan.to_dict()["breadth_score"] is None
    assert DailyPlan.from_dict(plan.to_dict()).breadth_score is None


def test_daily_plan_non_finite_breadth_serializes_as_null():
    assert _plan(breadth_score=float("nan")).to_dict()["breadth_score"] is None


def test_daily_plan_from_dict_ignores_unknown_keys_and_coerces():
    plan = DailyPlan.from_dict({
        "date": "2026-09-25", "regime": 7, "spy_change_pct": "1.5", "vix": None,
        "breadth_score": "0.5", "calendar_flags": "high:08:30:CPI",
        "long_ok": 1, "short_ok": 0, "risk_budget_pct": "x", "max_trades": "4",
        "sit_out": "yes", "sit_out_reason": None, "in_play": [{"symbol": "AAPL"}, "junk"],
        "notes": None, "future_field": [1, 2],
    })
    assert plan.regime == "unknown" and plan.spy_change_pct == 1.5 and math.isnan(plan.vix)
    assert plan.breadth_score == 0.5 and plan.calendar_flags == ("high:08:30:CPI",)
    assert plan.long_ok is True and plan.short_ok is False
    assert plan.risk_budget_pct == 0.0 and plan.max_trades == 4
    assert plan.sit_out is True and plan.sit_out_reason == ""
    assert len(plan.in_play) == 1 and plan.in_play[0].symbol == "AAPL"
    assert plan.notes == ()


def test_plan_inputs_round_trip_and_unknown_keys():
    inputs = PlanInputs(
        date="2026-09-25", regime="risk_on", spy_change_pct=0.5, vix=15.0,
        vix_change_pct=1.0, breadth_score=0.5, calendar_flags=("high:14:00:FOMC",),
        losing_day_streak=1, prev_day_pnl_r=-0.5, fit_stale=True,
        gate_enforcing=False, max_daily_loss_pct=2.0, allows_long=True, allows_short=False)
    d = inputs.to_dict()
    d["unexpected"] = "ignored"
    assert PlanInputs.from_dict(d) == inputs


# ────────────────────────────────────────────────────────────────────
# store
# ────────────────────────────────────────────────────────────────────

def test_plan_path_is_dated_json_under_the_desk_dir(tmp_path):
    p = plan_path(str(tmp_path), "2026-09-25")
    assert p.endswith("2026-09-25.json") and str(tmp_path) in p


def test_save_and_load_round_trip(tmp_path):
    plan = _plan()
    written = save_plan(plan, str(tmp_path))
    assert written == plan_path(str(tmp_path), plan.date) and os.path.exists(written)
    assert load_plan(str(tmp_path), plan.date) == plan


def test_save_plan_leaves_no_temp_file(tmp_path):
    save_plan(_plan(), str(tmp_path))
    leftovers = [str(p) for p in tmp_path.rglob("*") if p.is_file() and ".tmp" in p.name]
    assert leftovers == []


def test_save_plan_overwrites_in_place(tmp_path):
    save_plan(_plan(max_trades=4), str(tmp_path))
    save_plan(_plan(max_trades=2), str(tmp_path))
    assert load_plan(str(tmp_path), "2026-09-25").max_trades == 2
    files = [p.name for p in tmp_path.rglob("*.json")]
    assert files == ["2026-09-25.json"]


def test_save_plan_creates_missing_directories(tmp_path):
    target = str(tmp_path / "deep" / "desk")
    assert save_plan(_plan(), target) is not None
    assert load_plan(target, "2026-09-25") is not None


def test_load_plan_missing_file_returns_none(tmp_path):
    assert load_plan(str(tmp_path), "2026-09-25") is None


def test_load_plan_malformed_json_returns_none(tmp_path, caplog):
    path = plan_path(str(tmp_path), "2026-09-25")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("{ not json")
    with caplog.at_level("WARNING"):
        assert load_plan(str(tmp_path), "2026-09-25") is None
    assert caplog.records


def test_load_plan_non_dict_payload_returns_none(tmp_path):
    path = plan_path(str(tmp_path), "2026-09-25")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump([1, 2, 3], fh)
    assert load_plan(str(tmp_path), "2026-09-25") is None


def test_save_plan_on_unwritable_dir_returns_none_and_does_not_raise(tmp_path, caplog):
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    with caplog.at_level("ERROR"):
        assert save_plan(_plan(), str(blocker)) is None
    assert caplog.records
