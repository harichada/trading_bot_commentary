"""v-playbook-desk-b2-2026-09-25: DailyPlan rules and the economic-calendar reader.

Every rule in the plan brief gets a test that fires it alone, plus precedence
tests: the sit-out reason is the *first* rule that matches, so a day that trips
several rules must still name the one a human trader would say out loud.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from core.desk.daily_plan import build_daily_plan, load_calendar_flags, refresh_plan
from core.desk.models import InPlayRow, PlanInputs

NOW = datetime(2026, 9, 25, 9, 0)
LATER = datetime(2026, 9, 25, 9, 35)


def _inputs(**kw) -> PlanInputs:
    base = dict(
        date="2026-09-25",
        regime="risk_on",
        spy_change_pct=0.5,
        vix=15.0,
        vix_change_pct=1.0,
        breadth_score=0.5,
        calendar_flags=(),
        losing_day_streak=0,
        prev_day_pnl_r=0.5,
        fit_stale=False,
        gate_enforcing=False,
        max_daily_loss_pct=2.0,
        allows_long=True,
        allows_short=True,
    )
    base.update(kw)
    return PlanInputs(**base)


# ────────────────────────────────────────────────────────────────────
# sit-out rules
# ────────────────────────────────────────────────────────────────────

def test_quiet_day_does_not_sit_out():
    plan = build_daily_plan(_inputs(), NOW)
    assert plan.sit_out is False and plan.sit_out_reason == ""
    assert plan.date == "2026-09-25" and plan.regime == "risk_on"
    assert plan.built_at == NOW.isoformat() and plan.refreshed_at == ""


def test_sit_out_calendar_high_impact_before_1030():
    plan = build_daily_plan(_inputs(calendar_flags=("high:08:30:CPI",)), NOW)
    assert plan.sit_out is True
    assert plan.sit_out_reason == "calendar_high_impact_before_1030"
    assert any("calendar" in n for n in plan.notes)


def test_calendar_high_impact_after_1030_does_not_sit_out():
    plan = build_daily_plan(_inputs(calendar_flags=("high:10:30:FOMC",)), NOW)
    assert plan.sit_out is False and plan.sit_out_reason == ""


def test_calendar_low_impact_before_1030_does_not_sit_out():
    plan = build_daily_plan(_inputs(calendar_flags=("medium:08:30:Claims",)), NOW)
    assert plan.sit_out is False


def test_calendar_flag_with_unparseable_time_is_ignored():
    plan = build_daily_plan(_inputs(calendar_flags=("high:nope:Mystery",)), NOW)
    assert plan.sit_out is False


def test_sit_out_vix_spike():
    plan = build_daily_plan(_inputs(vix_change_pct=15.5), NOW)
    assert plan.sit_out is True and plan.sit_out_reason == "vix_spike"


def test_vix_change_at_threshold_does_not_sit_out():
    plan = build_daily_plan(_inputs(vix_change_pct=15.0), NOW)
    assert plan.sit_out is False


def test_sit_out_two_losing_days():
    plan = build_daily_plan(_inputs(losing_day_streak=2), NOW)
    assert plan.sit_out is True and plan.sit_out_reason == "two_losing_days"


def test_one_losing_day_does_not_sit_out():
    plan = build_daily_plan(_inputs(losing_day_streak=1), NOW)
    assert plan.sit_out is False


def test_sit_out_fit_stale_gate_enforcing():
    plan = build_daily_plan(_inputs(fit_stale=True, gate_enforcing=True), NOW)
    assert plan.sit_out is True and plan.sit_out_reason == "fit_stale_gate_enforcing"


def test_fit_stale_without_enforcing_gate_does_not_sit_out():
    plan = build_daily_plan(_inputs(fit_stale=True, gate_enforcing=False), NOW)
    assert plan.sit_out is False


def test_sit_out_precedence_calendar_first():
    plan = build_daily_plan(
        _inputs(calendar_flags=("high:08:30:CPI",), vix_change_pct=30.0,
                losing_day_streak=3, fit_stale=True, gate_enforcing=True),
        NOW,
    )
    assert plan.sit_out_reason == "calendar_high_impact_before_1030"


def test_sit_out_precedence_vix_before_streak():
    plan = build_daily_plan(
        _inputs(vix_change_pct=30.0, losing_day_streak=3, fit_stale=True, gate_enforcing=True),
        NOW,
    )
    assert plan.sit_out_reason == "vix_spike"


def test_sit_out_precedence_streak_before_fit_stale():
    plan = build_daily_plan(
        _inputs(losing_day_streak=3, fit_stale=True, gate_enforcing=True), NOW)
    assert plan.sit_out_reason == "two_losing_days"


def test_sit_out_note_names_the_reason():
    plan = build_daily_plan(_inputs(vix_change_pct=40.0), NOW)
    assert any("sit out" in n and "vix_spike" in n for n in plan.notes)


# ────────────────────────────────────────────────────────────────────
# risk budget / max trades
# ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("regime,mult", [
    ("risk_on", 1.0), ("mixed", 0.6), ("risk_off", 0.4),
    ("unknown", 0.4), ("something_else", 0.4),
])
def test_risk_budget_table(regime, mult):
    plan = build_daily_plan(_inputs(regime=regime, max_daily_loss_pct=2.0), NOW)
    assert plan.risk_budget_pct == pytest.approx(2.0 * mult)


@pytest.mark.parametrize("regime,mult", [("risk_on", 1.0), ("mixed", 0.6), ("risk_off", 0.4)])
def test_risk_budget_halved_after_losing_day(regime, mult):
    plan = build_daily_plan(
        _inputs(regime=regime, max_daily_loss_pct=2.0, prev_day_pnl_r=-0.3), NOW)
    assert plan.risk_budget_pct == pytest.approx(2.0 * mult / 2.0)
    assert any("halved" in n for n in plan.notes)


def test_risk_budget_not_halved_on_flat_day():
    plan = build_daily_plan(_inputs(prev_day_pnl_r=0.0, max_daily_loss_pct=2.0), NOW)
    assert plan.risk_budget_pct == pytest.approx(2.0)
    assert not any("halved" in n for n in plan.notes)


@pytest.mark.parametrize("regime,n", [
    ("risk_on", 6), ("mixed", 4), ("risk_off", 2), ("unknown", 2), ("garbage", 2),
])
def test_max_trades_table(regime, n):
    assert build_daily_plan(_inputs(regime=regime), NOW).max_trades == n


# ────────────────────────────────────────────────────────────────────
# direction flags
# ────────────────────────────────────────────────────────────────────

def test_direction_flags_with_healthy_breadth():
    plan = build_daily_plan(_inputs(breadth_score=0.5), NOW)
    assert plan.long_ok is True and plan.short_ok is True


def test_long_off_when_breadth_weak():
    plan = build_daily_plan(_inputs(breadth_score=0.39), NOW)
    assert plan.long_ok is False and plan.short_ok is True
    assert any("long" in n.lower() for n in plan.notes)


def test_short_off_when_breadth_strong():
    plan = build_daily_plan(_inputs(breadth_score=0.61), NOW)
    assert plan.long_ok is True and plan.short_ok is False
    assert any("short" in n.lower() for n in plan.notes)


def test_breadth_at_thresholds_keeps_both_sides():
    plan = build_daily_plan(_inputs(breadth_score=0.4), NOW)
    assert plan.long_ok is True
    plan = build_daily_plan(_inputs(breadth_score=0.6), NOW)
    assert plan.short_ok is True


@pytest.mark.parametrize("breadth", [None, float("nan")])
def test_missing_breadth_defers_to_market_context(breadth):
    plan = build_daily_plan(_inputs(breadth_score=breadth), NOW)
    assert plan.long_ok is True and plan.short_ok is True
    assert plan.breadth_score is None
    plan = build_daily_plan(
        _inputs(breadth_score=breadth, allows_long=False, allows_short=True), NOW)
    assert plan.long_ok is False and plan.short_ok is True


def test_market_context_veto_wins_over_breadth():
    plan = build_daily_plan(_inputs(breadth_score=0.5, allows_long=False, allows_short=False), NOW)
    assert plan.long_ok is False and plan.short_ok is False


# ────────────────────────────────────────────────────────────────────
# refresh
# ────────────────────────────────────────────────────────────────────

def test_refresh_plan_replaces_in_play_and_keeps_everything_else():
    plan = build_daily_plan(_inputs(regime="mixed", calendar_flags=("high:14:00:FOMC",)), NOW)
    rows = (InPlayRow(symbol="AAPL", score=0.9, rvol_open15=3.0, gap_pct=2.0,
                      catalyst=1.0, rs_20=4.0, reasons=("rvol",)),)
    refreshed = refresh_plan(plan, rows, LATER)

    assert refreshed.in_play == rows
    assert refreshed.refreshed_at == LATER.isoformat()
    assert refreshed.built_at == plan.built_at
    for field in ("date", "regime", "spy_change_pct", "vix", "vix_change_pct",
                  "breadth_score", "calendar_flags", "long_ok", "short_ok",
                  "risk_budget_pct", "max_trades", "sit_out", "sit_out_reason", "notes"):
        assert getattr(refreshed, field) == getattr(plan, field)


def test_refresh_plan_does_not_mutate_the_original():
    plan = build_daily_plan(_inputs(), NOW)
    refresh_plan(plan, (InPlayRow(symbol="MSFT"),), LATER)
    assert plan.in_play == () and plan.refreshed_at == ""


def test_refresh_plan_accepts_a_list_and_stores_a_tuple():
    plan = build_daily_plan(_inputs(), NOW)
    refreshed = refresh_plan(plan, [InPlayRow(symbol="MSFT")], LATER)
    assert isinstance(refreshed.in_play, tuple) and refreshed.in_play[0].symbol == "MSFT"


# ────────────────────────────────────────────────────────────────────
# calendar
# ────────────────────────────────────────────────────────────────────

def _write(tmp_path, payload):
    path = tmp_path / "calendar.json"
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return str(path)


def test_load_calendar_flags_filters_by_date(tmp_path):
    path = _write(tmp_path, [
        {"date": "2026-09-25", "time_et": "08:30", "event": "CPI", "impact": "high"},
        {"date": "2026-09-26", "time_et": "08:30", "event": "PPI", "impact": "high"},
    ])
    assert load_calendar_flags(path, "2026-09-25") == ("high:08:30:CPI",)


def test_load_calendar_flags_missing_file_is_empty(tmp_path):
    assert load_calendar_flags(str(tmp_path / "nope.json"), "2026-09-25") == ()


def test_load_calendar_flags_skips_malformed_rows(tmp_path, caplog):
    path = _write(tmp_path, [
        "not a dict",
        {"date": "2026-09-25", "event": "no time", "impact": "high"},
        {"date": "2026-09-25", "time_et": "09:00", "impact": "high"},
        {"date": "2026-09-25", "time_et": "08:30", "event": "CPI", "impact": "HIGH"},
    ])
    with caplog.at_level("WARNING"):
        flags = load_calendar_flags(path, "2026-09-25")
    assert flags == ("high:08:30:CPI",)
    assert caplog.records


def test_load_calendar_flags_bad_json_is_empty(tmp_path, caplog):
    path = _write(tmp_path, "{not json")
    with caplog.at_level("WARNING"):
        assert load_calendar_flags(path, "2026-09-25") == ()


def test_load_calendar_flags_non_list_payload_is_empty(tmp_path):
    assert load_calendar_flags(_write(tmp_path, {"rows": []}), "2026-09-25") == ()


def test_calendar_flags_feed_the_sit_out_rule(tmp_path):
    path = _write(tmp_path, [
        {"date": "2026-09-25", "time_et": "08:30", "event": "CPI", "impact": "high"}])
    flags = load_calendar_flags(path, "2026-09-25")
    assert build_daily_plan(_inputs(calendar_flags=flags), NOW).sit_out is True
