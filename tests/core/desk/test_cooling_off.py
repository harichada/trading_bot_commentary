"""v-cooling-off-2026-10-07: a served `two_losing_days` sit-out resets the streak
clock and the next session resumes at half risk. Before this, a sat-out session
had no trades, so the streak never cleared and the desk sat out every day
(2026-10-02 → 10-07 on two losing days worth $32)."""
from __future__ import annotations

import asyncio
from datetime import datetime

from core.desk.daily_plan import build_daily_plan
from core.desk.models import DailyPlan, PlanInputs
from core.desk.store import save_plan
from core.loops.desk_trade_stats import cooling_off
from tests.core.desk.test_desk_plan_audit import _CTX, _Engine, _cfg
from core.loops.desk_loop import DeskLoop
from core.loops import desk_steps as steps

NOW = datetime(2026, 10, 7, 9, 0)


def _plan(date, sit_out=False, reason=""):
    return DailyPlan(date=date, sit_out=sit_out, sit_out_reason=reason, built_at="x")


def test_cooling_off_reads_prior_plans():
    plans = [_plan("2026-10-02", True, "two_losing_days"), _plan("2026-10-05", True, "two_losing_days"),
             _plan("2026-10-06", True, "two_losing_days"), _plan("2026-10-07", True, "two_losing_days")]
    assert cooling_off(plans, "2026-10-07") == ("2026-10-06", True)
    assert cooling_off(plans + [_plan("2026-10-08")], "2026-10-09") == ("2026-10-07", False)
    assert cooling_off([_plan("2026-10-06", True, "vix_spike")], "2026-10-07") == ("", False)
    assert cooling_off([], "2026-10-07") == ("", False)
    assert cooling_off("garbage", "2026-10-07") == ("", False)


def test_plan_resumes_at_half_risk_after_a_served_sit_out():
    base = dict(date="2026-10-07", regime="risk_on", breadth_score=0.5, losing_day_streak=2,
                prev_day_pnl_r=0.0, max_daily_loss_pct=2.0)
    stuck = build_daily_plan(PlanInputs(**base), NOW)
    assert stuck.sit_out and stuck.sit_out_reason == "two_losing_days"
    resumed = build_daily_plan(PlanInputs(**base, prior_session_sat_out=True), NOW)
    assert resumed.sit_out is False
    assert resumed.risk_budget_pct == stuck.risk_budget_pct / 2
    assert any("resuming after" in n for n in resumed.notes)
    assert PlanInputs.from_dict(PlanInputs(**base, prior_session_sat_out=True).to_dict()).prior_session_sat_out is True


def test_loop_restarts_the_streak_clock_after_the_served_day(tmp_path):
    cfg = _cfg(tmp_path)
    save_plan(DailyPlan(date="2026-10-06", sit_out=True, sit_out_reason="two_losing_days", built_at="x"),
              cfg.DESK_DATA_DIR)
    trades = [{"exit_time": "2026-09-29 15:00", "pnl_r": -0.9, "strategy": "day_trade_momentum"},
              {"exit_time": "2026-10-01 15:00", "pnl_r": -1.7, "strategy": "day_trade_momentum"}]
    engine = _Engine(trades)
    loop = DeskLoop(engine, config_fn=lambda: cfg, context_reader=lambda symbol: _CTX)
    now = datetime(2026, 10, 7, 9, 0, tzinfo=steps.ET_TZ)
    asyncio.run(loop._step_plan(cfg, now, "2026-10-07"))
    details = [a[4] for a in engine.audits if a[0] == "desk_plan"][0]
    assert details["streak_since"] == "2026-10-07" and details["prior_sat_out"] is True
    assert details["sit_out"] is False
    assert details["risk_budget_pct"] == 2.5            # 5% cap × 1.0 risk_on, halved
