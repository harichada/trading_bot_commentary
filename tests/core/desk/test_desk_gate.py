"""v-playbook-desk-b2-2026-09-25 (Task 5): the desk-limit router gate.

`evaluate_desk_gate` is the pure rule the signal router consults before the
strategy-fit gate: it turns the morning plan plus today's entry count into
allow / shadow / skip. Shadow is the default posture — with
`ENABLE_DESK_LIMITS` off the decision is audited and nothing is blocked.
"""
from __future__ import annotations

from core.desk.daily_plan import evaluate_desk_gate
from core.desk.models import DailyPlan


def _plan(**kw) -> DailyPlan:
    base = dict(date="2026-09-25", regime="risk_on", long_ok=True, short_ok=True,
                max_trades=4, sit_out=False, sit_out_reason="")
    base.update(kw)
    return DailyPlan(**base)


# ── no plan / unusable input: never governed ────────────────────────

def test_no_plan_is_not_governed():
    d = evaluate_desk_gate(None, 0, "long", True)
    assert (d.action, d.reason, d.governed) == ("allow", "no_desk_plan", False)


def test_garbage_plan_is_not_governed():
    d = evaluate_desk_gate({"sit_out": True}, 99, "long", True)
    assert (d.action, d.governed) == ("allow", False)


def test_unknown_side_is_not_governed():
    d = evaluate_desk_gate(_plan(), 0, "sideways", True)
    assert (d.action, d.reason, d.governed) == ("allow", "desk_side_unknown", False)


# ── sit out ─────────────────────────────────────────────────────────

def test_sit_out_skips_when_enabled():
    d = evaluate_desk_gate(_plan(sit_out=True, sit_out_reason="vix_spike"), 0, "long", True)
    assert (d.action, d.reason, d.governed) == ("skip", "desk_sit_out:vix_spike", True)


def test_sit_out_only_shadows_when_disabled():
    d = evaluate_desk_gate(_plan(sit_out=True, sit_out_reason="vix_spike"), 0, "long", False)
    assert (d.action, d.reason, d.governed) == ("shadow", "desk_sit_out:vix_spike", True)


def test_sit_out_without_a_reason_still_names_the_rule():
    d = evaluate_desk_gate(_plan(sit_out=True, sit_out_reason=""), 0, "short", True)
    assert d.reason == "desk_sit_out:unknown"


# ── max trades ──────────────────────────────────────────────────────

def test_max_trades_blocks_at_the_cap():
    d = evaluate_desk_gate(_plan(max_trades=4), 4, "long", True)
    assert (d.action, d.reason, d.governed) == ("skip", "desk_max_trades", True)
    assert d.audit_fields()["trades_today"] == 4
    assert d.audit_fields()["max_trades"] == 4


def test_under_the_cap_is_allowed():
    d = evaluate_desk_gate(_plan(max_trades=4), 3, "long", True)
    assert (d.action, d.reason) == ("allow", "desk_ok")


def test_zero_max_trades_is_not_a_cap():
    """A plan with max_trades 0 means "the rules never set one" — not "no trades"."""
    d = evaluate_desk_gate(_plan(max_trades=0), 7, "long", True)
    assert d.action == "allow"


def test_unusable_trade_count_never_blocks():
    d = evaluate_desk_gate(_plan(max_trades=4), None, "long", True)
    assert d.action == "allow"
    d2 = evaluate_desk_gate(_plan(max_trades=4), "many", "long", True)
    assert d2.action == "allow"


# ── direction ───────────────────────────────────────────────────────

def test_long_off_blocks_longs_only():
    plan = _plan(long_ok=False, short_ok=True)
    assert evaluate_desk_gate(plan, 0, "long", True).reason == "desk_direction_long_off"
    assert evaluate_desk_gate(plan, 0, "short", True).action == "allow"


def test_short_off_blocks_shorts_only():
    plan = _plan(long_ok=True, short_ok=False)
    assert evaluate_desk_gate(plan, 0, "short", True).reason == "desk_direction_short_off"
    assert evaluate_desk_gate(plan, 0, "long", True).action == "allow"


def test_direction_only_shadows_when_disabled():
    d = evaluate_desk_gate(_plan(long_ok=False), 0, "long", False)
    assert (d.action, d.governed) == ("shadow", True)


# ── precedence ──────────────────────────────────────────────────────

def test_sit_out_wins_over_max_trades_and_direction():
    plan = _plan(sit_out=True, sit_out_reason="two_losing_days", max_trades=1, long_ok=False)
    assert evaluate_desk_gate(plan, 9, "long", True).reason == "desk_sit_out:two_losing_days"


def test_max_trades_wins_over_direction():
    plan = _plan(max_trades=1, long_ok=False)
    assert evaluate_desk_gate(plan, 3, "long", True).reason == "desk_max_trades"


# ── audit payload ───────────────────────────────────────────────────

def test_audit_fields_carry_the_plan_context():
    d = evaluate_desk_gate(_plan(sit_out=True, sit_out_reason="vix_spike", max_trades=2),
                           1, "long", True)
    fields = d.audit_fields()
    assert fields["sit_out_reason"] == "vix_spike"
    assert fields["plan_date"] == "2026-09-25"
    assert fields["regime"] == "risk_on"
    assert fields["trades_today"] == 1 and fields["max_trades"] == 2
