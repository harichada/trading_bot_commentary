"""v-playbook-desk-b2-2026-09-25 (Task 3): `carry_forward` and `evaluate_playbook_gate`.

Split out of `test_playbook.py` (fix r2) so both modules stay well under 400 lines; the rules,
clamp and store tests stay there.
"""
import pytest

from core.desk.playbook import (
    PROBATION_SIZE_MULT,
    SUSPENSION_LAPSE_DAYS,
    Playbook,
    PlaybookEntry,
    carry_forward,
    evaluate_playbook_gate,
)

NOW = "2026-09-25T20:00:00"


def _entry(**kw):
    base = dict(family="orb", side="long", regime="risk_on", status="enabled", size_mult=1.0,
                exp_r_recent=0.2, t_stat_recent=2.0, n_recent=50, updated_at=NOW, reason="enabled_normal")
    base.update(kw)
    return PlaybookEntry(**base)


# ────────────────────────── carry_forward (suspensions lapse) ──────────────────────────

def test_a_suspension_lapses_to_probation_without_recent_trades():
    """r1 finding 3: a suspended key stops trading, so it can only stay suspended on
    evidence. With no resolved trade in 30 days it goes back on probation."""
    assert SUSPENSION_LAPSE_DAYS == 30
    prev = _entry(status="suspended", size_mult=0.0, reason="suspended_negative_expectancy")

    e = carry_forward(prev, "2026-10-30T20:00:00", has_recent_trades=False)

    assert e.status == "probation" and e.size_mult == PROBATION_SIZE_MULT
    assert e.reason == "suspension_lapsed_no_recent_trades"
    assert e.updated_at == "2026-10-30T20:00:00"
    # The measurement that suspended it is kept, so the journal can still explain why.
    assert (e.exp_r_recent, e.t_stat_recent, e.n_recent) == (prev.exp_r_recent, prev.t_stat_recent,
                                                            prev.n_recent)
    assert prev.status == "suspended"          # never mutated


def test_a_suspension_with_recent_trades_is_untouched():
    prev = _entry(status="suspended", size_mult=0.0, reason="suspended_negative_expectancy")
    assert carry_forward(prev, NOW, has_recent_trades=True) == prev


@pytest.mark.parametrize("status", ["enabled", "probation"])
def test_carry_forward_only_touches_suspensions(status):
    prev = _entry(status=status, size_mult=1.0 if status == "enabled" else 0.5)
    assert carry_forward(prev, NOW, has_recent_trades=False) == prev


# ────────────────────────── gate ──────────────────────────

def _pb(*entries):
    pb = Playbook(entries={}, updated_at=NOW)
    for e in entries:
        pb = pb.with_entry(e)
    return pb


def test_gate_allows_when_there_is_no_entry():
    d = evaluate_playbook_gate(_pb(), "orb", "long", "risk_on", True)

    assert (d.action, d.reason, d.governed) == ("allow", "no_playbook_entry", False)
    assert "size_mult" not in d.audit_fields()


def test_gate_skips_a_suspended_key_when_enabled():
    pb = _pb(_entry(status="suspended", size_mult=0.0, reason="suspended_negative_expectancy"))
    d = evaluate_playbook_gate(pb, "orb", "long", "risk_on", True)

    assert (d.action, d.reason, d.governed) == ("skip", "playbook_suspended", True)
    fields = d.audit_fields()
    assert fields["family"] == "orb" and fields["regime"] == "risk_on"
    assert fields["status"] == "suspended" and fields["size_mult"] == 0.0


def test_gate_only_shadows_a_suspended_key_when_disabled():
    pb = _pb(_entry(status="suspended", size_mult=0.0))
    d = evaluate_playbook_gate(pb, "orb", "long", "risk_on", False)

    assert (d.action, d.reason, d.governed) == ("shadow", "playbook_suspended", True)


def test_gate_allows_probation_with_the_half_size_multiplier():
    pb = _pb(_entry(status="probation", size_mult=0.5))
    d = evaluate_playbook_gate(pb, "orb", "long", "risk_on", True)

    assert (d.action, d.reason, d.governed) == ("allow", "playbook_probation", True)
    assert d.audit_fields()["size_mult"] == 0.5


def test_gate_allows_an_enabled_key_with_its_multiplier():
    pb = _pb(_entry(status="enabled", size_mult=1.25))
    d = evaluate_playbook_gate(pb, "orb", "long", "risk_on", True)

    assert (d.action, d.reason, d.governed) == ("allow", "playbook_enabled", True)
    assert d.audit_fields()["size_mult"] == 1.25


def test_gate_clamps_a_tampered_multiplier_in_the_audit():
    pb = _pb(_entry(status="enabled", size_mult=99.0))
    assert evaluate_playbook_gate(pb, "orb", "long", "risk_on", True).audit_fields()["size_mult"] == 1.5


def test_gate_unknown_regime_falls_back_to_the_mixed_entry():
    pb = _pb(_entry(regime="mixed", status="suspended", size_mult=0.0))

    d = evaluate_playbook_gate(pb, "orb", "long", "unknown", True)
    assert (d.action, d.reason) == ("skip", "playbook_suspended")
    assert d.audit_fields()["regime"] == "mixed"
    assert d.audit_fields()["playbook_fallback"] == "mixed"

    assert evaluate_playbook_gate(pb, "orb", "long", None, True).action == "skip"


def test_gate_unknown_regime_prefers_its_own_entry_over_mixed():
    pb = _pb(_entry(regime="unknown", status="enabled", size_mult=1.0),
             _entry(regime="mixed", status="suspended", size_mult=0.0))
    d = evaluate_playbook_gate(pb, "orb", "long", "unknown", True)

    assert d.action == "allow" and d.audit_fields()["regime"] == "unknown"


def test_gate_falls_back_to_the_unknown_entry_for_a_known_regime():
    """r1 finding 1b: the `unknown` entry is the all-regimes fallback — until the shadow
    tape carries a regime, it is the only measurement most keys have."""
    pb = _pb(_entry(regime="unknown", status="suspended", size_mult=0.0))
    d = evaluate_playbook_gate(pb, "orb", "long", "risk_off", True)

    assert (d.action, d.reason, d.governed) == ("skip", "playbook_suspended", True)
    fields = d.audit_fields()
    assert fields["regime"] == "unknown" and fields["playbook_fallback"] == "unknown"


def test_gate_never_uses_the_mixed_entry_for_a_known_regime():
    """`mixed` is a *measured* regime, so it may only stand in where there is no regime
    evidence at all; risk_off must not be governed by the mixed row."""
    pb = _pb(_entry(regime="mixed", status="suspended", size_mult=0.0))
    d = evaluate_playbook_gate(pb, "orb", "long", "risk_off", True)

    assert (d.action, d.reason, d.governed) == ("allow", "no_playbook_entry", False)


def test_gate_prefers_the_exact_regime_over_every_fallback():
    pb = _pb(_entry(regime="risk_off", status="enabled", size_mult=1.0),
             _entry(regime="unknown", status="suspended", size_mult=0.0))
    d = evaluate_playbook_gate(pb, "orb", "long", "risk_off", True)

    assert d.action == "allow"
    assert d.audit_fields()["regime"] == "risk_off" and "playbook_fallback" not in d.audit_fields()


def test_gate_fallback_order_is_exact_then_mixed_then_unknown():
    """For an unknown live regime: its own entry, then `mixed`, then — nothing else,
    because `unknown` *was* the exact lookup."""
    exact = _pb(_entry(regime="unknown", status="enabled", size_mult=1.0),
                _entry(regime="mixed", status="suspended", size_mult=0.0))
    d = evaluate_playbook_gate(exact, "orb", "long", "unknown", True)
    assert d.action == "allow" and "playbook_fallback" not in d.audit_fields()

    only_mixed = _pb(_entry(regime="mixed", status="probation", size_mult=0.5))
    d2 = evaluate_playbook_gate(only_mixed, "orb", "long", "unknown", True)
    assert d2.reason == "playbook_probation"
    assert d2.audit_fields()["playbook_fallback"] == "mixed"


def test_gate_decision_is_a_strategy_fit_gate_decision():
    from core.strategy_fit_gate import GateDecision

    assert isinstance(evaluate_playbook_gate(_pb(), "orb", "long", "risk_on", True), GateDecision)
