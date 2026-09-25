"""v-playbook-desk-b2-2026-09-25 (Task 3): AdaptivePlaybook rules, store and gate.

Every transition of `update_entry`, the size-multiplier clamp, the
serialization round-trip (including the never-raises load contract) and
`evaluate_playbook_gate` incl. the unknown-regime fallback.
"""
import json

import pytest

from core.desk.playbook import (
    DEFAULT_SIZE_MULT,
    PROBATION_SIZE_MULT,
    SIZE_MULT_MAX,
    SIZE_MULT_MIN,
    UPSIZE_SIZE_MULT,
    Playbook,
    PlaybookEntry,
    ResolvedTrade,
    clamp_size_mult,
    evaluate_playbook_gate,
    load_playbook,
    playbook_path,
    save_playbook,
    update_entry,
)

NOW = "2026-09-25T20:00:00"


def _trades(rs, family="orb", side="long", regime="risk_on"):
    """ResolvedTrade list from R-multiples, newest last (exit_ts increasing)."""
    return [
        ResolvedTrade(family=family, side=side, regime=regime, r=float(r),
                      exit_ts=f"2026-09-{(i % 27) + 1:02d}T15:00:00", source="shadow")
        for i, r in enumerate(rs)
    ]


def _entry(**kw):
    base = dict(family="orb", side="long", regime="risk_on", status="enabled", size_mult=1.0,
                exp_r_recent=0.2, t_stat_recent=2.0, n_recent=50, updated_at=NOW, reason="enabled_normal")
    base.update(kw)
    return PlaybookEntry(**base)


# ────────────────────────── update_entry transitions ──────────────────────────

def test_suspends_on_a_significant_negative_sample():
    recent = _trades([-1.0, -0.8] * 6)          # n=12, exp_r=-0.9, t far below -1
    e = update_entry(None, recent, "FAIL", NOW)

    assert e.status == "suspended"
    assert e.size_mult == 0.0
    assert e.n_recent == 12
    assert e.exp_r_recent == pytest.approx(-0.9)
    assert e.t_stat_recent < -1
    assert "suspend" in e.reason
    assert (e.family, e.side, e.regime) == ("orb", "long", "risk_on")
    assert e.updated_at == NOW


def test_probation_when_negative_but_not_significant():
    recent = _trades([-2.0, 1.9] * 6)           # n=12, exp_r=-0.05, t ≈ -0.09
    e = update_entry(None, recent, "WATCH", NOW)

    assert e.status == "probation"
    assert e.size_mult == PROBATION_SIZE_MULT == 0.5
    assert e.t_stat_recent > -1
    assert "negative" in e.reason


def test_probation_on_a_small_sample_even_when_positive():
    e = update_entry(None, _trades([0.5] * 5), "PASS", NOW)

    assert e.status == "probation" and e.size_mult == 0.5
    assert e.n_recent == 5 and "sample" in e.reason


def test_probation_with_no_recent_trades_uses_the_previous_key():
    prev = _entry(status="enabled")
    e = update_entry(prev, [], None, NOW)

    assert (e.family, e.side, e.regime) == (prev.family, prev.side, prev.regime)
    assert e.status == "probation" and e.n_recent == 0
    assert e.exp_r_recent == 0.0 and e.t_stat_recent == 0.0


def test_update_entry_without_a_key_is_an_explicit_error():
    with pytest.raises(ValueError):
        update_entry(None, [], "PASS", NOW)


@pytest.mark.parametrize("verdict", ["PASS", "WATCH"])
def test_upsizes_a_large_positive_sample_with_a_fit_verdict(verdict):
    e = update_entry(None, _trades([0.2] * 40), verdict, NOW)

    assert e.status == "enabled"
    assert e.size_mult == UPSIZE_SIZE_MULT == 1.25
    assert e.n_recent == 40 and e.exp_r_recent == pytest.approx(0.2)
    assert "upsize" in e.reason


@pytest.mark.parametrize("verdict", ["FAIL", None, "SOMETHING_ELSE"])
def test_no_upsize_without_a_pass_or_watch_verdict(verdict):
    e = update_entry(None, _trades([0.2] * 40), verdict, NOW)

    assert e.status == "enabled" and e.size_mult == DEFAULT_SIZE_MULT == 1.0
    assert e.reason == "enabled_normal"


def test_no_upsize_below_the_expectancy_floor():
    e = update_entry(None, _trades([0.14] * 40), "PASS", NOW)
    assert e.status == "enabled" and e.size_mult == 1.0


def test_no_upsize_below_the_sample_floor():
    e = update_entry(None, _trades([0.2] * 39), "PASS", NOW)
    assert e.status == "enabled" and e.size_mult == 1.0


def test_enabled_normal_on_a_flat_ten_trade_sample():
    e = update_entry(None, _trades([0.0] * 10), None, NOW)
    assert e.status == "enabled" and e.size_mult == 1.0 and e.reason == "enabled_normal"


def test_update_entry_measures_exactly_what_it_is_handed():
    """`update_entry` is pure: the caller (the nightly updater) owns the
    `RECENT_WINDOW` slice, so the rules must measure the whole list given —
    no second, hidden truncation. See RECENT_WINDOW's note about the
    30-trade window vs the 40-trade upsize floor."""
    from core.desk.playbook import RECENT_WINDOW, UPSIZE_MIN_N

    assert RECENT_WINDOW == 30 and UPSIZE_MIN_N == 40
    e = update_entry(None, _trades([0.2] * 45), "PASS", NOW)
    assert e.n_recent == 45


def test_newest_last_ordering_does_not_change_the_measurement():
    """Order only matters for the caller's slice; mean and t are order-free."""
    a = update_entry(None, _trades([-0.5, -0.3] * 15), "PASS", NOW)
    b = update_entry(None, _trades([-0.3, -0.5] * 15), "PASS", NOW)

    assert a.n_recent == b.n_recent == 30
    assert a.exp_r_recent == b.exp_r_recent == pytest.approx(-0.4)
    assert a.status == b.status == "suspended"


def test_t_stat_is_zero_below_two_trades_and_without_dispersion():
    assert update_entry(None, _trades([-1.0]), None, NOW).t_stat_recent == 0.0
    assert update_entry(None, _trades([-1.0] * 12), None, NOW).t_stat_recent == 0.0


def test_a_zero_dispersion_loser_is_probation_not_suspended():
    """t_stat 0 means "no measurement", so the suspension rule can't fire on it."""
    e = update_entry(None, _trades([-1.0] * 12), None, NOW)
    assert e.status == "probation"


def test_previous_entry_is_never_mutated():
    prev = _entry()
    before = prev.to_dict()
    update_entry(prev, _trades([-1.0, -0.8] * 6), "FAIL", NOW)
    assert prev.to_dict() == before


# ────────────────────────── clamp ──────────────────────────

def test_clamp_size_mult_bounds():
    assert (SIZE_MULT_MIN, SIZE_MULT_MAX) == (0.0, 1.5)
    assert clamp_size_mult(-3.0) == 0.0
    assert clamp_size_mult(9.0) == 1.5
    assert clamp_size_mult(0.75) == 0.75
    assert clamp_size_mult(float("nan")) == 0.0
    assert clamp_size_mult("bogus") == 0.0


def test_from_dict_clamps_a_hand_edited_size_mult():
    e = PlaybookEntry.from_dict({**_entry().to_dict(), "size_mult": 99.0})
    assert e.size_mult == 1.5


# ────────────────────────── serialization ──────────────────────────

def test_playbook_round_trip_uses_pipe_joined_keys():
    pb = Playbook(entries={}, updated_at=NOW).with_entry(_entry()).with_entry(
        _entry(family="dt_momentum", side="short", regime="mixed", status="suspended", size_mult=0.0))

    d = pb.to_dict()
    assert set(d["entries"]) == {"orb|long|risk_on", "dt_momentum|short|mixed"}
    assert json.loads(json.dumps(d)) == d           # strict-JSON safe

    back = Playbook.from_dict(d)
    assert back.updated_at == NOW
    assert back.entries == pb.entries
    assert back.get("dt_momentum", "short", "mixed").status == "suspended"


def test_entry_from_dict_ignores_unknown_keys_and_non_finite_floats():
    e = PlaybookEntry.from_dict({**_entry().to_dict(), "surprise": 1, "exp_r_recent": float("inf")})
    assert e.exp_r_recent == 0.0
    assert e.to_dict()["exp_r_recent"] == 0.0


def test_with_entry_returns_a_new_playbook():
    pb = Playbook(entries={}, updated_at=NOW)
    pb2 = pb.with_entry(_entry())
    assert pb.entries == {} and len(pb2.entries) == 1


def test_save_and_load_round_trip_leaves_no_tmp_file(tmp_path):
    pb = Playbook(entries={}, updated_at=NOW).with_entry(_entry())
    p = playbook_path(tmp_path)
    save_playbook(pb, p)

    assert p.exists()
    assert list(tmp_path.glob("*.tmp")) == [] and list(tmp_path.glob("*.tmp*")) == []
    assert load_playbook(p).entries == pb.entries


def test_load_playbook_never_raises(tmp_path):
    missing = tmp_path / "nope" / "playbook.json"
    assert load_playbook(missing).entries == {}

    bad = tmp_path / "playbook.json"
    bad.write_text("{not json")
    assert load_playbook(bad).entries == {}

    wrong_type = tmp_path / "list.json"
    wrong_type.write_text("[1, 2, 3]")
    assert load_playbook(wrong_type).entries == {}

    bad_entry = tmp_path / "bad_entry.json"
    bad_entry.write_text(json.dumps({"updated_at": NOW, "entries": {"orb|long": "nope"}}))
    assert load_playbook(bad_entry).entries == {}


def test_playbook_path_default_is_the_desk_dir():
    assert playbook_path().as_posix() == "data/desk/playbook.json"


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

    assert evaluate_playbook_gate(pb, "orb", "long", None, True).action == "skip"


def test_gate_unknown_regime_prefers_its_own_entry_over_mixed():
    pb = _pb(_entry(regime="unknown", status="enabled", size_mult=1.0),
             _entry(regime="mixed", status="suspended", size_mult=0.0))
    d = evaluate_playbook_gate(pb, "orb", "long", "unknown", True)

    assert d.action == "allow" and d.audit_fields()["regime"] == "unknown"


def test_gate_does_not_fall_back_for_a_known_regime():
    pb = _pb(_entry(regime="mixed", status="suspended", size_mult=0.0))
    d = evaluate_playbook_gate(pb, "orb", "long", "risk_off", True)

    assert (d.action, d.reason, d.governed) == ("allow", "no_playbook_entry", False)


def test_gate_decision_is_a_strategy_fit_gate_decision():
    from core.strategy_fit_gate import GateDecision

    assert isinstance(evaluate_playbook_gate(_pb(), "orb", "long", "risk_on", True), GateDecision)
