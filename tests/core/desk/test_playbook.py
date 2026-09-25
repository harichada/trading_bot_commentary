"""v-playbook-desk-b2-2026-09-25 (Task 3): AdaptivePlaybook rules, clamp and store.

Every transition of `update_entry`, the size-multiplier clamp and the serialization
round-trip (including the never-raises load contract). `carry_forward` and
`evaluate_playbook_gate` are in `test_playbook_gate.py`.
"""
import json

import pytest

from core.desk.playbook import (
    DEFAULT_SIZE_MULT,
    PROBATION_SIZE_MULT,
    RECENT_WINDOW,
    SIZE_MULT_MAX,
    SIZE_MULT_MIN,
    UPSIZE_SIZE_MULT,
    Playbook,
    PlaybookEntry,
    ResolvedTrade,
    clamp_size_mult,
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
    no second, hidden truncation."""
    from core.desk.playbook import UPSIZE_MIN_N

    # r1 finding 2: the window is 60, so the 40-trade upsize floor is reachable.
    assert RECENT_WINDOW == 60 and UPSIZE_MIN_N == 40 and UPSIZE_MIN_N < RECENT_WINDOW
    e = update_entry(None, _trades([0.2] * 45), "PASS", NOW)
    assert e.n_recent == 45


def test_a_full_window_of_winners_with_a_pass_verdict_upsizes():
    """r1 finding 2: a 60-trade window at exp_R 0.2 and a PASS row reaches 1.25."""
    e = update_entry(None, _trades([0.2] * RECENT_WINDOW), "PASS", NOW)

    assert e.n_recent == 60 and e.status == "enabled"
    assert e.size_mult == UPSIZE_SIZE_MULT == 1.25 and e.reason == "enabled_upsize_fit_confirmed"


def test_newest_last_ordering_does_not_change_the_measurement():
    """Order only matters for the caller's slice; mean and t are order-free."""
    a = update_entry(None, _trades([-0.5, -0.3] * 15), "PASS", NOW)
    b = update_entry(None, _trades([-0.3, -0.5] * 15), "PASS", NOW)

    assert a.n_recent == b.n_recent == 30
    assert a.exp_r_recent == b.exp_r_recent == pytest.approx(-0.4)
    assert a.status == b.status == "suspended"


def test_t_stat_is_zero_below_two_trades():
    assert update_entry(None, _trades([-1.0]), None, NOW).t_stat_recent == 0.0


def test_twelve_exact_stop_outs_suspend_on_an_infinite_t_stat():
    """r1 finding 4: sd == 0 with a non-zero mean is the *most* certain result there is
    (every trade the same loss), so t is -inf and the key suspends."""
    e = update_entry(None, _trades([-1.0] * 12), None, NOW)

    assert e.t_stat_recent == float("-inf")
    assert e.status == "suspended" and e.size_mult == 0.0


def test_twelve_exact_winners_have_a_positive_infinite_t_stat():
    e = update_entry(None, _trades([1.0] * 12), None, NOW)
    assert e.t_stat_recent == float("inf") and e.status == "enabled"


def test_a_zero_mean_zero_dispersion_sample_has_a_zero_t_stat():
    e = update_entry(None, _trades([0.0] * 12), None, NOW)
    assert e.t_stat_recent == 0.0 and e.status == "enabled"


def test_a_non_finite_t_stat_serializes_as_null_and_reads_back_as_zero():
    """JSON has no Infinity: `null` is the honest wire value, and the nightly recompute
    restores the real number."""
    e = update_entry(None, _trades([-1.0] * 12), None, NOW)
    d = e.to_dict()

    assert d["t_stat_recent"] is None
    assert json.loads(json.dumps(d)) == d
    assert PlaybookEntry.from_dict(d).t_stat_recent == 0.0


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
