"""v-playbook-desk-b2-2026-09-25 (Task 5): the playbook size step in the sizing chain.

`risk.manager.playbook_size_multiplier` is the whole rule: read
`signal.reasoning["strategy_size_multiplier"]` (written by the engine's playbook
hook, and only when `ENABLE_PLAYBOOK_GATE` is on), clamp it to [0.0, 1.5], and
return 1.0 — a no-op — for every off/absent/unusable case. It takes `enabled`
as an argument rather than reading Config so the rule is testable without the
main module's Config singleton.
"""
from __future__ import annotations

from risk.manager import (PLAYBOOK_SIZE_MULT_MAX, PLAYBOOK_SIZE_MULT_MIN,
                          playbook_size_multiplier, playbook_size_step)


def test_disabled_is_always_a_no_op():
    assert playbook_size_multiplier({"strategy_size_multiplier": 0.5}, False) == 1.0
    assert playbook_size_multiplier({"strategy_size_multiplier": 0.0}, False) == 1.0


def test_absent_multiplier_is_a_no_op():
    assert playbook_size_multiplier({}, True) == 1.0
    assert playbook_size_multiplier({"strategy_size_multiplier": None}, True) == 1.0
    assert playbook_size_multiplier(None, True) == 1.0
    assert playbook_size_multiplier("nonsense", True) == 1.0


def test_a_normal_multiplier_passes_through():
    assert playbook_size_multiplier({"strategy_size_multiplier": 0.5}, True) == 0.5
    assert playbook_size_multiplier({"strategy_size_multiplier": 1.25}, True) == 1.25
    assert playbook_size_multiplier({"strategy_size_multiplier": "0.5"}, True) == 0.5


def test_the_multiplier_is_clamped_both_ways():
    assert playbook_size_multiplier({"strategy_size_multiplier": 9.0}, True) == PLAYBOOK_SIZE_MULT_MAX
    assert playbook_size_multiplier({"strategy_size_multiplier": -3.0}, True) == PLAYBOOK_SIZE_MULT_MIN
    assert (PLAYBOOK_SIZE_MULT_MIN, PLAYBOOK_SIZE_MULT_MAX) == (0.0, 1.5)


def test_unusable_values_are_a_no_op():
    for bad in ("abc", float("nan"), float("inf"), [], {}):
        assert playbook_size_multiplier({"strategy_size_multiplier": bad}, True) == 1.0


def test_a_suspended_key_clamps_to_zero_not_to_a_negative():
    assert playbook_size_multiplier({"strategy_size_multiplier": 0.0}, True) == 0.0


# ────────────────────────────────────────────────────────────────────
# fix round 1, finding 1: a zero multiplier means *no trade*
# ────────────────────────────────────────────────────────────────────

def test_zero_multiplier_yields_no_trade():
    """A suspended (or size-less) playbook entry must not become a 1-share order.

    `playbook_size_step` is the sizing chain's step: with the gate on and a
    multiplier at or below zero the position is 0 shares, which
    `engine._execute_signal` reads as "no trade".
    """
    shares, mult = playbook_size_step(10, {"strategy_size_multiplier": 0.0}, True)
    assert (shares, mult) == (0, 0.0)


def test_a_negative_multiplier_also_yields_no_trade():
    shares, mult = playbook_size_step(10, {"strategy_size_multiplier": -2.0}, True)
    assert (shares, mult) == (0, 0.0)


def test_positive_multipliers_keep_the_chain_floor_of_one_share():
    assert playbook_size_step(10, {"strategy_size_multiplier": 0.5}, True) == (5, 0.5)
    assert playbook_size_step(3, {"strategy_size_multiplier": 0.5}, True) == (1, 0.5)
    assert playbook_size_step(10, {"strategy_size_multiplier": 1.25}, True) == (12, 1.25)
    # Clamped upper bound still applies through the step.
    assert playbook_size_step(10, {"strategy_size_multiplier": 9.0}, True) == (15, 1.5)


def test_the_step_is_inert_when_the_gate_is_off():
    assert playbook_size_step(10, {"strategy_size_multiplier": 0.0}, False) == (10, 1.0)
    assert playbook_size_step(10, {"strategy_size_multiplier": 0.5}, False) == (10, 1.0)


def test_the_step_is_inert_without_a_multiplier():
    assert playbook_size_step(7, {}, True) == (7, 1.0)
    assert playbook_size_step(7, None, True) == (7, 1.0)
    assert playbook_size_step(7, {"strategy_size_multiplier": "abc"}, True) == (7, 1.0)


def test_zero_shares_in_yields_zero_shares_out():
    assert playbook_size_step(0, {"strategy_size_multiplier": 0.5}, True) == (1, 0.5)
    assert playbook_size_step(0, {"strategy_size_multiplier": 0.0}, True) == (0, 0.0)
