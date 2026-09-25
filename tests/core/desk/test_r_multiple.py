"""v-playbook-desk-b2-2026-09-25 (Task 3b): `core/desk/r_multiple.derive_pnl_r`.

Pure function, no I/O — the same `pnl_r` derivation both live-trade readers
(`research/strategy_hunt/playbook_live.py`, `data_providers/db_logger.py`) must use so a
trade on the migrated `bot_trades` schema (has `pnl_r`) and the same trade on the
un-migrated schema (has `stop_loss` instead) read the same R.
"""
from __future__ import annotations

import math

from core.desk.r_multiple import derive_pnl_r


def test_stored_pnl_r_wins_when_present_and_finite():
    assert derive_pnl_r(1.25, 999.0, 1.0, 2.0, 3.0) == 1.25


def test_stored_pnl_r_zero_is_still_used_not_treated_as_missing():
    assert derive_pnl_r(0.0, 999.0, 1.0, 2.0, 3.0) == 0.0


def test_derives_from_stop_distance_when_pnl_r_column_absent():
    # long: entry 100, stop 98 -> risk/share 2.0; qty 10 -> denom 20; pnl 40 -> R 2.0
    assert derive_pnl_r(None, 40.0, 100.0, 98.0, 10.0) == 2.0


def test_sign_follows_pnl_not_side():
    # a losing trade: pnl negative, denominator always positive -> R negative
    assert derive_pnl_r(None, -20.0, 100.0, 98.0, 10.0) == -1.0


def test_short_side_stop_above_entry_still_uses_abs_distance():
    # short: entry 50, stop 52 -> risk/share 2.0; qty 5 -> denom 10; pnl 15 -> R 1.5
    assert derive_pnl_r(None, 15.0, 50.0, 52.0, 5.0) == 1.5


def test_both_schemas_yield_the_same_r_for_the_same_trade():
    migrated = derive_pnl_r(2.0, 999.0, 999.0, 999.0, 999.0)
    un_migrated = derive_pnl_r(None, 40.0, 100.0, 98.0, 10.0)
    assert migrated == un_migrated == 2.0


def test_degenerate_stop_equal_to_entry_is_none():
    assert derive_pnl_r(None, 40.0, 100.0, 100.0, 10.0) is None


def test_missing_stop_loss_is_none():
    assert derive_pnl_r(None, 40.0, 100.0, None, 10.0) is None


def test_missing_entry_price_is_none():
    assert derive_pnl_r(None, 40.0, None, 98.0, 10.0) is None


def test_missing_quantity_is_none():
    assert derive_pnl_r(None, 40.0, 100.0, 98.0, None) is None


def test_missing_pnl_is_none():
    assert derive_pnl_r(None, None, 100.0, 98.0, 10.0) is None


def test_zero_quantity_is_none():
    assert derive_pnl_r(None, 40.0, 100.0, 98.0, 0.0) is None


def test_nan_pnl_r_falls_back_to_derivation():
    assert derive_pnl_r(float("nan"), 40.0, 100.0, 98.0, 10.0) == 2.0


def test_infinite_pnl_r_falls_back_to_derivation():
    assert derive_pnl_r(float("inf"), 40.0, 100.0, 98.0, 10.0) == 2.0


def test_non_numeric_pnl_r_falls_back_to_derivation():
    assert derive_pnl_r("not-a-number", 40.0, 100.0, 98.0, 10.0) == 2.0


def test_non_finite_fallback_inputs_are_none():
    assert derive_pnl_r(None, float("nan"), 100.0, 98.0, 10.0) is None
    assert derive_pnl_r(None, 40.0, float("inf"), 98.0, 10.0) is None


def test_result_is_a_plain_finite_float():
    r = derive_pnl_r(None, 40.0, 100.0, 98.0, 10.0)
    assert isinstance(r, float) and math.isfinite(r)
