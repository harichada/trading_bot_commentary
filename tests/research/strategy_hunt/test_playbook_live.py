"""v-playbook-desk-b2-2026-09-25 (Task 3b): the live tape on the un-migrated `bot_trades`
schema — no `pnl_r`, `setup_type`, `initial_stop` or `initial_risk_per_share` column
(those come from `scripts/migrate_ledger_integrity.py`, gated by `LEDGER_INTEGRITY`,
default off).

`test_playbook_update.py` already covers the migrated schema (rows carry `pnl_r`
directly) end to end through `resolved_from_live`/`live_resolved_trades`; this file
covers the un-migrated schema (rows carry `pnl`/`entry_price`/`stop_loss`/`quantity`
instead) and the cross-schema equivalence and SQL-shape rulings from the task brief.
"""
from __future__ import annotations

import pandas as pd

from research.strategy_hunt.playbook_live import (
    LIVE_TRADES_SQL,
    resolved_from_live,
)


def _un_migrated_row(**overrides):
    """One `bot_trades` row shaped like the live, un-migrated schema: no `pnl_r`
    column at all (so it is simply absent from the dict, exactly as
    `DataFrame.to_dict("records")` would omit a column pandas never read)."""
    row = {
        "symbol": "AAA", "side": "long", "strategy": "orb_contraction_rvol",
        "entry_time": "2026-09-01T10:00:00", "exit_time": "2026-09-01T11:00:00",
        "pnl": 40.0, "entry_price": 100.0, "stop_loss": 98.0, "quantity": 10.0,
        "mode": "live",
    }
    row.update(overrides)
    return row


def test_un_migrated_schema_derives_r_from_stop_distance():
    trades = pd.DataFrame([_un_migrated_row()])
    out = resolved_from_live(trades)
    assert [t.r for t in out] == [2.0]  # 40 / (abs(100-98) * 10)


def test_un_migrated_schema_degenerate_stop_is_dropped():
    trades = pd.DataFrame([_un_migrated_row(stop_loss=100.0)])  # stop == entry
    assert resolved_from_live(trades) == []


def test_un_migrated_schema_missing_stop_loss_column_is_dropped():
    row = _un_migrated_row()
    del row["stop_loss"]
    trades = pd.DataFrame([row])
    assert resolved_from_live(trades) == []


def test_migrated_and_un_migrated_schema_yield_the_same_r_for_the_same_trade():
    migrated = pd.DataFrame([{
        "symbol": "AAA", "side": "long", "strategy": "orb_contraction_rvol",
        "entry_time": "2026-09-01T10:00:00", "exit_time": "2026-09-01T11:00:00",
        "pnl_r": 2.0,
    }])
    un_migrated = pd.DataFrame([_un_migrated_row()])  # pnl 40 / (2 * 10) = 2.0 too

    r_migrated = [t.r for t in resolved_from_live(migrated)]
    r_un_migrated = [t.r for t in resolved_from_live(un_migrated)]
    assert r_migrated == r_un_migrated == [2.0]


def test_stored_pnl_r_wins_over_derivation_when_both_present():
    trades = pd.DataFrame([_un_migrated_row(pnl_r=9.0)])  # would derive to 2.0
    out = resolved_from_live(trades)
    assert [t.r for t in out] == [9.0]


# ── SQL shape (Task 3b ruling) ──

def test_live_trades_sql_never_names_pnl_r():
    """Regression: the old query named `pnl_r` directly and errored with
    `column "pnl_r" does not exist` on the un-migrated schema. `SELECT *` (or an
    introspected column list) must never repeat that mistake."""
    assert "pnl_r" not in LIVE_TRADES_SQL


def test_live_trades_sql_never_names_any_migration_only_column():
    for col in ("setup_type", "initial_stop", "initial_risk_per_share"):
        assert col not in LIVE_TRADES_SQL


def test_live_trades_sql_filters_to_live_mode():
    assert "mode = 'live'" in LIVE_TRADES_SQL


def test_live_trades_sql_still_bounds_by_exit_time():
    assert "exit_time >= :since" in LIVE_TRADES_SQL
