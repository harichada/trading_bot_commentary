"""v-go-live-5k-2026-09-28 (Task 1): source-level wiring for the allocation cap.

The engine is not constructible in tests (see
`tests/core/test_strategy_fit_wiring.py`), so the one engine touch point — the
per-cycle `risk_manager.bot_open_notional` assignment computed from bot-managed
positions — is asserted against the source text, next to the existing
`bot_daily_pnl` assignment it rides along with.
"""
from __future__ import annotations

import importlib
from pathlib import Path

ENGINE = Path("core/engine.py").read_text()
RISK = Path("risk/manager.py").read_text()


def test_engine_sets_bot_open_notional_next_to_bot_daily_pnl():
    pnl = ENGINE.index("self.risk_manager.bot_daily_pnl = self._compute_bot_daily_pnl()")
    notional = ENGINE.index(
        "self.risk_manager.bot_open_notional = self._compute_bot_open_notional()")
    # Same per-cycle block: the notional assignment follows the P&L one closely.
    assert 0 < notional - pnl < 1200


def test_engine_has_exactly_one_open_notional_helper():
    assert ENGINE.count("def _compute_bot_open_notional") == 1
    assert ENGINE.count("self.risk_manager.bot_open_notional = ") == 1


def test_the_open_notional_helper_filters_to_bot_managed_positions():
    start = ENGINE.index("def _compute_bot_open_notional")
    block = ENGINE[start:start + 2600]
    assert "managed_by_bot" in block
    assert "is_external" in block
    assert "is_manually_managed" in block
    assert "is_long_term" in block
    assert "HANDS_OFF_DENYLIST" in block


def test_risk_manager_reads_the_allocation_from_the_shared_helpers():
    assert "from risk.allocation import" in RISK
    assert "Config().BOT_CAPITAL_ALLOCATION" in RISK


def test_risk_manager_seeds_bot_open_notional():
    assert "self.bot_open_notional = 0.0" in RISK


def test_the_allocation_step_runs_after_the_buying_power_cap():
    bp_cap = RISK.index("bp_cap = self.buying_power * Config().MAX_POSITION_VALUE_BP_PCT")
    alloc = RISK.index("position_size = apply_allocation_cap(")
    assert bp_cap < alloc


def test_the_allocation_step_is_skipped_when_the_flag_is_off():
    """The inert-by-default pin, at source level: the whole step sits behind
    `if allocation > 0`, and `coerce_allocation` maps every off/odd value to 0."""
    alloc = RISK.index("position_size = apply_allocation_cap(")
    assert "if allocation > 0:" in RISK[max(0, alloc - 400):alloc]


def test_modules_import():
    importlib.import_module("risk.allocation")
    importlib.import_module("risk.manager")
    importlib.import_module("core.engine")
