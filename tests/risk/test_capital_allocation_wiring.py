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
    assert 0 < notional - pnl < 2000


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
    bp_cap = RISK.index("bp_cap = eff_buying_power * Config().MAX_POSITION_VALUE_BP_PCT")
    alloc = RISK.index("position_size = apply_allocation_cap(")
    assert bp_cap < alloc


# ── fix round 1: the cap is read-time, so no consumer may read the raw
#    attributes and every engine consumer goes through the accessors ──

def test_the_manager_exposes_the_two_read_time_accessors():
    assert "def effective_balance(self)" in RISK
    assert "def effective_buying_power(self)" in RISK


def test_no_risk_decision_reads_the_raw_attributes():
    """`account_balance` / `buying_power` are write targets only. Every
    remaining mention in risk/manager.py must be an assignment, the sync
    machinery, the sync log line, or the accessors themselves — never a
    consumer, or a write site would defeat the cap again."""
    allowed_fragments = (
        "self.account_balance = ",
        "self.buying_power = ",
        "self.schwab_buying_power = self.buying_power",
        "schwab_account_info.get('buying_power', self.buying_power)",
        "schwab_account_info.get('balance', self.account_balance)",
        "SyncedBalances(balance=self.account_balance,",
        "buying_power=self.buying_power,",
        "self.account_balance, self.buying_power, self.bot_open_notional,",
        'f"Buying Power=${self.buying_power:.2f}, Balance=${self.account_balance:.2f}")',
        "return effective_value(self.account_balance, self._allocation())",
        "return effective_value(self.buying_power, self._allocation())",
        "eff_balance = effective_value(self.account_balance, allocation)",
        "eff_buying_power = effective_value(self.buying_power, allocation)",
    )
    offenders = []
    for line in RISK.splitlines():
        if "self.account_balance" not in line and "self.buying_power" not in line:
            continue
        if any(fragment in line for fragment in allowed_fragments):
            continue
        offenders.append(line.strip())
    assert not offenders, offenders


def test_the_engine_reads_through_the_accessors_only():
    """Writes stay raw (three live write sites); reads must not be."""
    reads = [line.strip() for line in ENGINE.splitlines()
             if ("risk_manager.account_balance" in line
                 or "risk_manager.buying_power" in line)
             and "risk_manager.buying_power = " not in line
             and "risk_manager.account_balance = " not in line]
    assert not reads, reads
    assert "self.risk_manager.effective_balance()" in ENGINE
    assert "self.risk_manager.effective_buying_power()" in ENGINE


def test_the_emergency_stop_uses_the_effective_equity():
    """The -$1,395-instead-of-$150 class of bug: the 5% emergency stop is a
    risk decision and must see the allocation."""
    stop = ENGINE.index("if _emrg_pnl < 0 and abs(_emrg_pnl) >")
    assert "self.risk_manager.effective_balance() * 0.05" in ENGINE[stop:stop + 200]


def test_the_open_notional_helper_skips_non_finite_values():
    start = ENGINE.index("def _compute_bot_open_notional")
    block = ENGINE[start:start + 2900]
    assert "np.isfinite(qty)" in block and "np.isfinite(price)" in block


def test_the_allocation_step_only_short_circuits_when_it_reduced_the_size():
    assert "if position_size == 0 and _pre_allocation_size > 0:" in RISK


def test_the_allocation_step_is_skipped_when_the_flag_is_off():
    """The inert-by-default pin, at source level: the whole step sits behind
    `if allocation > 0`, and `coerce_allocation` maps every off/odd value to 0."""
    alloc = RISK.index("position_size = apply_allocation_cap(")
    assert "if allocation > 0:" in RISK[max(0, alloc - 400):alloc]


def test_modules_import():
    importlib.import_module("risk.allocation")
    importlib.import_module("risk.manager")
    importlib.import_module("core.engine")
