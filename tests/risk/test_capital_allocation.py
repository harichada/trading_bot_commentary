"""v-go-live-5k-2026-09-28 (Task 1): the BOT_CAPITAL_ALLOCATION cap.

The allocation is a hard ceiling on the slice of a bigger brokerage account the
bot is allowed to work with. With `BOT_CAPITAL_ALLOCATION` at its default 0.0
every path here is byte-identical to the pre-allocation behaviour — that is the
property the `*_off_*` tests pin. With the allocation on:

  * `sync_with_schwab_data` caps `account_balance` and `buying_power` at the
    allocation and keeps the raw Schwab numbers on `schwab_equity` /
    `schwab_buying_power`;
  * sizing caps a new position's notional at `allocation - bot_open_notional`
    and returns 0 shares (`allocation_exhausted`) when there is not even one
    share of headroom;
  * the daily-loss circuit trips at `MAX_DAILY_LOSS x allocation`.

The pure rules live in `risk/allocation.py` so they are testable without the
main module's Config singleton; `risk.manager.Config` is monkeypatched with a
stub for the manager-level tests.
"""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from risk.allocation import (ALLOCATION_EXHAUSTED, allocation_capped_shares,
                             allocation_headroom, capped_balance_pair,
                             coerce_allocation, effective_value, finite_float)


# ────────────────────────────────────────────────────────────────────
# pure rules: risk/allocation.py
# ────────────────────────────────────────────────────────────────────

def test_finite_float_accepts_numbers_and_numeric_strings():
    assert finite_float(5000) == 5000.0
    assert finite_float(5000.5) == 5000.5
    assert finite_float("5000") == 5000.0


def test_finite_float_rejects_odd_inputs():
    for bad in (None, float("nan"), float("inf"), float("-inf"), "abc", [], {}, True):
        assert finite_float(bad) is None


def test_coerce_allocation_default_is_off():
    assert coerce_allocation(0.0) == 0.0
    assert coerce_allocation(None) == 0.0
    assert coerce_allocation("") == 0.0


def test_coerce_allocation_rejects_negative_and_unusable_values():
    assert coerce_allocation(-5000) == 0.0
    assert coerce_allocation(float("nan")) == 0.0
    assert coerce_allocation("nonsense") == 0.0


def test_coerce_allocation_passes_a_real_allocation_through():
    assert coerce_allocation(5000) == 5000.0
    assert coerce_allocation("5000") == 5000.0


def test_capped_balance_pair_caps_at_the_allocation():
    capped, raw, odd = capped_balance_pair(46500.0, 5000.0, prev_raw=0.0, prev_capped=0.0)
    assert (capped, raw, odd) == (5000.0, 46500.0, False)


def test_capped_balance_pair_leaves_a_smaller_balance_alone():
    capped, raw, odd = capped_balance_pair(1200.0, 5000.0, prev_raw=0.0, prev_capped=0.0)
    assert (capped, raw, odd) == (1200.0, 1200.0, False)


def test_capped_balance_pair_keeps_the_previous_values_on_odd_input():
    for bad in (None, float("nan"), "abc"):
        capped, raw, odd = capped_balance_pair(
            bad, 5000.0, prev_raw=46500.0, prev_capped=5000.0)
        assert (capped, raw, odd) == (5000.0, 46500.0, True)


def test_allocation_headroom_never_goes_negative():
    assert allocation_headroom(5000.0, 0.0) == 5000.0
    assert allocation_headroom(5000.0, 1200.0) == 3800.0
    assert allocation_headroom(5000.0, 5000.0) == 0.0
    assert allocation_headroom(5000.0, 9000.0) == 0.0


def test_allocation_headroom_treats_odd_open_notional_as_fully_deployed():
    """A broken notional must never *widen* the allowance."""
    for bad in (None, float("nan"), "abc"):
        assert allocation_headroom(5000.0, bad) == 0.0


def test_allocation_capped_shares_leaves_a_fitting_size_alone():
    assert allocation_capped_shares(10, 100.0, 5000.0) == 10


def test_allocation_capped_shares_trims_to_the_headroom():
    assert allocation_capped_shares(50, 100.0, 1250.0) == 12


def test_allocation_capped_shares_is_zero_below_one_share():
    assert allocation_capped_shares(10, 100.0, 99.99) == 0
    assert allocation_capped_shares(10, 100.0, 0.0) == 0


def test_allocation_capped_shares_never_raises_on_odd_inputs():
    assert allocation_capped_shares(10, 0.0, 5000.0) == 10
    assert allocation_capped_shares(10, float("nan"), 5000.0) == 10
    assert allocation_capped_shares(10, 100.0, float("nan")) == 0


def test_the_exhaustion_reason_is_the_documented_string():
    assert ALLOCATION_EXHAUSTED == "allocation_exhausted"


# ────────────────────────────────────────────────────────────────────
# test doubles
# ────────────────────────────────────────────────────────────────────

class FakeCommentary:
    def __init__(self):
        self.entries = []

    def add_commentary(self, commentary):
        self.entries.append(commentary)

    def reasons(self):
        out = []
        for entry in self.entries:
            data = getattr(entry, 'data', None) or {}
            if data.get('reason'):
                out.append(data['reason'])
        return out


class FakeConfig:
    """Only the knobs `risk/manager.py` reads during sizing / the circuit."""

    def __init__(self, **overrides):
        self.BOT_CAPITAL_ALLOCATION = 0.0
        self.RISK_PER_TRADE_PCT = 0.01
        self.SIZING_STRENGTH_WEIGHT = 0.5
        self.ATR_REWARD_RISK_RATIO = 2.0
        self.KELLY_FLOOR = 0.50
        self.ENABLE_PLAYBOOK_GATE = False
        self.ENABLE_MARKET_CONTEXT_SIZING = False
        self.LIVE_SIZE_MULTIPLIER = 1.0
        # fix round 1, minor 5: match Config().yaml (trading.max_position_value).
        self.MAX_POSITION_VALUE = 10000
        self.MAX_POSITION_VALUE_BP_PCT = 0.20
        self.MIN_POSITION_SIZE = 1
        self.ENABLE_BOT_ONLY_PNL_CIRCUIT = True
        self.MAX_DAILY_LOSS = 0.03
        self.MAX_CONSECUTIVE_LOSSES = 3
        self.DAILY_LOSS_ACTION = "pause_entries"
        self.DAILY_LOSS_BOT_MANAGED_ONLY = False
        for key, value in overrides.items():
            setattr(self, key, value)

    def STRATEGY_SIZE_MULTIPLIER(self, _name):
        return 1.0


@pytest.fixture
def cfg(monkeypatch):
    """Install a stub Config into risk.manager and hand it back for tuning."""
    stub = FakeConfig()
    monkeypatch.setattr('risk.manager.Config', lambda: stub)
    return stub


def make_manager(commentary=None, balance=46500.0):
    from risk.manager import RiskManagerWithCommentary
    return RiskManagerWithCommentary(balance, commentary or FakeCommentary())


def make_signal(price=100.0, stop_pct=0.03, strength=0.8, confidence=0.8):
    return SimpleNamespace(
        symbol='AAPL',
        strength=strength,
        confidence=confidence,
        entry_price=price,
        stop_loss=price * (1.0 - stop_pct),
        take_profit=price * (1.0 + 2 * stop_pct),
        reasoning={},
    )


def size(manager, signal, price):
    return asyncio.run(
        manager.calculate_position_size_with_commentary(signal, price))


SCHWAB = {'balance': 46500.0, 'buying_power': 93000.0, 'day_pnl': -12.0}


# ────────────────────────────────────────────────────────────────────
# allocation OFF — today's behaviour, byte for byte
# ────────────────────────────────────────────────────────────────────

def test_sync_off_leaves_the_schwab_balances_untouched(cfg):
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    assert mgr.account_balance == 46500.0
    assert mgr.buying_power == 93000.0
    assert mgr.schwab_daily_pnl == -12.0


def test_sync_off_still_records_the_raw_schwab_numbers(cfg):
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    assert mgr.schwab_equity == 46500.0
    assert mgr.schwab_buying_power == 93000.0


def test_sync_off_keeps_the_legacy_missing_key_fallbacks(cfg):
    mgr = make_manager()
    before_bp, before_bal = mgr.buying_power, mgr.account_balance
    mgr.sync_with_schwab_data({'day_pnl': 0})
    assert (mgr.buying_power, mgr.account_balance) == (before_bp, before_bal)


def test_bot_open_notional_defaults_to_zero(cfg):
    assert make_manager().bot_open_notional == 0.0


def test_sizing_off_is_unaffected_by_open_notional(cfg):
    """The regression pin: with the flag off the allocation step cannot fire."""
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    baseline = size(mgr, make_signal(), 100.0)
    assert baseline > 0

    mgr2 = make_manager()
    mgr2.sync_with_schwab_data(dict(SCHWAB))
    mgr2.bot_open_notional = 10_000_000.0
    assert size(mgr2, make_signal(), 100.0) == baseline


def test_circuit_off_uses_the_full_schwab_balance(cfg):
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    mgr.bot_daily_pnl = -200.0        # 3% of 46,500 is 1,395 — nowhere near
    allowed, reason = mgr.check_trading_allowed()
    assert allowed, reason


# ────────────────────────────────────────────────────────────────────
# allocation ON
# ────────────────────────────────────────────────────────────────────

def test_sync_on_caps_balance_and_buying_power(cfg):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    assert mgr.account_balance == 5000.0
    assert mgr.buying_power == 5000.0
    assert mgr.schwab_equity == 46500.0
    assert mgr.schwab_buying_power == 93000.0


def test_sync_on_logs_the_allocation_once(cfg, caplog):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    with caplog.at_level('INFO', logger='TradingBot'):
        mgr.sync_with_schwab_data(dict(SCHWAB))
    lines = [r for r in caplog.records if 'bot_capital_allocation' in r.getMessage()]
    assert len(lines) == 1


def test_sync_on_never_raises_on_odd_balances(cfg, caplog):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    with caplog.at_level('WARNING', logger='TradingBot'):
        mgr.sync_with_schwab_data({'balance': None, 'buying_power': float('nan'),
                                   'day_pnl': 3.0})
    # Previous (capped and raw) values survive the bad payload.
    assert mgr.account_balance == 5000.0
    assert mgr.buying_power == 5000.0
    assert mgr.schwab_equity == 46500.0
    assert mgr.schwab_buying_power == 93000.0
    assert mgr.schwab_daily_pnl == 3.0
    warnings = [r for r in caplog.records
                if 'allocation_sync_odd_balance' in r.getMessage()]
    assert len(warnings) == 1


def test_sync_on_keeps_a_smaller_real_balance(cfg):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    mgr.sync_with_schwab_data({'balance': 1200.0, 'buying_power': 900.0, 'day_pnl': 0})
    assert (mgr.account_balance, mgr.buying_power) == (1200.0, 900.0)


def test_circuit_on_uses_the_capped_balance(cfg):
    """3% of a $5,000 allocation is $150 — not 3% of the $46.5k account."""
    # v-daily-loss-pause-entries-2026-11-05: check_trading_allowed only
    # refuses on a daily-loss breach under the legacy `halt` action.
    cfg.DAILY_LOSS_ACTION = "halt"
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))

    mgr.bot_daily_pnl = -149.0
    allowed, _ = mgr.check_trading_allowed()
    assert allowed

    mgr.bot_daily_pnl = -150.0
    allowed, reason = mgr.check_trading_allowed()
    assert not allowed and 'Bot daily loss' in reason


def test_sizing_on_a_5k_allocation(cfg):
    """$5,000 allocation, 1% risk, $100 stock, 3% stop.

    risk/share $3.00 -> risk budget $50 -> 16 shares -> Kelly 0.7 -> 11
    shares ($1,100). The per-position BP cap (20% of the capped $5,000
    buying power = $1,000) then binds at 10 shares / $1,000 notional.
    """
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    assert size(mgr, make_signal(), 100.0) == 10


def test_sizing_on_caps_the_notional_at_the_remaining_headroom(cfg):
    """Only $450 of the allocation is left, so at most 4 shares of a $100 stock."""
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    cfg.MAX_POSITION_VALUE_BP_PCT = 1.0        # let the allocation be the binding cap
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    mgr.bot_open_notional = 4550.0
    assert size(mgr, make_signal(), 100.0) == 4


def test_sizing_on_reports_the_cap_in_commentary(cfg):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    cfg.MAX_POSITION_VALUE_BP_PCT = 1.0
    commentary = FakeCommentary()
    mgr = make_manager(commentary)
    mgr.sync_with_schwab_data(dict(SCHWAB))
    mgr.bot_open_notional = 4550.0
    size(mgr, make_signal(), 100.0)
    assert 'allocation_cap' in commentary.reasons()


def test_sizing_on_returns_zero_when_the_allocation_is_exhausted(cfg):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    commentary = FakeCommentary()
    mgr = make_manager(commentary)
    mgr.sync_with_schwab_data(dict(SCHWAB))
    mgr.bot_open_notional = 4950.0             # $50 left, one share costs $100
    assert size(mgr, make_signal(), 100.0) == 0
    assert ALLOCATION_EXHAUSTED in commentary.reasons()


def test_sizing_on_returns_zero_when_the_allocation_is_overshot(cfg):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    commentary = FakeCommentary()
    mgr = make_manager(commentary)
    mgr.sync_with_schwab_data(dict(SCHWAB))
    mgr.bot_open_notional = 7000.0
    assert size(mgr, make_signal(), 100.0) == 0
    assert ALLOCATION_EXHAUSTED in commentary.reasons()


def test_sizing_on_never_raises_on_an_odd_open_notional(cfg):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    mgr.bot_open_notional = float('nan')
    assert size(mgr, make_signal(), 100.0) == 0


def test_headroom_with_no_open_positions_is_the_whole_allocation(cfg):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    cfg.MAX_POSITION_VALUE_BP_PCT = 1.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    shares = size(mgr, make_signal(), 100.0)
    assert shares > 0 and shares * 100.0 <= 5000.0


# ────────────────────────────────────────────────────────────────────
# core/config.py: the flag and the two env overrides
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def real_config(monkeypatch):
    from core.config import Config as RealConfig
    for name in ('BOT_CAPITAL_ALLOCATION', 'MAX_DAILY_LOSS', 'RISK_PER_TRADE_PCT'):
        monkeypatch.delenv(name, raising=False)
    return RealConfig()


def test_bot_capital_allocation_defaults_to_off(real_config):
    assert real_config.BOT_CAPITAL_ALLOCATION == 0.0


def test_bot_capital_allocation_reads_the_env(monkeypatch, real_config):
    monkeypatch.setenv('BOT_CAPITAL_ALLOCATION', '5000')
    assert real_config.BOT_CAPITAL_ALLOCATION == 5000.0


def test_bot_capital_allocation_ignores_junk_and_negatives(monkeypatch, real_config):
    for bad in ('abc', '-5000', 'nan', ''):
        monkeypatch.setenv('BOT_CAPITAL_ALLOCATION', bad)
        assert real_config.BOT_CAPITAL_ALLOCATION == 0.0


def test_max_daily_loss_env_override(monkeypatch, real_config):
    yaml_value = real_config.MAX_DAILY_LOSS
    assert isinstance(yaml_value, (int, float))
    monkeypatch.setenv('MAX_DAILY_LOSS', '0.03')
    assert real_config.MAX_DAILY_LOSS == 0.03
    monkeypatch.setenv('MAX_DAILY_LOSS', 'abc')
    assert real_config.MAX_DAILY_LOSS == yaml_value


def test_max_daily_loss_refuses_a_non_fraction_env_value(monkeypatch, real_config):
    """The live-safety pin: `.env` on this machine carries MAX_DAILY_LOSS="500"
    (dollars). Honouring that as a fraction would set the circuit to 500x
    equity — i.e. switch it off. Only (0, 1] is a fraction."""
    yaml_value = real_config.MAX_DAILY_LOSS
    for bad in ('500', '1.5', '0', '-0.03', 'inf', ''):
        monkeypatch.setenv('MAX_DAILY_LOSS', bad)
        assert real_config.MAX_DAILY_LOSS == yaml_value, bad
    monkeypatch.setenv('MAX_DAILY_LOSS', '1.0')
    assert real_config.MAX_DAILY_LOSS == 1.0


def test_risk_per_trade_pct_env_override(monkeypatch, real_config):
    yaml_value = real_config.RISK_PER_TRADE_PCT
    monkeypatch.setenv('RISK_PER_TRADE_PCT', '0.01')
    assert real_config.RISK_PER_TRADE_PCT == 0.01
    for bad in ('nan', '5', '0', '-1', 'abc'):
        monkeypatch.setenv('RISK_PER_TRADE_PCT', bad)
        assert real_config.RISK_PER_TRADE_PCT == yaml_value, bad


def test_the_daily_loss_limit_on_a_5k_allocation_is_150(monkeypatch, real_config):
    monkeypatch.setenv('MAX_DAILY_LOSS', '0.03')
    monkeypatch.setenv('BOT_CAPITAL_ALLOCATION', '5000')
    limit = real_config.MAX_DAILY_LOSS * real_config.BOT_CAPITAL_ALLOCATION
    assert math.isclose(limit, 150.0)


# ════════════════════════════════════════════════════════════════════
# fix round 1 — the cap is applied at READ time
#
# Round 0 applied the cap only inside `sync_with_schwab_data`, and three live
# sites overwrite the raw attributes with whole-account Schwab numbers right
# afterwards (`core/engine.py` router before sizing, `_check_buying_power`, and
# the `/ws` position-cache refresh every 45s). Measured effect: an $18,600
# per-position cap and a -$1,395 daily stop instead of $1,000 / -$150.
#
# `effective_balance()` / `effective_buying_power()` are now the only things
# any consumer reads; the plain attributes stay as write targets, so a write
# site cannot defeat the cap.
# ════════════════════════════════════════════════════════════════════

def test_effective_value_is_the_raw_object_when_the_allocation_is_off():
    """Off must be byte-identical — including for values that would raise
    downstream today (a `None` buying power still raises the same TypeError)."""
    assert effective_value(46500.0, 0.0) == 46500.0
    assert effective_value(None, 0.0) is None
    assert effective_value("odd", 0.0) == "odd"
    nan = float('nan')
    assert effective_value(nan, 0.0) is nan


def test_effective_value_caps_when_the_allocation_is_on():
    assert effective_value(46500.0, 5000.0) == 5000.0
    assert effective_value(1200.0, 5000.0) == 1200.0
    assert effective_value(5000.0, 5000.0) == 5000.0


def test_effective_value_fails_closed_on_odd_input_under_a_cap():
    """Under a cap an unusable balance means "no capital", not "all of it"."""
    for bad in (None, float('nan'), float('inf'), 'abc', [], True):
        assert effective_value(bad, 5000.0) == 0.0


def test_effective_readers_track_the_flag(cfg):
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    assert mgr.effective_balance() == 46500.0
    assert mgr.effective_buying_power() == 93000.0

    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    assert mgr.effective_balance() == 5000.0
    assert mgr.effective_buying_power() == 5000.0


def overwrite_with_raw_schwab(mgr):
    """What all three live write sites do: raw whole-account numbers, straight
    onto the attributes, after the capped sync has run."""
    mgr.buying_power = SCHWAB['buying_power']
    mgr.account_balance = SCHWAB['balance']


def test_a_write_site_cannot_defeat_the_sizing_cap(cfg):
    """$100 stock, 3% stop, $5,000 allocation → 10 shares, even after the
    router / _check_buying_power / dashboard refresh clobber the attributes."""
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    overwrite_with_raw_schwab(mgr)
    assert size(mgr, make_signal(), 100.0) == 10


def test_a_write_site_cannot_defeat_the_daily_loss_stop(cfg):
    """3% of the $5,000 allocation is -$150, not 3% of the $46.5k account."""
    # v-daily-loss-pause-entries-2026-11-05: check_trading_allowed only
    # refuses on a daily-loss breach under the legacy `halt` action.
    cfg.DAILY_LOSS_ACTION = "halt"
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    overwrite_with_raw_schwab(mgr)

    mgr.bot_daily_pnl = -149.0
    assert mgr.check_trading_allowed()[0]

    mgr.bot_daily_pnl = -150.0
    allowed, reason = mgr.check_trading_allowed()
    assert not allowed and 'Bot daily loss' in reason


def test_a_write_site_cannot_defeat_the_notional_headroom(cfg):
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    commentary = FakeCommentary()
    mgr = make_manager(commentary)
    mgr.sync_with_schwab_data(dict(SCHWAB))
    overwrite_with_raw_schwab(mgr)
    mgr.bot_open_notional = 4950.0
    assert size(mgr, make_signal(), 100.0) == 0
    assert ALLOCATION_EXHAUSTED in commentary.reasons()


def test_the_raw_attributes_are_still_the_write_targets(cfg):
    """Write sites keep working — they just no longer decide anything."""
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    overwrite_with_raw_schwab(mgr)
    assert mgr.buying_power == 93000.0
    assert mgr.account_balance == 46500.0


def test_read_time_clamping_is_inert_with_the_flag_off(cfg):
    """The regression pin for round 1: with the flag off, a raw overwrite is
    honoured exactly as it is today (bigger account → bigger position)."""
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    overwrite_with_raw_schwab(mgr)
    # abs cap $10,000 vs bp_cap 20% x $93,000 = $18,600 → $10,000 binds.
    assert size(mgr, make_signal(), 100.0) == 100
    mgr.bot_daily_pnl = -1000.0
    assert mgr.check_trading_allowed()[0]      # -$1,395 is the off-flag limit


def test_the_insufficient_buying_power_gate_uses_the_effective_value(cfg):
    """A $50 allocation leaves the bot under the $100 floor even though the
    account has $93k of buying power."""
    cfg.BOT_CAPITAL_ALLOCATION = 50.0
    mgr = make_manager()
    mgr.sync_with_schwab_data(dict(SCHWAB))
    overwrite_with_raw_schwab(mgr)
    allowed, reason = mgr.check_trading_allowed()
    assert not allowed and 'Insufficient buying power' in reason


# ── fix round 1, minor 4: the min-size path is not bypassed ──────────

def test_min_position_size_still_applies_when_the_allocation_did_not_bind(cfg):
    """The allocation step must only short-circuit when it actually reduced the
    size — otherwise a size of 0 reaching it would skip `Position Too Small`."""
    cfg.BOT_CAPITAL_ALLOCATION = 5000.0
    cfg.MIN_POSITION_SIZE = 50                 # far above the 10 shares sized
    commentary = FakeCommentary()
    mgr = make_manager(commentary)
    mgr.sync_with_schwab_data(dict(SCHWAB))
    assert size(mgr, make_signal(), 100.0) == 0
    titles = [getattr(c, 'title', '') for c in commentary.entries]
    assert any('Position Too Small' in t for t in titles)
    assert ALLOCATION_EXHAUSTED not in commentary.reasons()


# ════════════════════════════════════════════════════════════════════
# fix round 1, important 2 — `_compute_bot_open_notional` must be NaN-safe
#
# A NaN quantity or price made the whole sum NaN, which `allocation_headroom`
# reads as "fully deployed" → every entry refused as allocation_exhausted.
# The engine is not constructible in tests, so the method is called unbound
# against a stub `self`.
# ════════════════════════════════════════════════════════════════════

def open_notional(positions):
    from core.engine import TradingEngineWithCommentary
    stub = SimpleNamespace(positions=positions)
    return TradingEngineWithCommentary._compute_bot_open_notional(stub)


def bot_position(quantity=10, price=100.0, **flags):
    return SimpleNamespace(quantity=quantity, current_price=price,
                           managed_by_bot=flags.pop('managed_by_bot', True),
                           is_external=flags.pop('is_external', False),
                           is_manually_managed=flags.pop('is_manually_managed', False),
                           is_long_term=flags.pop('is_long_term', False))


def test_open_notional_sums_bot_managed_market_value():
    assert open_notional({'AAPL': bot_position(10, 100.0),
                          'MSFT': bot_position(5, 200.0)}) == 2000.0


def test_open_notional_counts_a_short_as_deployed_capital():
    assert open_notional({'AAPL': bot_position(-10, 100.0)}) == 1000.0


def test_open_notional_is_nan_safe():
    """The bug: one NaN made the whole sum NaN, so headroom went to 0 and every
    entry was refused."""
    for bad in (float('nan'), float('inf'), float('-inf')):
        assert open_notional({'AAPL': bot_position(10, bad)}) == 0.0
        assert open_notional({'AAPL': bot_position(bad, 100.0)}) == 0.0
        mixed = open_notional({'AAPL': bot_position(10, bad),
                               'MSFT': bot_position(5, 200.0)})
        assert mixed == 1000.0, bad


def test_open_notional_skips_positions_the_bot_does_not_manage():
    assert open_notional({'AAPL': bot_position(managed_by_bot=False)}) == 0.0
    assert open_notional({'AAPL': bot_position(is_external=True)}) == 0.0
    assert open_notional({'AAPL': bot_position(is_manually_managed=True)}) == 0.0
    assert open_notional({'AAPL': bot_position(is_long_term=True)}) == 0.0


def test_open_notional_skips_the_hands_off_denylist():
    from core.config import Config as RealConfig
    denylist = RealConfig().HANDS_OFF_DENYLIST
    assert denylist, "expected a non-empty hands-off denylist to test against"
    symbol = sorted(denylist)[0]
    assert open_notional({symbol: bot_position(10, 100.0)}) == 0.0


def test_open_notional_never_raises_on_a_malformed_position():
    assert open_notional({'AAPL': SimpleNamespace()}) == 0.0
    assert open_notional({'AAPL': bot_position('abc', 100.0)}) == 0.0
    assert open_notional({}) == 0.0
