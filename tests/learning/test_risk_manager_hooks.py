"""v-learning-loop-2026-10-22: the sizing-chain hooks in risk/manager.py.

Same stub-Config harness as tests/risk/test_fit_lane_allowlist_sizing.py: the real
sizing chain, flags toggled on a FakeConfig.
"""
import asyncio
from types import SimpleNamespace

from core.models import SignalType
from tests.risk.test_fit_lane_allowlist_sizing import FakeCommentary, FakeConfig, make_manager

SCHWAB = {'balance': 100000.0, 'buying_power': 200000.0, 'day_pnl': 0.0}


def _sig(stop, side=SignalType.BUY, price=100.0):
    return SimpleNamespace(symbol="NVDA", signal_type=side, stop_loss=stop, take_profit=price * 1.1,
                           confidence=0.9, strength=0.9, reasoning={"strategy": "day_trade_momentum"})


def _size(monkeypatch, sig, price=100.0, **over):
    stub = FakeConfig(**{"RISK_PER_TRADE_PCT": 0.01, "MAX_POSITION_VALUE": 1e9, "MAX_POSITION_VALUE_BP_PCT": 1.0, **over})
    monkeypatch.setattr('risk.manager.Config', lambda: stub)
    m = make_manager(FakeCommentary(), balance=100000.0)
    m.sync_with_schwab_data(dict(SCHWAB))
    return asyncio.run(m.calculate_position_size_with_commentary(sig, price))


def test_flags_off_is_byte_identical(monkeypatch):
    base = _size(monkeypatch, _sig(98.0))
    assert base > 75   # $1,000 risk budget / $2 -> far more than the 75-share 1R cap
    assert _size(monkeypatch, _sig(98.0), ENABLE_RISK_CAP_PER_TRADE=False, REQUIRE_STOP_R_SHADOW=False) == base


def test_risk_cap_clamps_to_150_dollars(monkeypatch):
    n = _size(monkeypatch, _sig(98.0), ENABLE_RISK_CAP_PER_TRADE=True, RISK_CAP_PER_TRADE_USD=150.0)
    assert n == 75 and n * 2.0 <= 150


def test_risk_cap_zero_when_one_share_exceeds(monkeypatch):
    assert _size(monkeypatch, _sig(800.0, price=1000.0), price=1000.0,
                 ENABLE_RISK_CAP_PER_TRADE=True, RISK_CAP_PER_TRADE_USD=150.0) == 0


def test_require_stop_enforce_rejects_wrong_side_stop(monkeypatch):
    assert _size(monkeypatch, _sig(101.0), REQUIRE_STOP_R_ENFORCE=True) == 0
    assert _size(monkeypatch, _sig(99.0, side=SignalType.SELL), REQUIRE_STOP_R_ENFORCE=True) == 0
    assert _size(monkeypatch, _sig(101.0, side=SignalType.SELL), REQUIRE_STOP_R_ENFORCE=True) > 0


def test_require_stop_shadow_never_blocks(monkeypatch):
    off = _size(monkeypatch, _sig(101.0))
    assert _size(monkeypatch, _sig(101.0), REQUIRE_STOP_R_SHADOW=True) == off
