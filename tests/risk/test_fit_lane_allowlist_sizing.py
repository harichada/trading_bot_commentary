"""v-go-live-5k-2026-09-28 (Task 2, fix round 1): what an allowlisted sleeve signal actually sizes to.

An allowlisted entry has no walk-forward score behind it — for an unmeasured symbol the fit row
is absent and its score is NaN, which `_clamp` floors at strength 0.2, and `risk/manager.py` then
caps Kelly at 0.20 for anything under strength 0.25. That is a score-derived haircut applied to a
signal that has no score, so the sleeve must instead size on the risk-per-trade path (1% of the
allocation against the ATR stop) at a neutral strength, `FIT_LANE_ALLOWLIST_STRENGTH`.

The stub Config mirrors `tests/risk/test_capital_allocation.py`'s (Task 1's harness, same knobs
and defaults) rather than importing it: that file is owned by another task and still moving, and a
sizing assertion must not turn red because a helper there was renamed. The code under test is the
real `risk/manager.py` chain, including Task 1's allocation cap (c3ee3c2).

Go-live values: allocation 5,000, RISK_PER_TRADE_PCT 1%, MAX_POSITION_VALUE_BP_PCT 0.20,
a $100 stock with a 3% stop.
"""
from __future__ import annotations

import asyncio

import pytest

SCHWAB = {'balance': 46500.0, 'buying_power': 93000.0, 'day_pnl': -12.0}


class FakeCommentary:
    def __init__(self):
        self.entries = []

    def add_commentary(self, commentary):
        self.entries.append(commentary)


class FakeConfig:
    """Only the knobs `risk/manager.py` reads while sizing."""

    def __init__(self, **overrides):
        self.BOT_CAPITAL_ALLOCATION = 0.0
        self.RISK_PER_TRADE_PCT = 0.01
        self.SIZING_STRENGTH_WEIGHT = 0.5
        self.ATR_REWARD_RISK_RATIO = 2.0
        self.KELLY_FLOOR = 0.50
        self.ENABLE_PLAYBOOK_GATE = False
        self.ENABLE_MARKET_CONTEXT_SIZING = False
        self.LIVE_SIZE_MULTIPLIER = 1.0
        self.MAX_POSITION_VALUE = 10000          # Config().yaml trading.max_position_value
        self.MAX_POSITION_VALUE_BP_PCT = 0.20
        self.MIN_POSITION_SIZE = 1
        self.ENABLE_BOT_ONLY_PNL_CIRCUIT = True
        self.MAX_DAILY_LOSS = 0.03
        self.MAX_CONSECUTIVE_LOSSES = 3
        for key, value in overrides.items():
            setattr(self, key, value)

    def STRATEGY_SIZE_MULTIPLIER(self, _name):
        return 1.0


def make_manager(commentary=None, balance=46500.0):
    from risk.manager import RiskManagerWithCommentary
    return RiskManagerWithCommentary(balance, commentary or FakeCommentary())

PRICE = 100.0
STOP = 97.0                 # 3% stop -> $3 risk per share
ALLOCATION = 5000.0
# 1% of the 5,000 allocation is $50 of risk / $3 a share = 16 shares before any Kelly haircut;
# the buying-power cap (5,000 x 0.20 = $1,000) is what binds a full-Kelly signal at 10 shares.
RISK_PER_TRADE = 0.01
BP_PCT = 0.20


def _cfg(monkeypatch, **over):
    settings = {"BOT_CAPITAL_ALLOCATION": ALLOCATION, "RISK_PER_TRADE_PCT": RISK_PER_TRADE,
                "MAX_POSITION_VALUE_BP_PCT": BP_PCT}
    settings.update(over)
    stub = FakeConfig(**settings)
    monkeypatch.setattr('risk.manager.Config', lambda: stub)
    return stub


def _fit_signal(admission, verdict="", score=float("nan"), side="long"):
    """A real fit-lane signal, built by the code under test rather than hand-rolled, so the
    strength/confidence the risk manager reads are the ones the lane actually emits."""
    import pandas as pd
    from strategies.families.models import Entry
    from strategies.fit_lane import entry_to_signal
    entry = Entry("", side, pd.Timestamp("2025-12-01 09:35"), PRICE, STOP, PRICE * 1.6, 20,
                  "multiday", "mom_12_1_long")
    return entry_to_signal(entry, "mom_12_1", {"rank_min": 0.8}, verdict, score, "NVDA",
                           admission=admission, regime="risk_on")


def _size(monkeypatch, signal):
    mgr = make_manager(FakeCommentary())
    mgr.sync_with_schwab_data(dict(SCHWAB))
    assert mgr.account_balance == ALLOCATION and mgr.buying_power == ALLOCATION
    return asyncio.run(mgr.calculate_position_size_with_commentary(signal, PRICE))


def test_allowlisted_signal_sizes_on_the_risk_per_trade_path(monkeypatch):
    """The fix: a neutral 0.6 strength puts the sleeve on the risk-per-trade path — 8 shares
    (~$800 notional, $24 = 0.48% of the allocation at risk), against the 3 shares (~$300) the
    old NaN-score path produced. The $1,000 buying-power cap is the next binding limit (10
    shares), which a strength of 0.8 or more reaches — see the table test below."""
    # The live .env pins FIT_LANE_ALLOWLIST_STRENGTH=0.8; this test asserts the default.
    monkeypatch.setenv("FIT_LANE_ALLOWLIST_STRENGTH", "0.6")
    _cfg(monkeypatch)
    from strategies.fit_lane_admission import ADMIT_ALLOWLIST
    shares = _size(monkeypatch, _fit_signal(ADMIT_ALLOWLIST))
    assert shares == 8
    assert shares * PRICE == pytest.approx(800.0)
    assert shares * (PRICE - STOP) == pytest.approx(24.0)


def test_the_old_nan_score_path_was_the_problem(monkeypatch):
    """Regression witness: the same entry admitted with a NaN score and no neutral strength
    sizes to a third of that, purely because the absent row had no score."""
    _cfg(monkeypatch)
    from strategies.fit_lane_admission import ADMIT_PASS
    assert _size(monkeypatch, _fit_signal(ADMIT_PASS)) == 3


@pytest.mark.parametrize("strength,expected", [(0.6, 8), (0.7, 8), (0.8, 10), (1.0, 10)])
def test_share_count_by_allowlist_strength(monkeypatch, strength, expected):
    """The whole curve under the go-live values, so the controller can pick the knob knowing what
    it buys. The Kelly map is quality - (1-quality)/2 with a 0.5 floor, so 0.6 and 0.7 both land
    on 8 shares ($800) and only 0.8+ reaches the $1,000 buying-power cap at 10 shares."""
    _cfg(monkeypatch)
    import strategies.fit_lane_admission as A
    monkeypatch.setattr(A, "allowlist_strength", lambda: strength)
    assert _size(monkeypatch, _fit_signal(A.ADMIT_ALLOWLIST)) == expected


def test_allocation_still_binds_on_an_allowlisted_signal(monkeypatch):
    """Neutral sizing does not escape Task 1's portfolio cap: with the allocation already spent,
    an allowlisted sleeve signal takes no position at all."""
    _cfg(monkeypatch)
    from strategies.fit_lane_admission import ADMIT_ALLOWLIST
    mgr = make_manager(FakeCommentary())
    mgr.sync_with_schwab_data(dict(SCHWAB))
    mgr.bot_open_notional = ALLOCATION
    assert asyncio.run(mgr.calculate_position_size_with_commentary(
        _fit_signal(ADMIT_ALLOWLIST), PRICE)) == 0


def test_neutral_strength_does_not_change_sizing_with_the_allocation_off(monkeypatch):
    """Sanity: the neutral strength is a property of the signal, not of the allocation — with the
    cap off the same signal just sizes against the whole account (and hits its buying-power
    cap), so nothing about this fix depends on Task 1 being on."""
    _cfg(monkeypatch, BOT_CAPITAL_ALLOCATION=0.0)
    from strategies.fit_lane_admission import ADMIT_ALLOWLIST
    mgr = make_manager(FakeCommentary())
    mgr.sync_with_schwab_data(dict(SCHWAB))
    shares = asyncio.run(mgr.calculate_position_size_with_commentary(
        _fit_signal(ADMIT_ALLOWLIST), PRICE))
    assert shares > 8                      # a 46.5k account, not the 5k slice
