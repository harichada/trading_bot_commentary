"""v-wedge-fix-2026-09-28 fix round 2 (Important 1): the open-trade desk is bounded.

`ActiveOpenDesk._get_indicators` called `engine.data_provider.get_market_data`
inline, on the desk loop, every ACTIVE_OPEN_DESK_INTERVAL_SEC (5 s) for EVERY
open position — and this module issues real exits. One hung Schwab bar fetch
there stalls proactive exit management for the whole book, which is the same
failure as 2026-09-28 sitting on a different loop.

The fetch now goes through the engine's bounded helper, and a timeout means
"skip this position's evaluation this tick" rather than "evaluate its exits
against no indicators".
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from analysis.active_open_desk import ActiveOpenDesk, BarsUnavailable
from core import provider_calls

BUDGET_SEC = 0.2


class HangingProvider:
    def __init__(self, hang: float = 30.0) -> None:
        self.calls = 0
        self.hang = hang

    def get_market_data(self, symbol, **kwargs):
        self.calls += 1
        time.sleep(self.hang)
        return pd.DataFrame({"Close": [1.0]})


class GoodProvider:
    def get_market_data(self, symbol, **kwargs):
        return pd.DataFrame({"Close": [1.0, 2.0], "High": [1.0, 2.0],
                             "Low": [1.0, 2.0], "Volume": [10, 20]})


class FakeAnalyzer:
    async def analyze_with_commentary(self, data, symbol):
        return {"rsi": 55.0}


class FakeEngine:
    """Only what the desk touches. `_provider_call` is the real helper, so
    these tests exercise the production bounding path."""

    def __init__(self, provider) -> None:
        self.data_provider = provider
        self.technical_analyzer = FakeAnalyzer()

    async def _provider_call(self, fn, *args, timeout=None, site="provider", **kwargs):
        return await provider_calls.provider_call(
            fn, *args, timeout=timeout, site=site, **kwargs
        )


@pytest.fixture(autouse=True)
def _tight_budget(monkeypatch):
    provider_calls.reset_state()
    monkeypatch.setattr(provider_calls, "history_timeout", lambda: BUDGET_SEC)
    yield
    provider_calls.reset_state()


def test_a_hung_bar_fetch_raises_bars_unavailable_fast():
    provider = HangingProvider()
    desk = ActiveOpenDesk(FakeEngine(provider))
    started = time.monotonic()
    with pytest.raises(BarsUnavailable):
        asyncio.run(desk._get_indicators("NVDA"))
    elapsed = time.monotonic() - started
    assert elapsed < BUDGET_SEC + 0.5, f"took {elapsed:.2f}s — not bounded"
    assert provider.calls == 1


def test_bars_unavailable_is_an_exception_the_caller_already_catches():
    assert issubclass(BarsUnavailable, Exception)


def test_indicators_still_come_back_when_the_provider_answers():
    desk = ActiveOpenDesk(FakeEngine(GoodProvider()))
    out = asyncio.run(desk._get_indicators("NVDA"))
    assert out == {"rsi": 55.0}


def test_an_empty_frame_is_no_indicators_not_a_skip():
    class EmptyProvider:
        def get_market_data(self, symbol, **kwargs):
            return pd.DataFrame()

    desk = ActiveOpenDesk(FakeEngine(EmptyProvider()))
    assert asyncio.run(desk._get_indicators("NVDA")) is None


def test_no_provider_is_no_indicators():
    desk = ActiveOpenDesk(FakeEngine(None))
    assert asyncio.run(desk._get_indicators("NVDA")) is None


def test_the_position_evaluation_is_skipped_on_a_bar_timeout():
    """The desk must not reach its exit rules with no indicators — the whole
    tick for that position is skipped."""
    @dataclass
    class Pos:
        symbol: str = "NVDA"
        entry_price: float = 100.0
        stop_loss: float = 95.0
        take_profit: float = 110.0
        side: str = "long"
        quantity: int = 10
        managed_by_bot: bool = True
        is_external: bool = False
        is_manually_managed: bool = False
        is_long_term: bool = False
        current_price: float = 101.0
        entry_time: datetime = field(default_factory=datetime.now)
        original_stop: Optional[float] = 95.0
        reasoning: Dict[str, Any] = field(default_factory=dict)

    desk = ActiveOpenDesk(FakeEngine(HangingProvider()))
    desk._get_market_context = lambda symbol: {}
    started = time.monotonic()
    decision = asyncio.run(desk._evaluate_position("NVDA", Pos()))
    assert decision is None
    assert time.monotonic() - started < BUDGET_SEC + 0.6
