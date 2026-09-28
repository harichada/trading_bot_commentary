"""v-wedge-fix-2026-09-28 fix round 1 (Important 1): the strategy alias is bounded.

`core/engine.py` hands `self.data_provider` to the day-trade strategies as
`_schwab_provider`, and `strategies/builtin.py` passes it into
`core/market_context.py` from inside `generate_signal_with_commentary` — which
the engine awaits on its own loop. Those five `schwab_provider.get_quote(...)` /
`.get_market_data(...)` calls were the last unbounded synchronous provider calls
reachable from the event loop: exactly the shape that wedged the process on
2026-09-28, just behind a different attribute name.

Every test here uses a provider that hangs for 30 s. With the helper in place
each context function returns its documented fallback in well under a second;
without it, these tests hang until the suite is killed.
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from core import market_context as mc
from core import provider_calls


HANG_SEC = 30.0
BUDGET_SEC = 0.2


class HangingProvider:
    """Stands in for a Schwab provider whose socket never comes back."""

    def __init__(self) -> None:
        self.quote_calls = 0
        self.bar_calls = 0

    def get_quote(self, symbol):
        self.quote_calls += 1
        time.sleep(HANG_SEC)
        return {"netPercentChange": 99.0, "open": 999.0}

    def get_market_data(self, symbol, **kwargs):
        self.bar_calls += 1
        time.sleep(HANG_SEC)
        return pd.DataFrame()


class GoodProvider:
    def get_quote(self, symbol):
        return {"netPercentChange": 1.25, "open": 100.0}

    def get_market_data(self, symbol, **kwargs):
        return pd.DataFrame()


@pytest.fixture(autouse=True)
def _tight_budgets(monkeypatch):
    """Shrink both provider budgets so a hang is observable in a fast test."""
    provider_calls.reset_state()
    monkeypatch.setattr(provider_calls, "default_timeout", lambda: BUDGET_SEC)
    monkeypatch.setattr(provider_calls, "history_timeout", lambda: BUDGET_SEC)
    mc.clear_symbol_day_change_cache()
    mc._SPY_VWAP_CACHE.clear()
    mc._SPY_BARS_CACHE.clear()
    yield
    provider_calls.reset_state()
    mc.clear_symbol_day_change_cache()
    mc._SPY_VWAP_CACHE.clear()
    mc._SPY_BARS_CACHE.clear()


def test_symbol_day_change_falls_back_to_the_bar_estimate_on_a_hang():
    provider = HangingProvider()
    started = time.monotonic()
    pct, source = mc.get_symbol_day_change_pct(
        "NVDA", bar_open=100.0, bar_close=101.0, schwab_provider=provider,
    )
    elapsed = time.monotonic() - started
    assert source == "bar_estimate"
    assert pct == pytest.approx(1.0)
    assert elapsed < BUDGET_SEC + 0.5
    assert provider.quote_calls == 1


def test_symbol_day_change_still_prefers_the_schwab_quote_when_it_answers():
    pct, source = mc.get_symbol_day_change_pct(
        "NVDA", bar_open=100.0, bar_close=101.0, schwab_provider=GoodProvider(),
    )
    assert source == "schwab_quote"
    assert pct == pytest.approx(1.25)


def test_spy_index_context_returns_without_the_rth_open_or_vwap_on_a_hang():
    provider = HangingProvider()
    started = time.monotonic()
    ctx = mc.get_spy_index_context(schwab_provider=provider)
    elapsed = time.monotonic() - started
    assert ctx.spy_rth_open is None
    assert ctx.spy_session_vwap is None
    # Two bounded calls (quote + bars), not two open-ended ones.
    assert elapsed < 2 * BUDGET_SEC + 0.6
    assert provider.quote_calls == 1 and provider.bar_calls == 1


def test_rs_from_open_context_degrades_to_unavailable_on_a_hang():
    spy_ctx = mc.get_spy_index_context(schwab_provider=None)
    provider = HangingProvider()
    started = time.monotonic()
    ctx = mc.get_rs_from_open_context(
        "NVDA", sym_last=100.0, rs_prior_close=0.0,
        spy_ctx=spy_ctx, schwab_provider=provider,
    )
    elapsed = time.monotonic() - started
    assert ctx.sym_rth_open is None
    assert ctx.rs_from_open is None
    assert ctx.source == "unavailable"
    assert elapsed < BUDGET_SEC + 0.5


def test_inmove_turn_context_degrades_on_a_hang():
    spy_ctx = mc.get_spy_index_context(schwab_provider=None)
    provider = HangingProvider()
    started = time.monotonic()
    ctx = mc.get_inmove_turn_context(spy_ctx, schwab_provider=provider)
    elapsed = time.monotonic() - started
    assert ctx.inmove_turn_active is False
    assert elapsed < BUDGET_SEC + 0.5
    assert provider.bar_calls == 1


def test_a_provider_error_still_reaches_the_existing_fallback():
    """Non-timeout exceptions propagate out of the helper, so the module's own
    try/except keeps handling them — behaviour unchanged."""
    class BoomProvider:
        def get_quote(self, symbol):
            raise RuntimeError("schwab exploded")

    pct, source = mc.get_symbol_day_change_pct(
        "NVDA", bar_open=100.0, bar_close=102.0, schwab_provider=BoomProvider(),
    )
    assert source == "bar_estimate"
    assert pct == pytest.approx(2.0)
