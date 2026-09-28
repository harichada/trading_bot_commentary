"""v-wedge-fix-2026-09-28 fix round 1 (Minor 8): nested timeout budgets.

`RealTimeDataProvider` (the yfinance fallback provider) is called through the
engine's bounded helper, which itself allows 3 s for a quote and 10 s for
history. The isolated subprocess call underneath must use the SAME shape of
budget, otherwise the inner call outlives the outer one and the engine's quote
timeout is decided by a 10 s history budget it never asked for.

Lives under tests/core so it runs with the mandated core/strategies/risk sweep;
the subject is data_providers/realtime.py.
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.config import Config
from data_providers import realtime as rt


@pytest.fixture
def recorder(monkeypatch):
    """Capture every isolated call's fn name and timeout, return nothing."""
    calls = []

    def _fake(fn, *args, timeout=None, site=None, **kwargs):
        calls.append({"fn": fn.__name__, "timeout": timeout, "site": site,
                      "args": args})
        return None

    monkeypatch.setattr(rt, "run_isolated_sync", _fake)
    return calls


@pytest.fixture
def provider():
    return rt.RealTimeDataProvider(commentary_system=None, schwab_client=None)


def test_get_quote_uses_the_quote_budget(recorder, provider):
    assert provider.get_quote("NVDA") is None
    assert len(recorder) == 1
    assert recorder[0]["fn"] == "yf_fast_quote"
    assert recorder[0]["timeout"] == pytest.approx(Config().QUOTE_FETCH_TIMEOUT_SEC)


def test_get_market_data_uses_the_history_budget(recorder, provider):
    out = provider.get_market_data("NVDA")
    assert isinstance(out, pd.DataFrame) and out.empty
    assert recorder[0]["fn"] == "yf_history"
    assert recorder[0]["timeout"] == pytest.approx(
        Config().PROVIDER_HISTORY_TIMEOUT_SEC
    )


def test_batched_calls_use_the_history_budget(recorder, provider):
    """Batched multi-symbol reads walk one ticker at a time in the child, so
    they get the longer budget even though they return quotes."""
    assert provider.get_multiple_quotes(["SPY", "QQQ"]) == {}
    breadth = provider.calculate_market_breadth()
    assert breadth["advance_decline_ratio"] == 1.0       # neutral fallback
    for call in recorder:
        assert call["timeout"] == pytest.approx(
            Config().PROVIDER_HISTORY_TIMEOUT_SEC
        ), call


def test_the_quote_budget_is_strictly_tighter_than_the_history_budget():
    assert Config().QUOTE_FETCH_TIMEOUT_SEC < Config().PROVIDER_HISTORY_TIMEOUT_SEC
