"""v-wedge-fix-2026-09-28 fix round 2 (Important 2): the index strip is bounded.

`MarketIndicesCache.refresh` is called from the engine loop every 10 s with a
raw `schwab_client.get_quotes(...)` — synchronous, no timeout. One hung batch
quote call there stalls the engine loop, and the strip it feeds is purely
cosmetic. The call is now bounded and a timeout keeps the previous snapshot.
"""
from __future__ import annotations

import logging
import time

import pytest

from core import provider_calls
from core.market_indices import MarketIndicesCache


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _payload(symbols, last=100.0, change=1.5):
    return {s: {"quote": {"lastPrice": last, "netPercentChange": change}}
            for s in symbols}


class GoodClient:
    def __init__(self):
        self.calls = 0

    def get_quotes(self, symbols):
        self.calls += 1
        return _Response(_payload(symbols))


class HangingClient:
    def __init__(self, hang=30.0):
        self.calls = 0
        self.hang = hang

    def get_quotes(self, symbols):
        self.calls += 1
        time.sleep(self.hang)
        return _Response(_payload(symbols))


@pytest.fixture(autouse=True)
def _fresh():
    MarketIndicesCache.reset_for_tests()
    provider_calls.reset_state()
    yield
    MarketIndicesCache.reset_for_tests()
    provider_calls.reset_state()


def test_a_good_refresh_populates_the_snapshot():
    cache = MarketIndicesCache.instance()
    cache.refresh(GoodClient())
    snap = cache.snapshot()
    assert snap["error"] is None
    assert snap["indices"] and snap["last_updated"] is not None


def test_a_hung_quote_call_is_bounded_and_keeps_the_previous_snapshot(caplog):
    cache = MarketIndicesCache.instance()
    cache.refresh(GoodClient())
    before = cache.snapshot()
    assert before["indices"]

    client = HangingClient()
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="TradingBot"):
        cache.refresh(client, timeout=0.2)
    elapsed = time.monotonic() - started

    assert elapsed < 0.8, f"refresh blocked for {elapsed:.2f}s"
    assert client.calls == 1
    after = cache.snapshot()
    # Previous values survive; only the error field changes.
    assert after["indices"] == before["indices"]
    assert after["last_updated"] == before["last_updated"]
    assert "timeout" in (after["error"] or "")
    assert any("market_indices" in r.getMessage() for r in caplog.records)


def test_a_timeout_on_a_cold_cache_leaves_it_empty_not_broken():
    cache = MarketIndicesCache.instance()
    cache.refresh(HangingClient(), timeout=0.2)
    snap = cache.snapshot()
    assert snap["indices"] == [] or all(
        q.get("last") in (None, 0, 0.0) for q in snap["indices"]
    )
    assert snap["stale"] is True
    assert "timeout" in (snap["error"] or "")


def test_the_engine_passes_the_quote_budget_not_the_broker_one():
    """The strip is cosmetic and sits on the engine loop: 3 s, not 15 s."""
    from pathlib import Path

    engine = Path("core/engine.py").read_text()
    idx = engine.index("cache.refresh(")
    assert "Config().QUOTE_FETCH_TIMEOUT_SEC" in engine[idx:idx + 300]
