"""v-wedge-fix-2026-09-28 (Task 4, ruling 1): the bounded provider-call helper.

The engine froze on 2026-09-28 because synchronous provider calls ran inline
on the event loop with no timeout. These tests pin the contract of the helper
that replaces them:

  - a hung callable never blocks the caller for longer than the timeout
  - a timeout yields None (callers keep their own fallback) and is logged
    at most once per minute per call site
  - non-timeout exceptions still propagate, so every existing try/except
    around a provider call keeps behaving exactly as it does today
  - the executor is dedicated (never asyncio's default pool), so abandoned
    provider threads cannot starve unrelated to_thread work
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from core import provider_calls


@pytest.fixture(autouse=True)
def _fresh_helper_state():
    provider_calls.reset_state()
    yield
    provider_calls.reset_state()


def test_executor_is_dedicated_and_bounded():
    ex = provider_calls.get_executor()
    assert isinstance(ex, ThreadPoolExecutor)
    assert ex is provider_calls.get_executor()      # one pool, reused
    assert ex._max_workers >= 1
    assert "provider" in (ex._thread_name_prefix or "")


def test_async_timeout_returns_none_without_blocking_the_loop():
    async def _run():
        started = time.monotonic()
        ticks = 0

        async def _ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        tick_task = asyncio.create_task(_ticker())
        out = await provider_calls.provider_call(
            lambda: time.sleep(30), timeout=0.2, site="unit_hang",
        )
        elapsed = time.monotonic() - started
        tick_task.cancel()
        return out, elapsed, ticks

    out, elapsed, ticks = asyncio.run(_run())
    assert out is None
    assert elapsed < 0.7          # timeout + 0.5s budget
    assert ticks > 1              # the loop kept running while it hung


def test_async_success_passes_args_and_kwargs_through():
    def _fn(symbol, *, frequency=1):
        return {"symbol": symbol, "frequency": frequency}

    out = asyncio.run(
        provider_calls.provider_call(_fn, "MRVL", frequency=5, site="unit_ok")
    )
    assert out == {"symbol": "MRVL", "frequency": 5}


def test_non_timeout_exceptions_propagate_to_the_caller():
    class Boom(RuntimeError):
        pass

    def _fn():
        raise Boom("provider exploded")

    with pytest.raises(Boom):
        asyncio.run(provider_calls.provider_call(_fn, site="unit_raise"))


def test_sync_variant_times_out_and_returns_none():
    started = time.monotonic()
    out = provider_calls.provider_call_sync(
        lambda: time.sleep(30), timeout=0.2, site="unit_sync_hang",
    )
    assert out is None
    assert time.monotonic() - started < 0.7


def test_sync_variant_returns_the_value_and_propagates_errors():
    assert provider_calls.provider_call_sync(lambda a: a * 2, 21, site="s") == 42
    with pytest.raises(ValueError):
        provider_calls.provider_call_sync(
            lambda: (_ for _ in ()).throw(ValueError("x")), site="s2",
        )


def test_timeout_logging_is_rate_limited_per_call_site(caplog):
    clock = {"t": 1000.0}
    provider_calls.set_clock(lambda: clock["t"])
    with caplog.at_level(logging.WARNING, logger="TradingBot"):
        for _ in range(3):
            provider_calls.provider_call_sync(
                lambda: time.sleep(5), timeout=0.05, site="rate_limited",
            )
        first = [r for r in caplog.records if "provider_call_timeout" in r.getMessage()]
        assert len(first) == 1
        assert "site=rate_limited" in first[0].getMessage()

        clock["t"] += 61.0        # past the one-minute window
        provider_calls.provider_call_sync(
            lambda: time.sleep(5), timeout=0.05, site="rate_limited",
        )
        again = [r for r in caplog.records if "provider_call_timeout" in r.getMessage()]
        assert len(again) == 2
        # The suppressed calls are accounted for, not silently dropped.
        assert "suppressed=2" in again[1].getMessage()


def test_default_timeout_comes_from_config():
    from core.config import Config

    assert provider_calls.default_timeout() == pytest.approx(
        Config().QUOTE_FETCH_TIMEOUT_SEC
    )
    assert provider_calls.history_timeout() == pytest.approx(
        Config().PROVIDER_HISTORY_TIMEOUT_SEC
    )
