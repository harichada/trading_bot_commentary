"""v-playbook-desk-b2-final-2026-09-25 (whole-branch review, M3 + M4): the engine's
desk hook readers.

The engine is not constructible in tests, so the three readers are exercised on a
stand-in object that borrows exactly those unbound methods (`tests/core/test_desk_wiring.py`
covers their placement in the router; this file covers their behaviour):

* M4 — `_desk_plan_today` / `_desk_playbook` do file I/O on the async router path, so
  each read must go through `asyncio.to_thread`, keeping the same 60 s cache.
* M3 — `_desk_trades_today` caches an untrustworthy count (None) on *any* failure, not
  only a timeout, so a broken Postgres costs one read a minute instead of one a signal.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

import core.engine as engine_mod
from core.engine import TradingEngineWithCommentary as Engine


class _Hooks:
    """Only the desk readers, on an object we can construct."""

    _DESK_CACHE_SEC = Engine._DESK_CACHE_SEC
    _DESK_DB_TIMEOUT_SEC = Engine._DESK_DB_TIMEOUT_SEC
    _desk_plan_today = Engine._desk_plan_today
    _desk_playbook = Engine._desk_playbook
    _desk_trades_today = Engine._desk_trades_today
    _desk_hook_warn = Engine._desk_hook_warn

    def __init__(self, db_logger=None, internal=()):
        self.db_logger = db_logger
        self._internal = list(internal)

    def get_internal_trades_today(self):
        return list(self._internal)


_PLAN = object()
_BOOK = object()


# ── M4: the file reads are awaited off the event-loop thread ─────────

def test_the_plan_and_playbook_readers_are_coroutines():
    assert asyncio.iscoroutinefunction(Engine._desk_plan_today)
    assert asyncio.iscoroutinefunction(Engine._desk_playbook)


def test_the_plan_read_runs_in_a_thread_and_is_cached(monkeypatch):
    seen = {"calls": 0, "thread": None}

    def fake_load(desk_dir, date):
        seen["calls"] += 1
        seen["thread"] = threading.current_thread()
        return _PLAN

    monkeypatch.setattr(engine_mod, "load_desk_plan", fake_load)
    hooks = _Hooks()

    async def main():
        return await hooks._desk_plan_today(), await hooks._desk_plan_today()

    first, second = asyncio.run(main())
    assert first is _PLAN and second is _PLAN
    assert seen["calls"] == 1                                  # cached for 60 s
    assert seen["thread"] is not threading.main_thread()       # off the router's thread


def test_the_playbook_read_runs_in_a_thread_and_is_cached(monkeypatch):
    seen = {"calls": 0, "thread": None}

    def fake_load(path):
        seen["calls"] += 1
        seen["thread"] = threading.current_thread()
        return _BOOK

    monkeypatch.setattr(engine_mod, "load_playbook", fake_load)
    hooks = _Hooks()

    async def main():
        return await hooks._desk_playbook(), await hooks._desk_playbook()

    first, second = asyncio.run(main())
    assert first is _BOOK and second is _BOOK
    assert seen["calls"] == 1
    assert seen["thread"] is not threading.main_thread()


def test_a_stale_plan_cache_is_re_read(monkeypatch):
    """The cache is a 60 s window, not a one-shot: an expired entry must be replaced."""
    calls = {"n": 0}

    def fake_load(desk_dir, date):
        calls["n"] += 1
        return _PLAN

    monkeypatch.setattr(engine_mod, "load_desk_plan", fake_load)
    hooks = _Hooks()

    async def main():
        await hooks._desk_plan_today()
        date, _stamp, plan = hooks._desk_plan_cache
        hooks._desk_plan_cache = (date, 0.0, plan)             # 1970: long expired
        await hooks._desk_plan_today()

    asyncio.run(main())
    assert calls["n"] == 2


# ── M3: any failed trade-count read caches as untrustworthy ─────────

class _BoomLogger:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    async def get_todays_bot_entries(self, symbols=None):
        self.calls += 1
        raise self.exc


class _SlowLogger:
    def __init__(self):
        self.calls = 0

    async def get_todays_bot_entries(self, symbols=None):
        self.calls += 1
        await asyncio.sleep(5.0)
        return {}


@pytest.mark.parametrize("exc", [RuntimeError("connection reset"),
                                 ValueError("bad row"),
                                 OSError("socket closed")])
def test_a_failed_trade_count_read_is_none_and_cached(exc):
    """M3: only `asyncio.TimeoutError` was caught, so any other failure propagated out
    of the helper *and cached nothing* — the hook swallowed it and the next signal ran
    the same broken query. A failure must read as an untrustworthy count (None, which
    `evaluate_desk_gate` does not apply the max-trades rule to) and be cached."""
    db_logger = _BoomLogger(exc)
    hooks = _Hooks(db_logger=db_logger)

    async def main():
        return await hooks._desk_trades_today(), await hooks._desk_trades_today()

    assert asyncio.run(main()) == (None, None)
    assert db_logger.calls == 1                    # retried at most once per 60 s


def test_a_slow_trade_count_read_still_times_out_to_none():
    db_logger = _SlowLogger()
    hooks = _Hooks(db_logger=db_logger)
    hooks._DESK_DB_TIMEOUT_SEC = 0.01

    async def main():
        return await hooks._desk_trades_today(), await hooks._desk_trades_today()

    assert asyncio.run(main()) == (None, None)
    assert db_logger.calls == 1


def test_a_healthy_trade_count_read_is_counted_and_cached():
    class _Logger:
        def __init__(self):
            self.calls = 0

        async def get_todays_bot_entries(self, symbols=None):
            self.calls += 1
            return {"NVDA": 1, "AMD": 1}

    db_logger = _Logger()
    hooks = _Hooks(db_logger=db_logger)

    async def main():
        return await hooks._desk_trades_today(), await hooks._desk_trades_today()

    assert asyncio.run(main()) == (2, 2)
    assert db_logger.calls == 1
