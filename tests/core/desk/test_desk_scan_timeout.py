"""v-wedge-fix-2026-09-28 fix round 2 (Important 3): the desk scan is bounded.

`DeskLoop._row_for` fetched today's 5-minute bars through
`asyncio.to_thread(steps.fetch_today_bars5, ...)` with no timeout. `to_thread`
keeps the loop responsive but the *await* never returns, so one hung symbol
stalls the whole in-play scan (and, with a bounded gather, everything behind
it). A timeout now produces the same all-missing row the scan already uses to
show a hole rather than silently shortening its list.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from core import provider_calls
from core.desk.models import InPlayRow
from core.loops import desk_steps as steps
from core.loops.desk_loop import DeskLoop

BUDGET_SEC = 0.2


def _et(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=steps.ET_TZ)


@pytest.fixture(autouse=True)
def _tight(monkeypatch):
    provider_calls.reset_state()
    monkeypatch.setattr(provider_calls, "history_timeout", lambda: BUDGET_SEC)
    yield
    provider_calls.reset_state()


def _loop(monkeypatch, fetch):
    monkeypatch.setattr(steps, "fetch_today_bars5", fetch)
    engine = SimpleNamespace(data_provider=object(), market_state={},
                             dynamic_watchlist=[], db_logger=None,
                             _audit=lambda *a, **k: None)
    return DeskLoop(engine, config_fn=lambda: SimpleNamespace(
        STRATEGY_HUNT_DATA_ROOT="",
    ))


def test_a_hung_bar_fetch_yields_a_missing_row_fast(monkeypatch):
    calls = {"n": 0}

    def _hang(provider, symbol):
        calls["n"] += 1
        # 3s, not 30: the 0.2s budget is what is under test, and the abandoned
        # worker thread delays asyncio.run's teardown by however long it sleeps.
        time.sleep(3)
        return pd.DataFrame()

    loop = _loop(monkeypatch, _hang)
    cfg = loop._config_fn()

    async def _timed():
        # Time the ROW, not the event-loop teardown: the abandoned worker
        # thread outlives the await (that is the documented limit of a thread
        # timeout) and asyncio.run waits for the default executor on exit.
        started = time.monotonic()
        row = await loop._row_for(cfg, "NVDA", _et(2026, 9, 29, 9, 40), {})
        return row, time.monotonic() - started

    row, elapsed = asyncio.run(_timed())

    assert isinstance(row, InPlayRow)
    assert row.symbol == "NVDA"
    assert "missing:today_bars" in row.reasons
    assert elapsed < BUDGET_SEC + 0.6, f"scan blocked {elapsed:.2f}s"
    assert calls["n"] == 1


def test_the_timeout_is_labelled_distinctly_from_a_provider_error(monkeypatch):
    def _hang(provider, symbol):
        time.sleep(3)

    loop = _loop(monkeypatch, _hang)
    cfg = loop._config_fn()
    row = asyncio.run(loop._row_for(cfg, "NVDA", _et(2026, 9, 29, 9, 40), {}))
    assert "timeout:today_bars" in row.reasons

    def _boom(provider, symbol):
        raise RuntimeError("provider said no")

    loop = _loop(monkeypatch, _boom)
    row = asyncio.run(loop._row_for(cfg, "NVDA", _et(2026, 9, 29, 9, 40), {}))
    assert "error:scan" in row.reasons
    assert "timeout:today_bars" not in row.reasons


def test_a_healthy_fetch_still_produces_a_real_row(monkeypatch):
    idx = pd.date_range("2026-09-29 09:30", periods=12, freq="5min")
    frame = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "volume": 10_000}, index=idx,
    )
    loop = _loop(monkeypatch, lambda provider, symbol: frame)
    cfg = loop._config_fn()
    row = asyncio.run(loop._row_for(cfg, "NVDA", _et(2026, 9, 29, 9, 40), {}))
    assert isinstance(row, InPlayRow)
    assert "timeout:today_bars" not in row.reasons
