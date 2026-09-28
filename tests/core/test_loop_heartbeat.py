"""v-wedge-fix-2026-09-28 (Task 4, ruling 3): per-loop heartbeat + watchdog.

On 2026-09-28 the process was wedged for 16 hours with no log line saying so.
The heartbeat gives each event loop a one-second pulse and a daemon thread
that shouts `loop_lag` when a pulse goes missing. Tests drive a fake clock so
nothing sleeps.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from core.loop_heartbeat import LoopHeartbeat, get_heartbeat


def test_beat_records_zero_lag():
    clock = {"t": 100.0}
    hb = LoopHeartbeat(clock=lambda: clock["t"], warn_sec=5.0)
    hb.beat("engine")
    assert hb.lag("engine") == pytest.approx(0.0)
    clock["t"] += 2.0
    assert hb.lag("engine") == pytest.approx(2.0)


def test_unknown_loop_has_no_lag():
    hb = LoopHeartbeat(clock=lambda: 0.0, warn_sec=5.0)
    assert hb.lag("nope") is None
    assert hb.check_once() == []


def test_lag_above_the_threshold_logs_a_warning(caplog):
    clock = {"t": 0.0}
    hb = LoopHeartbeat(clock=lambda: clock["t"], warn_sec=5.0, startup_grace_sec=0.0)
    hb.beat("position_loop")
    clock["t"] = 4.9
    with caplog.at_level(logging.WARNING, logger="TradingBot"):
        assert hb.check_once() == []
        clock["t"] = 12.5
        lagging = hb.check_once()
    assert lagging and lagging[0][0] == "position_loop"
    msgs = [r.getMessage() for r in caplog.records if "loop_lag" in r.getMessage()]
    assert len(msgs) == 1
    assert "loop=position_loop" in msgs[0]
    assert "lag=12.5" in msgs[0]
    assert "threads=" in msgs[0]


def test_warnings_are_rate_limited_per_loop(caplog):
    clock = {"t": 0.0}
    hb = LoopHeartbeat(clock=lambda: clock["t"], warn_sec=5.0,
                       warn_interval_sec=30.0, startup_grace_sec=0.0)
    hb.beat("engine")
    with caplog.at_level(logging.WARNING, logger="TradingBot"):
        clock["t"] = 10.0
        hb.check_once()
        clock["t"] = 12.0
        hb.check_once()          # inside the rate-limit window
        assert len([r for r in caplog.records if "loop_lag" in r.getMessage()]) == 1
        clock["t"] = 45.0
        hb.check_once()
        assert len([r for r in caplog.records if "loop_lag" in r.getMessage()]) == 2


def test_a_recovered_loop_stops_warning(caplog):
    clock = {"t": 0.0}
    hb = LoopHeartbeat(clock=lambda: clock["t"], warn_sec=5.0, startup_grace_sec=0.0)
    hb.beat("engine")
    clock["t"] = 20.0
    assert hb.check_once()
    hb.beat("engine")
    assert hb.check_once() == []


def test_run_beats_until_stopped():
    clock = {"t": 0.0}
    hb = LoopHeartbeat(clock=lambda: clock["t"], warn_sec=5.0)
    beats = {"n": 0}

    def _tick():
        beats["n"] += 1
        return beats["n"] < 3

    async def _go():
        await hb.run("engine", interval=0.0, is_running=_tick)

    asyncio.run(_go())
    assert beats["n"] == 3
    # fix round 1 (Minor 6): a returned heartbeat task forgets its loop, so
    # there is no lag left to warn about. See the dedicated test below.
    assert hb.lag("engine") is None


def test_watchdog_thread_is_a_daemon_and_starts_once():
    hb = LoopHeartbeat(clock=lambda: 0.0, warn_sec=5.0)
    t1 = hb.start_watchdog(interval=60.0)
    t2 = hb.start_watchdog(interval=60.0)
    assert t1 is t2
    assert t1.daemon is True


def test_module_singleton_is_shared():
    assert get_heartbeat() is get_heartbeat()


def test_docstring_documents_the_gil_caveat():
    import core.loop_heartbeat as mod

    doc = (mod.__doc__ or "").lower()
    assert "gil" in doc
    assert "watchdog" in doc


# ── fix round 1 (Minor 6): stopped loops and the start-up grace ────────

def test_a_loop_whose_task_returns_is_forgotten():
    """A stopped loop (engine shutdown, supervisor stop) must not warn
    forever — `run()` drops its registration on the way out."""
    clock = {"t": 0.0}
    hb = LoopHeartbeat(clock=lambda: clock["t"], warn_sec=5.0, startup_grace_sec=0.0)

    async def _go():
        await hb.run("engine", interval=0.0, is_running=lambda: False)

    hb.beat("engine")
    asyncio.run(_go())
    assert hb.lag("engine") is None
    clock["t"] = 100.0
    assert hb.check_once() == []


def test_a_cancelled_heartbeat_task_is_also_forgotten():
    clock = {"t": 0.0}
    hb = LoopHeartbeat(clock=lambda: clock["t"], warn_sec=5.0, startup_grace_sec=0.0)

    async def _go():
        task = asyncio.ensure_future(hb.run("api", interval=0.01))
        await asyncio.sleep(0.05)
        assert hb.lag("api") is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_go())
    assert hb.lag("api") is None


def test_no_warning_during_the_first_30_seconds_of_a_loop():
    """Start-up is legitimately lumpy (the engine constructor alone takes
    ~13 s). Warning through it would train the operator to ignore the line."""
    clock = {"t": 1000.0}
    hb = LoopHeartbeat(clock=lambda: clock["t"], warn_sec=5.0)
    assert hb.startup_grace_sec == 30.0
    hb.beat("engine")
    clock["t"] += 20.0            # 20s of lag, still inside the grace
    assert hb.check_once() == []
    clock["t"] += 15.0            # 35s since the first beat, grace over
    stalled = hb.check_once()
    assert stalled and stalled[0][0] == "engine"


def test_the_grace_is_measured_from_the_first_beat_not_the_last():
    clock = {"t": 0.0}
    hb = LoopHeartbeat(clock=lambda: clock["t"], warn_sec=5.0, startup_grace_sec=30.0)
    for _ in range(4):
        hb.beat("engine")
        clock["t"] += 10.0        # healthy for 40s
    clock["t"] += 6.0            # then stalls
    assert hb.check_once()       # grace already spent — this warns
