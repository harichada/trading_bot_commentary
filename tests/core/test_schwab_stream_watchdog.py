"""Regression tests for v-stream-watchdog-recover-2026-05-12.

Companion to docs/incidents/2026-05-12-stream-watchdog.md. The
watchdog must be:

  1. Observable — heartbeat log every 30 s during healthy operation.
  2. Supervised — silent task death surfaced via add_done_callback.
  3. Dead-man-switched — process exit(42) if the watchdog itself dies.
  4. Pure-decision-logicked — silent-death rule unit-testable without
     asyncio or wall-clock.
  5. Recovered — full reconnect path executes on silent-tick gap.

The four tests required by the task spec live here. ``os._exit`` is
patched in the exit-42 test so we don't kill the test process.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


# ── 1. Pure decision logic ────────────────────────────────────────────


class TestPureSilentDeathDetector:
    """Table-driven test of ``is_stream_silently_dead`` across edge cases."""

    def _now(self) -> datetime:
        return datetime(2026, 5, 12, 14, 50, 0, tzinfo=timezone.utc)

    def test_no_tick_inside_grace_returns_false(self):
        from core.stream_health import is_stream_silently_dead

        now = self._now()
        connect_at = now - timedelta(seconds=30)
        assert is_stream_silently_dead(
            last_tick_at=None,
            now=now,
            threshold_sec=120.0,
            connect_at=connect_at,
            connect_grace_sec=60.0,
        ) is False

    def test_no_tick_past_grace_returns_true(self):
        from core.stream_health import is_stream_silently_dead

        now = self._now()
        connect_at = now - timedelta(seconds=90)
        assert is_stream_silently_dead(
            last_tick_at=None,
            now=now,
            threshold_sec=120.0,
            connect_at=connect_at,
            connect_grace_sec=60.0,
        ) is True

    def test_no_tick_no_connect_returns_true(self):
        """Without a connect timestamp there's no grace to apply."""
        from core.stream_health import is_stream_silently_dead

        now = self._now()
        assert is_stream_silently_dead(
            last_tick_at=None,
            now=now,
            threshold_sec=120.0,
            connect_at=None,
        ) is True

    def test_just_under_threshold_returns_false(self):
        from core.stream_health import is_stream_silently_dead

        now = self._now()
        last_tick = now - timedelta(seconds=119.5)
        assert is_stream_silently_dead(
            last_tick_at=last_tick, now=now, threshold_sec=120.0,
        ) is False

    def test_just_over_threshold_returns_true(self):
        from core.stream_health import is_stream_silently_dead

        now = self._now()
        last_tick = now - timedelta(seconds=120.5)
        assert is_stream_silently_dead(
            last_tick_at=last_tick, now=now, threshold_sec=120.0,
        ) is True

    def test_exactly_at_threshold_returns_false(self):
        """Strict > comparison — boundary stays alive to prevent flap."""
        from core.stream_health import is_stream_silently_dead

        now = self._now()
        last_tick = now - timedelta(seconds=120.0)
        assert is_stream_silently_dead(
            last_tick_at=last_tick, now=now, threshold_sec=120.0,
        ) is False

    def test_future_last_tick_returns_false(self):
        """Clock skew protection — never declare death on a future tick."""
        from core.stream_health import is_stream_silently_dead

        now = self._now()
        last_tick = now + timedelta(seconds=5)
        assert is_stream_silently_dead(
            last_tick_at=last_tick, now=now, threshold_sec=120.0,
        ) is False

    def test_75_minute_gap_matches_2026_05_12_incident(self):
        """Reproduce the operator-observed gap. Watchdog must fire."""
        from core.stream_health import is_stream_silently_dead

        now = datetime(2026, 5, 12, 16, 1, 0, tzinfo=timezone.utc)
        last_tick = datetime(2026, 5, 12, 14, 46, 29, tzinfo=timezone.utc)
        assert is_stream_silently_dead(
            last_tick_at=last_tick, now=now, threshold_sec=120.0,
        ) is True


# ── 2. Watchdog heartbeat cadence ─────────────────────────────────────


@pytest.mark.asyncio
async def test_watchdog_heartbeat_cadence(caplog):
    """A healthy watchdog must emit a heartbeat log line on every
    period. We tick three periods and assert three heartbeats.

    Uses a tight period (0.05 s) so the test runs in well under a
    second. The pure function and the supervisor are exercised in
    other tests; here we only assert the *observability* contract.
    """
    import logging

    from core.schwab_stream_watchdog import StreamWatchdog

    stop = asyncio.Event()
    now = datetime.now(timezone.utc)

    # Healthy: a fresh tick on every check.
    def _last_tick():
        return datetime.now(timezone.utc)

    wd = StreamWatchdog(
        period_sec=0.05,
        threshold_sec=10.0,
        connect_at=now,
        connect_grace_sec=0.0,
        get_last_tick=_last_tick,
        on_death=lambda: None,
        stop_event=stop,
        is_rth_now=lambda: True,
    )

    with caplog.at_level(logging.INFO, logger="TradingBot"):
        task = asyncio.create_task(wd.run())
        await asyncio.sleep(0.18)   # ≈ 3 periods
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)

    heartbeats = [
        r for r in caplog.records
        if "schwab_stream.watchdog: heartbeat" in r.getMessage()
    ]
    # 3 periods elapsed → expect 3 heartbeats (±1 for scheduling jitter).
    assert 2 <= len(heartbeats) <= 4, (
        f"expected ~3 heartbeats, got {len(heartbeats)}"
    )


# ── 3. Silent tick gap triggers reconnect ─────────────────────────────


@pytest.mark.asyncio
async def test_silent_tick_gap_triggers_reconnect():
    """Stop sending ticks. The watchdog must invoke ``on_death`` (the
    reconnect trigger) within threshold + one period."""
    from core.schwab_stream_watchdog import StreamWatchdog

    stop = asyncio.Event()
    now = datetime.now(timezone.utc)

    # Last tick was long ago — already dead from the start.
    frozen_last_tick = now - timedelta(seconds=10)
    def _last_tick():
        return frozen_last_tick

    triggered = asyncio.Event()
    def _on_death():
        triggered.set()

    wd = StreamWatchdog(
        period_sec=0.05,
        threshold_sec=0.5,
        connect_at=now - timedelta(seconds=10),   # past grace
        connect_grace_sec=1.0,
        get_last_tick=_last_tick,
        on_death=_on_death,
        stop_event=stop,
        is_rth_now=lambda: True,
    )

    task = asyncio.create_task(wd.run())
    # Threshold 0.5s + one period (0.05s) plus generous schedule slack.
    await asyncio.wait_for(triggered.wait(), timeout=2.0)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)


# ── 4. Dead-man's switch — watchdog death triggers exit(42) ───────────


@pytest.mark.asyncio
async def test_dead_watchdog_triggers_exit_42():
    """If the watchdog stops emitting heartbeats for ``dead_after_sec``,
    the dead-man's switch calls ``os._exit(42)``. We patch ``os._exit``
    so the test process survives."""
    from core.schwab_stream_watchdog import DeadMansSwitch

    stop = asyncio.Event()
    heartbeat_counter = {"n": 0}

    def _heartbeats() -> int:
        return heartbeat_counter["n"]   # never increments → watchdog "dead"

    with patch("os._exit") as mock_exit:
        switch = DeadMansSwitch(
            check_period_sec=0.05,
            dead_after_sec=0.3,
            get_heartbeat_count=_heartbeats,
            stop_event=stop,
        )
        task = asyncio.create_task(switch.run())
        try:
            # Give the switch enough time to (a) take an initial reading
            # and (b) confirm dead_after_sec without an increment.
            await asyncio.sleep(0.6)
        finally:
            stop.set()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                task.cancel()

        mock_exit.assert_called_once_with(42)
