"""Observable, supervised, dead-man-switched stream watchdog.

v-stream-watchdog-recover-2026-05-12. See
``docs/incidents/2026-05-12-stream-watchdog.md`` for the failure
this module exists to make impossible.

Two cooperating classes:

  ``StreamWatchdog``
    Periodic ticker that observes the stream's last-tick timestamp,
    emits a heartbeat log line on every period (healthy or not), and
    invokes ``on_death`` when ``is_stream_silently_dead`` says the
    stream has gone silent during RTH. The heartbeat counter is the
    observability anchor — every external supervisor reads it.

  ``DeadMansSwitch``
    Sibling task that watches the heartbeat counter. If the counter
    fails to advance for ``dead_after_sec``, the watchdog itself is
    presumed dead (silent cancellation, hidden exception in the
    handler, event-loop starvation). The switch logs CRITICAL and
    calls ``os._exit(42)`` — the process supervisor restarts the bot.
    Crash loud, not silent.

Both are dependency-injected (callables for tick reader, RTH check,
death handler) so unit tests run with no real Schwab client and no
wall-clock dependence.
"""
from __future__ import annotations

import asyncio
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Callable, Optional

from core.stream_health import is_stream_silently_dead

logger = logging.getLogger("TradingBot")


class StreamWatchdog:
    """Observable tick-freshness watchdog.

    Every ``period_sec`` it:
      1. Increments ``heartbeat_count`` (the dead-man-switch reads this).
      2. Emits an INFO-level heartbeat log line — *always*, regardless
         of stream health, so the absence of heartbeats is itself a
         diagnosable signal.
      3. If in RTH, runs the pure ``is_stream_silently_dead`` check.
         On death: logs WARNING, invokes ``on_death()``, returns.
    """

    def __init__(
        self,
        *,
        period_sec: float,
        threshold_sec: float,
        connect_at: datetime,
        connect_grace_sec: float,
        get_last_tick: Callable[[], Optional[datetime]],
        on_death: Callable[[], None],
        stop_event: asyncio.Event,
        is_rth_now: Callable[[], bool],
    ) -> None:
        self.period_sec = period_sec
        self.threshold_sec = threshold_sec
        self.connect_at = connect_at
        self.connect_grace_sec = connect_grace_sec
        self.get_last_tick = get_last_tick
        self.on_death = on_death
        self.stop_event = stop_event
        self.is_rth_now = is_rth_now
        self.heartbeat_count: int = 0

    async def run(self) -> None:
        while not self.stop_event.is_set():
            # Bounded wait — stop_event lets us shut down promptly,
            # timeout drives the periodic check. No bare await.
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=self.period_sec
                )
                return   # stop signalled
            except asyncio.TimeoutError:
                pass     # normal period elapsed

            self.heartbeat_count += 1
            last_tick = self._safe_get_last_tick()
            now = datetime.now(timezone.utc)
            rth = self._safe_is_rth()

            age_str = "never"
            if last_tick is not None:
                age_str = f"{(now - last_tick).total_seconds():.0f}s"
            logger.info(
                "schwab_stream.watchdog: heartbeat age=%s threshold=%.0fs "
                "rth=%s heartbeat_count=%d",
                age_str, self.threshold_sec, rth, self.heartbeat_count,
            )

            if not rth:
                continue

            if is_stream_silently_dead(
                last_tick_at=last_tick,
                now=now,
                threshold_sec=self.threshold_sec,
                connect_at=self.connect_at,
                connect_grace_sec=self.connect_grace_sec,
            ):
                logger.warning(
                    "schwab_stream.watchdog: silent death detected "
                    "(last tick %s, threshold %.0fs) — invoking on_death",
                    age_str, self.threshold_sec,
                )
                try:
                    self.on_death()
                except Exception:
                    logger.critical(
                        "schwab_stream.watchdog: on_death handler raised:\n%s",
                        traceback.format_exc(),
                    )
                return

    def _safe_get_last_tick(self) -> Optional[datetime]:
        try:
            return self.get_last_tick()
        except Exception:
            logger.warning(
                "schwab_stream.watchdog: get_last_tick raised, treating as None"
            )
            return None

    def _safe_is_rth(self) -> bool:
        try:
            return bool(self.is_rth_now())
        except Exception:
            logger.warning(
                "schwab_stream.watchdog: is_rth_now raised, defaulting to False"
            )
            return False


class DeadMansSwitch:
    """Watchdog-watchdog. Reads ``get_heartbeat_count``; if the value
    fails to advance for ``dead_after_sec``, calls ``os._exit(42)``.

    The exit code 42 is the contract with the external process
    supervisor: "I died, please restart me, this was deliberate." Any
    PID monitor / systemd / Docker restart-policy treats it the same
    as any other non-zero exit; the value just makes the cause
    grep-able in logs.
    """

    def __init__(
        self,
        *,
        check_period_sec: float,
        dead_after_sec: float,
        get_heartbeat_count: Callable[[], int],
        stop_event: asyncio.Event,
    ) -> None:
        self.check_period_sec = check_period_sec
        self.dead_after_sec = dead_after_sec
        self.get_heartbeat_count = get_heartbeat_count
        self.stop_event = stop_event

    async def run(self) -> None:
        last_count = self._safe_count()
        last_change_at = datetime.now(timezone.utc)

        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=self.check_period_sec
                )
                return
            except asyncio.TimeoutError:
                pass

            current = self._safe_count()
            now = datetime.now(timezone.utc)

            if current != last_count:
                last_count = current
                last_change_at = now
                continue

            since = (now - last_change_at).total_seconds()
            if since >= self.dead_after_sec:
                logger.critical(
                    "schwab_stream.watchdog: DEAD MAN'S SWITCH — "
                    "heartbeat counter stuck at %d for %.1fs "
                    "(threshold %.1fs). Calling os._exit(42).",
                    last_count, since, self.dead_after_sec,
                )
                os._exit(42)
                return   # unreachable in production; lets tests patch _exit

    def _safe_count(self) -> int:
        try:
            return int(self.get_heartbeat_count())
        except Exception:
            logger.warning(
                "schwab_stream.watchdog: get_heartbeat_count raised, treating as 0"
            )
            return 0


def supervise_task(task: asyncio.Task, name: str) -> None:
    """Attach a done-callback that logs CRITICAL with traceback if the
    task exits unexpectedly. Use this for every watchdog/switch task
    so silent cancellation cannot happen.

    Cancellation initiated by us (during shutdown) is *not* an
    unexpected exit; we tolerate ``CancelledError`` quietly.
    """
    def _done(t: asyncio.Task) -> None:
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.critical(
                "schwab_stream.watchdog: supervisor probe on %s raised:\n%s",
                name, traceback.format_exc(),
            )
            return
        if exc is None:
            logger.info("schwab_stream.watchdog: task %s exited cleanly", name)
            return
        logger.critical(
            "schwab_stream.watchdog: task %s died with %s:\n%s",
            name, type(exc).__name__,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    task.add_done_callback(_done)
