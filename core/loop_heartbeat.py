"""v-wedge-fix-2026-09-28 (Task 4, ruling 3): per-loop heartbeat + watchdog.

WHAT IT IS
----------
Each event loop in the process (the engine's loop on its daemon thread, and
uvicorn's loop) runs one cheap task that stamps a monotonic timestamp every
second. A daemon `threading.Thread` compares those stamps to `now` and logs

    loop_lag loop=<name> lag=<seconds> threads=<n>

at WARNING when a loop has not pulsed for `Config().LOOP_LAG_WARN_SEC`,
rate-limited to one line per loop per minute. That is the missing signal
from 2026-09-28: the bot was wedged for a full session and the log said
nothing at all — `/api/status` simply never answered.

THE GIL CAVEAT (why an external restarter also exists)
------------------------------------------------------
This watchdog is a *thread*. A call that hangs while holding the GIL — a
curl_cffi transfer stalled in a TLS write, the mechanism behind the
2026-09-28 freeze — freezes this thread too, so no `loop_lag` line ever
gets written. The heartbeat therefore catches the *survivable* class of
stall (a blocking socket that released the GIL, a runaway synchronous
section, a starved loop) and nothing more. Recovery from a GIL-holding
hang needs a different process: `core/isolated_calls.py` keeps such calls
out of this process in the first place, and `scripts/bot_watchdog.sh`
restarts the bot from outside when `/api/status` stops answering.

USAGE
-----
    hb = get_heartbeat()
    hb.start_watchdog()                      # once per process
    await hb.run("engine", is_running=lambda: self.is_running)   # per loop
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("TradingBot")

#: One `loop_lag` line per loop per minute.
DEFAULT_WARN_INTERVAL_SEC: float = 60.0
#: How often the watchdog thread wakes up.
DEFAULT_CHECK_INTERVAL_SEC: float = 2.0
#: v-wedge-fix-2026-09-28 fix round 1 (Minor 6): no warning in a loop's first
#: half-minute. Start-up is legitimately lumpy — building the engine alone takes
#: ~13 s — and a line that cries wolf at every boot trains the operator to
#: ignore it.
DEFAULT_STARTUP_GRACE_SEC: float = 30.0


def _config_warn_sec() -> float:
    try:
        from core.config import Config

        return float(Config().LOOP_LAG_WARN_SEC)
    except Exception:                                # pragma: no cover
        return 5.0


class LoopHeartbeat:
    """Registry of per-loop pulses plus the watchdog that reads them.

    Thread-safe by construction: beats and reads take a plain lock and do
    nothing else, so the beat costs microseconds on the loop it guards.
    """

    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        warn_sec: Optional[float] = None,
        warn_interval_sec: float = DEFAULT_WARN_INTERVAL_SEC,
        startup_grace_sec: float = DEFAULT_STARTUP_GRACE_SEC,
    ) -> None:
        self._clock: Callable[[], float] = clock or time.monotonic
        self._warn_sec: Optional[float] = warn_sec
        self._warn_interval_sec = warn_interval_sec
        self.startup_grace_sec = startup_grace_sec
        self._lock = threading.Lock()
        self._last_beat: Dict[str, float] = {}
        self._first_beat: Dict[str, float] = {}
        self._last_warn: Dict[str, float] = {}
        self._watchdog: Optional[threading.Thread] = None

    # ── producer side (runs on the guarded loop) ──────────────────────

    def beat(self, name: str) -> None:
        with self._lock:
            now = self._clock()
            self._last_beat[name] = now
            self._first_beat.setdefault(name, now)

    def forget(self, name: str) -> None:
        """Drop a loop's registration.

        Called when a heartbeat task returns or is cancelled: a loop that has
        legitimately stopped must not produce `loop_lag` warnings for the rest
        of the process's life.
        """
        with self._lock:
            self._last_beat.pop(name, None)
            self._first_beat.pop(name, None)
            self._last_warn.pop(name, None)

    async def run(
        self,
        name: str,
        interval: float = 1.0,
        is_running: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Pulse `name` every `interval` seconds until `is_running()` is
        false. Intended to be registered as a supervised task.

        Always forgets the loop on the way out — a clean stop, a cancellation
        and a crash all mean "this loop is no longer pulsing on purpose".
        """
        try:
            while True:
                if is_running is not None and not is_running():
                    return
                self.beat(name)
                await asyncio.sleep(interval)
        finally:
            self.forget(name)

    # ── consumer side (runs on the watchdog thread) ───────────────────

    def warn_sec(self) -> float:
        return _config_warn_sec() if self._warn_sec is None else self._warn_sec

    def lag(self, name: str) -> Optional[float]:
        """Seconds since `name` last pulsed, or None if it never has."""
        with self._lock:
            last = self._last_beat.get(name)
        return None if last is None else self._clock() - last

    def lags(self) -> Dict[str, float]:
        now = self._clock()
        with self._lock:
            return {name: now - last for name, last in self._last_beat.items()}

    def check_once(self) -> List[Tuple[str, float]]:
        """Log a rate-limited `loop_lag` line for every stalled loop.

        Returns the (name, lag) pairs found over threshold — including the
        ones whose log line was rate-limited away, so callers and tests can
        see the full picture. Never raises: the watchdog must outlive any
        bookkeeping error.
        """
        threshold = self.warn_sec()
        stalled: List[Tuple[str, float]] = []
        try:
            now = self._clock()
            with self._lock:
                first_beats = dict(self._first_beat)
            for name, lag in sorted(self.lags().items()):
                if lag < threshold:
                    continue
                first = first_beats.get(name)
                if first is not None and (now - first) < self.startup_grace_sec:
                    continue          # start-up grace — see the module notes
                stalled.append((name, lag))
                with self._lock:
                    last_warn = self._last_warn.get(name)
                    if (
                        last_warn is not None
                        and (now - last_warn) < self._warn_interval_sec
                    ):
                        continue
                    self._last_warn[name] = now
                logger.warning(
                    "loop_lag loop=%s lag=%.1f threads=%d threshold=%.1f",
                    name, lag, threading.active_count(), threshold,
                )
        except Exception as exc:                     # pragma: no cover
            logger.debug("loop_heartbeat: check failed: %s", exc)
        return stalled

    def start_watchdog(
        self, interval: float = DEFAULT_CHECK_INTERVAL_SEC,
    ) -> threading.Thread:
        """Start (once) the daemon thread that calls `check_once()`.

        Daemon so it can never hold up interpreter shutdown, and idempotent
        so the engine and the API can both ask for it.
        """
        with self._lock:
            if self._watchdog is not None and self._watchdog.is_alive():
                return self._watchdog
            thread = threading.Thread(
                target=self._watchdog_body,
                args=(interval,),
                name="loop-heartbeat-watchdog",
                daemon=True,
            )
            self._watchdog = thread
        thread.start()
        logger.info(
            "loop_heartbeat: watchdog started interval=%.1fs warn_at=%.1fs",
            interval, self.warn_sec(),
        )
        return thread

    def _watchdog_body(self, interval: float) -> None:
        while True:
            self.check_once()
            time.sleep(interval)


_singleton: Optional[LoopHeartbeat] = None
_singleton_lock = threading.Lock()


def get_heartbeat() -> LoopHeartbeat:
    """Process-wide heartbeat registry shared by every loop."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = LoopHeartbeat()
        return _singleton
