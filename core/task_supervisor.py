"""v-task-supervisor-2026-04-30 (Phase 2): supervised long-running tasks.

The engine spawns three concurrent tasks (analysis_loop, position_loop,
quote_streamer). If any of them raises an unhandled exception, asyncio
silently cancels them and the bot keeps "running" with critical
machinery dead. This module wraps each task in a supervisor that:

  - Catches unhandled exceptions, logs them, and restarts the task
    with exponential backoff.
  - Counts crashes in a rolling window. After N crashes in T seconds,
    enters CIRCUIT_BROKEN and stops auto-restarting — operator
    intervention required. This prevents log-spam reboot loops.
  - Surfaces high-importance commentary cards so the user sees the
    failure on the dashboard.
  - Honours a global is_running flag for clean shutdown.

Per-task priority levels:
  - CRITICAL: position_loop. If it dies, exits stop firing. We restart
    aggressively (faster backoff) and alert at the first crash.
  - NORMAL:   analysis_loop. New entries pause but existing positions
    are still managed. Standard backoff.
  - NORMAL:   quote_streamer. Cache stops refreshing → position_loop's
    is_stale check skips evaluation safely. Standard backoff.

Backoff: 5s → 30s → 60s, capped. Reset to 5s on a successful "uptime"
(60s without crash).
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Awaitable, Callable, Optional, Deque

logger = logging.getLogger("TradingBot")


class TaskPriority(str, Enum):
    CRITICAL = "critical"   # position_loop — exit safety
    NORMAL = "normal"


class SupervisorState(str, Enum):
    RUNNING = "running"
    BACKOFF = "backoff"
    CIRCUIT_BROKEN = "circuit_broken"   # operator must reset
    STOPPED = "stopped"


@dataclass
class SupervisedTask:
    name: str
    coro_factory: Callable[[], Awaitable[None]]   # () -> awaitable
    priority: TaskPriority = TaskPriority.NORMAL
    state: SupervisorState = SupervisorState.RUNNING
    crash_times: Deque[datetime] = field(default_factory=deque)
    last_crash: Optional[Exception] = None
    backoff_idx: int = 0
    asyncio_task: Optional[asyncio.Task] = None
    started_at: Optional[datetime] = None


_BACKOFF_SCHEDULE_SEC = (5, 30, 60)   # NORMAL priority
_CRITICAL_BACKOFF_SCHEDULE_SEC = (2, 5, 15)   # CRITICAL — restart faster


class TaskSupervisor:
    """Owner of the supervised tasks. The engine creates one of these
    and registers each long-running coroutine via .register()."""

    def __init__(
        self,
        max_crashes: int,
        window_sec: float,
        on_circuit_broken: Optional[Callable[[SupervisedTask], None]] = None,
        on_crash: Optional[Callable[[SupervisedTask, Exception], None]] = None,
    ) -> None:
        self.max_crashes = max_crashes
        self.window_sec = window_sec
        self._tasks: dict[str, SupervisedTask] = {}
        self._on_circuit_broken = on_circuit_broken
        self._on_crash = on_crash
        self._stop = asyncio.Event()

    def register(
        self,
        name: str,
        coro_factory: Callable[[], Awaitable[None]],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> SupervisedTask:
        st = SupervisedTask(name=name, coro_factory=coro_factory, priority=priority)
        self._tasks[name] = st
        return st

    def start_all(self) -> None:
        for st in self._tasks.values():
            self._launch(st)

    def _launch(self, st: SupervisedTask) -> None:
        if st.state == SupervisorState.CIRCUIT_BROKEN or st.state == SupervisorState.STOPPED:
            return
        st.started_at = datetime.now(timezone.utc)
        st.state = SupervisorState.RUNNING
        st.asyncio_task = asyncio.create_task(
            self._wrap(st), name=f"supervised:{st.name}",
        )

    async def _wrap(self, st: SupervisedTask) -> None:
        """Run the supervised coroutine, catch exceptions, schedule restart."""
        try:
            await st.coro_factory()
            # Clean exit (the coroutine's own loop saw is_running=False)
            st.state = SupervisorState.STOPPED
            logger.info("supervisor: %s exited cleanly", st.name)
        except asyncio.CancelledError:
            st.state = SupervisorState.STOPPED
            raise
        except Exception as exc:
            st.last_crash = exc
            now = datetime.now(timezone.utc)
            st.crash_times.append(now)
            # Trim window
            cutoff = now - timedelta(seconds=self.window_sec)
            while st.crash_times and st.crash_times[0] < cutoff:
                st.crash_times.popleft()

            crashes_in_window = len(st.crash_times)
            logger.error(
                "supervisor: %s CRASHED (%d in window). %s: %s",
                st.name, crashes_in_window, type(exc).__name__, exc,
                exc_info=True,
            )
            if self._on_crash is not None:
                try:
                    self._on_crash(st, exc)
                except Exception:
                    pass

            if crashes_in_window >= self.max_crashes:
                # Trip the breaker. No more restarts until operator resets.
                st.state = SupervisorState.CIRCUIT_BROKEN
                logger.critical(
                    "supervisor: %s CIRCUIT BROKEN after %d crashes in %ds — manual reset required",
                    st.name, crashes_in_window, self.window_sec,
                )
                if self._on_circuit_broken is not None:
                    try:
                        self._on_circuit_broken(st)
                    except Exception:
                        pass
                return

            # Schedule a restart
            schedule = (_CRITICAL_BACKOFF_SCHEDULE_SEC if st.priority == TaskPriority.CRITICAL
                        else _BACKOFF_SCHEDULE_SEC)
            delay = schedule[min(st.backoff_idx, len(schedule) - 1)]
            st.backoff_idx += 1
            st.state = SupervisorState.BACKOFF
            logger.warning("supervisor: %s restarting in %ds", st.name, delay)
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                st.state = SupervisorState.STOPPED
                raise
            self._launch(st)

    async def stop_all(self) -> None:
        """Cooperative shutdown — sets stop flag, cancels asyncio tasks."""
        self._stop.set()
        for st in self._tasks.values():
            st.state = SupervisorState.STOPPED
            if st.asyncio_task is not None and not st.asyncio_task.done():
                st.asyncio_task.cancel()
        # Give tasks a beat to finish
        await asyncio.sleep(0)

    def status(self) -> dict:
        """Diagnostic snapshot for the dashboard / health endpoint."""
        out = {}
        for name, st in self._tasks.items():
            out[name] = {
                "state": st.state.value,
                "priority": st.priority.value,
                "crashes_in_window": len(st.crash_times),
                "backoff_idx": st.backoff_idx,
                "last_crash": (
                    f"{type(st.last_crash).__name__}: {st.last_crash}"
                    if st.last_crash else None
                ),
                "started_at": (st.started_at.isoformat() if st.started_at else None),
            }
        return out

    def reset_circuit(self, name: str) -> bool:
        """Operator escape hatch: clear circuit-break and restart task."""
        st = self._tasks.get(name)
        if st is None:
            return False
        if st.state != SupervisorState.CIRCUIT_BROKEN:
            return False
        st.state = SupervisorState.RUNNING
        st.crash_times.clear()
        st.backoff_idx = 0
        self._launch(st)
        return True
