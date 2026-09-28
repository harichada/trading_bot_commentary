"""v-wedge-fix-2026-09-28 (Task 4, ruling 2): run yfinance in a subprocess.

WHY A SUBPROCESS AND NOT A THREAD
---------------------------------
`core.provider_calls` bounds a synchronous call by abandoning the *await*.
That is enough for Schwab (a stuck socket parks a thread and nothing else).
It is NOT enough for yfinance-on-curl_cffi: a transfer stalled inside a TLS
write holds the GIL, and a thread that holds the GIL freezes every other
thread in the process — the 2026-09-28 wedge (247/251 threads in
futex_wait, main thread in poll, 1.5 KB stuck in a Yahoo send queue, CPU
0 %). No in-process timeout can recover from that. Another process can:
the parent's `wait_for` fires, the child is killed, the pool is rebuilt.

THE CONTRACT
------------
`await run_isolated(fn, *args, timeout=..., site=...)`

  * `fn` must be a module-level function and its return value picklable —
    see `yf_isolated.py`, which holds every entry point and deliberately
    does not import `core`;
  * spawn context, `max_workers=2`; the pool is torn down (children
    KILLed) and recreated whenever a call times out;
  * returns None on timeout, on a broken pool, or on a child exception —
    every caller already has a "no data" fallback;
  * while a freshly created pool is still booting (spawn re-imports the
    main module in the child, which for this repo is seconds, not
    milliseconds) calls return None immediately instead of burning their
    timeout on a cold child and then killing it mid-boot. Use
    `wait_ready()` if you need the pool up before you call.

KILL SWITCH
-----------
`ISOLATED_YF_ENABLED=0` routes these calls back through the bounded thread
executor (still timeout-bounded, just not GIL-proof). It exists so an
operator can rule the subprocess pool in or out without a code change.
"""
from __future__ import annotations

import asyncio
import logging
import functools
import multiprocessing
import os
import signal
import threading
import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Optional

from core import provider_calls
from core.config import Config

logger = logging.getLogger("TradingBot")

#: Ruling 2: two workers, no more. Each child is a full interpreter.
POOL_WORKERS: int = 2
#: v-wedge-fix-2026-09-28 fix round 1 (Important 2): after a probe failure the
#: pool is torn down so the NEXT call rebuilds it — but not instantly on every
#: call. A child that cannot even run a trivial function means a broken
#: environment, and respawning two interpreters per call would make that worse.
PROBE_RETRY_COOLDOWN_SEC: float = 60.0

_lock = threading.Lock()
_pool: Optional[ProcessPoolExecutor] = None
_ready = threading.Event()
_probe_failed = threading.Event()
_probe_failed_at: Optional[float] = None
#: Bumped on every pool creation. A probe callback from a superseded pool must
#: not report unavailability or tear down its successor — the probe of a pool we
#: killed on purpose always fails, and acting on that would kill the new pool.
_generation: int = 0


# ── pool ownership ────────────────────────────────────────────────────

def is_enabled() -> bool:
    return bool(Config().ISOLATED_YF_ENABLED)


def default_timeout() -> float:
    return provider_calls.history_timeout()


def get_pool_if_created() -> Optional[ProcessPoolExecutor]:
    """The live pool, or None when none has been built. Never builds one."""
    return _pool


def get_pool() -> ProcessPoolExecutor:
    """The spawn-context pool, created on first use.

    Creation submits a readiness probe; `is_ready()` flips once a child has
    answered it, which is how callers tell a booted pool from a cold one.
    """
    global _pool, _generation
    with _lock:
        if _pool is None:
            import yf_isolated                      # child-safe entry points

            _generation += 1
            generation = _generation
            ctx = multiprocessing.get_context("spawn")
            _pool = ProcessPoolExecutor(
                max_workers=POOL_WORKERS,
                mp_context=ctx,
                initializer=yf_isolated.child_init,
            )
            _ready.clear()
            _probe_failed.clear()
            logger.info(
                "isolated_calls: spawn pool created workers=%d (booting)",
                POOL_WORKERS,
            )
            try:
                probe = _pool.submit(yf_isolated.probe)
                probe.add_done_callback(
                    functools.partial(_on_probe_done, generation=generation)
                )
            except Exception as exc:                # pragma: no cover
                logger.warning("isolated_calls: probe submit failed: %s", exc)
        return _pool


def _on_probe_done(future: Any, generation: int = 0) -> None:
    """Probe completion handler.

    v-wedge-fix-2026-09-28 fix round 1 (Important 2): a failed probe used to
    just log at INFO and leave the pool alive-but-never-ready, which disabled
    isolated calls for the whole session. Now it is a rate-limited WARNING and
    the pool is torn down so the next call (after the cooldown) rebuilds it.
    The teardown runs on its own thread because this callback can fire inline
    inside `get_pool()` while the module lock is held.
    """
    global _probe_failed_at
    try:
        if future.cancelled():
            return
        exc = future.exception()
    except Exception:                               # pragma: no cover
        return
    if generation != _generation:
        # Superseded pool (we tore it down, or it was rebuilt). Its probe's
        # outcome says nothing about the pool that is live now.
        return
    if exc is not None:
        _probe_failed_at = time.monotonic()
        _probe_failed.set()
        provider_calls.log_rate_limited(
            "isolated_probe_failed",
            lambda suppressed, exc=exc: (
                f"isolated_pool_unavailable err={type(exc).__name__}: {exc} "
                f"suppressed={suppressed} — pool torn down, retry in "
                f"{PROBE_RETRY_COOLDOWN_SEC:.0f}s; yfinance callers fall back"
            ),
        )
        threading.Thread(
            target=shutdown_pool, args=("probe_failed",),
            name="isolated-pool-teardown", daemon=True,
        ).start()
        return
    _probe_failed.clear()
    _ready.set()
    logger.info("isolated_calls: spawn pool ready")


def is_ready() -> bool:
    return _ready.is_set()


def current_generation() -> int:
    """The live pool's generation. Bumped on creation AND on teardown, so a
    callback holding an older number knows its pool is gone."""
    return _generation


def probe_failed() -> bool:
    """True when the last readiness probe raised in the child."""
    return _probe_failed.is_set()


def _in_probe_cooldown() -> bool:
    if _probe_failed_at is None:
        return False
    return (time.monotonic() - _probe_failed_at) < PROBE_RETRY_COOLDOWN_SEC


def _ensure_pool() -> Optional[ProcessPoolExecutor]:
    """The pool for a real call: None while a failed probe is cooling down."""
    if _pool is None and _in_probe_cooldown():
        return None
    return get_pool()


def wait_ready(timeout: float = 120.0) -> bool:
    """Block until a child has answered the probe, or the probe fails.

    For start-up warming and for tests — never call this from an event loop.
    Returns as soon as the probe fails rather than burning the whole timeout;
    an explicit wait also clears the cooldown, because the caller is asking
    for the pool now.
    """
    global _probe_failed_at
    _probe_failed_at = None
    get_pool()
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if _ready.wait(0.1):
            return True
        if _probe_failed.is_set():
            return False
    return _ready.is_set()


def shutdown_pool(reason: str) -> None:
    """Kill the children and drop the pool.

    `shutdown(cancel_futures=True)` only unqueues work that has not
    started; the process that is *currently* hung must be signalled, so we
    walk the pool's children and KILL them. Anything less leaves the wedge
    in place, which is the entire failure we are fixing.
    """
    global _pool, _generation
    with _lock:
        pool, _pool = _pool, None
        _ready.clear()
        # v-wedge-fix-2026-09-28 fix round 2 (Minor 5): invalidate the
        # generation here, not only on the next creation. Between a teardown
        # and the next `get_pool()` the dead pool's probe can still fire, and
        # it must not be able to report unavailability for a pool nobody owns.
        _generation += 1
    if pool is None:
        return
    logger.warning("isolated_calls: tearing down spawn pool reason=%s", reason)
    for process in list(getattr(pool, "_processes", {}).values()):
        try:
            pid = process.pid
            if pid is None or not process.is_alive():
                continue
            os.kill(pid, signal.SIGKILL)
            logger.warning("isolated_calls: killed child pid=%s", pid)
        except (ProcessLookupError, PermissionError, AttributeError) as exc:
            logger.debug("isolated_calls: child kill skipped: %s", exc)
        except Exception as exc:                    # pragma: no cover
            logger.debug("isolated_calls: child kill failed: %s", exc)
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:                        # pragma: no cover
        logger.debug("isolated_calls: pool shutdown failed: %s", exc)


# ── the helpers ───────────────────────────────────────────────────────

def _log_not_ready(site: str) -> None:
    """Booting is normal and logs at INFO; an unavailable pool (probe failed,
    or cooling down after one) is a WARNING — see _on_probe_done."""
    if _probe_failed.is_set() or _in_probe_cooldown():
        provider_calls.log_rate_limited(
            f"isolated_unavailable:{site}",
            lambda suppressed: (
                f"isolated_pool_unavailable site={site} suppressed={suppressed} "
                f"— probe failed, caller falls back"
            ),
        )
        return
    provider_calls.log_rate_limited(
        f"isolated_warming:{site}",
        lambda suppressed: (
            f"isolated_call_warming site={site} suppressed={suppressed} "
            f"— spawn pool still booting, caller falls back"
        ),
        level=logging.INFO,
    )


def _log_timeout(site: str, timeout: float) -> None:
    provider_calls.log_rate_limited(
        f"isolated_timeout:{site}",
        lambda suppressed: (
            f"isolated_call_timeout site={site} timeout={timeout:.1f}s "
            f"suppressed={suppressed} — child killed, pool rebuilt"
        ),
    )


def _log_failure(site: str, exc: BaseException) -> None:
    provider_calls.log_rate_limited(
        f"isolated_error:{site}",
        lambda suppressed: (
            f"isolated_call_error site={site} err={type(exc).__name__}: {exc} "
            f"suppressed={suppressed}"
        ),
    )


async def run_isolated(
    fn: Callable[..., Any],
    *args: Any,
    timeout: Optional[float] = None,
    site: str = "isolated",
    **kwargs: Any,
) -> Optional[Any]:
    """Run `fn` in the isolated pool, bounded by `timeout`. None on failure."""
    budget = default_timeout() if timeout is None else float(timeout)
    if not is_enabled():
        return await provider_calls.provider_call(
            fn, *args, timeout=budget, site=f"{site}:thread", **kwargs
        )
    pool = _ensure_pool()
    if pool is None or not is_ready():
        _log_not_ready(site)
        return None
    try:
        future = pool.submit(fn, *args, **kwargs)
    except (BrokenExecutor, RuntimeError) as exc:
        shutdown_pool(f"submit_failed:{type(exc).__name__}")
        _log_failure(site, exc)
        return None
    try:
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=budget)
    except asyncio.TimeoutError:
        shutdown_pool(f"timeout:{site}")
        _log_timeout(site, budget)
        return None
    except Exception as exc:
        if isinstance(exc, BrokenExecutor):
            shutdown_pool(f"broken:{site}")
        _log_failure(site, exc)
        return None


def run_isolated_sync(
    fn: Callable[..., Any],
    *args: Any,
    timeout: Optional[float] = None,
    site: str = "isolated",
    **kwargs: Any,
) -> Optional[Any]:
    """Blocking twin of :func:`run_isolated` for synchronous call sites
    (the fallback data provider's methods are plain functions)."""
    budget = default_timeout() if timeout is None else float(timeout)
    if not is_enabled():
        return provider_calls.provider_call_sync(
            fn, *args, timeout=budget, site=f"{site}:thread", **kwargs
        )
    pool = _ensure_pool()
    if pool is None or not is_ready():
        _log_not_ready(site)
        return None
    try:
        future = pool.submit(fn, *args, **kwargs)
    except (BrokenExecutor, RuntimeError) as exc:
        shutdown_pool(f"submit_failed:{type(exc).__name__}")
        _log_failure(site, exc)
        return None
    try:
        return future.result(timeout=budget)
    except FutureTimeoutError:
        shutdown_pool(f"timeout:{site}")
        _log_timeout(site, budget)
        return None
    except Exception as exc:
        if isinstance(exc, BrokenExecutor):
            shutdown_pool(f"broken:{site}")
        _log_failure(site, exc)
        return None
