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
import multiprocessing
import os
import signal
import threading
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Optional

from core import provider_calls
from core.config import Config

logger = logging.getLogger("TradingBot")

#: Ruling 2: two workers, no more. Each child is a full interpreter.
POOL_WORKERS: int = 2

_lock = threading.Lock()
_pool: Optional[ProcessPoolExecutor] = None
_ready = threading.Event()


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
    global _pool
    with _lock:
        if _pool is None:
            import yf_isolated                      # child-safe entry points

            ctx = multiprocessing.get_context("spawn")
            _pool = ProcessPoolExecutor(
                max_workers=POOL_WORKERS,
                mp_context=ctx,
                initializer=yf_isolated.child_init,
            )
            _ready.clear()
            logger.info(
                "isolated_calls: spawn pool created workers=%d (booting)",
                POOL_WORKERS,
            )
            try:
                probe = _pool.submit(yf_isolated.probe)
                probe.add_done_callback(_on_probe_done)
            except Exception as exc:                # pragma: no cover
                logger.warning("isolated_calls: probe submit failed: %s", exc)
        return _pool


def _on_probe_done(future: Any) -> None:
    try:
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.warning("isolated_calls: pool probe failed: %s", exc)
            return
    except Exception:                               # pragma: no cover
        return
    _ready.set()
    logger.info("isolated_calls: spawn pool ready")


def is_ready() -> bool:
    return _ready.is_set()


def wait_ready(timeout: float = 120.0) -> bool:
    """Block until a child has answered the probe. For start-up warming
    and for tests — never call this from an event loop."""
    get_pool()
    return _ready.wait(timeout)


def shutdown_pool(reason: str) -> None:
    """Kill the children and drop the pool.

    `shutdown(cancel_futures=True)` only unqueues work that has not
    started; the process that is *currently* hung must be signalled, so we
    walk the pool's children and KILL them. Anything less leaves the wedge
    in place, which is the entire failure we are fixing.
    """
    global _pool
    with _lock:
        pool, _pool = _pool, None
        _ready.clear()
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
    pool = get_pool()
    if not is_ready():
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
    pool = get_pool()
    if not is_ready():
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
