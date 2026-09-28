"""v-wedge-fix-2026-09-28 (Task 4, ruling 1): bounded synchronous provider calls.

WHY THIS EXISTS
---------------
2026-09-28 09:37 ET, live process pid 1083970: `/api/status` stopped
answering, news ticks stretched to 16-25 s, the 09:00 desk plan was never
built, CPU sat at 0 % and 247 of 251 threads were parked in `futex_wait`
while the main thread sat in `do_sys_poll`. One TCP socket to Yahoo had
~1.5 KB stuck in its send queue. The engine was wedged for the whole
session by a single HTTP call that never came back.

The mechanism: `core/engine.py` called `self.data_provider.get_quote(...)`
and `.get_market_data(...)` *inline* from coroutines. A synchronous HTTP
call on the event loop has no timeout of its own, so a stalled socket
stops every other task on that loop — position management, exits, the API.

THE CONTRACT
------------
`await provider_call(fn, *args, timeout=..., site=...)`

  * runs `fn` in a DEDICATED, bounded `ThreadPoolExecutor` — never
    asyncio's default pool, so an abandoned provider thread cannot starve
    `asyncio.to_thread` work elsewhere in the process;
  * bounds the *await* with `asyncio.wait_for`;
  * on timeout returns None and logs at most once per minute per call
    site (with the suppressed count), so each caller keeps its existing
    fallback behaviour (skip the symbol, keep the last quote, …);
  * lets every other exception propagate unchanged, so the try/except
    that already wraps each call site keeps behaving exactly as today.

`provider_call_sync(...)` is the same contract for call sites that are
themselves synchronous (e.g. functions handed to a callback or already
running in a worker thread). It bounds the *caller*, not the worker.

THE BROKER CONTRACT IS DIFFERENT (fix round 2)
----------------------------------------------
`broker_call(...)` / `broker_call_sync(...)` bound Schwab order and account
calls the same way, but on timeout they RAISE `BrokerTimeoutError` instead of
returning None. A market-data timeout means "no data, use your fallback"; an
order timeout means "we do not know whether the broker accepted this order".
Returning None there would read as "no order placed" — the one conclusion the
caller is not entitled to draw — so it raises, every existing `except
Exception` handler treats it as a broker error, and no retry is implied.

`last_call_timed_out(site)` reports whether the LAST call at a given site
timed out, so a caller that gets None can tell "the provider timed out" from
"this data legitimately does not exist" and fail closed on the former.

HONEST LIMIT
------------
`asyncio.wait_for` abandons the await; it cannot abandon the thread. A
hung provider thread keeps running until its socket gives up (news
strategy installs `socket.setdefaulttimeout(15)` as a global backstop).
That is why the pool is separate and bounded: leaked threads are capped
at PROVIDER_EXECUTOR_WORKERS and cannot touch anything else. A call that
hangs while *holding the GIL* (curl_cffi in a TLS write) freezes the whole
interpreter regardless of threads — that class of call belongs in
`core.isolated_calls` (a subprocess), not here.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, Optional

from core.config import Config

logger = logging.getLogger("TradingBot")

#: One WARNING per call site per minute; the rest are counted and reported.
TIMEOUT_LOG_INTERVAL_SEC: float = 60.0

_lock = threading.Lock()
_executor: Optional[ThreadPoolExecutor] = None
_last_log_at: Dict[str, float] = {}
_suppressed: Dict[str, int] = {}
_clock: Callable[[], float] = time.monotonic
#: Per-site outcome of the most recent call: True when it timed out. Callers
#: that receive None use this to tell a timeout from legitimately absent data
#: (fix round 2, Minor 6 — the market-context entry vetoes fail closed on a
#: timeout and stay fail-open when the data simply is not there).
_timed_out: Dict[str, bool] = {}


class BrokerTimeoutError(TimeoutError):
    """A broker order/account call did not answer inside its budget.

    Subclasses the builtin `TimeoutError` (itself an `OSError`) so that every
    existing `except Exception` handler around a Schwab call treats it as a
    broker error without modification, while code that cares can catch it
    precisely — the close path audits `broker_timeout_close` on it, and the
    entry path audits `broker_timeout_entry` and never retries, because after
    a timeout the order's fate at the broker is unknown.
    """

    def __init__(self, site: str, timeout: float) -> None:
        self.site = site
        self.timeout = float(timeout)
        super().__init__(
            f"broker_call_timeout site={site} timeout={self.timeout:.1f}s "
            f"— order/account state at the broker is UNKNOWN"
        )


# ── configuration helpers ─────────────────────────────────────────────

def default_timeout() -> float:
    """Quote-shaped calls. Same budget the quote streamer already uses."""
    return float(Config().QUOTE_FETCH_TIMEOUT_SEC)


def history_timeout() -> float:
    """Bar/history-shaped calls — a year of dailies legitimately takes
    longer than a quote, so they get their own (larger) budget."""
    return float(Config().PROVIDER_HISTORY_TIMEOUT_SEC)


def broker_timeout() -> float:
    """Order/account calls. Generous on purpose: cutting an order call short
    is worse than waiting for it, because the outcome is then unknown."""
    return float(Config().BROKER_CALL_TIMEOUT_SEC)


# ── executor ownership ────────────────────────────────────────────────

def get_executor() -> ThreadPoolExecutor:
    """The one dedicated provider pool. Created on first use because
    Config() reads .env/yaml and must not run at import time."""
    global _executor
    with _lock:
        if _executor is None:
            workers = max(1, int(Config().PROVIDER_EXECUTOR_WORKERS))
            _executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="provider-call",
            )
            logger.info("provider_calls: executor created workers=%d", workers)
        return _executor


def set_clock(clock: Callable[[], float]) -> None:
    """Test seam for the rate limiter. Production never calls this."""
    global _clock
    _clock = clock


def reset_state() -> None:
    """Drop the rate-limiter state and the pool (tests only). The old
    pool is released without waiting — abandoned provider threads must
    never block a teardown."""
    global _executor, _clock
    with _lock:
        old, _executor = _executor, None
        _last_log_at.clear()
        _suppressed.clear()
        _timed_out.clear()
        _clock = time.monotonic
    if old is not None:
        old.shutdown(wait=False)


# ── the helpers ───────────────────────────────────────────────────────

def log_rate_limited(
    key: str,
    build_message: Callable[[int], str],
    level: int = logging.WARNING,
) -> bool:
    """Log at most one line per `key` per TIMEOUT_LOG_INTERVAL_SEC.

    `build_message(suppressed)` is only called when the line is actually
    emitted, and receives how many reports were swallowed since the last
    one — a degraded provider must be visible without flooding the log.
    Returns True when it logged. Never raises: a logging failure must not
    turn a degraded fetch into a crashed loop.
    """
    try:
        now = _clock()
        with _lock:
            last = _last_log_at.get(key)
            if last is not None and (now - last) < TIMEOUT_LOG_INTERVAL_SEC:
                _suppressed[key] = _suppressed.get(key, 0) + 1
                return False
            suppressed = _suppressed.pop(key, 0)
            _last_log_at[key] = now
        logger.log(level, "%s", build_message(suppressed))
        return True
    except Exception as exc:                      # pragma: no cover - defensive
        logger.debug("provider_calls: rate-limited log failed: %s", exc)
        return False


def _log_timeout(site: str, timeout: float) -> None:
    log_rate_limited(
        f"timeout:{site}",
        lambda suppressed: (
            f"provider_call_timeout site={site} timeout={timeout:.1f}s "
            f"suppressed={suppressed} threads={threading.active_count()} "
            f"— caller falls back"
        ),
    )


def last_call_timed_out(site: str) -> bool:
    """Did the most recent call at `site` time out?

    False for a site that has never been called. Deliberately last-outcome,
    not ever-timed-out: a caller asks this to interpret the None it just got.
    """
    with _lock:
        return bool(_timed_out.get(site, False))


def _record_outcome(site: str, timed_out: bool) -> None:
    with _lock:
        _timed_out[site] = timed_out


async def provider_call(
    fn: Callable[..., Any],
    *args: Any,
    timeout: Optional[float] = None,
    site: str = "provider",
    **kwargs: Any,
) -> Optional[Any]:
    """Run a synchronous provider call off the event loop, bounded.

    Returns the call's value, or None when it did not answer within
    `timeout`. Non-timeout exceptions propagate to the caller.
    """
    budget = default_timeout() if timeout is None else float(timeout)
    call = functools.partial(fn, *args, **kwargs)
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(get_executor(), call), timeout=budget,
        )
    except asyncio.TimeoutError:
        _record_outcome(site, True)
        _log_timeout(site, budget)
        return None
    _record_outcome(site, False)
    return result


def provider_call_sync(
    fn: Callable[..., Any],
    *args: Any,
    timeout: Optional[float] = None,
    site: str = "provider",
    **kwargs: Any,
) -> Optional[Any]:
    """Blocking twin of :func:`provider_call` for synchronous call sites.

    The caller is bounded by `timeout`; the worker thread may outlive it
    (see HONEST LIMIT above). Must never be called from *inside* the
    provider pool — that would deadlock a saturated pool. All current
    callers run either on an event loop or in asyncio's default pool.
    """
    budget = default_timeout() if timeout is None else float(timeout)
    future = get_executor().submit(functools.partial(fn, *args, **kwargs))
    try:
        result = future.result(timeout=budget)
    except FutureTimeoutError:
        future.cancel()          # only helps if it never started
        _record_outcome(site, True)
        _log_timeout(site, budget)
        return None
    _record_outcome(site, False)
    return result


# ── broker calls: raise, never return None ────────────────────────────

def _log_broker_timeout(site: str, timeout: float) -> None:
    log_rate_limited(
        f"broker_timeout:{site}",
        lambda suppressed: (
            f"broker_call_timeout site={site} timeout={timeout:.1f}s "
            f"suppressed={suppressed} threads={threading.active_count()} "
            f"— outcome at the broker is UNKNOWN; raising BrokerTimeoutError"
        ),
        level=logging.ERROR,
    )


async def broker_call(
    fn: Callable[..., Any],
    *args: Any,
    timeout: Optional[float] = None,
    site: str = "broker",
    **kwargs: Any,
) -> Any:
    """Await a Schwab order/account call off the loop, bounded.

    Returns the broker's response object. Raises `BrokerTimeoutError` if it
    does not answer in time — see THE BROKER CONTRACT above. Every other
    exception propagates unchanged.
    """
    budget = broker_timeout() if timeout is None else float(timeout)
    call = functools.partial(fn, *args, **kwargs)
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(get_executor(), call), timeout=budget,
        )
    except asyncio.TimeoutError:
        _record_outcome(site, True)
        _log_broker_timeout(site, budget)
        raise BrokerTimeoutError(site, budget) from None
    _record_outcome(site, False)
    return result


def broker_call_sync(
    fn: Callable[..., Any],
    *args: Any,
    timeout: Optional[float] = None,
    site: str = "broker",
    **kwargs: Any,
) -> Any:
    """Blocking twin of :func:`broker_call` for synchronous call sites."""
    budget = broker_timeout() if timeout is None else float(timeout)
    future = get_executor().submit(functools.partial(fn, *args, **kwargs))
    try:
        result = future.result(timeout=budget)
    except FutureTimeoutError:
        future.cancel()
        _record_outcome(site, True)
        _log_broker_timeout(site, budget)
        raise BrokerTimeoutError(site, budget) from None
    _record_outcome(site, False)
    return result
