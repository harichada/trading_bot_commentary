"""Swing (multi-day) exit helpers: time-stop, earnings-eve flatten.

Pure helpers with no engine or research dependencies, wired into
`core/engine.py` under the `v-strategy-fit-2026-09-25` comments.
"""
from __future__ import annotations

import functools
import logging
import time
from datetime import date, datetime, timedelta
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_TIME_STOP_CUTOFF = (15, 50)  # 15:50 local (ET) — same cutoff as flatten checks


def sessions_held(entry_time: datetime, now: datetime) -> int:
    """Number of weekdays from entry_time.date() to now.date(), inclusive (min 1)."""
    start = entry_time.date()
    end = now.date()
    if end < start:
        return 1
    count = 0
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon..Fri
            count += 1
        d += timedelta(days=1)
    return max(count, 1)


def is_multiday(reasoning: dict) -> bool:
    return bool(reasoning) and reasoning.get("hold_kind") == "multiday"


def should_time_stop(
    reasoning: dict, entry_time: datetime, now: datetime, default_max: int
) -> Tuple[bool, str]:
    """Time-stop for multi-day (swing) positions.

    Fires once the position has been held for `max_hold` sessions or more
    and it's at/after the 15:50 cutoff.

    v-strategy-fit-review2-2026-09-25 (Important 6): `max_hold_sessions`
    comes from a signal's `reasoning` dict, which is persisted to
    trading_state.json and reloaded across restarts — so it can be a
    string, None, or anything else a past writer left there. A non-numeric
    value falls back to `default_max` rather than raising into the exit
    check, where a ValueError would have skipped the hard SL/TP checks
    that run after this one.
    """
    if not is_multiday(reasoning or {}):
        return False, ""
    try:
        max_hold = int(reasoning.get("max_hold_sessions") or default_max)
    except (TypeError, ValueError):
        logger.debug(
            "should_time_stop: non-numeric max_hold_sessions %r; using default %s",
            reasoning.get("max_hold_sessions"), default_max,
        )
        max_hold = int(default_max)
    held = sessions_held(entry_time, now)
    cutoff = now.replace(
        hour=_TIME_STOP_CUTOFF[0], minute=_TIME_STOP_CUTOFF[1], second=0, microsecond=0
    ).time()
    if held >= max_hold and now.time() >= cutoff:
        return True, "swing_time_stop"
    return False, ""


def next_weekday(d: date) -> date:
    """Next calendar weekday (Mon-Fri) strictly after `d`."""
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


@functools.lru_cache(maxsize=256)
def _load_earnings_dates_cached(root: str, symbol: str, hour_bucket: int) -> frozenset:
    path = f"{root}/earnings/{symbol.upper()}.parquet"
    try:
        df = pd.read_parquet(path, columns=["date"])
        dates = pd.to_datetime(df["date"]).dt.date
        return frozenset(dates.tolist())
    except Exception:
        # Never raise into the exit-check path: missing dir, malformed
        # parquet, garbage date values, or a missing parquet engine
        # (ImportError) must all fail open to an empty earnings set.
        logger.debug("load_earnings_dates: failed to load %s", path, exc_info=True)
        return frozenset()


def load_earnings_dates(root: str, symbol: str) -> set:
    """Read `<root>/earnings/<SYM>.parquet` (`date` column) if present, else empty set.

    Cached per process with a 1-hour TTL via an hour-bucketed lru_cache key.
    """
    hour_bucket = int(time.time() // 3600)
    return set(_load_earnings_dates_cached(root, symbol, hour_bucket))


def should_earnings_eve_flatten(symbol: str, now: datetime, earnings: set) -> Tuple[bool, str]:
    """Flatten before an earnings print: fires on the eve of an earnings date at/after 15:50."""
    cutoff = now.replace(
        hour=_TIME_STOP_CUTOFF[0], minute=_TIME_STOP_CUTOFF[1], second=0, microsecond=0
    ).time()
    if next_weekday(now.date()) in earnings and now.time() >= cutoff:
        return True, "earnings_eve_flatten"
    return False, ""
