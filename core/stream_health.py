"""Pure-function decision logic for Schwab stream health.

v-stream-watchdog-recover-2026-05-12. Extracted from the watchdog in
``core/schwab_stream.py`` so the silent-death-detection rule can be
exercised in isolation, with no asyncio, no clock, no I/O. The
watchdog itself becomes a thin shell that calls this and logs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional


def is_stream_silently_dead(
    last_tick_at: Optional[datetime],
    now: datetime,
    threshold_sec: float,
    connect_at: Optional[datetime] = None,
    connect_grace_sec: float = 60.0,
) -> bool:
    """Return True iff the stream should be treated as silently dead.

    Inputs
    ------
    last_tick_at
        Timestamp of the most recent tick from any subscribed symbol.
        ``None`` means no tick has ever arrived on this connection.
    now
        Reference clock. Caller passes ``datetime.now(timezone.utc)``;
        tests pass synthetic values.
    threshold_sec
        Max allowed tick-gap before declaring death. Production is
        120.0 s (one minute is too aggressive — illiquid-symbol prints
        legitimately gap that long; two minutes during RTH is suspect).
    connect_at
        Timestamp the current connection was established. Used to apply
        ``connect_grace_sec`` — we don't fire on a fresh connection
        that's still waiting for its first tick. Pass ``None`` to
        disable the grace check (then a ``None`` ``last_tick_at`` is
        immediately considered dead regardless of how recent the
        connect was).
    connect_grace_sec
        Seconds after ``connect_at`` during which a ``None``
        ``last_tick_at`` is tolerated. Default 60 s.

    Output
    ------
    bool
        ``True`` if the caller should treat this as silent death and
        force a reconnect. ``False`` otherwise.

    Decision table (all times in seconds)
    -------------------------------------
    | last_tick_at | connect_age | rule                              | result |
    |--------------|-------------|-----------------------------------|--------|
    | None         | < grace     | still inside connect grace        | False  |
    | None         | ≥ grace     | no ticks since connect, dead      | True   |
    | now - 0      | any         | just ticked                       | False  |
    | now - thresh | any         | exactly at threshold (boundary)   | False  |
    | now - >thresh| any         | past threshold                    | True   |
    | future       | any         | clock skew, treat conservatively  | False  |

    The boundary is ``> threshold_sec`` (strict). At exactly the
    threshold we consider the stream still alive — this prevents a
    flapping edge from churning reconnects.
    """
    if last_tick_at is None:
        if connect_at is None:
            return True
        connect_age = (now - connect_at).total_seconds()
        return connect_age >= connect_grace_sec

    tick_age = (now - last_tick_at).total_seconds()
    if tick_age < 0:
        return False
    return tick_age > threshold_sec
