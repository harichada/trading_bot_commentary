"""Pure frame/verdict helpers for the live fit lane.

Extracted from `strategies/fit_lane.py`
(v-strategy-fit-review2-2026-09-25) to keep both modules inside the
project's 400-line file budget. Everything here is pure: frames in,
frames out, no I/O, no engine imports, no mutation of the caller's
inputs. `strategies/fit_lane.py` re-exports these names, so
`from strategies.fit_lane import with_synthetic_next_bar` still works.
"""
from __future__ import annotations

import math
from datetime import time as dt_time
from typing import Any, Iterable, Optional

import pandas as pd

_NEWS_FRAME_COLUMNS = ["created_at", "headline", "source", "sentiment"]

# v-strategy-fit-review-2026-09-25: bar snapshot must be no older than this,
# and the synthetic "now" bar must not extend past the RTH close, or the
# adapter skips rather than computing entries on stale data.
_FRESHNESS_LIMIT = pd.Timedelta(minutes=15)
_RTH_CLOSE = dt_time(15, 55)


def _to_naive_et(ts: pd.Timestamp) -> pd.Timestamp:
    """Tz-aware timestamps are converted to ET then stripped; naive ones
    are assumed to already be ET."""
    if ts.tzinfo is not None:
        return ts.tz_convert("America/New_York").tz_localize(None)
    return ts


def news_frame_from_items(items: Iterable[Any]) -> pd.DataFrame:
    """Pure conversion of NewsBus items (ScoredNewsItem, or anything with
    similar attributes) into the DataFrame shape families expect:
    created_at (naive ET), headline, source, sentiment. Empty frame for
    no items, read defensively via getattr for attribute variance.
    """
    rows = []
    for item in items or []:
        pub = getattr(item, "published_time", None)
        if pub is None:
            pub = getattr(item, "created_at", None)
        ts = _to_naive_et(pd.Timestamp(pub)) if pub is not None else pd.NaT
        sentiment = getattr(item, "sentiment_score", None)
        if sentiment is None:
            sentiment = getattr(item, "sentiment", 0.0)
        rows.append({
            "created_at": ts,
            "headline": getattr(item, "headline", ""),
            "source": getattr(item, "source", ""),
            "sentiment": float(sentiment or 0.0),
        })
    return pd.DataFrame(rows, columns=_NEWS_FRAME_COLUMNS)


def with_synthetic_next_bar(bars5: pd.DataFrame, now_px: float,
                            next_ts: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Append one synthetic bar representing "right now":
    open=high=low=close=now_px, volume=0. Families produce entries at bar
    i+1's open, so an entry landing on this synthetic timestamp means
    "enter now at the current price".

    `next_ts` defaults to `bars5.index[-1] + 5min`; the adapter passes
    `next_rth_bar(...)` so a 15:55 last closed bar yields the next
    session's 09:30 open instead of a nonexistent 16:00 bar
    (v-strategy-fit-review2-2026-09-25, Important 4).

    v-strategy-fit-review-2026-09-25: pd.concat drops DataFrame.attrs, so
    attrs are copied through explicitly (the adapter relies on
    bars.attrs["symbol"] surviving this call).
    """
    if bars5.empty:
        return bars5
    next_ts = bars5.index[-1] + pd.Timedelta(minutes=5) if next_ts is None else pd.Timestamp(next_ts)
    row = pd.DataFrame(
        {"open": [now_px], "high": [now_px], "low": [now_px], "close": [now_px], "volume": [0]},
        index=pd.Index([next_ts], name=bars5.index.name),
    )
    out = pd.concat([bars5, row])
    out.attrs = dict(bars5.attrs)
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    """v-strategy-fit-review2-2026-09-25 (Minor): NaN must clamp to the
    floor, not the ceiling. `min(hi, nan)` returns nan and `max(lo, nan)`
    returns nan, so an unguarded clamp turned a missing/NaN score into
    strength=nan — and `nan * 2.0` compared False against every bound,
    silently sizing a cold-start row as if it were maximally strong."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(v):
        return lo
    return max(lo, min(hi, v))


def live_eligible(row: Any) -> bool:
    """v-strategy-fit-review2-2026-09-25 (Critical 1): a fit row may back a
    LIVE entry only when the walk-forward pipeline actually demonstrated an
    edge for that (symbol, family, side) — verdict PASS from source
    "walkforward". WATCH, FAIL, and cold-start PASS-shaped rows (which the
    cold-start classifier can never emit as PASS, but defend anyway) all
    stay shadow-only, no matter what FIT_LANE_LIVE_ENTRIES_ENABLED says.
    """
    if row is None:
        return False
    return str(getattr(row, "verdict", "")) == "PASS" and str(getattr(row, "source", "")) == "walkforward"


def next_rth_bar(last_closed_ts: pd.Timestamp) -> pd.Timestamp:
    """v-strategy-fit-review2-2026-09-25 (Important 4): timestamp of the bar
    that follows `last_closed_ts` on the RTH 5-minute grid.

    A last closed bar at 15:55 (the session's final bar) is followed by the
    *next weekday's* 09:30 open, not by 16:00 — that's the bar session-open
    families (`pullback_trend`, `mr_20d`, `vol_breakout`, `post_news_drift`)
    evaluate at (`bar_of_day == 0`). Every other bar is followed by +5min.
    """
    ts = pd.Timestamp(last_closed_ts)
    if ts.time() >= _RTH_CLOSE:
        nxt = ts.normalize() + pd.Timedelta(days=1)
        while nxt.weekday() >= 5:  # Sat/Sun -> next weekday
            nxt += pd.Timedelta(days=1)
        return nxt + pd.Timedelta(hours=9, minutes=30)
    return ts + pd.Timedelta(minutes=5)


def with_daily_placeholder(daily: pd.DataFrame, session_date: pd.Timestamp) -> pd.DataFrame:
    """v-strategy-fit-review2-2026-09-25 (Important 4): ensure `daily` has a
    row for `session_date`.

    `daily_features()` shifts the daily frame by one row, so row D carries
    values computed through D-1. `indicator_frame` then reindexes those onto
    `bars5.index.normalize()`. On a session-open synthetic bar the new
    session's date is not in the provider's daily frame yet (today's daily
    bar doesn't exist until the close), so every `d_*` feature would be NaN
    and `bar_of_day == 0` families would bail on the NaN check. A
    placeholder row (NaN prices, volume 0) for that date gives the
    one-session lag something to resolve against without inventing any
    price data.
    """
    if daily is None or daily.empty:
        return daily
    date = pd.Timestamp(session_date).normalize()
    if date in daily.index:
        return daily
    row = pd.DataFrame(
        {c: [0.0 if c == "volume" else float("nan")] for c in daily.columns},
        index=pd.Index([date], name=daily.index.name),
    )
    out = pd.concat([daily, row]).sort_index()
    out.attrs = dict(daily.attrs)
    return out
