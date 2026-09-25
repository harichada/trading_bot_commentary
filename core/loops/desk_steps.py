"""v-playbook-desk-b2-2026-09-25 (Task 5): the desk loop's steps, as total functions.

`core/loops/desk_loop.py` is a schedule: every 30 s it asks "has the ET clock
crossed 09:00 / 09:35 / 16:05 yet, and did this step already fire today?" and
then calls one step. Everything that step needs to *decide* — the clock maths,
the breadth reading, the candidate list, the provider's opening bars — lives
here so it can be tested without an engine. The trade-row statistics (the
losing-day streak, the journal's trailing summary and playbook changes) live in
the sibling `core/loops/desk_trade_stats.py` to keep both files under the
400-line limit, and are re-exported here so callers import one module.

Contract for every function in this module: it never raises. The desk runs on
the live trading path; a malformed Parquet file, a provider that returns None or
a breadth dict in an unexpected shape degrades to "no reading" (None / [] / 0),
never to an exception. The loop's own try/except is the second line of defence,
not the first.

Nothing here imports `research`: the nightly context Parquet is read as a *data*
file, the same one-way arrangement `core/strategy_fit_bars.py` has with that
tree.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from core.loops.desk_trade_stats import (daily_r_stats, playbook_changes_today,
                                          trailing_summary)

logger = logging.getLogger(__name__)

__all__ = [
    "ET_TZ", "SCAN_CONCURRENCY", "CONTEXT_RELPATH",
    "now_et", "session_date_str", "is_session_day", "parse_et_time", "crossed",
    "breadth_score", "daily_loss_pct",
    "scan_symbols", "load_scan_universe", "fetch_today_bars5",
    # Re-exported from core/loops/desk_trade_stats.py (split for file size):
    "daily_r_stats", "trailing_summary", "playbook_changes_today",
]

ET_TZ = ZoneInfo("America/New_York")
SCAN_CONCURRENCY = 4                    # provider calls in flight during a scan
CONTEXT_RELPATH = ("context", "universe_daily.parquet")
RTH_START, RTH_END = "09:30", "15:55"   # the only bars the in-play features may see
AD_SCALE = 20.0                         # signed advance-decline that counts as one-sided


# ────────────────────────────────────────────────────────────────────
# clock
# ────────────────────────────────────────────────────────────────────

def now_et() -> datetime:
    """The wall clock the desk runs on."""
    return datetime.now(ET_TZ)


def session_date_str(now: datetime) -> str:
    """`now`'s ET calendar date as `YYYY-MM-DD` ("" when the clock is unusable)."""
    try:
        return now.strftime("%Y-%m-%d")
    except (AttributeError, TypeError, ValueError):
        logger.debug("desk_steps: unusable clock %r", now)
        return ""


def is_session_day(now: datetime) -> bool:
    """Is this a day the desk works?

    Weekday only. The repo has no holiday calendar (checked: `is_market_hours`
    and `_is_market_open` in `core/engine.py` are both weekday-based), so the
    desk matches the engine rather than inventing a second, divergent notion of
    "closed". A holiday therefore builds a plan nobody trades on — harmless, and
    the journal will say the day was empty.
    """
    try:
        return now.weekday() < 5
    except (AttributeError, TypeError):
        return False


def parse_et_time(value: Any, default: str) -> Tuple[int, int]:
    """`"09:35"` → `(9, 35)`, falling back to `default` (then to 00:00)."""
    for candidate in (value, default):
        if not isinstance(candidate, str):
            continue
        parts = candidate.strip().split(":")
        if len(parts) < 2:
            continue
        try:
            hh, mm = int(parts[0]), int(parts[1])
        except (TypeError, ValueError):
            continue
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh, mm
    logger.warning("desk_steps: unusable schedule time %r (default %r)", value, default)
    return 0, 0


def crossed(now: datetime, target: Tuple[int, int]) -> bool:
    """Has `now` reached `target` (hh, mm) on its own date?"""
    try:
        hh, mm = target
        return now >= now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    except (AttributeError, TypeError, ValueError):
        return False


# ────────────────────────────────────────────────────────────────────
# readings
# ────────────────────────────────────────────────────────────────────

def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(value, 1.0))


def breadth_score(breadth: Any) -> Optional[float]:
    """`engine.market_state["breadth"]` as a 0–1 score, or None when unreadable.

    The providers disagree on shape — `data_providers/realtime.py` emits
    `advancing_percent`/`advances`/`declines` (live) or a signed
    `advance_decline` (dummy), `data_providers/schwab.py` emits a signed
    `advance_decline` plus `new_highs`/`new_lows`. Each is mapped to the same
    0–1 scale, most direct reading first. None means "no breadth reading",
    which `build_daily_plan` treats differently from a low one.
    """
    if not isinstance(breadth, Mapping):
        return None

    pct = _finite(breadth.get("advancing_percent"))
    if pct is not None:
        return _clamp01(pct / 100.0)

    adv, dec = _finite(breadth.get("advances")), _finite(breadth.get("declines"))
    if adv is not None and dec is not None and (adv + dec) > 0:
        return _clamp01(adv / (adv + dec))

    ratio = _finite(breadth.get("advance_decline_ratio"))
    if ratio is not None and ratio >= 0:
        return _clamp01(ratio / (1.0 + ratio))

    highs, lows = _finite(breadth.get("new_highs")), _finite(breadth.get("new_lows"))
    if highs is not None and lows is not None and (highs + lows) > 0:
        return _clamp01(highs / (highs + lows))

    signed = _finite(breadth.get("advance_decline"))
    if signed is not None:
        return _clamp01(0.5 + max(-AD_SCALE, min(signed, AD_SCALE)) / (2 * AD_SCALE))
    return None


def daily_loss_pct(value: Any) -> float:
    """`Config().MAX_DAILY_LOSS` as a *percent* for the plan's risk budget.

    The config value is a fraction (0.05 = 5%), but a future edit could write
    the percent directly; anything at or below 1.0 is read as a fraction, above
    that as a percent. Unusable or negative input is 0.0 — a plan with no risk
    budget is safe, a plan with a garbage one is not.
    """
    number = _finite(value)
    if number is None or number <= 0:
        return 0.0
    return number * 100.0 if number <= 1.0 else number


# ────────────────────────────────────────────────────────────────────
# candidates
# ────────────────────────────────────────────────────────────────────

def _symbols(values: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(values, (list, tuple, set)):
        return out
    for item in values:
        if not isinstance(item, str):
            continue
        symbol = item.strip().upper()
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def scan_symbols(pinned: Any, watchlist: Any, universe: Any, limit: Any) -> List[str]:
    """The 09:35 scan list: pinned, then the live watchlist, then the universe.

    `limit` caps only the universe slice — pinned symbols and whatever the
    screener is already tracking are never dropped, because the desk must be
    able to rank what the bot can actually trade.
    """
    try:
        cap = max(0, int(limit))
    except (TypeError, ValueError):
        cap = 0
    out = _symbols(pinned)
    for symbol in _symbols(watchlist):
        if symbol not in out:
            out.append(symbol)
    added = 0
    for symbol in _symbols(universe):
        if added >= cap:
            break
        if symbol not in out:
            out.append(symbol)
            added += 1
    return out


def load_scan_universe(root: Any, limit: Any) -> List[str]:
    """Yesterday's most-in-play symbols from the nightly context Parquet.

    Ranked by `rvol_rank5` ascending (rank 1 = the highest opening relative
    volume that session) and then by `adv_dollars_20` descending, over the
    newest date in the file. Never raises: a missing tree, a missing Parquet
    engine or an unexpected schema returns [] and the scan falls back to pinned
    + watchlist.
    """
    try:
        cap = max(0, int(limit))
    except (TypeError, ValueError):
        return []
    if cap == 0 or root is None:
        return []
    try:
        path = Path(root).joinpath(*CONTEXT_RELPATH)
        if not path.is_file():
            return []
        frame = pd.read_parquet(path, engine="pyarrow",
                                columns=["date", "symbol", "rvol_rank5", "adv_dollars_20"])
        if frame is None or frame.empty:
            return []
        frame = frame.assign(date=pd.to_datetime(frame["date"], errors="coerce"))
        newest = frame["date"].max()
        if pd.isna(newest):
            return []
        latest = frame[frame["date"] == newest].copy()
        latest["rvol_rank5"] = pd.to_numeric(latest["rvol_rank5"], errors="coerce")
        latest["adv_dollars_20"] = pd.to_numeric(latest["adv_dollars_20"], errors="coerce")
        latest = latest.sort_values(["rvol_rank5", "adv_dollars_20"],
                                   ascending=[True, False], na_position="last")
        return _symbols(list(latest["symbol"]))[:cap]
    except Exception:
        logger.debug("desk_steps: load_scan_universe failed under %r", root, exc_info=True)
        return []


def fetch_today_bars5(provider: Any, symbol: str) -> Optional[pd.DataFrame]:
    """The provider's recent 5-minute regular-hours window, ET-naive.

    Mirrors `engine._fit_bars_source`'s normalization (lower-cased columns, ET
    index with the timezone dropped, `between_time` 09:30–15:55) because the
    in-play features must never see pre- or post-market bars. None on any
    failure — the scan reports the symbol as `missing:today_bars` instead.
    """
    try:
        frame = provider.get_market_data(symbol, period_type="day", period=10,
                                         frequency_type="minute", frequency=5)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        frame = frame.rename(columns={c: str(c).lower() for c in frame.columns})
        index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if getattr(index, "tz", None) is not None:
            index = index.tz_convert(ET_TZ).tz_localize(None)
        frame = frame.copy()
        frame.index = index
        frame = frame.sort_index().between_time(RTH_START, RTH_END)
        return None if frame.empty else frame
    except Exception:
        logger.debug("desk_steps: fetch_today_bars5 failed for %r", symbol, exc_info=True)
        return None
