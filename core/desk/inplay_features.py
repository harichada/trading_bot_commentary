"""v-playbook-desk-b2-2026-09-25: the opening features behind the in-play score.

At 09:35 the desk asks of each candidate: is it trading more than its own
normal, did it reprice overnight, is there a reason, and has it been leading or
lagging the market? Those four readings are `InPlayRow`'s features; this module
computes them from bars plus the nightly context row, and `inplay_ranker`
turns them into a score.

Bar conventions are the ones `strategies/families/indicators.py` uses: 5-minute
bars with a naive-ET `DatetimeIndex`, sessions grouped by `index.normalize()`,
`bar_of_day` 0 at 09:30 — so the opening 15 minutes is a session's first three
bars.

Nothing here raises. A missing frame, a short history, a column that is not
there, a garbage context value: each degrades to a NaN field plus a
`missing:<cause>` reason, because a feature failure for one symbol must never
take down the 09:35 scan. And nothing here imports `research`: the nightly
context Parquet is read as a *data* file, the same one-way arrangement
`core/strategy_fit_bars.py` has with the research tree.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.desk.models import InPlayRow

logger = logging.getLogger(__name__)

__all__ = ["opening_features", "load_prior_context", "OPEN_BARS",
           "HISTORY_SESSIONS", "MIN_HISTORY_SESSIONS", "NEWS_CATALYST_MIN",
           "CONTEXT_RELPATH"]

OPEN_BARS = 3                  # 09:30, 09:35, 09:40 — the opening 15 minutes
HISTORY_SESSIONS = 14          # window the opening volume is compared against
MIN_HISTORY_SESSIONS = 5       # below this the median is noise, so rvol stays NaN
NEWS_CATALYST_MIN = 3          # headlines in 24h that count as a catalyst
CONTEXT_RELPATH = ("context", "universe_daily.parquet")
ET = "America/New_York"
NAN = float("nan")

_TRUTHY_STRINGS = ("1", "true", "yes", "on", "y", "t")


# ────────────────────────────────────────────────────────────────────
# coercion helpers — all total, none raise
# ────────────────────────────────────────────────────────────────────

def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    try:
        return bool(value)
    except Exception:  # exotic objects (e.g. an array) must not break the scan
        return False


def _symbol(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _add(reasons: List[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _naive_et_frame(frame: Any) -> Optional[pd.DataFrame]:
    """A time-sorted copy indexed by naive ET timestamps, or None when the input
    is not a usable bar frame."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if getattr(index, "tz", None) is not None:
            index = index.tz_convert(ET).tz_localize(None)
    except (TypeError, ValueError):
        logger.debug("inplay_features: unusable bar index", exc_info=True)
        return None
    out = frame.copy()
    out.index = index
    return out.sort_index()


def _session_date(now_et: Any, today_bars: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
    """Today's session date from the clock, falling back to the newest date in
    the frame the caller handed us."""
    try:
        stamp = pd.Timestamp(now_et)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(ET).tz_localize(None)
        if not pd.isna(stamp):
            return stamp.normalize()
    except (TypeError, ValueError):
        logger.debug("inplay_features: unusable now_et %r", now_et, exc_info=True)
    if today_bars is not None:
        return pd.Timestamp(today_bars.index.max()).normalize()
    return None


def _sessions(frame: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.DataFrame]]:
    """(session date, bars) pairs, oldest first — the grouping indicators.py uses."""
    return [(pd.Timestamp(day), group)
            for day, group in frame.groupby(frame.index.normalize(), sort=True)]


# ────────────────────────────────────────────────────────────────────
# the four features
# ────────────────────────────────────────────────────────────────────

def _open_volume(session: pd.DataFrame) -> Optional[float]:
    """Volume of a session's first `OPEN_BARS` bars; None when it cannot be read."""
    if "volume" not in session.columns or len(session) < OPEN_BARS:
        return None
    values = pd.to_numeric(session["volume"].iloc[:OPEN_BARS], errors="coerce")
    if values.isna().any():
        return None
    return _finite(values.sum())


def _median_open_volume(prior: List[Tuple[pd.Timestamp, pd.DataFrame]]) -> Optional[float]:
    """Median opening volume over the last `HISTORY_SESSIONS` prior sessions."""
    window = prior[-HISTORY_SESSIONS:]
    volumes = [v for v in (_open_volume(group) for _, group in window) if v is not None]
    if len(volumes) < MIN_HISTORY_SESSIONS:
        return None
    median = _finite(pd.Series(volumes).median())
    return median if median and median > 0 else None


def _first_open(session: pd.DataFrame) -> Optional[float]:
    if "open" not in session.columns or session.empty:
        return None
    return _finite(session["open"].iloc[0])


def _last_close(session: pd.DataFrame) -> Optional[float]:
    if "close" not in session.columns or session.empty:
        return None
    return _finite(session["close"].iloc[-1])


def _catalyst(prior_ctx: Dict[str, Any]) -> float:
    """1.0 when there is a reason for the move: earnings today or yesterday, or
    a burst of headlines."""
    if _truthy(prior_ctx.get("earnings_today")) or _truthy(prior_ctx.get("earnings_prev")):
        return 1.0
    news = _finite(prior_ctx.get("news_count_24h"))
    return 1.0 if news is not None and news >= NEWS_CATALYST_MIN else 0.0


def opening_features(symbol: str, today_bars5: Any, history_bars5: Any,
                     prior_ctx: Optional[Dict[str, Any]], now_et: datetime) -> InPlayRow:
    """One candidate's opening features, unscored (`score` stays NaN).

    `today_bars5` is today's 5-minute session, `history_bars5` the prior
    sessions (extra sessions are fine — only those before today are used), and
    `prior_ctx` the symbol's last row from the nightly universe context.
    """
    reasons: List[str] = []
    rvol = gap = rs_20 = NAN
    catalyst = 0.0

    today = _naive_et_frame(today_bars5)
    history = _naive_et_frame(history_bars5)
    session_date = _session_date(now_et, today)

    if today is not None and session_date is not None:
        today = today[today.index.normalize() == session_date]
        if today.empty:
            today = None
    if today is None:
        _add(reasons, "missing:today_bars")

    prior: List[Tuple[pd.Timestamp, pd.DataFrame]] = []
    if history is not None:
        prior = [(day, group) for day, group in _sessions(history)
                 if session_date is None or day < session_date]

    prev_close = _last_close(prior[-1][1]) if prior else None
    if prev_close is None or prev_close <= 0:
        prev_close = None
        _add(reasons, "missing:prev_close")

    if today is not None:
        open_volume = _open_volume(today)
        median = _median_open_volume(prior)
        if open_volume is not None and median is not None:
            rvol = open_volume / median
        else:
            _add(reasons, "missing:rvol_open15")

        session_open = _first_open(today)
        if session_open is not None and prev_close is not None:
            gap = (session_open / prev_close - 1.0) * 100.0
        elif session_open is None:
            _add(reasons, "missing:session_open")

    if isinstance(prior_ctx, dict):
        catalyst = _catalyst(prior_ctx)
        rs_value = _finite(prior_ctx.get("rs_20"))
        if rs_value is None:
            _add(reasons, "missing:rs_20")
        else:
            rs_20 = rs_value
    else:
        _add(reasons, "missing:prior_ctx")

    return InPlayRow(symbol=_symbol(symbol), rvol_open15=rvol, gap_pct=gap,
                     catalyst=catalyst, rs_20=rs_20, reasons=tuple(reasons))


# ────────────────────────────────────────────────────────────────────
# the nightly context row
# ────────────────────────────────────────────────────────────────────

def _plain(value: Any) -> Any:
    """numpy/pandas scalars as plain Python, so the row is JSON-friendly."""
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", ()) == ():
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def _date_str(value: Any) -> str:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return ""
    return "" if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")


def load_prior_context(root: Any, symbol: str) -> Optional[Dict[str, Any]]:
    """The symbol's most recent row of `<root>/context/universe_daily.parquet`
    as a plain dict, or None.

    Read with a pushdown filter so one symbol never pulls the whole universe
    into memory. Never raises: a missing root, a symbol that is not in the
    universe, a truncated file or a missing Parquet engine all return None, and
    `opening_features` then reports `missing:prior_ctx`.
    """
    try:
        sym = _symbol(symbol) or _symbol(str(symbol))
        if not sym:
            return None
        path = Path(root).joinpath(*CONTEXT_RELPATH)
        if not path.is_file():
            return None
        frame = pd.read_parquet(path, engine="pyarrow",
                                filters=[("symbol", "==", sym)])
        if frame is None or frame.empty:
            return None
        if "date" in frame.columns:
            frame = frame.assign(date=pd.to_datetime(frame["date"], errors="coerce"))
            frame = frame.sort_values("date")
        row = {str(k): _plain(v) for k, v in frame.iloc[-1].to_dict().items()}
        if "date" in row:
            row["date"] = _date_str(row["date"])
        return row
    except Exception:
        # Missing pyarrow, unreadable/corrupt file, a root that is not a path,
        # a frame with no `symbol` column — the scan carries on without context.
        logger.debug("load_prior_context: failed for %r under %r", symbol, root,
                     exc_info=True)
        return None
