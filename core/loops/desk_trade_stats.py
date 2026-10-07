"""v-playbook-desk-b2-2026-09-25 (Task 5): trade-row statistics for the desk.

Split out of `core/loops/desk_steps.py` to keep both files under the 400-line
limit. Everything here reads *rows* — closed `bot_trades` rows as
`core/desk/journal_data.trades_for` shapes them, and playbook entries — and
turns them into the numbers the plan rules and the journal need:

  `daily_r_stats`          the losing-day streak and prior-session R that
                           `build_daily_plan` conditions on
  `trailing_summary`       the journal's `trailing` block (expectancy by
                           family x regime, sit-out days, max-trades days)
  `playbook_changes_today` the journal's "Playbook Changes" section

Same contract as its sibling: no function here raises, and unusable input
degrades to an empty/neutral reading. `core/loops/desk_steps.py` re-exports
these three, so callers can keep importing one module.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from core.desk.models import DailyPlan

logger = logging.getLogger(__name__)

__all__ = ["daily_r_stats", "trailing_summary", "playbook_changes_today",
           "TRAILING_MIN_PF_DENOM"]

TRAILING_MIN_PF_DENOM = 1e-12


def _finite(value: Any) -> Optional[float]:
    """Local copy of the sibling's coercion helper: importing it back would make
    the two modules circular, and it is five lines."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _row_date(row: Any) -> str:
    """A trade row's session date (`exit_time`, else `entry_time`, else `date`)."""
    if not isinstance(row, Mapping):
        return ""
    for key in ("exit_time", "entry_time", "date"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if len(text) >= 10:
                return text[:10]
            continue
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            continue
        if not pd.isna(stamp):
            return stamp.strftime("%Y-%m-%d")
    return ""


def _row_r(row: Any) -> Optional[float]:
    if not isinstance(row, Mapping):
        return None
    for key in ("r", "pnl_r"):
        value = _finite(row.get(key))
        if value is not None:
            return value
    return None


def _r_by_date(trades: Any) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    if not isinstance(trades, (list, tuple)):
        return out
    for row in trades:
        date, r = _row_date(row), _row_r(row)
        if date and r is not None:
            out[date].append(r)
    return out


def cooling_off(plans: Any, today: str, streak_reason: str = "two_losing_days") -> Tuple[str, bool]:
    """v-cooling-off-2026-10-07: `(served_date, prior_session_sat_out)` from
    prior sessions' plans. `served_date` is the latest session before `today`
    the desk sat out for the losing streak — trades on or before it no longer
    count toward a new streak (one cooling-off day per streak, not one per
    day forever: with no trades on a sat-out day the streak could never clear
    on its own, which sidelined the desk 2026-10-02 → 10-07).
    `prior_session_sat_out` is True when the most recent prior plan is that
    sit-out, so the plan rules resume at half risk. Never raises."""
    prior: List[DailyPlan] = []
    for p in plans if isinstance(plans, (list, tuple)) else []:
        if isinstance(p, DailyPlan) and p.date and p.date < str(today):
            prior.append(p)
    if not prior:
        return "", False
    prior.sort(key=lambda p: p.date)
    served = [p.date for p in prior if p.sit_out and p.sit_out_reason == streak_reason]
    latest = prior[-1]
    return (served[-1] if served else ""), bool(latest.sit_out and latest.sit_out_reason == streak_reason)


def daily_r_stats(trades: Any, today: str) -> Tuple[int, float]:
    """`(losing_day_streak, prev_day_pnl_r)` from closed trades before `today`.

    The streak counts back from the most recent prior session with trades; a
    session that netted >= 0 ends it. Sessions with no trades are not in the
    data and simply do not count either way. `(0, 0.0)` when there is nothing
    usable to read — a fresh account must not look like a losing streak.
    """
    by_date = _r_by_date(trades)
    dates = sorted((d for d in by_date if d < str(today)), reverse=True)
    if not dates:
        return 0, 0.0

    prev_r = sum(by_date[dates[0]])
    streak = 0
    for date in dates:
        if sum(by_date[date]) < 0:
            streak += 1
        else:
            break
    return streak, prev_r


# ────────────────────────────────────────────────────────────────────
# journal inputs
# ────────────────────────────────────────────────────────────────────

def _expectancy_rows(trades: Any) -> List[dict]:
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    if isinstance(trades, (list, tuple)):
        for row in trades:
            r = _row_r(row)
            if r is None or not isinstance(row, Mapping):
                continue
            family = str(row.get("family") or "unknown")
            regime = str(row.get("regime") or "unknown")
            groups[(family, regime)].append(r)

    out: List[dict] = []
    for (family, regime), rs in sorted(groups.items()):
        gains = sum(r for r in rs if r > 0)
        losses = -sum(r for r in rs if r < 0)
        pf = gains / losses if losses > TRAILING_MIN_PF_DENOM else None
        out.append({"family": family, "regime": regime, "n": len(rs),
                    "exp_r": sum(rs) / len(rs), "pf": pf})
    return out


def trailing_summary(trades: Any, plans: Any) -> Dict[str, List[dict]]:
    """The journal's `trailing` block: expectancy, sit-out days, max-trades days.

    `trades` are the trailing window's closed trades (as
    `journal_data.trades_for` shapes them) and `plans` a `{date: DailyPlan}`
    map for the same window. Shapes exactly what `core/desk/journal.py` reads —
    `render_journal` derives the three "what to change" items from these rows.
    """
    by_date = _r_by_date(trades)
    summary: Dict[str, List[dict]] = {"expectancy": _expectancy_rows(trades),
                                      "sit_out_days": [], "max_trades_days": []}
    if not isinstance(plans, Mapping):
        return summary

    for date in sorted(str(d) for d in plans):
        plan = plans.get(date)
        if not isinstance(plan, DailyPlan):
            continue
        rs = by_date.get(date, [])
        if plan.sit_out:
            summary["sit_out_days"].append({
                "date": date,
                "reason": plan.sit_out_reason or "unknown",
                "would_have_r": sum(rs),
            })
        if plan.max_trades > 0 and len(rs) >= plan.max_trades:
            summary["max_trades_days"].append({
                "date": date, "max_trades": plan.max_trades, "trades_taken": len(rs),
            })
    return summary


def playbook_changes_today(playbook: Any, date: str) -> List[dict]:
    """Playbook entries the nightly updater stamped on `date`, for the journal.

    The playbook file keeps only the current state, so a "change" is any entry
    whose `updated_at` falls on this date; `old_status` is left empty rather
    than guessed.
    """
    entries = getattr(playbook, "entries", None)
    if not isinstance(entries, Mapping):
        return []
    out: List[dict] = []
    for entry in entries.values():
        stamp = str(getattr(entry, "updated_at", "") or "")
        if not stamp.startswith(str(date)):
            continue
        out.append({"family": str(getattr(entry, "family", "") or "unknown"),
                    "side": str(getattr(entry, "side", "") or "unknown"),
                    "regime": str(getattr(entry, "regime", "") or "unknown"),
                    "old_status": "",
                    "new_status": str(getattr(entry, "status", "") or "unknown"),
                    "reason": str(getattr(entry, "reason", "") or "")})
    return sorted(out, key=lambda r: (r["family"], r["side"], r["regime"]))
