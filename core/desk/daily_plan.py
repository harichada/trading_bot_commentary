"""v-playbook-desk-b2-2026-09-25: the desk's morning routine, as pure rules.

`build_daily_plan(inputs, now_et)` turns the tape's state into the day's plan
— sit out or trade, how much risk, how many trades, which way — and
`refresh_plan(plan, in_play, now_et)` attaches the 09:35 in-play list without
touching a single other decision.

Both functions are pure: no clock, no I/O, no globals. `load_calendar_flags`
is the one reader here, and it never raises — a missing or broken calendar
file means "no scheduled events", which is what a trader assumes when the
calendar does not load.

Rules (spec: docs/superpowers/specs/2026-09-25-playbook-desk-design.md §7):

  sit out, first match wins
    calendar_high_impact_before_1030 — a `high:` flag scheduled before 10:30
    vix_spike                        — VIX up more than 15%
    two_losing_days                  — two or more losing days in a row
    fit_stale_gate_enforcing         — stale fit snapshot while the gate enforces

  risk_budget_pct = max_daily_loss_pct × regime multiplier, halved after a
                    losing day
  max_trades      = per-regime cap
  long_ok         = market context allows longs and breadth >= 0.40
  short_ok        = market context allows shorts and breadth <= 0.60
                    (no breadth reading → market context alone decides)
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.desk.models import DailyPlan, InPlayRow, PlanInputs

logger = logging.getLogger(__name__)

__all__ = [
    "build_daily_plan",
    "refresh_plan",
    "load_calendar_flags",
    "REGIME_RISK_MULT",
    "REGIME_MAX_TRADES",
]

# ── Rule constants (named so the journal and the tests can quote them) ──
REGIME_RISK_MULT: Dict[str, float] = {
    "risk_on": 1.0, "mixed": 0.6, "risk_off": 0.4, "unknown": 0.4,
}
REGIME_MAX_TRADES: Dict[str, int] = {
    "risk_on": 6, "mixed": 4, "risk_off": 2, "unknown": 2,
}
_DEFAULT_RISK_MULT = REGIME_RISK_MULT["unknown"]
_DEFAULT_MAX_TRADES = REGIME_MAX_TRADES["unknown"]

HIGH_IMPACT_PREFIX = "high:"
CALENDAR_CUTOFF_MINUTES = 10 * 60 + 30      # 10:30 ET
VIX_SPIKE_PCT = 15.0                        # strictly greater than
LOSING_STREAK_SIT_OUT = 2
BREADTH_LONG_MIN = 0.4
BREADTH_SHORT_MAX = 0.6

SIT_OUT_CALENDAR = "calendar_high_impact_before_1030"
SIT_OUT_VIX = "vix_spike"
SIT_OUT_STREAK = "two_losing_days"
SIT_OUT_FIT_STALE = "fit_stale_gate_enforcing"


# ────────────────────────────────────────────────────────────────────
# Calendar
# ────────────────────────────────────────────────────────────────────

def _time_to_minutes(value: Any) -> Optional[int]:
    """"HH:MM" → minutes since midnight; None for anything unparseable."""
    if not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    if len(parts) < 2:
        return None
    try:
        hh, mm = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return hh * 60 + mm


def load_calendar_flags(path: str, date: str) -> Tuple[str, ...]:
    """Read the economic calendar and return this date's flags.

    File format: a JSON list of `{"date", "time_et", "event", "impact"}`.
    Each matching row becomes `"<impact>:<HH:MM>:<event>"` (impact lowercased)
    so downstream rules can match on a plain string prefix.

    Never raises: a missing file gives `()`; a malformed file or row is logged
    and skipped.
    """
    if not path or not os.path.exists(path):
        return ()
    try:
        with open(path, "r") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("desk calendar unreadable path=%s err=%s", path, exc)
        return ()
    if not isinstance(payload, list):
        logger.warning("desk calendar is not a JSON list path=%s type=%s",
                       path, type(payload).__name__)
        return ()

    flags: List[str] = []
    skipped = 0
    for row in payload:
        if not isinstance(row, dict):
            skipped += 1
            continue
        if str(row.get("date", "")).strip() != str(date).strip():
            continue
        minutes = _time_to_minutes(row.get("time_et"))
        event = row.get("event")
        impact = row.get("impact")
        if minutes is None or not isinstance(event, str) or not event.strip() \
                or not isinstance(impact, str) or not impact.strip():
            skipped += 1
            continue
        flags.append(f"{impact.strip().lower()}:{minutes // 60:02d}:{minutes % 60:02d}:"
                     f"{event.strip()}")
    if skipped:
        logger.warning("desk calendar skipped %d malformed row(s) path=%s", skipped, path)
    return tuple(flags)


def _high_impact_before_cutoff(flags: Sequence[str]) -> Optional[str]:
    """The first `high:` flag scheduled before the cutoff, if any."""
    for flag in flags:
        if not isinstance(flag, str) or not flag.lower().startswith(HIGH_IMPACT_PREFIX):
            continue
        parts = flag.split(":")
        if len(parts) < 3:
            continue
        minutes = _time_to_minutes(f"{parts[1]}:{parts[2]}")
        if minutes is not None and minutes < CALENDAR_CUTOFF_MINUTES:
            return flag
    return None


# ────────────────────────────────────────────────────────────────────
# Sub-rules
# ────────────────────────────────────────────────────────────────────

def _sit_out(inputs: PlanInputs) -> Tuple[bool, str, str]:
    """(sit_out, reason, note). First matching rule wins."""
    flag = _high_impact_before_cutoff(inputs.calendar_flags)
    if flag is not None:
        return True, SIT_OUT_CALENDAR, (
            f"sit out ({SIT_OUT_CALENDAR}): high-impact calendar event {flag} "
            f"lands before 10:30 ET")

    if math.isfinite(inputs.vix_change_pct) and inputs.vix_change_pct > VIX_SPIKE_PCT:
        return True, SIT_OUT_VIX, (
            f"sit out ({SIT_OUT_VIX}): VIX up {inputs.vix_change_pct:.1f}%, "
            f"above the {VIX_SPIKE_PCT:.0f}% spike line")

    if inputs.losing_day_streak >= LOSING_STREAK_SIT_OUT:
        return True, SIT_OUT_STREAK, (
            f"sit out ({SIT_OUT_STREAK}): {inputs.losing_day_streak} losing days in a row")

    if inputs.fit_stale and inputs.gate_enforcing:
        return True, SIT_OUT_FIT_STALE, (
            f"sit out ({SIT_OUT_FIT_STALE}): the strategy-fit snapshot is stale "
            f"while the fit gate is enforcing")

    return False, "", ""


def _risk_budget(inputs: PlanInputs) -> Tuple[float, List[str]]:
    mult = REGIME_RISK_MULT.get(inputs.regime, _DEFAULT_RISK_MULT)
    loss_cap = inputs.max_daily_loss_pct if math.isfinite(inputs.max_daily_loss_pct) else 0.0
    budget = loss_cap * mult
    notes = [f"regime {inputs.regime}: risk budget {mult:.2f}x of the "
             f"{loss_cap:.2f}% daily loss cap = {budget:.2f}%"]
    if math.isfinite(inputs.prev_day_pnl_r) and inputs.prev_day_pnl_r < 0:
        budget /= 2.0
        notes.append(f"prior session closed {inputs.prev_day_pnl_r:.2f}R: "
                     f"risk budget halved to {budget:.2f}%")
    return budget, notes


def _directions(inputs: PlanInputs) -> Tuple[bool, bool, List[str]]:
    breadth = inputs.breadth_score
    if breadth is not None and not math.isfinite(breadth):
        breadth = None

    notes: List[str] = []
    long_ok = bool(inputs.allows_long)
    short_ok = bool(inputs.allows_short)

    if not long_ok:
        notes.append("longs off: market context does not allow longs")
    elif breadth is not None and breadth < BREADTH_LONG_MIN:
        long_ok = False
        notes.append(f"longs off: breadth {breadth:.2f} below {BREADTH_LONG_MIN:.2f}")

    if not short_ok:
        notes.append("shorts off: market context does not allow shorts")
    elif breadth is not None and breadth > BREADTH_SHORT_MAX:
        short_ok = False
        notes.append(f"shorts off: breadth {breadth:.2f} above {BREADTH_SHORT_MAX:.2f}")

    if breadth is None:
        notes.append("no breadth reading: direction left to market context")
    return long_ok, short_ok, notes


# ────────────────────────────────────────────────────────────────────
# Public rules
# ────────────────────────────────────────────────────────────────────

def build_daily_plan(inputs: PlanInputs, now_et: datetime) -> DailyPlan:
    """Build the day's plan from pre-market state. Pure; `now_et` stamps it."""
    sit_out, reason, sit_out_note = _sit_out(inputs)
    budget, budget_notes = _risk_budget(inputs)
    long_ok, short_ok, direction_notes = _directions(inputs)
    max_trades = REGIME_MAX_TRADES.get(inputs.regime, _DEFAULT_MAX_TRADES)

    notes: List[str] = []
    if sit_out_note:
        notes.append(sit_out_note)
    notes.extend(budget_notes)
    notes.append(f"max trades {max_trades} for regime {inputs.regime}")
    notes.extend(direction_notes)
    if inputs.calendar_flags:
        notes.append("calendar today: " + ", ".join(inputs.calendar_flags))

    breadth = inputs.breadth_score
    if breadth is not None and not math.isfinite(breadth):
        breadth = None

    return DailyPlan(
        date=inputs.date,
        regime=inputs.regime,
        spy_change_pct=inputs.spy_change_pct,
        vix=inputs.vix,
        vix_change_pct=inputs.vix_change_pct,
        breadth_score=breadth,
        calendar_flags=tuple(inputs.calendar_flags),
        long_ok=long_ok,
        short_ok=short_ok,
        risk_budget_pct=budget,
        max_trades=max_trades,
        sit_out=sit_out,
        sit_out_reason=reason,
        in_play=(),
        notes=tuple(notes),
        built_at=now_et.isoformat(),
        refreshed_at="",
    )


def refresh_plan(plan: DailyPlan, in_play: Sequence[InPlayRow],
                 now_et: datetime) -> DailyPlan:
    """Return a copy of `plan` carrying the 09:35 in-play list.

    Only `in_play` and `refreshed_at` change — the morning's risk, direction
    and sit-out decisions stand for the day.
    """
    return plan.with_in_play(tuple(in_play), now_et.isoformat())
