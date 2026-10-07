"""v-playbook-desk-b2-2026-09-25: the desk's value types.

`InPlayRow` (one ranked stock), `DailyPlan` (the morning plan a rule set
produces and the loop persists) and `PlanInputs` (everything the rules read).

All three are frozen: the rules return new objects rather than mutating, so a
plan handed to the router can never change underneath it.

Serialization contract, because the *live* loop reads these files:
`from_dict` never raises — unknown keys are ignored (a newer writer may add
fields), and every value is coerced defensively, degrading to nan/0/""/False/()
rather than blowing up the trading path. `to_dict` emits plain JSON: nan and
±inf become `null`, so `json.dumps` never writes the `NaN`/`Infinity`
literals that strict JSON readers reject.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = ["InPlayRow", "DailyPlan", "PlanInputs"]

# Regime → (risk-budget multiplier, max trades) lives in daily_plan.py; the
# models stay pure data.


# ────────────────────────────────────────────────────────────────────
# Defensive coercion helpers (shared by every from_dict/to_dict)
# ────────────────────────────────────────────────────────────────────

_TRUTHY = ("1", "true", "yes", "on", "y", "t")


def _as_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_symbol(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_opt_float(value: Any) -> Optional[float]:
    """None for missing/unusable/non-finite; a float otherwise."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_str_tuple(value: Any) -> Tuple[str, ...]:
    """A lone string counts as a one-element tuple; anything else non-iterable is empty."""
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if v is not None)
    return ()


def _json_float(value: Any) -> Optional[float]:
    """Non-finite (nan/inf) and unusable floats serialize as JSON null."""
    return _as_opt_float(value)


# ────────────────────────────────────────────────────────────────────
# InPlayRow
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InPlayRow:
    """One stock on the in-play list, with the features that put it there.

    `score` is filled in by the ranker (Task 2); `catalyst` is 0.0/1.0;
    `rs_20` and `gap_pct` are percentages; `reasons` are short human tags
    ("rvol", "pinned", "missing:rvol_open15") shown in the journal.
    """
    symbol: str = ""
    score: float = float("nan")
    rvol_open15: float = float("nan")
    gap_pct: float = float("nan")
    catalyst: float = 0.0
    rs_20: float = float("nan")
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": _json_float(self.score),
            "rvol_open15": _json_float(self.rvol_open15),
            "gap_pct": _json_float(self.gap_pct),
            "catalyst": _json_float(self.catalyst),
            "rs_20": _json_float(self.rs_20),
            "reasons": list(self.reasons),
        }

    @staticmethod
    def from_dict(d: Any) -> "InPlayRow":
        if not isinstance(d, dict):
            return InPlayRow()
        return InPlayRow(
            symbol=_as_symbol(d.get("symbol")),
            score=_as_float(d.get("score")),
            rvol_open15=_as_float(d.get("rvol_open15")),
            gap_pct=_as_float(d.get("gap_pct")),
            catalyst=_as_float(d.get("catalyst"), 0.0),
            rs_20=_as_float(d.get("rs_20")),
            reasons=_as_str_tuple(d.get("reasons")),
        )


def _as_rows(value: Any) -> Tuple[InPlayRow, ...]:
    """Parse a list of row dicts, dropping entries that carry no symbol."""
    if not isinstance(value, (list, tuple)):
        return ()
    rows = []
    for item in value:
        row = item if isinstance(item, InPlayRow) else InPlayRow.from_dict(item)
        if row.symbol:
            rows.append(row)
    return tuple(rows)


# ────────────────────────────────────────────────────────────────────
# DailyPlan
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DailyPlan:
    """The morning plan: what the tape looks like and what the desk may do.

    `breadth_score` is Optional because "no breadth reading" is a real state
    the direction rules treat differently from a low reading.
    """
    date: str = ""
    regime: str = "unknown"
    spy_change_pct: float = float("nan")
    vix: float = float("nan")
    vix_change_pct: float = float("nan")
    breadth_score: Optional[float] = None
    calendar_flags: Tuple[str, ...] = ()
    long_ok: bool = False
    short_ok: bool = False
    risk_budget_pct: float = 0.0
    max_trades: int = 0
    sit_out: bool = False
    sit_out_reason: str = ""
    in_play: Tuple[InPlayRow, ...] = ()
    notes: Tuple[str, ...] = ()
    built_at: str = ""
    refreshed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "regime": self.regime,
            "spy_change_pct": _json_float(self.spy_change_pct),
            "vix": _json_float(self.vix),
            "vix_change_pct": _json_float(self.vix_change_pct),
            "breadth_score": _json_float(self.breadth_score),
            "calendar_flags": list(self.calendar_flags),
            "long_ok": self.long_ok,
            "short_ok": self.short_ok,
            "risk_budget_pct": _json_float(self.risk_budget_pct),
            "max_trades": self.max_trades,
            "sit_out": self.sit_out,
            "sit_out_reason": self.sit_out_reason,
            "in_play": [r.to_dict() for r in self.in_play],
            "notes": list(self.notes),
            "built_at": self.built_at,
            "refreshed_at": self.refreshed_at,
        }

    @staticmethod
    def from_dict(d: Any) -> "DailyPlan":
        if not isinstance(d, dict):
            return DailyPlan()
        return DailyPlan(
            date=_as_str(d.get("date")),
            regime=_as_str(d.get("regime"), "unknown") or "unknown",
            spy_change_pct=_as_float(d.get("spy_change_pct")),
            vix=_as_float(d.get("vix")),
            vix_change_pct=_as_float(d.get("vix_change_pct")),
            breadth_score=_as_opt_float(d.get("breadth_score")),
            calendar_flags=_as_str_tuple(d.get("calendar_flags")),
            long_ok=_as_bool(d.get("long_ok")),
            short_ok=_as_bool(d.get("short_ok")),
            risk_budget_pct=_as_float(d.get("risk_budget_pct"), 0.0),
            max_trades=_as_int(d.get("max_trades"), 0),
            sit_out=_as_bool(d.get("sit_out")),
            sit_out_reason=_as_str(d.get("sit_out_reason")),
            in_play=_as_rows(d.get("in_play")),
            notes=_as_str_tuple(d.get("notes")),
            built_at=_as_str(d.get("built_at")),
            refreshed_at=_as_str(d.get("refreshed_at")),
        )

    def with_in_play(self, rows: Sequence[InPlayRow], refreshed_at: str) -> "DailyPlan":
        """Immutable update used by `daily_plan.refresh_plan`."""
        return replace(self, in_play=tuple(rows), refreshed_at=refreshed_at)


# ────────────────────────────────────────────────────────────────────
# PlanInputs
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlanInputs:
    """Everything `build_daily_plan` reads — gathered by the loop, never by the rules.

    `allows_long`/`allows_short` come from market context; `max_daily_loss_pct`
    from risk config; `fit_stale`/`gate_enforcing` from the strategy-fit reader
    and its flags.
    """
    date: str = ""
    regime: str = "unknown"
    spy_change_pct: float = float("nan")
    vix: float = float("nan")
    vix_change_pct: float = float("nan")
    breadth_score: Optional[float] = None
    calendar_flags: Tuple[str, ...] = ()
    losing_day_streak: int = 0
    prev_day_pnl_r: float = 0.0
    fit_stale: bool = False
    gate_enforcing: bool = False
    max_daily_loss_pct: float = 0.0
    allows_long: bool = True
    allows_short: bool = True
    # v-cooling-off-2026-10-07: the session before this one was a `two_losing_days`
    # sit-out — the cooling-off day has been served, so the desk resumes (at half
    # risk) instead of sitting out again on the same two old losses.
    prior_session_sat_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "regime": self.regime,
            "spy_change_pct": _json_float(self.spy_change_pct),
            "vix": _json_float(self.vix),
            "vix_change_pct": _json_float(self.vix_change_pct),
            "breadth_score": _json_float(self.breadth_score),
            "calendar_flags": list(self.calendar_flags),
            "losing_day_streak": self.losing_day_streak,
            "prev_day_pnl_r": _json_float(self.prev_day_pnl_r),
            "fit_stale": self.fit_stale,
            "gate_enforcing": self.gate_enforcing,
            "max_daily_loss_pct": _json_float(self.max_daily_loss_pct),
            "allows_long": self.allows_long,
            "allows_short": self.allows_short,
            "prior_session_sat_out": self.prior_session_sat_out,
        }

    @staticmethod
    def from_dict(d: Any) -> "PlanInputs":
        if not isinstance(d, dict):
            return PlanInputs()
        return PlanInputs(
            date=_as_str(d.get("date")),
            regime=_as_str(d.get("regime"), "unknown") or "unknown",
            spy_change_pct=_as_float(d.get("spy_change_pct")),
            vix=_as_float(d.get("vix")),
            vix_change_pct=_as_float(d.get("vix_change_pct")),
            breadth_score=_as_opt_float(d.get("breadth_score")),
            calendar_flags=_as_str_tuple(d.get("calendar_flags")),
            losing_day_streak=_as_int(d.get("losing_day_streak"), 0),
            prev_day_pnl_r=_as_float(d.get("prev_day_pnl_r"), 0.0),
            fit_stale=_as_bool(d.get("fit_stale")),
            gate_enforcing=_as_bool(d.get("gate_enforcing")),
            max_daily_loss_pct=_as_float(d.get("max_daily_loss_pct"), 0.0),
            allows_long=_as_bool(d.get("allows_long"), True),
            allows_short=_as_bool(d.get("allows_short"), True),
            prior_session_sat_out=_as_bool(d.get("prior_session_sat_out")),
        )
