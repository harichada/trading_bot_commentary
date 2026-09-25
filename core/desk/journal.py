"""v-playbook-desk-b2-2026-09-25 (Task 4): the post-close journal — a pure
renderer that turns the day's plan, governed decisions, closed trades and
playbook changes into a markdown report.

Pure and defensive by design: this module does no I/O (see
`core/desk/journal_data.py` for the adapters and the atomic file writer) and
`render_journal` never raises. Every section degrades to a "none" line when
its inputs are missing, empty or malformed — a bad or partial upstream read
must still produce a readable journal, not a stack trace on the 16:05 ET job.

Decision-dict vocabulary (see `core/strategy_fit_gate.GateDecision` /
`core/desk/playbook.evaluate_playbook_gate`): `action` is one of "proceed",
"shadow" or "skip". "skip" blocked live; "shadow" is the same governed block
recorded while the gate isn't enforcing yet — i.e. it "would have blocked"
had the gate been on. Both accept an optional `component`, `reason`,
`symbol`.

Trade-dict vocabulary: `symbol`, `side`, `family` (falls back to `strategy`),
`r` (falls back to `pnl_r`), `exit_reason`, `regime` (falls back to
"unknown"), and either `hold` directly or `entry_time`/`exit_time` to derive
it from.

`trailing` (precomputed by the caller over the last 20 sessions — see
`core/desk/journal_data.py`) is a dict that may carry:
  - "expectancy": rows of {"family", "regime", "n", "exp_r", "pf"}
  - "sit_out_days": rows of {"date", "reason", "would_have_r"}
  - "max_trades_days": rows of {"date", "max_trades", "trades_taken"}
Any key may be absent or empty; that means "no trailing evidence", not zero,
so those rules simply produce no items rather than firing on absent data.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["render_journal"]

NONE_LINE = "none"
_SUSPEND_MIN_N = 20


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` off a dict or an attribute-bearing object; `default` for
    anything else (None, a plain string, a malformed row)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _str(value: Any, default: str = "") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _num(obj: Any, *keys: str) -> Optional[float]:
    """First finite float found under any of `keys` on `obj`; None otherwise."""
    for key in keys:
        raw = _field(obj, key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        return value
    return None


def _fmt_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return NONE_LINE
    try:
        f = float(value)
    except (TypeError, ValueError):
        return NONE_LINE
    if math.isnan(f):
        return NONE_LINE
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f"{f:.{digits}f}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return NONE_LINE
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return NONE_LINE


def _to_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _bulleted(lines: Sequence[str], empty_label: str = NONE_LINE, indent: str = "  ") -> List[str]:
    if not lines:
        return [f"{indent}- {empty_label}"]
    return [f"{indent}- {line}" for line in lines]


def _rows(trailing: Any, key: str) -> Sequence[Any]:
    """A trailing sub-list by key, or () for anything not actually a list/tuple
    (missing key, None, or a malformed type) — never iterated as anything else."""
    if not isinstance(trailing, dict):
        return ()
    value = trailing.get(key)
    return value if isinstance(value, (list, tuple)) else ()


def _bias_label(long_ok: bool, short_ok: bool) -> str:
    if long_ok and short_ok:
        return "long+short"
    if long_ok:
        return "long only"
    if short_ok:
        return "short only"
    return "neutral (no bias)"


def _in_play_line(row: Any) -> str:
    symbol = _str(_field(row, "symbol"), "?")
    score = _fmt_float(_field(row, "score"))
    rvol = _fmt_float(_field(row, "rvol_open15"))
    gap = _fmt_float(_field(row, "gap_pct"))
    reasons = _field(row, "reasons") or ()
    reasons_str = ",".join(str(r) for r in reasons) if reasons else NONE_LINE
    return f"{symbol} score={score} rvol={rvol} gap={gap}% reasons={reasons_str}"


def _render_plan(plan: Any) -> str:
    lines = ["## Plan"]
    if plan is None:
        lines.append(f"- {NONE_LINE}")
        return "\n".join(lines)

    regime = _str(_field(plan, "regime"), "unknown")
    long_ok = bool(_field(plan, "long_ok", False))
    short_ok = bool(_field(plan, "short_ok", False))
    budget = _num(plan, "risk_budget_pct")
    max_trades = _field(plan, "max_trades")
    sit_out = bool(_field(plan, "sit_out", False))
    sit_out_reason = _str(_field(plan, "sit_out_reason"), "")

    lines.append(f"- regime: {regime}")
    lines.append(f"- bias: {_bias_label(long_ok, short_ok)}")
    lines.append(f"- risk budget: {_fmt_float(budget)}%")
    lines.append(f"- max trades: {_fmt_int(max_trades)}")
    lines.append(f"- sit-out: {'yes (' + (sit_out_reason or NONE_LINE) + ')' if sit_out else 'no'}")

    in_play = list(_field(plan, "in_play") or ())[:10]
    lines.append("- in-play (top 10):")
    lines.extend(_bulleted([_in_play_line(row) for row in in_play]))
    return "\n".join(lines)


def _render_decisions(decisions: Sequence[Any]) -> str:
    lines = ["## Decisions"]
    if not decisions:
        lines.append(f"- {NONE_LINE}")
        return "\n".join(lines)

    counts: Counter = Counter()
    skips: List[str] = []
    shadow_blocks: List[str] = []
    for d in decisions:
        component = _str(_field(d, "component"), "unknown")
        action = _str(_field(d, "action"), "unknown")
        reason = _str(_field(d, "reason"), "")
        symbol = _str(_field(d, "symbol"), "")
        tag = f"{symbol} " if symbol else ""
        counts[(component, action)] += 1
        if action == "skip":
            skips.append(f"{tag}{component}: {reason or NONE_LINE}")
        elif action == "shadow":
            shadow_blocks.append(f"{tag}{component}: {reason or NONE_LINE}")

    lines.append("- by component/action:")
    lines.extend(_bulleted([f"{component}/{action}: {n}"
                             for (component, action), n in sorted(counts.items())]))
    lines.append("- skips:")
    lines.extend(_bulleted(skips))
    lines.append("- shadow audits that would have blocked:")
    lines.extend(_bulleted(shadow_blocks))
    return "\n".join(lines)


def _hold_str(trade: Any) -> str:
    hold = _field(trade, "hold")
    if hold:
        return str(hold)
    entry_dt = _to_dt(_field(trade, "entry_time"))
    exit_dt = _to_dt(_field(trade, "exit_time"))
    if entry_dt is None or exit_dt is None:
        return NONE_LINE
    minutes = int((exit_dt - entry_dt).total_seconds() // 60)
    if minutes < 0:
        return NONE_LINE
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins}m" if hours else f"{mins}m"


def _trade_family(trade: Any) -> str:
    return _str(_field(trade, "family") or _field(trade, "strategy"), "unknown")


def _render_trades(trades: Sequence[Any]) -> str:
    lines = ["## Trades"]
    if not trades:
        lines.append(f"- {NONE_LINE}")
        return "\n".join(lines)

    rows = []
    for t in trades:
        symbol = _str(_field(t, "symbol"), "?")
        side = _str(_field(t, "side"), NONE_LINE)
        family = _trade_family(t)
        r = _fmt_float(_num(t, "r", "pnl_r"))
        exit_reason = _str(_field(t, "exit_reason"), NONE_LINE)
        hold = _hold_str(t)
        rows.append(f"{symbol} {side} family={family} R={r} exit={exit_reason} hold={hold}")
    lines.extend(_bulleted(rows))
    return "\n".join(lines)


def _stats(rs: Sequence[float]) -> Tuple[int, Optional[float], Optional[float]]:
    n = len(rs)
    if n == 0:
        return 0, None, None
    exp_r = sum(rs) / n
    gains = sum(r for r in rs if r > 0)
    losses = -sum(r for r in rs if r < 0)
    if losses > 0:
        pf: Optional[float] = gains / losses
    elif gains > 0:
        pf = float("inf")
    else:
        pf = None
    return n, exp_r, pf


def _today_expectancy(trades: Sequence[Any]) -> Dict[Tuple[str, str], List[float]]:
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for t in trades:
        r = _num(t, "r", "pnl_r")
        if r is None:
            continue
        family = _trade_family(t)
        regime = _str(_field(t, "regime"), "unknown")
        groups[(family, regime)].append(r)
    return groups


def _render_expectancy(trades: Sequence[Any], trailing: Dict[str, Any]) -> str:
    lines = ["## Expectancy by Setup x Regime"]

    today_groups = _today_expectancy(trades)
    today_rows = []
    for (family, regime), rs in sorted(today_groups.items()):
        n, exp_r, pf = _stats(rs)
        today_rows.append(f"{family}/{regime}: n={n} exp_R={_fmt_float(exp_r)} PF={_fmt_float(pf)}")
    lines.append("- today:")
    lines.extend(_bulleted(today_rows))

    trailing_rows = []
    for row in _rows(trailing, "expectancy"):
        family = _str(_field(row, "family"), "unknown")
        regime = _str(_field(row, "regime"), "unknown")
        n = _fmt_int(_field(row, "n"))
        exp_r = _fmt_float(_num(row, "exp_r"))
        pf = _fmt_float(_num(row, "pf"))
        trailing_rows.append(f"{family}/{regime}: n={n} exp_R={exp_r} PF={pf}")
    lines.append("- trailing 20 sessions:")
    lines.extend(_bulleted(trailing_rows))
    return "\n".join(lines)


def _render_playbook_changes(changes: Sequence[Any]) -> str:
    lines = ["## Playbook Changes"]
    if not changes:
        lines.append(f"- {NONE_LINE}")
        return "\n".join(lines)
    rows = []
    for c in changes:
        family = _str(_field(c, "family"), "?")
        side = _str(_field(c, "side"), "?")
        regime = _str(_field(c, "regime"), "?")
        old = _str(_field(c, "old_status"), NONE_LINE)
        new = _str(_field(c, "new_status"), NONE_LINE)
        reason = _str(_field(c, "reason"), NONE_LINE)
        rows.append(f"{family}/{side}/{regime}: {old} -> {new} ({reason})")
    lines.extend(_bulleted(rows))
    return "\n".join(lines)


def _suspend_candidates(trailing: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for row in _rows(trailing, "expectancy"):
        n_raw = _field(row, "n")
        exp_r = _num(row, "exp_r")
        if n_raw is None or exp_r is None:
            continue
        try:
            n = int(n_raw)
        except (TypeError, ValueError):
            continue
        if n >= _SUSPEND_MIN_N and exp_r < 0:
            family = _str(_field(row, "family"), "unknown")
            regime = _str(_field(row, "regime"), "unknown")
            out.append(f"{family}/{regime}: suspend candidate (n={n}, exp_R={_fmt_float(exp_r)})")
    return out


def _sit_out_reviews(trailing: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for row in _rows(trailing, "sit_out_days"):
        would = _num(row, "would_have_r")
        if would is None or would <= 0:
            continue
        date = _str(_field(row, "date"), "unknown")
        reason = _str(_field(row, "reason"), "")
        out.append(f"{date}: review sit-out rule (would-have R={_fmt_float(would)}, reason={reason or NONE_LINE})")
    return out


def _max_trades_reviews(trailing: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for row in _rows(trailing, "max_trades_days"):
        try:
            max_trades = int(_field(row, "max_trades"))
            taken = int(_field(row, "trades_taken"))
        except (TypeError, ValueError):
            continue
        if max_trades > 0 and taken >= max_trades:
            date = _str(_field(row, "date"), "unknown")
            out.append(f"{date}: raise/lower max_trades (hit {taken}/{max_trades})")
    return out


def _render_what_to_change(trailing: Dict[str, Any]) -> str:
    lines = ["## What to Change"]
    items: List[str] = []
    items.extend(_suspend_candidates(trailing))
    items.extend(_sit_out_reviews(trailing))
    items.extend(_max_trades_reviews(trailing))
    lines.extend(_bulleted(items))
    return "\n".join(lines)


def render_journal(plan: Any, decisions: Any, trades: Any, playbook_changes: Any,
                    trailing: Any) -> str:
    """Render the full post-close journal as markdown. Pure, never raises:
    every section degrades to a "none" line on missing/malformed input."""
    decisions = decisions if isinstance(decisions, (list, tuple)) else []
    trades = trades if isinstance(trades, (list, tuple)) else []
    playbook_changes = playbook_changes if isinstance(playbook_changes, (list, tuple)) else []
    trailing = trailing if isinstance(trailing, dict) else {}

    sections = [
        _render_plan(plan),
        _render_decisions(decisions),
        _render_trades(trades),
        _render_expectancy(trades, trailing),
        _render_playbook_changes(playbook_changes),
        _render_what_to_change(trailing),
    ]
    return "\n\n".join(sections) + "\n"
