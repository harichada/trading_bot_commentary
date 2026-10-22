"""v-learning-loop-2026-10-22: hard risk caps + stop/R entry validation (pure).

Wired from `risk/manager.py` (per-trade clamp, stop/R check) and
`core/engine.py::_process_signal_with_commentary` (daily realized-loss halt,
new entries only). Every caller gates on its Config flag; with the flags off
nothing here runs. Hands-off denylist symbols (trading.hands_off_denylist)
never count toward the realized total.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Iterable, Optional, Tuple


def clamp_shares_to_risk(shares: int, risk_per_share: float, max_risk_usd: float) -> int:
    """Largest share count <= `shares` with shares * risk_per_share <= max_risk_usd.
    Never increases size. Bad inputs leave `shares` unchanged (fail-safe for a
    clamp is 'no extra clamp', the other caps still bind)."""
    try:
        shares = int(shares)
        rps = float(risk_per_share)
        cap = float(max_risk_usd)
    except (TypeError, ValueError):
        return shares
    if shares <= 0 or not (rps > 0) or not (cap > 0) or math.isinf(rps):
        return shares
    return max(0, min(shares, int(math.floor(cap / rps + 1e-9))))


def validate_stop(side: str, entry: Any, stop: Any) -> Tuple[bool, str]:
    """Is R derivable for a new entry? long: 0 < stop < entry; short: stop > entry."""
    try:
        e, s = float(entry), float(stop)
    except (TypeError, ValueError):
        return False, "stop_missing"
    if not (math.isfinite(e) and math.isfinite(s)) or e <= 0 or s <= 0:
        return False, "stop_missing"
    if side == "long" and not s < e:
        return False, "stop_wrong_side"
    if side == "short" and not s > e:
        return False, "stop_wrong_side"
    return True, "stop_ok"


def _exit_date(v: Any) -> Optional[date]:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str) and v:
        try:
            return datetime.fromisoformat(v).date()
        except ValueError:
            return None
    return None


def realized_bot_pnl_today(trade_history: Iterable[Any], today: date, denylist: Iterable[str] = ()) -> float:
    """Sum of realized `pnl` for bot-managed trades closed `today`, excluding the
    hands-off denylist. Mirrors the realized half of engine._compute_bot_daily_pnl."""
    deny = {str(s).upper() for s in (denylist or ())}
    total = 0.0
    for t in trade_history or ():
        try:
            if not isinstance(t, dict) or _exit_date(t.get("exit_time")) != today:
                continue
            if str(t.get("symbol", "")).upper() in deny or not t.get("managed_by_bot", True):
                continue
            if t.get("is_external") or t.get("is_hands_off"):
                continue
            total += float(t.get("pnl") or t.get("realized_pnl") or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def daily_halt(realized_today: float, limit_usd: float) -> Tuple[bool, str]:
    try:
        if float(realized_today) <= -abs(float(limit_usd)):
            return True, f"daily_realized_loss_halt pnl={float(realized_today):.2f} limit=-{abs(float(limit_usd)):.2f}"
    except (TypeError, ValueError):
        pass
    return False, "ok"
