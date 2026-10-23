"""SPEC §4 sizing + caps and §5 exit rules (pure functions; no orders)."""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

from core.swing_lane.config import (DAILY_LOSS_LIMIT_USD, EARNINGS_BLACKOUT_DAYS, MAX_NOTIONAL,
                                    MAX_OPEN, MAX_OPEN_RISK_USD, RISK_USD, TIME_STOP_DAYS)


def shares(fill: float, stop: float) -> int:
    d = fill - stop
    if not (d > 0) or not (fill > 0):
        return 0
    return int(math.floor(min(RISK_USD / d, MAX_NOTIONAL / fill)))


def next_trading_days(d: date, n: int) -> list:
    out, x = [], d
    while len(out) < n:
        x += timedelta(days=1)
        if x.weekday() < 5:
            out.append(x)
    return out


def earnings_check(signal_date: date, earnings: Optional[set], blackout_on: bool):
    """-> (blocked: bool, reason: str|None, earnings_next: date|None).
    Missing calendar => blocked (fail closed) when the blackout is on."""
    if earnings is None:
        return (True, "earnings_data_missing", None) if blackout_on else (False, None, None)
    upcoming = sorted(e for e in earnings if e > signal_date)
    nxt = upcoming[0] if upcoming else None
    if blackout_on and nxt is not None and nxt in set(next_trading_days(signal_date, EARNINGS_BLACKOUT_DAYS)):
        return True, "earnings_blackout", nxt
    return False, None, nxt


def slot_available(open_positions: int, open_risk_usd: float, new_risk_usd: float) -> bool:
    return open_positions < MAX_OPEN and open_risk_usd + new_risk_usd <= MAX_OPEN_RISK_USD + 1e-9


def daily_loss_blocked(realized_today: Optional[float]) -> bool:
    return realized_today is not None and realized_today <= -DAILY_LOSS_LIMIT_USD


def b_trail_stop(stop: float, fill: float, highest_close: float, atr: float, last_close: float) -> float:
    """B trail: once close >= fill + 1R, stop = max(stop, fill, highest_close - 2*ATR)."""
    r = fill - stop
    if r > 0 and last_close >= fill + r:
        return max(stop, fill, highest_close - 2 * atr)
    return stop


def time_stop_due(hold_day: int) -> bool:
    """Exit at the close of trading day 10 (entry day = 1). Never extended."""
    return hold_day >= TIME_STOP_DAYS
