"""Rolling walk-forward windows with an embargo between in-sample and out-of-sample."""
from __future__ import annotations

import pandas as pd

from research.strategy_hunt.models import Window

GRACE_DAYS = 7  # tolerate an in-sample start up to this many days before the first available bar


def _month_start_with_grace(ts: pd.Timestamp) -> pd.Timestamp:
    """First day of ts's month if ts is within GRACE_DAYS of that month start (bounded early-data
    tolerance so the in-sample window doesn't silently start too far before the first bar);
    otherwise the first day of the next month (ceiling)."""
    ts = pd.Timestamp(ts).normalize()
    first = ts.replace(day=1)
    return first if ts.day <= GRACE_DAYS + 1 else first + pd.DateOffset(months=1)


def build_windows(first_bar: pd.Timestamp, last_bar: pd.Timestamp, is_months: int = 12, oos_months: int = 3,
                  step_months: int = 3, embargo_days: int = 7, start: str = "2022-01-01") -> list[Window]:
    first_bar, last_bar = pd.Timestamp(first_bar), pd.Timestamp(last_bar)
    earliest = max(pd.Timestamp(start), first_bar.normalize() + pd.DateOffset(months=is_months))
    oos_start = _month_start_with_grace(earliest)
    out: list[Window] = []
    while oos_start < last_bar:
        is_start = oos_start - pd.DateOffset(months=is_months)
        is_end = oos_start - pd.Timedelta(days=embargo_days)
        oos_end = min(oos_start + pd.DateOffset(months=oos_months), last_bar)
        out.append(Window(len(out), is_start, is_end, oos_start, oos_end))
        oos_start = oos_start + pd.DateOffset(months=step_months)
    return out
