"""Rolling walk-forward windows with an embargo between in-sample and out-of-sample."""
from __future__ import annotations

import pandas as pd

from research.strategy_hunt.models import Window


def _month_start_on_or_after(ts: pd.Timestamp) -> pd.Timestamp:
    """Nearest month start: rounds up (next month) when >= 15 days into the month, down otherwise."""
    ts = pd.Timestamp(ts).normalize()
    first = ts.replace(day=1)
    return first + pd.DateOffset(months=1) if ts.day >= 15 else first


def build_windows(first_bar: pd.Timestamp, last_bar: pd.Timestamp, is_months: int = 12, oos_months: int = 3,
                  step_months: int = 3, embargo_days: int = 7, start: str = "2022-01-01") -> list[Window]:
    first_bar, last_bar = pd.Timestamp(first_bar), pd.Timestamp(last_bar)
    earliest = max(pd.Timestamp(start), first_bar.normalize() + pd.DateOffset(months=is_months))
    oos_start = _month_start_on_or_after(earliest)
    out: list[Window] = []
    while oos_start < last_bar:
        is_start = oos_start - pd.DateOffset(months=is_months)
        is_end = oos_start - pd.Timedelta(days=embargo_days)
        oos_end = min(oos_start + pd.DateOffset(months=oos_months), last_bar)
        out.append(Window(len(out), is_start, is_end, oos_start, oos_end))
        oos_start = oos_start + pd.DateOffset(months=step_months)
    return out
