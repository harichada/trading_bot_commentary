"""Read-only data access via the repo's existing Parquet tree (no new feeds).

Daily bars: core.strategy_fit_bars.load_history -> `<root>/daily/<SYM>.parquet`.
Earnings:   `<root>/earnings/<SYM>.parquet` (`date` column), the same file
            core.swing_exits.load_earnings_dates reads. Unlike that helper we
            distinguish *missing* (None -> fail closed) from *no dates* (empty set).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("swing_lane")
_COLS = ("open", "high", "low", "close", "volume")


def load_daily(root: str, symbol: str) -> Optional[pd.DataFrame]:
    from core.strategy_fit_bars import load_history
    _, daily = load_history(root, symbol)
    if daily is None or daily.empty:
        return None
    df = daily.rename(columns={c: c.lower() for c in daily.columns})
    if not all(c in df.columns for c in _COLS):
        return None
    return df[list(_COLS)].astype(float).sort_index()


def load_earnings(root: str, symbol: str) -> Optional[set]:
    """Set of earnings dates, or None when the calendar is unavailable for `symbol`."""
    path = Path(root) / "earnings" / f"{symbol.upper()}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["date"])
        return set(pd.to_datetime(df["date"]).dt.date.tolist())
    except Exception:
        logger.debug("swing_lane: unreadable earnings file %s", path, exc_info=True)
        return None


def drop_partial(df: pd.DataFrame, now_et) -> pd.DataFrame:
    """Use only completed bars: drop today's bar before 16:15 ET (SPEC §2)."""
    from core.swing_lane.config import COMPLETE_BAR_AFTER
    if df is None or df.empty:
        return df
    last = pd.Timestamp(df.index[-1]).date()
    if last >= now_et.date() and (now_et.hour, now_et.minute) < COMPLETE_BAR_AFTER:
        return df[[pd.Timestamp(i).date() < now_et.date() for i in df.index]]
    return df
