"""Merge the nightly research Parquet history with the broker's short
recent window, for the live fit lanes.

v-strategy-fit-review2-2026-09-25 (Important 3): `core/engine.py`'s
`_fit_bars_source` fetched `period_type="day", period=10` of 5-minute
bars — about 7 sessions, because Schwab's price-history API caps minute
data at 10 days. But `strategies/families/indicators.py` needs 20 prior
sessions for `rvol` and 10 for `or{15,30}_range_avg20`, so the `orb` and
`news_catalyst` families could never fire live: their gating columns were
always NaN. Asking Schwab for more history is not an option (the cap is
the provider's), so the deeper history comes from the research pipeline's
own nightly export instead, and this module splices the two together.

Pure pandas, no engine or research imports, no network: `core/` must not
depend on `research/` (see `core/strategy_fit.py`), it just reads the
Parquet files that pipeline writes. `load_history` never raises — a
missing root, missing file, or unreadable Parquet all degrade to "no
history", leaving the caller with the provider's recent window alone
(exactly the pre-fix behavior).

Frame contract (same as the rest of the fit system): 5-minute bars carry a
naive-ET `DatetimeIndex` named `ts` with columns open/high/low/close/
volume; daily frames are indexed by normalized `date`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

BARS5_SUBDIR = "bars5"
DAILY_SUBDIR = "daily"
_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


def _naive_et_index(df: pd.DataFrame, index_name: str, normalize: bool) -> pd.DataFrame:
    """Copy of `df` whose index is a naive-ET `DatetimeIndex` named
    `index_name` (normalized to midnight for daily frames)."""
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    if normalize:
        idx = idx.normalize()
    idx.name = index_name
    out = df.copy()
    out.index = idx
    return out


def _merge(hist: Optional[pd.DataFrame], recent: Optional[pd.DataFrame], index_name: str,
           normalize: bool) -> pd.DataFrame:
    """Concatenate `hist` then `recent`, drop duplicate timestamps keeping the
    later (recent) row, and sort ascending.

    `recent` wins every overlapping timestamp: the broker's view of a bar is
    the live one, and the nightly export may predate an adjustment. Either
    side may be None or empty; the result keeps `recent`'s columns and attrs
    when it has any.
    """
    frames = [f for f in (hist, recent) if f is not None and not f.empty]
    if not frames:
        return recent if recent is not None else pd.DataFrame()
    # Each index is coerced to naive ET *before* concatenating: mixing a naive
    # and a tz-aware DatetimeIndex in pd.concat yields an object-dtype index
    # that pd.to_datetime then refuses outright.
    frames = [_naive_et_index(f, index_name, normalize) for f in frames]
    if len(frames) == 1:
        out = frames[0]
    else:
        # Align on the recent frame's columns so a stale export with extra or
        # differently-ordered columns can't reshape what the caller gets.
        cols = list(frames[-1].columns)
        out = pd.concat([f.reindex(columns=cols) for f in frames])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if recent is not None:
        out.attrs = dict(recent.attrs)
    return out


def merge_history(hist5: Optional[pd.DataFrame], recent5: Optional[pd.DataFrame]) -> pd.DataFrame:
    """5-minute bars: nightly research history spliced under the provider's
    recent window. Index named `ts`, naive ET, ascending, no duplicate
    timestamps (the recent frame wins)."""
    return _merge(hist5, recent5, index_name="ts", normalize=False)


def merge_daily(hist_d: Optional[pd.DataFrame], recent_d: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Daily bars: same splice, with the index normalized to midnight and
    named `date` (what `daily_features` reindexes against)."""
    return _merge(hist_d, recent_d, index_name="date", normalize=True)


def _read_parquet(path: Path, index_name: str, normalize: bool) -> Optional[pd.DataFrame]:
    """One Parquet file as a bars frame, or None if it can't be read.

    The research exporter writes the timestamp as a column (`ts`/`date`), so
    it is promoted to the index here when present.
    """
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        # Missing pyarrow, truncated/corrupt file, permission error — history
        # is an optimization, never a requirement.
        logger.debug("strategy_fit_bars: unreadable parquet %s", path, exc_info=True)
        return None
    try:
        df = df.rename(columns={c: str(c).lower() for c in df.columns})
        if index_name in df.columns:
            df = df.set_index(index_name)
        elif "ts" in df.columns:
            df = df.set_index("ts")
        elif "date" in df.columns:
            df = df.set_index("date")
        keep = [c for c in _BAR_COLUMNS if c in df.columns]
        if not keep:
            return None
        df = df[keep]
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        if normalize:
            idx = idx.normalize()
        idx.name = index_name
        df.index = idx
        return df[~df.index.duplicated(keep="last")].sort_index()
    except Exception:
        logger.debug("strategy_fit_bars: unparseable parquet %s", path, exc_info=True)
        return None


def load_history(root: str, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """`(bars5, daily)` from `<root>/bars5/<SYM>.parquet` and
    `<root>/daily/<SYM>.parquet`; either element is None when absent or
    unreadable. Never raises — this runs on the trading path."""
    try:
        base = Path(root)
        sym = str(symbol).upper()
        bars5 = _read_parquet(base / BARS5_SUBDIR / f"{sym}.parquet", "ts", normalize=False)
        daily = _read_parquet(base / DAILY_SUBDIR / f"{sym}.parquet", "date", normalize=True)
        return bars5, daily
    except Exception:
        logger.debug("strategy_fit_bars: load_history failed for %s under %s", symbol, root, exc_info=True)
        return None, None
