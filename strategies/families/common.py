"""Helpers shared by candidate families: Family descriptor, grids, entry construction, de-duplication."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from strategies.families.models import ENTRY_STYLES, HOLD_KINDS, Entry

EntryFn = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict], list[Entry]]
FLATTEN_BAR = 66  # 15:00 ET on a 5-min RTH grid (09:30 = bar 0)


@dataclass(frozen=True)
class Family:
    name: str
    fn: EntryFn
    grid: dict[str, list]
    hold_kind: str
    sides: tuple[str, ...]


def grid_combos(grid: dict[str, list]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    values = [sorted(grid[k], key=lambda v: (str(type(v)), v)) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def median_params(grid: dict[str, list]) -> dict[str, Any]:
    out = {}
    for k in sorted(grid):
        vals = sorted(grid[k], key=lambda v: (str(type(v)), v))
        out[k] = vals[len(vals) // 2]
    return out


def make_entry(bars5: pd.DataFrame, i: int, side: str, stop_px: float, target_px: float,
               time_stop: int, hold_kind: str, tag: str, symbol: Optional[str] = None,
               entry_style: str = "market") -> Optional[Entry]:
    """Entry at the open of bar i+1. None if there is no next bar, or (intraday) it belongs to another session.

    `hold_kind` must be one of HOLD_KINDS ("intraday", "multiday", "overnight"); an unknown kind
    raises here rather than producing an Entry the simulator would have to guess how to exit.
    Only "intraday" is confined to a single session: an "overnight" entry is allowed to be a
    session's last bar (its whole point is being held past the close).

    `entry_style` (Batch-2 Task B) is one of ENTRY_STYLES and defaults to the market fill every
    family used before, so an existing call site produces exactly the Entry it produced before.
    It is validated here as well as in `Entry` so a typo in a grid fails at the family boundary.
    Note that the returned `entry_px` is the bar's open for *both* styles: it is the candidate's
    reference price (what the brackets are drawn from), and only the simulator knows whether the
    limit was reached and at what price the fill therefore happened."""
    if hold_kind not in HOLD_KINDS:
        raise ValueError(f"hold_kind must be one of {HOLD_KINDS}, got {hold_kind!r}")
    if entry_style not in ENTRY_STYLES:
        raise ValueError(f"entry_style must be one of {ENTRY_STYLES}, got {entry_style!r}")
    j = i + 1
    if j >= len(bars5):
        return None
    ts_i, ts_j = bars5.index[i], bars5.index[j]
    if hold_kind == "intraday" and ts_i.normalize() != ts_j.normalize():
        return None
    px = float(bars5["open"].iloc[j])
    if px <= 0 or stop_px <= 0 or target_px <= 0:
        return None
    if (side == "long" and not (stop_px < px < target_px)) or (side == "short" and not (target_px < px < stop_px)):
        return None
    return Entry(symbol or str(bars5.attrs.get("symbol", "")), side, ts_j, px, float(stop_px), float(target_px),
                 int(time_stop), hold_kind, tag, entry_style)


def one_per_session(entries: list[Entry]) -> list[Entry]:
    seen: set[tuple[pd.Timestamp, str]] = set()
    kept: list[Entry] = []
    for e in sorted(entries, key=lambda x: (x.entry_ts, x.side)):
        key = (e.entry_ts.normalize(), e.side)
        if key in seen:
            continue
        seen.add(key)
        kept.append(e)
    return kept


def spaced_by_sessions(entries: list[Entry], sessions: int) -> list[Entry]:
    """Keep an entry only if `sessions` distinct session-dates have elapsed since the last kept entry of that side."""
    dates = sorted({e.entry_ts.normalize() for e in entries})
    rank = {d: i for i, d in enumerate(dates)}
    last_rank: dict[str, int] = {}
    kept: list[Entry] = []
    for e in sorted(entries, key=lambda x: (x.entry_ts, x.side)):
        r = rank[e.entry_ts.normalize()]
        if e.side in last_rank and r - last_rank[e.side] < sessions:
            continue
        last_rank[e.side] = r
        kept.append(e)
    return kept


def bars_to_1500(bar_of_day: int) -> int:
    return max(1, FLATTEN_BAR - int(bar_of_day))


def candidate_positions(mask: np.ndarray) -> np.ndarray:
    """Indices where mask is True, excluding the last bar (no next-bar open to enter on)."""
    pos = np.flatnonzero(np.asarray(mask, dtype=bool))
    return pos[pos < len(mask) - 1]


def col(ind: pd.DataFrame, name: str) -> np.ndarray:
    """Column as a float ndarray; an all-NaN array of the right length if the column is missing."""
    if name not in ind.columns:
        return np.full(len(ind), np.nan)
    return ind[name].to_numpy(dtype=float)


def session_opens(ind: pd.DataFrame) -> list[int]:
    """Index positions where a new session starts (bar_of_day == 0), excluding the very first
    session in the frame (there is no prior bar to build a make_entry(..., j-1, ...) trade from).
    Missing bar_of_day is treated as all-NaN (via col), so this returns [] rather than raising."""
    return [int(i) for i in np.flatnonzero(col(ind, "bar_of_day") == 0) if i > 0]
