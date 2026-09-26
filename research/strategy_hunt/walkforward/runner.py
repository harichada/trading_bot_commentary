"""Precompute trades per parameter set once; select per window in-sample; collect out-of-sample."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd

from research.strategy_hunt.models import SimTrade, Window
from research.strategy_hunt.sim.costs import CostModel
from research.strategy_hunt.sim.simulator import SessionTable, build_session_table, simulate
from strategies.families.common import Family, grid_combos, median_params
from strategies.families.indicators import indicator_frame
from strategies.families.models import Entry

logger = logging.getLogger("strategy_hunt.walkforward.runner")
LOOKBACK_DAYS = 120


def coerce_pin(raw: str, values: list) -> Any:
    """`raw` typed like the grid axis it replaces (Batch-2 Task F).

    The types come from the axis itself rather than from the string's shape: `"3"` on a float axis
    must be `3.0`, not `3`, or `grid_combos`/`median_params` would sort a mixed-type axis and the
    family would receive an `int` where every other combo carries a `float`. A mixed int/float axis
    widens to float -- truncating would silently pin a value the operator did not ask for. A value
    that cannot be typed raises, because a pin the run ignored is worse than a run that never
    started."""
    types = {type(v) for v in values}
    try:
        if bool in types:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        if str in types or not types:
            return raw
        if float in types:
            return float(raw)
        if int in types:
            return int(raw)
    except ValueError as exc:
        raise ValueError(f"grid pin value {raw!r} is not a valid {sorted(t.__name__ for t in types)}: {exc}") from exc
    return raw


def pin_grid(grid: dict[str, list], pins: Mapping[str, str]) -> dict[str, list]:
    """A new grid with every axis named in `pins` collapsed to that single typed value.

    Axes the pin does not name are copied through untouched, and a pin naming an axis this grid
    does not have is ignored -- pinning `entry_style` across a run means "in every family that has
    the choice", not "only families that have it may run"."""
    if not pins:
        return dict(grid)
    out: dict[str, list] = {}
    for key, values in grid.items():
        if key in pins:
            value = coerce_pin(str(pins[key]), list(values))
            if value not in list(values):
                logger.warning("grid pin %s=%r is not one of that axis's values %r; pinning it anyway",
                               key, value, list(values))
            out[key] = [value]
        else:
            out[key] = list(values)
    return out


def pinned_family(family: Family, pins: Optional[Mapping[str, str]]) -> Family:
    """`family` with its grid pinned. The same object when nothing changes, so an unpinned run is
    byte-for-byte the run it was before this flag existed."""
    if not pins or not any(k in family.grid for k in pins):
        return family
    return replace(family, grid=pin_grid(family.grid, pins))


def entries_in(entries: list[Entry], start: pd.Timestamp, end: pd.Timestamp) -> list[Entry]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return [x for x in entries if s <= x.entry_ts < e]


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def precompute(family: Family, symbol: str, bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame,
               cost_model: CostModel, earnings: set, regime_lookup: Optional[Callable[[pd.Timestamp], str]] = None,
               table: Optional[SessionTable] = None) -> dict[str, list[SimTrade]]:
    bars = bars5.copy()
    bars.attrs["symbol"] = symbol
    tb = table or build_session_table(bars5)
    out: dict[str, list[SimTrade]] = {}
    for params in grid_combos(family.grid):
        k = _key(params)
        entries = family.fn(bars, ind, daily, news, params)
        out[k] = simulate(entries, bars5, cost_model, earnings, family.name, -1, k, regime_lookup, tb)
    return out


def _in(trades: list[SimTrade], start: pd.Timestamp, end: pd.Timestamp) -> list[SimTrade]:
    return [t for t in trades if start <= t.entry_ts < end]


def select_params(pre: dict[str, list[SimTrade]], window: Window, min_trades: int = 30, grid: Optional[dict] = None) -> str:
    best: Optional[tuple[float, str]] = None
    for k in sorted(pre):
        is_trades = _in(pre[k], window.is_start, window.is_end)
        if len(is_trades) < min_trades:
            continue
        score = float(np.mean([t.r_multiple for t in is_trades]))
        if best is None or score > best[0]:
            best = (score, k)
    if best:
        return best[1]
    return _key(median_params(grid or {}))


def run_symbol_family(family: Family, symbol: str, bars5: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame,
                      cost_model: CostModel, earnings: set, windows: list[Window],
                      regime_lookup: Optional[Callable[[pd.Timestamp], str]] = None, spy5: Optional[pd.DataFrame] = None,
                      ind: Optional[pd.DataFrame] = None, table: Optional[SessionTable] = None,
                      context: Optional[pd.DataFrame] = None) -> tuple[list[SimTrade], list[dict[str, Any]]]:
    """`context` (v-playbook-desk-b1-2026-09-25) is the symbol's universe-context slice from
    `research.strategy_hunt.context.load_context_slice`; only consulted when `ind` isn't already
    precomputed (a caller that hands in `ind` has already decided whether to join it)."""
    if bars5.empty or not windows:
        return [], []
    ind = ind if ind is not None else indicator_frame(bars5, daily, spy5, context)
    pre = precompute(family, symbol, bars5, ind, daily, news, cost_model, earnings, regime_lookup, table)
    trades: list[SimTrade] = []
    choices: list[dict[str, Any]] = []
    for w in windows:
        k = select_params(pre, w, grid=family.grid)
        oos = _in(pre.get(k, []), w.oos_start, w.oos_end)
        trades.extend(replace(t, window_id=w.window_id) for t in oos)
        choices.append({"window_id": w.window_id, "params": json.loads(k), "n_is": len(_in(pre.get(k, []), w.is_start, w.is_end))})
    return trades, choices
