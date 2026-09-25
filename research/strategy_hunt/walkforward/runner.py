"""Precompute trades per parameter set once; select per window in-sample; collect out-of-sample."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from research.strategy_hunt.models import SimTrade, Window
from research.strategy_hunt.sim.costs import CostModel
from research.strategy_hunt.sim.simulator import SessionTable, build_session_table, simulate
from strategies.families.common import Family, grid_combos, median_params
from strategies.families.indicators import indicator_frame
from strategies.families.models import Entry

LOOKBACK_DAYS = 120


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
                      ind: Optional[pd.DataFrame] = None, table: Optional[SessionTable] = None) -> tuple[list[SimTrade], list[dict[str, Any]]]:
    if bars5.empty or not windows:
        return [], []
    ind = ind if ind is not None else indicator_frame(bars5, daily, spy5)
    pre = precompute(family, symbol, bars5, ind, daily, news, cost_model, earnings, regime_lookup, table)
    trades: list[SimTrade] = []
    choices: list[dict[str, Any]] = []
    for w in windows:
        k = select_params(pre, w, grid=family.grid)
        oos = _in(pre.get(k, []), w.oos_start, w.oos_end)
        trades.extend(replace(t, window_id=w.window_id) for t in oos)
        choices.append({"window_id": w.window_id, "params": json.loads(k), "n_is": len(_in(pre.get(k, []), w.is_start, w.is_end))})
    return trades, choices
