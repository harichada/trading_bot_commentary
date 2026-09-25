# research/strategy_hunt/sim/simulator.py
"""Array-based execution of Entry objects: stops (gap-through), targets, time stops, eod, earnings-eve, borrow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from research.strategy_hunt.models import SimTrade
from research.strategy_hunt.sim.costs import CostModel
from strategies.families.models import Entry


@dataclass(frozen=True)
class SessionTable:
    session_id: np.ndarray
    first_pos: np.ndarray
    last_pos: np.ndarray
    dates: pd.DatetimeIndex

    def next_is_earnings(self, earnings: set) -> np.ndarray:
        nxt = list(self.dates[1:]) + [None]
        return np.array([d is not None and pd.Timestamp(d) in earnings for d in nxt], dtype=bool)


def build_session_table(bars5: pd.DataFrame) -> SessionTable:
    days = bars5.index.normalize()
    codes, uniques = pd.factorize(days, sort=False)
    first = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    last = np.r_[first[1:] - 1, len(codes) - 1]
    return SessionTable(codes.astype(np.int64), first, last, pd.DatetimeIndex(uniques))


def _limit(entry: Entry, start: int, tb: SessionTable, earn_next: np.ndarray, n: int) -> tuple[int, str]:
    s0 = int(tb.session_id[start])
    if entry.hold_kind == "intraday":
        eod = int(tb.last_pos[s0])
        ts_pos = start + int(entry.time_stop)
        return (ts_pos, "time_stop") if ts_pos <= eod else (eod, "eod")
    last_session = s0 + int(entry.time_stop) - 1
    for s in range(s0, min(last_session, len(tb.last_pos) - 1) + 1):
        if earn_next[s]:
            return int(tb.last_pos[s]), "earnings_eve"
    if last_session <= len(tb.last_pos) - 1:
        return int(tb.last_pos[last_session]), "time_stop"
    return n - 1, "data_end"


def _first_hit(side: str, o, h, l, stop: float, target: float) -> tuple[Optional[int], Optional[float], Optional[str]]:
    if side == "long":
        stop_hit, tgt_hit = l <= stop, h >= target
    else:
        stop_hit, tgt_hit = h >= stop, l <= target
    i_s = int(np.argmax(stop_hit)) if stop_hit.any() else None
    i_t = int(np.argmax(tgt_hit)) if tgt_hit.any() else None
    if i_s is not None and (i_t is None or i_s <= i_t):
        gap = (o[i_s] <= stop) if side == "long" else (o[i_s] >= stop)
        return i_s, float(o[i_s]) if gap else stop, "stop"
    if i_t is not None:
        return i_t, target, "target"
    return None, None, None


def simulate(entries: list[Entry], bars5: pd.DataFrame, cost_model: CostModel, earnings: set, family: str,
             window_id: int = -1, params_json: str = "{}",
             regime_lookup: Optional[Callable[[pd.Timestamp], str]] = None,
             table: Optional[SessionTable] = None) -> list[SimTrade]:
    if bars5.empty or not entries:
        return []
    tb = table or build_session_table(bars5)
    earn_next = tb.next_is_earnings(earnings)
    o, h, l, c, v = (bars5[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    idx = bars5.index
    pos_of = pd.Series(np.arange(len(idx)), index=idx)
    n = len(idx)
    out: list[SimTrade] = []
    for e in entries:
        ts = pd.Timestamp(e.entry_ts)
        if ts not in pos_of.index:
            continue
        start = int(pos_of[ts])
        if start + 1 >= n:
            exit_pos, exit_px, reason = start, float(c[start]), "data_end"
        else:
            limit_pos, limit_reason = _limit(e, start, tb, earn_next, n)
            limit_pos = min(limit_pos, n - 1)
            sl = slice(start + 1, limit_pos + 1)
            i, px, why = _first_hit(e.side, o[sl], h[sl], l[sl], e.stop_px, e.target_px)
            if i is None:
                exit_pos, exit_px, reason = limit_pos, float(c[limit_pos]), limit_reason
            else:
                exit_pos, exit_px, reason = start + 1 + i, float(px), why
        hold_sessions = int(tb.session_id[exit_pos] - tb.session_id[start]) + 1
        sign = 1.0 if e.side == "long" else -1.0
        entry_fill = cost_model.entry_fill(e.side, e.entry_px, ts, float(v[start]))
        exit_ts = idx[exit_pos]
        exit_fill = cost_model.exit_fill(e.side, exit_px, exit_ts, float(v[exit_pos]), hold_sessions)
        r = sign * (exit_fill - entry_fill) / e.risk_per_share
        pnl_pct = sign * (exit_fill - entry_fill) / entry_fill * 100.0
        cost_pct = ((entry_fill - e.entry_px) * sign + (exit_px - exit_fill) * sign) / e.entry_px * 100.0
        out.append(SimTrade(e.symbol, family, e.side, ts, exit_ts, float(e.entry_px), float(e.stop_px), float(e.target_px),
                            exit_px, float(entry_fill), float(exit_fill), float(r), float(pnl_pct), float(cost_pct), reason,
                            int(exit_pos - start), hold_sessions, e.tag,
                            regime_lookup(ts) if regime_lookup else "unknown", window_id, params_json))
    return out
