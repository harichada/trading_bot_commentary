# research/strategy_hunt/sim/simulator.py
"""Array-based execution of Entry objects: stops (gap-through), targets, time stops, eod, earnings-eve, borrow.

Entry styles (Batch-2 Task B). `Entry.entry_style == "market"` is the Batch-1 lane: the entry is
taken at the fill bar's open and pays the full taker friction. `"limit_open"` instead rests a limit
`LIMIT_OFFSET_BPS` inside that open -- below it for a long, above it for a short -- and is filled at
the limit price only if the fill bar trades through it; if the bar never reaches the limit there is
no trade at all. Everything after the fill (the stop/target scan window, the exit vocabulary, the
hold accounting, the exit fill) is identical for both styles.

The `limit_open` assumption, stated plainly because it flatters the style: a resting limit at the
open is treated as a *maker* order that pays no spread and no slippage (`CostModel.entry_fill`), and
there is no queue-position model -- a bar whose low merely touches the limit is assumed to have
filled the whole order. Real queue priority at the open is the worst case for that assumption, so
a `limit_open` result is an upper bound on what the style could earn, not an estimate of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from research.strategy_hunt.models import SimTrade
from research.strategy_hunt.sim.costs import CostModel
from strategies.families.models import Entry

# Exit reason for an "overnight" hold: the only reason whose fill is the limit bar's OPEN rather
# than its close. Pricing it at the 09:30 bar's close would credit an overnight family with five
# minutes of intraday drift it never held.
NEXT_OPEN = "next_open"

# How far inside the fill bar's open a `limit_open` order rests, in basis points. Fixed rather than
# volatility-scaled so the limit-vs-market delta is one number per family and not a second grid axis.
LIMIT_OFFSET_BPS = 10
LIMIT_OPEN = "limit_open"


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
    """The bar the position is closed on at the latest, and why. `NEXT_OPEN` exits at that bar's
    open; every other reason exits at its close (see `_limit_exit_px`)."""
    s0 = int(tb.session_id[start])
    if entry.hold_kind == "intraday":
        eod = int(tb.last_pos[s0])
        ts_pos = start + int(entry.time_stop)
        return (ts_pos, "time_stop") if ts_pos <= eod else (eod, "eod")
    if entry.hold_kind == "overnight":
        # Held past the close and closed at the next session's open. The earnings-eve rule still
        # applies: if that next session is an earnings date, flatten into this session's close
        # instead of carrying the report. (An entry filled on the session's *last* bar has nowhere
        # to flatten to and is skipped before it gets here -- see `_skip_earnings_eve_overnight`.)
        #
        # One night still spans two session ids (the fill session and the next), so `hold_sessions`
        # is 2 and a short pays two sessions of borrow -- 2/252 of the annual rate, per
        # `CostModel.borrow_cost`. That is deliberate: the position is on the books across one
        # settlement date, and charging a single session would under-count the short's carry.
        if earn_next[s0]:
            return int(tb.last_pos[s0]), "earnings_eve"
        if s0 + 1 <= len(tb.first_pos) - 1:
            return int(tb.first_pos[s0 + 1]), NEXT_OPEN
        return n - 1, "data_end"
    last_session = s0 + int(entry.time_stop) - 1
    for s in range(s0, min(last_session, len(tb.last_pos) - 1) + 1):
        if earn_next[s]:
            return int(tb.last_pos[s]), "earnings_eve"
    if last_session <= len(tb.last_pos) - 1:
        return int(tb.last_pos[last_session]), "time_stop"
    return n - 1, "data_end"


def _skip_earnings_eve_overnight(entry: Entry, start: int, tb: SessionTable, earn_next: np.ndarray) -> bool:
    """v-batch1-task1-r1 (Important 2): True for an overnight entry filled on the last bar of a
    session whose next session is an earnings date.

    The canonical overnight fill *is* the session's last bar (15:50 signal, 15:55 fill), so the
    earnings-eve rule would flatten such a position on the very bar it was opened on: `hold_bars` 0,
    R ~ 0, and both legs' costs paid. That is a phantom trade rather than one the family could have
    taken — live, the same earnings-eve check that would flatten it runs before the entry — so the
    entry is dropped and no trade is emitted. An overnight entry earlier in the session is a real
    hold and still exits `earnings_eve` at the close.
    """
    if entry.hold_kind != "overnight":
        return False
    s0 = int(tb.session_id[start])
    return bool(earn_next[s0]) and start >= int(tb.last_pos[s0])


def _limit_open_fill(side: str, bar_open: float, bar_low: float, bar_high: float) -> Optional[float]:
    """The `limit_open` fill price for a fill bar, or None if the bar never reached the limit.

    The limit is drawn from the *bar's* open rather than from `Entry.entry_px` (which is the same
    number for any entry `make_entry` built) so the rule reads off the data the fill happened in.
    A touch counts: `low == limit` means the bar printed the price."""
    offset = LIMIT_OFFSET_BPS / 10_000.0
    if side == "long":
        limit = bar_open * (1.0 - offset)
        return limit if bar_low <= limit else None
    limit = bar_open * (1.0 + offset)
    return limit if bar_high >= limit else None


def _limit_exit_px(reason: str, limit_pos: int, o: np.ndarray, c: np.ndarray) -> float:
    """Fill for a limit-driven exit: the bar's open for a next-open exit, its close otherwise."""
    return float(o[limit_pos] if reason == NEXT_OPEN else c[limit_pos])


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
        if _skip_earnings_eve_overnight(e, start, tb, earn_next):
            continue
        # The entry gate: a market order always fills, a limit order only if the bar traded through
        # it. `entry_ref` is the price the fill happened at; `e.entry_px` stays the candidate's
        # reference open, which the brackets and `risk_per_share` are drawn from.
        entry_ref = float(e.entry_px)
        if e.entry_style == LIMIT_OPEN:
            fill = _limit_open_fill(e.side, float(o[start]), float(l[start]), float(h[start]))
            if fill is None:
                continue
            entry_ref = fill
        if start + 1 >= n:
            exit_pos, exit_px, reason = start, float(c[start]), "data_end"
        else:
            limit_pos, limit_reason = _limit(e, start, tb, earn_next, n)
            limit_pos = min(limit_pos, n - 1)
            # A next-open exit happens at the limit bar's open, so that bar's intrabar range is
            # never available to the position: the stop/target scan stops one bar earlier. For the
            # canonical 15:50 signal / 15:55 fill there are no bars in between at all, which is
            # what makes the overnight stop protective-only -- an overnight gap through it is
            # realized at the open, the same fill a gap-through-stop would have produced.
            scan_end = limit_pos - 1 if limit_reason == NEXT_OPEN else limit_pos
            sl = slice(start + 1, max(scan_end, start) + 1)
            i, px, why = _first_hit(e.side, o[sl], h[sl], l[sl], e.stop_px, e.target_px)
            if i is None:
                exit_pos, exit_px, reason = limit_pos, _limit_exit_px(limit_reason, limit_pos, o, c), limit_reason
            else:
                exit_pos, exit_px, reason = start + 1 + i, float(px), why
        hold_sessions = int(tb.session_id[exit_pos] - tb.session_id[start]) + 1
        sign = 1.0 if e.side == "long" else -1.0
        entry_fill = cost_model.entry_fill(e.side, entry_ref, ts, float(v[start]), style=e.entry_style)
        exit_ts = idx[exit_pos]
        exit_fill = cost_model.exit_fill(e.side, exit_px, exit_ts, float(v[exit_pos]), hold_sessions)
        r = sign * (exit_fill - entry_fill) / e.risk_per_share
        pnl_pct = sign * (exit_fill - entry_fill) / entry_fill * 100.0
        cost_pct = ((entry_fill - e.entry_px) * sign + (exit_px - exit_fill) * sign) / e.entry_px * 100.0
        out.append(SimTrade(e.symbol, family, e.side, ts, exit_ts, float(e.entry_px), float(e.stop_px), float(e.target_px),
                            exit_px, float(entry_fill), float(exit_fill), float(r), float(pnl_pct), float(cost_pct), reason,
                            int(exit_pos - start), hold_sessions, e.tag,
                            regime_lookup(ts) if regime_lookup else "unknown", window_id, params_json,
                            e.entry_style))
    return out
