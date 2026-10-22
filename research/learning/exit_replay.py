"""v-learning-loop-2026-10-22: LEARNING_EXIT_SHADOW counterfactual exit replay.

Question: what would each closed live trade have made WITHOUT the proactive
early exits (MACD flip, RSI<50, RSI-extreme, trailing, manual) — i.e. holding
to the ORIGINAL stop, the ORIGINAL target, or the day-trade time flatten?

Rules (1-minute bars, naive ET):
  * replay starts on the first bar AFTER the entry minute (the entry-minute bar
    mixes pre-fill prices).
  * long: open <= stop -> exit at open (gap); low <= stop -> exit at stop;
    open >= target -> exit at open; high >= target -> exit at target.
    Short mirrored. Stop and target in the same bar -> STOP (conservative).
  * first bar at/after the flatten time (DAY_TRADE_FLATTEN_HOUR, 15:00) -> exit at its open.
  * no slippage/fees are modelled on either side (the actual fills already carry theirs).
Status: ok | no_bars (no bar after entry) | incomplete (bars end before an exit).
Report-only; nothing here touches orders or the live exit path.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from research.learning.data import DAY_TRADE_STRATEGIES, Trade, load_bars_csv, load_csv

DEFAULT_FLATTEN = dtime(15, 0)


@dataclass
class ReplayResult:
    id: str
    symbol: str
    side: str
    strategy: str
    session: str
    actual_exit_reason: str
    actual_pnl: float
    actual_r: Optional[float]
    status: str
    cf_exit_kind: str = ""
    cf_exit_ts: str = ""
    cf_exit_price: Optional[float] = None
    cf_pnl: Optional[float] = None
    cf_r: Optional[float] = None
    bars_used: int = 0
    note: str = ""


def replay_one(t: Trade, bars: Optional[pd.DataFrame], flatten: dtime = DEFAULT_FLATTEN) -> ReplayResult:
    base = dict(id=str(t.id), symbol=t.symbol, side=t.side, strategy=t.strategy, session=t.session,
                actual_exit_reason=t.exit_reason, actual_pnl=round(t.pnl, 2),
                actual_r=None if t.r is None else round(t.r, 3))
    if t.stop is None or t.target is None:
        return ReplayResult(status="no_stop_target", note="no stored stop/target", **base)
    if bars is None or bars.empty:
        return ReplayResult(status="no_bars", note="minute_bars missing for this session", **base)
    entry_min = t.entry_ts.floor("min")
    b = bars[bars["ts"] > entry_min].sort_values("ts")
    if b.empty:
        return ReplayResult(status="no_bars", note="no bar after entry minute", **base)
    s, tg, long_ = t.stop, t.target, t.side == "long"
    kind = px = None
    ts_hit = None
    n = 0
    for row in b.itertuples(index=False):
        n += 1
        ts = pd.Timestamp(row.ts)
        if ts.time() >= flatten:
            kind, px, ts_hit = "time_flatten", float(row.o), ts
            break
        o, h, l = float(row.o), float(row.h), float(row.l)
        if long_:
            if o <= s: kind, px = "stop_gap", o
            elif l <= s: kind, px = "stop", s
            elif o >= tg: kind, px = "target_gap", o
            elif h >= tg: kind, px = "target", tg
        else:
            if o >= s: kind, px = "stop_gap", o
            elif h >= s: kind, px = "stop", s
            elif o <= tg: kind, px = "target_gap", o
            elif l <= tg: kind, px = "target", tg
        if kind:
            ts_hit = ts
            break
    if kind is None:
        return ReplayResult(status="incomplete", bars_used=n,
                            note=f"bars end {b['ts'].iloc[-1]} before stop/target/flatten", **base)
    pnl = (px - t.entry) * t.qty * (1 if long_ else -1)
    r = pnl / (t.rps * t.qty) if t.rps else None
    return ReplayResult(status="ok", cf_exit_kind=kind, cf_exit_ts=str(ts_hit), cf_exit_price=round(px, 4),
                        cf_pnl=round(pnl, 2), cf_r=None if r is None else round(r, 3), bars_used=n, **base)


def replay(trades: Sequence[Trade], bars: Dict[str, pd.DataFrame], flatten: dtime = DEFAULT_FLATTEN) -> List[ReplayResult]:
    return [replay_one(t, bars.get(str(t.id)), flatten) for t in trades]


def summarize(results: Sequence[ReplayResult]) -> Dict[str, object]:
    ok = [r for r in results if r.status == "ok"]
    def s(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs), 2)
    out = {
        "n_total": len(results), "n_replayed": len(ok),
        "n_no_bars": sum(r.status == "no_bars" for r in results),
        "n_incomplete": sum(r.status == "incomplete" for r in results),
        "n_no_stop_target": sum(r.status == "no_stop_target" for r in results),
        "replayed_actual_pnl": s(r.actual_pnl for r in ok), "replayed_cf_pnl": s(r.cf_pnl for r in ok),
        "replayed_actual_r": s(r.actual_r for r in ok), "replayed_cf_r": s(r.cf_r for r in ok),
        "replayed_n_with_r": sum(r.actual_r is not None and r.cf_r is not None for r in ok),
        "unreplayed_actual_pnl": s(r.actual_pnl for r in results if r.status != "ok"),
        "unreplayed_actual_r": s(r.actual_r for r in results if r.status != "ok"),
        "cf_exit_kinds": {k: sum(r.cf_exit_kind == k for r in ok) for k in sorted({r.cf_exit_kind for r in ok})},
    }
    out["delta_pnl"] = round(out["replayed_cf_pnl"] - out["replayed_actual_pnl"], 2)
    out["delta_r"] = round(out["replayed_cf_r"] - out["replayed_actual_r"], 3)
    return out


def render(results: Sequence[ReplayResult], summary: Dict[str, object], title: str) -> str:
    L = [f"# {title}", "", "Counterfactual = hold to ORIGINAL stop / ORIGINAL target / 15:00 ET flatten; "
         "no MACD-flip, RSI<50, RSI-extreme, trailing or manual early exits. 1-min bars, next-bar start, "
         "stop wins ties, no slippage modelled.", "",
         "| id | sym | side | session | actual exit | act $ | act R | status | cf exit | cf $ | cf R |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r.id} | {r.symbol} | {r.side} | {r.session} | {r.actual_exit_reason} | {r.actual_pnl:+.2f} | "
                 f"{'' if r.actual_r is None else f'{r.actual_r:+.2f}'} | {r.status} | {r.cf_exit_kind} | "
                 f"{'' if r.cf_pnl is None else f'{r.cf_pnl:+.2f}'} | {'' if r.cf_r is None else f'{r.cf_r:+.2f}'} |")
    L += ["", "## Summary", "", "```json", json.dumps(summary, indent=1), "```"]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Counterfactual exit replay (report only).")
    ap.add_argument("--csv", required=True); ap.add_argument("--bars", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--all-strategies", action="store_true")
    ap.add_argument("--denylist", default="HQGE,SPCX")
    a = ap.parse_args(argv)
    trades = load_csv(a.csv, a.denylist.split(","))
    if not a.all_strategies:
        trades = [t for t in trades if t.strategy in DAY_TRADE_STRATEGIES]
    res = replay(trades, load_bars_csv(a.bars))
    summ = summarize(res)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res, summ, "Exit-shadow replay"))
    out.with_suffix(".json").write_text(json.dumps({"summary": summ, "trades": [asdict(r) for r in res]}, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
