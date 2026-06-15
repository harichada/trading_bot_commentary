"""Mean-reversion GATED to its regime — the decision test for promoting
the allocator from shadow to a live gate.

v-meanrev-gated-wf-2026-06-15. The bot's breadwinner is mean-reversion
(walk-forward 2026-06-09: PF 1.55 in its good window w6, 1.34 in w3),
but it bled -104.7% in w2 — a TRENDING window where a chop strategy
should never have traded. Five continuation-entry hunts failed; the
data keeps saying the edge is regime-gating, not the entry. This test
answers the actual roadmap question:

  Does gating mean-rev to its CHOPPY regime lift its OWN pooled PF
  past 1.30 by removing the trend-window disasters?

Gate is INVERTED vs pullback: mean-rev trades ONLY when SPY's 20-day
efficiency ratio reads CHOPPY (ER < threshold). Uses the REAL
mean-reversion strategy via the replay engine (not a reimplementation)
— trades captured with entry timestamps, then each entry's day matched
to the SPY ER series (shifted 1 day, no lookahead).

Compares baseline (all mean-rev trades) vs gated, across ER cuts
{0.25, 0.30, 0.35}.

DECISION GATE (pre-committed): promote the allocator to a LIVE gate
for mean-rev if gated pooled PF >= 1.30 AND the w2 disaster is
removed (w2 sum > -15%) AND no window sum < -25%, at a threshold that
also keeps >= 60% of the baseline's winning-window gains.

Run:
    /home/nvidia/anaconda3/envs/trading-bot/bin/python \
        research/meanrev_gated_walkforward_2026-06-15.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
logging.getLogger("TradingBot").setLevel(logging.WARNING)

import pandas as pd
from sqlalchemy import create_engine, text

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "walkforward_base", REPO_ROOT / "research" / "walkforward_2026-06-09.py")
_wf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_wf)

from backtest.engine import DEFAULT_DSN, _run_backtest_with_trades

FREQUENCY = 5
ER_CUTS = (0.25, 0.30, 0.35)
OUT = REPO_ROOT / "research" / "meanrev_gated_report_2026-06-15.json"


def spy_er(dsn: str) -> pd.Series:
    eng = create_engine(dsn)
    with eng.connect() as c:
        df = pd.read_sql(text("SELECT date, close FROM daily_bars "
                              "WHERE symbol='SPY' ORDER BY date"), c)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["close"]
    net = (s - s.shift(20)).abs()
    path = s.diff().abs().rolling(20).sum()
    return (net / path).where(path > 0, 0.0).shift(1)


def pf(rets):
    w = sum(r for r in rets if r > 0); l = -sum(r for r in rets if r < 0)
    return (w / l) if l else (float("inf") if w else 0.0)


def summ(rets):
    if not rets:
        return {"n_trades": 0}
    w = [r for r in rets if r > 0]
    return {"n_trades": len(rets), "win_rate_pct": round(100*len(w)/len(rets), 1),
            "profit_factor": round(pf(rets), 2), "sum_return_pct": round(sum(rets), 1)}


async def main():
    dsn = DEFAULT_DSN
    er = spy_er(dsn)
    windows = []
    end = _wf.DATA_END
    for _ in range(_wf.N_WINDOWS):
        start = end - timedelta(days=_wf.WINDOW_DAYS); windows.append((start, end)); end = start
    windows.reverse()

    # Replay each window once, capture mean_reversion trades + entry day + ER
    trades = []   # {window, day, er, ret}
    for i, (start, w_end) in enumerate(windows, 1):
        syms = _wf.window_symbols(dsn, start, w_end, _wf.TOP_N)
        _report, tr = await _run_backtest_with_trades(
            symbols=syms, days=_wf.WINDOW_DAYS, frequency=FREQUENCY, dsn=dsn,
            bars_loader=_wf.make_window_loader(dsn, w_end),
            run_id=f"mrg{i}", seed=42)
        for t in tr:
            if t.strategy != "mean_reversion":
                continue
            day = pd.Timestamp(t.entry_time).normalize()
            idx = er.index.searchsorted(day, side="right") - 1
            e = float(er.iloc[idx]) if idx >= 0 else float("nan")
            trades.append({"window": i, "er": e, "ret": float(t.return_pct)})
        print(f"  window {i} replayed: "
              f"{sum(1 for x in trades if x['window']==i)} mean-rev trades", flush=True)

    def per_window(sel):
        out = {}
        for w in range(1, _wf.N_WINDOWS + 1):
            out[w] = summ([x["ret"] for x in sel if x["window"] == w])
        return out

    report = {"baseline": {}, "thresholds": {}}
    base = trades
    report["baseline"] = {"pooled": summ([x["ret"] for x in base]),
                          "per_window": per_window(base)}
    print("\n=== baseline (all mean-rev) ===", flush=True)
    for w, s in report["baseline"]["per_window"].items():
        if s["n_trades"]:
            print(f"  w{w}: n={s['n_trades']:4} PF={s['profit_factor']:5.2f} "
                  f"sum={s['sum_return_pct']:+7.1f}%", flush=True)
    print(f"  POOLED PF={report['baseline']['pooled']['profit_factor']} "
          f"sum={report['baseline']['pooled']['sum_return_pct']}", flush=True)

    import math
    base_win_gain = sum(max(0, report["baseline"]["per_window"][w].get("sum_return_pct", 0))
                        for w in range(1, _wf.N_WINDOWS + 1))
    for thr in ER_CUTS:
        # mean-rev trades only on CHOPPY bars: ER < thr (NaN excluded)
        sel = [x for x in trades if not math.isnan(x["er"]) and x["er"] < thr]
        pw = per_window(sel)
        pooled = summ([x["ret"] for x in sel])
        w2 = pw[2].get("sum_return_pct", 0)
        worst = min((pw[w].get("sum_return_pct", 0) for w in range(1, _wf.N_WINDOWS+1)), default=0)
        win_gain = sum(max(0, pw[w].get("sum_return_pct", 0)) for w in range(1, _wf.N_WINDOWS+1))
        kept = (win_gain / base_win_gain) if base_win_gain else 0
        passed = (pooled.get("profit_factor", 0) >= 1.30 and w2 > -15.0
                  and worst > -25.0 and kept >= 0.60)
        report["thresholds"][str(thr)] = {"pooled": pooled, "per_window": pw,
            "w2_sum": w2, "worst": worst, "kept_win_gain_frac": round(kept, 2),
            "PASSED": passed}
        print(f"\n=== gated: mean-rev only when ER < {thr} (choppy) ===", flush=True)
        for w, s in pw.items():
            if s["n_trades"]:
                print(f"  w{w}: n={s['n_trades']:4} PF={s['profit_factor']:5.2f} "
                      f"sum={s['sum_return_pct']:+7.1f}%", flush=True)
            else:
                print(f"  w{w}: n=0 (gate blocked)", flush=True)
        print(f"  POOLED PF={pooled.get('profit_factor')} "
              f"sum={pooled.get('sum_return_pct')} n={pooled.get('n_trades')} | "
              f"w2={w2:+.0f} worst={worst:+.0f} kept_gains={kept:.0%} PASSED={passed}",
              flush=True)

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport → {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
