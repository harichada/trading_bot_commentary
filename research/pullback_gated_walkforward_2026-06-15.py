"""Pullback-continuation GATED by the regime allocator — the decisive test.

v-pullback-gated-wf-2026-06-15. Ungated pullback failed (PF 0.84):
great in trend windows (w3 +267%), ruinous in chop (w4 -843%). Four
strategies now die the same way. Thesis: the edge is regime-gating,
not the entry. This test gates the SAME pullback entry to bars where
SPY's 20-day Kaufman Efficiency Ratio (shifted 1 day, no lookahead)
reads trending — exactly what the live regime allocator would do.

If gated-pullback flips to PF >= 1.3 purely by NOT trading the choppy
windows, that proves regime-gating > entry-hunting and argues for
promoting the allocator from shadow to a live gate.

Tested at ER cuts {0.25, 0.30, 0.35} for robustness. Same ladder
exits and entry rules as the ungated run.

Run:
    /home/nvidia/anaconda3/envs/trading-bot/bin/python \
        research/pullback_gated_walkforward_2026-06-15.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import Counter
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

from backtest.engine import DEFAULT_DSN, MAX_HOLD_BARS, compute_indicators
from core.direction_reader import read_direction

FREQUENCY = 5
BE_R, TRAIL_ACT, TRAIL_W, RR = 0.5, 2.0, 1.5, 2.0
ER_CUTS = (0.25, 0.30, 0.35)
OUT = REPO_ROOT / "research" / "pullback_gated_report_2026-06-15.json"


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


def simulate(df: pd.DataFrame, er: pd.Series, thr: float) -> list[dict]:
    ind = compute_indicators(df)
    trades, pos = [], None
    req = ("rsi", "atr", "adx", "sma_50", "ema_20", "macd_histogram",
           "ema_20_slope_pct", "close_vs_sma50_pct", "obv_slope_pct")
    for i in range(50, len(df)):
        bar = df.iloc[i]
        row = ind.iloc[i].to_dict()
        if any(pd.isna(row.get(k)) for k in req):
            continue
        close, low, atr = float(bar["Close"]), float(bar["Low"]), float(row["atr"])
        if pos is not None:
            pos["hold"] += 1
            sh = low <= pos["stop"]; th = float(bar["High"]) >= pos["target"]
            er_, ep = None, None
            if sh:
                ep = pos["stop"]; er_ = "trail" if pos["stop"] > pos["orig"] else "stop"
            elif th:
                ep = pos["target"]; er_ = "target"
            elif pos["hold"] >= MAX_HOLD_BARS:
                ep = close; er_ = "timeout"
            if er_:
                trades.append({"r": (ep - pos["entry"]) / pos["entry"] * 100,
                               "x": er_})
                pos = None
            else:
                pk = max(pos["peak"], float(bar["High"])); pos["peak"] = pk
                sd = pos["entry"] - pos["orig"]
                if sd > 0 and pk >= pos["entry"] + BE_R * sd:
                    pos["stop"] = max(pos["stop"], pos["entry"])
                if atr > 0 and pk >= pos["entry"] + TRAIL_ACT * atr:
                    pos["stop"] = max(pos["stop"], close - TRAIL_W * atr)
                continue
        if pos is not None:
            continue
        # ---- regime gate ----
        day = df.index[i].normalize()
        idx = er.index.searchsorted(day, side="right") - 1
        cur_er = float(er.iloc[idx]) if idx >= 0 else float("nan")
        if not (cur_er >= thr):           # NaN or choppy → no entry
            continue
        try:
            dr = read_direction(close, row)
        except Exception:
            continue
        if not (dr.direction >= 3.0 and dr.phase in ("early", "middle")):
            continue
        ema = float(row["ema_20"])
        if ema <= 0 or atr <= 0:
            continue
        prev = df.iloc[i - 1]
        touched = (low <= ema * 1.002) or (float(prev["Low"]) <= ema * 1.002)
        if not (touched and close > ema and close > float(prev["Close"])
                and float(row["rsi"]) < 68.0):
            continue
        sl = min(low, float(prev["Low"])) - 0.1 * atr
        sd = close - sl
        if sd < 0.3 * atr:
            sl = close - 0.3 * atr; sd = 0.3 * atr
        if sd > 3.0 * atr:
            continue
        pos = {"entry": close, "stop": sl, "orig": sl,
               "target": close + RR * sd, "peak": close, "hold": 0}
    return trades


def summ(trades):
    if not trades:
        return {"n_trades": 0}
    rets = [t["r"] for t in trades]
    w = [r for r in rets if r > 0]; l = [-r for r in rets if r < 0]
    pf = (sum(w) / sum(l)) if l else float("inf")
    return {"n_trades": len(trades), "win_rate_pct": round(100*len(w)/len(rets), 1),
            "profit_factor": round(pf, 2), "sum_return_pct": round(sum(rets), 1),
            "exits": dict(Counter(t["x"] for t in trades))}


async def main():
    dsn = DEFAULT_DSN
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)
    er = spy_er(dsn)
    windows = []
    end = _wf.DATA_END
    for _ in range(_wf.N_WINDOWS):
        start = end - timedelta(days=_wf.WINDOW_DAYS); windows.append((start, end)); end = start
    windows.reverse()

    # preload bars once per window
    report = {"thresholds": {}}
    cache = {}
    for i, (start, w_end) in enumerate(windows, 1):
        syms = _wf.window_symbols(dsn, start, w_end, _wf.TOP_N)
        dfs = []
        for s in syms:
            try:
                df = provider.get_market_data(s, period_type="month",
                    period=max(1, _wf.WINDOW_DAYS // 30),
                    frequency_type="minute", frequency=FREQUENCY, end=w_end)
            except Exception:
                continue
            if df is not None and not df.empty and len(df) >= 100:
                dfs.append(df)
        cache[i] = (start, w_end, dfs)

    for thr in ER_CUTS:
        print(f"\n=== ER gate >= {thr} ===", flush=True)
        all_t = []; wins = []
        for i in sorted(cache):
            start, w_end, dfs = cache[i]
            wt = []
            for df in dfs:
                wt.extend(simulate(df, er, thr))
            s = summ(wt); s["window"] = i; wins.append(s); all_t.extend(wt)
            if s["n_trades"]:
                print(f"  w{i}: n={s['n_trades']:4} PF={s['profit_factor']:5.2f} "
                      f"WR={s['win_rate_pct']:5.1f}% sum={s['sum_return_pct']:+7.1f}%", flush=True)
            else:
                print(f"  w{i}: n=0 (gate closed all bars)", flush=True)
        pooled = summ(all_t)
        trend_ok = all(next(x for x in wins if x["window"]==w).get("sum_return_pct",0) > 0
                       for w in (1,2,3))
        worst = min((x.get("sum_return_pct",0) for x in wins), default=0)
        passed = (pooled.get("profit_factor",0) >= 1.30 and trend_ok
                  and worst > -25.0 and pooled.get("n_trades",0) >= 100)
        report["thresholds"][str(thr)] = {"windows": wins, "pooled": pooled,
            "trend_windows_positive": trend_ok, "worst": worst, "PASSED": passed}
        print(f"  POOLED PF={pooled.get('profit_factor')} "
              f"sum={pooled.get('sum_return_pct')} n={pooled.get('n_trades')} "
              f"trend_ok={trend_ok} worst={worst} PASSED={passed}", flush=True)

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport → {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
