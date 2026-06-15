"""Pullback-continuation strategy — walk-forward trial.

v-pullback-continuation-wf-2026-06-15. The coverage hole (2026-06-12):
the bot has a panic-buyer (mean-rev), an explosion-buyer (breakout),
and a headline-buyer (news), but NO continuation-buyer for the most
common profitable tape — a broad steady advance. On 2026-06-12, 23/40
panel symbols read direction >= +3 and the bot took nothing: breakout
gates reject already-extended runners, mean-rev can't enter strength,
momentum is dead (PF 0.84, twice-buried 2026-06-12).

This strategy fills the hole the RIGHT way — it does NOT chase
strength. It waits for a PULLBACK within an established uptrend:

  ENTRY (long only):
    * trend:    direction reader >= +3 AND phase in {early, middle}
                (the live Symbol-Intel definition of "strong, not
                 exhausted")
    * pullback: this bar or the prior bar dipped to/through EMA20
                (price came back to the mean)
    * resume:   close back above EMA20 AND rising vs prior close
    * room:     RSI < 68 (buying the dip, not the peak)
  STOP:    just below the pullback low, floored at 0.3xATR and
           capped at 3xATR of risk
  TARGET:  entry + 2R
  EXITS:   the live ladder — breakeven at +0.5R, 1.5xATR trail once
           +2xATR, target kept, MAX_HOLD timeout

vs momentum's "buy strength NOW", the pullback gives a real stop (the
swing low at the EMA) and avoids buying noise-tops — which is exactly
why momentum failed.

Six standard 60-day windows, per-window top-30 universe. No lookahead
(every signal uses data through bar i only; ladder updates after exit
checks).

DECISION GATE (pre-committed): enters shadow next ONLY if
  * pooled PF >= 1.30, AND
  * positive sum in EACH trending window (1, 2, 3), AND
  * no window with sum_return < -25%, AND
  * >= 100 total trades (not a rare-signal artifact)
Otherwise the coverage hole gets a different attack.

Run:
    /home/nvidia/anaconda3/envs/trading-bot/bin/python \
        research/pullback_continuation_walkforward_2026-06-15.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
logging.getLogger("TradingBot").setLevel(logging.WARNING)

import numpy as np
import pandas as pd

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "walkforward_base", REPO_ROOT / "research" / "walkforward_2026-06-09.py")
_wf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_wf)

from backtest.engine import DEFAULT_DSN, MAX_HOLD_BARS, compute_indicators
from core.direction_reader import read_direction

FREQUENCY = 5
BE_ACTIVATION_R = 0.5
TRAIL_ACT_ATR = 2.0
TRAIL_WIDTH_ATR = 1.5
RR = 2.0
OUT_PATH = REPO_ROOT / "research" / "pullback_continuation_report_2026-06-15.json"

# Decision-gate thresholds (pre-committed)
GATE_PF = 1.30
GATE_MIN_TRADES = 100
GATE_CATASTROPHE = -25.0
TREND_WINDOWS = (1, 2, 3)


def simulate_symbol(df: pd.DataFrame) -> list[dict]:
    ind = compute_indicators(df)
    trades = []
    pos = None
    req = ("rsi", "atr", "adx", "sma_50", "ema_20", "macd_histogram",
           "ema_20_slope_pct", "close_vs_sma50_pct", "obv_slope_pct")

    for i in range(50, len(df)):
        bar = df.iloc[i]
        row = ind.iloc[i].to_dict()
        if any(pd.isna(row.get(k)) for k in req):
            continue
        close = float(bar["Close"])
        low = float(bar["Low"])
        atr = float(row["atr"])

        # ---- manage open position (ladder) ----
        if pos is not None:
            pos["hold"] += 1
            stop_hit = low <= pos["stop"]
            target_hit = float(bar["High"]) >= pos["target"]
            exit_reason = exit_price = None
            if stop_hit:
                exit_price = pos["stop"]
                exit_reason = "trail" if pos["stop"] > pos["orig_stop"] else "stop"
            elif target_hit:
                exit_price = pos["target"]
                exit_reason = "target"
            elif pos["hold"] >= MAX_HOLD_BARS:
                exit_price = close
                exit_reason = "timeout"
            if exit_reason is not None:
                ret = (exit_price - pos["entry"]) / pos["entry"] * 100
                trades.append({"return_pct": ret, "exit_reason": exit_reason,
                               "hold": pos["hold"]})
                pos = None
            else:
                peak = max(pos["peak"], float(bar["High"]))
                pos["peak"] = peak
                sd = pos["entry"] - pos["orig_stop"]
                if sd > 0 and peak >= pos["entry"] + BE_ACTIVATION_R * sd:
                    pos["stop"] = max(pos["stop"], pos["entry"])
                if atr > 0 and peak >= pos["entry"] + TRAIL_ACT_ATR * atr:
                    pos["stop"] = max(pos["stop"], close - TRAIL_WIDTH_ATR * atr)
                continue  # one position per symbol; no same-bar re-entry

        # ---- entry scan ----
        if pos is not None:
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
        resuming = close > ema and close > float(prev["Close"])
        room = float(row["rsi"]) < 68.0
        if not (touched and resuming and room):
            continue

        swing_low = min(low, float(prev["Low"]))
        stop = swing_low - 0.1 * atr
        stop_dist = close - stop
        if stop_dist < 0.3 * atr:           # too tight — noise stop
            stop = close - 0.3 * atr
            stop_dist = 0.3 * atr
        if stop_dist > 3.0 * atr:           # too wide — skip, risk unclear
            continue
        pos = {"entry": close, "stop": stop, "orig_stop": stop,
               "target": close + RR * stop_dist, "peak": close,
               "hold": 0}
    return trades


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0}
    rets = [t["return_pct"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [-r for r in rets if r < 0]
    pf = (sum(wins) / sum(losses)) if losses else float("inf")
    from collections import Counter
    exits = Counter(t["exit_reason"] for t in trades)
    return {
        "n_trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / len(trades), 1),
        "profit_factor": round(pf, 2),
        "sum_return_pct": round(sum(rets), 1),
        "avg_return_pct": round(sum(rets) / len(trades), 3),
        "avg_hold_bars": round(sum(t["hold"] for t in trades) / len(trades), 1),
        "exit_breakdown": dict(exits),
    }


async def main() -> None:
    dsn = DEFAULT_DSN
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)

    windows = []
    end = _wf.DATA_END
    for _ in range(_wf.N_WINDOWS):
        start = end - timedelta(days=_wf.WINDOW_DAYS)
        windows.append((start, end))
        end = start
    windows.reverse()

    results = []
    all_trades = []
    for i, (start, w_end) in enumerate(windows, 1):
        symbols = _wf.window_symbols(dsn, start, w_end, _wf.TOP_N)
        win_trades = []
        for sym in symbols:
            try:
                df = provider.get_market_data(
                    sym, period_type="month",
                    period=max(1, _wf.WINDOW_DAYS // 30),
                    frequency_type="minute", frequency=FREQUENCY, end=w_end)
            except Exception:
                continue
            if df is None or df.empty or len(df) < 100:
                continue
            win_trades.extend(simulate_symbol(df))
        s = summarize(win_trades)
        results.append({"window": i,
                        "label": f"{start:%Y-%m-%d} → {w_end:%Y-%m-%d}", **s})
        all_trades.extend(win_trades)
        if s["n_trades"]:
            print(f"  w{i} {start:%Y-%m-%d}→{w_end:%Y-%m-%d}: "
                  f"n={s['n_trades']:4} PF={s['profit_factor']:5.2f} "
                  f"WR={s['win_rate_pct']:5.1f}% sum={s['sum_return_pct']:+7.1f}% "
                  f"exits={s['exit_breakdown']}", flush=True)
        else:
            print(f"  w{i}: n=0", flush=True)

    pooled = summarize(all_trades)
    # gate evaluation
    trend_ok = all(
        next(r for r in results if r["window"] == w).get("sum_return_pct", 0) > 0
        for w in TREND_WINDOWS)
    worst = min((r.get("sum_return_pct", 0) for r in results), default=0)
    passed = (pooled.get("profit_factor", 0) >= GATE_PF
              and trend_ok
              and worst > GATE_CATASTROPHE
              and pooled.get("n_trades", 0) >= GATE_MIN_TRADES)
    verdict = {
        "pooled": pooled,
        "trend_windows_all_positive": trend_ok,
        "worst_window_sum_return_pct": worst,
        "gate": {"pf": GATE_PF, "min_trades": GATE_MIN_TRADES,
                 "catastrophe": GATE_CATASTROPHE},
        "PASSED": passed,
    }
    out = {"generated": datetime.now().isoformat(),
           "ladder_params": {"be_r": BE_ACTIVATION_R, "trail_act_atr": TRAIL_ACT_ATR,
                             "trail_width_atr": TRAIL_WIDTH_ATR, "rr": RR},
           "windows": results, "verdict": verdict}
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print("\n--- VERDICT ---", flush=True)
    print(json.dumps(verdict, indent=2, default=str), flush=True)
    print(f"report → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
