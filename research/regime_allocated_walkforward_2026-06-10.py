"""Regime-allocated walk-forward — P1 validation, 2026-06-10.

Hypothesis (from the 2026-06-09 walk-forward): breakout and mean-rev
are near-anti-correlated across regimes, so allocating BETWEEN them by
a tape-character signal should beat both standalone. If this test
fails, the regime-allocator module does not get built.

Design
------
* Same six 60-day windows + per-window top-30 universes as
  walkforward_2026-06-09.py. One replay pass, trades captured with
  entry timestamps.
* Regime signal: SPY 20-day Kaufman Efficiency Ratio computed from
  daily_bars THROUGH THE PRIOR SESSION (shift 1 day — no lookahead).
      ER = |close[t-1] - close[t-21]| / Σ|daily deltas of those 20d|
  High ER = directional tape → breakout allowed, mean-rev blocked.
  Low ER  = choppy tape     → mean-rev allowed, breakout blocked.
* Threshold robustness: evaluated at ER cuts {0.20, 0.25, 0.30, 0.35,
  0.40}. Verdict requires allocation to help across MOST cuts, not at
  one cherry-picked value.
* Benchmarks per policy: n trades, PF, summed per-trade return.
    - baseline: every trade both strategies took
    - allocated(θ): trades kept only when entry-day regime permits
      that strategy
    - oracle: per-window best single strategy (upper bound, uses
      future knowledge — context only, not attainable).

Run:
    /home/nvidia/anaconda3/envs/trading-bot/bin/python \
        research/regime_allocated_walkforward_2026-06-10.py
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
from sqlalchemy import create_engine, text

from backtest.engine import DEFAULT_DSN, _run_backtest_with_trades

# The window/universe helpers live in a hyphenated filename — load by path.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "walkforward_base", REPO_ROOT / "research" / "walkforward_2026-06-09.py"
)
_wf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_wf)
DATA_END, N_WINDOWS, TOP_N, WINDOW_DAYS = (
    _wf.DATA_END, _wf.N_WINDOWS, _wf.TOP_N, _wf.WINDOW_DAYS,
)
make_window_loader, window_symbols = _wf.make_window_loader, _wf.window_symbols

FREQUENCY = 5
ER_CUTS = (0.20, 0.25, 0.30, 0.35, 0.40)
TRADES_PATH = REPO_ROOT / "research" / "walkforward_trades_2026-06-10.ndjson"
OUT_PATH = REPO_ROOT / "research" / "regime_allocated_report_2026-06-10.json"


def spy_efficiency_ratio(dsn: str) -> pd.Series:
    """Daily ER series indexed by date, shifted so date D uses data
    through D-1 only."""
    engine = create_engine(dsn)
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT date, close FROM daily_bars WHERE symbol='SPY' "
                 "ORDER BY date"),
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    close = df["close"]
    net = (close - close.shift(20)).abs()
    path = close.diff().abs().rolling(20).sum()
    er = (net / path).where(path > 0, 0.0)
    return er.shift(1)  # date D sees ER computed through D-1


def pf(returns: list[float]) -> float:
    gains = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return round(gains / losses, 3)


def policy_metrics(trades: list[dict]) -> dict:
    rets = [t["return_pct"] for t in trades]
    return {
        "n_trades": len(rets),
        "profit_factor": pf(rets),
        "sum_return_pct": round(sum(rets), 1),
        "win_rate_pct": round(
            100 * sum(1 for r in rets if r > 0) / len(rets), 1
        ) if rets else 0.0,
    }


async def main() -> None:
    dsn = DEFAULT_DSN
    er = spy_efficiency_ratio(dsn)

    windows = []
    end = DATA_END
    for _ in range(N_WINDOWS):
        start = end - timedelta(days=WINDOW_DAYS)
        windows.append((start, end))
        end = start
    windows.reverse()

    # ── Replay once, capture trades ──────────────────────────────────
    all_trades: list[dict] = []
    for i, (start, w_end) in enumerate(windows, 1):
        symbols = window_symbols(dsn, start, w_end, TOP_N)
        print(f"[window {i}/{N_WINDOWS}] {start:%Y-%m-%d} → {w_end:%Y-%m-%d}",
              flush=True)
        _report, trades = await _run_backtest_with_trades(
            symbols=symbols, days=WINDOW_DAYS, frequency=FREQUENCY,
            dsn=dsn, bars_loader=make_window_loader(dsn, w_end),
            run_id=f"rwf{i}", seed=42,
        )
        for t in trades:
            all_trades.append({
                "window": i,
                "strategy": t.strategy,
                "symbol": t.symbol,
                "side": t.side,
                "entry_time": str(t.entry_time),
                "exit_reason": t.exit_reason,
                "return_pct": float(t.return_pct),
            })

    with open(TRADES_PATH, "w") as f:
        for t in all_trades:
            f.write(json.dumps(t) + "\n")
    print(f"trades → {TRADES_PATH} ({len(all_trades)})", flush=True)

    # ── Attach entry-day regime ──────────────────────────────────────
    for t in all_trades:
        day = pd.Timestamp(t["entry_time"]).normalize()
        # ER index is daily; use most recent ER at-or-before entry day
        idx = er.index.searchsorted(day, side="right") - 1
        t["er"] = float(er.iloc[idx]) if idx >= 0 else float("nan")

    core = [t for t in all_trades
            if t["strategy"] in ("breakout", "mean_reversion")
            and not np.isnan(t["er"])]

    # ── Policies ─────────────────────────────────────────────────────
    def allocated(theta: float) -> list[dict]:
        keep = []
        for t in core:
            trending = t["er"] >= theta
            if t["strategy"] == "breakout" and trending:
                keep.append(t)
            elif t["strategy"] == "mean_reversion" and not trending:
                keep.append(t)
        return keep

    by_window_strat: dict[tuple[int, str], list[dict]] = {}
    for t in core:
        by_window_strat.setdefault((t["window"], t["strategy"]), []).append(t)
    oracle = []
    for w in range(1, N_WINDOWS + 1):
        best = max(
            ("breakout", "mean_reversion"),
            key=lambda s: sum(x["return_pct"]
                              for x in by_window_strat.get((w, s), [])),
        )
        oracle.extend(by_window_strat.get((w, best), []))

    report = {
        "generated": datetime.now().isoformat(),
        "n_core_trades": len(core),
        "policies": {
            "baseline_all_trades": policy_metrics(core),
            "oracle_per_window_best": policy_metrics(oracle),
            **{f"allocated_er>={theta}": policy_metrics(allocated(theta))
               for theta in ER_CUTS},
        },
        "per_window_allocated_0.30": {
            w: policy_metrics([t for t in allocated(0.30)
                               if t["window"] == w])
            for w in range(1, N_WINDOWS + 1)
        },
    }
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["policies"], indent=2), flush=True)
    print(f"report → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
