"""Walk-forward robustness test — 2026-06-09.

Question: is each strategy's edge consistent across time, or was the
single-window backtest carried by one favorable regime?

Design
------
Six sequential, NON-overlapping 60-day windows walking back from the
end of available minute_bars data (2026-04-17). For each window:

* Universe = top-30 symbols by dollar volume WITHIN that window
  (era-appropriate names, like the live screener would have picked —
  avoids survivorship bias from today's hot list). Leveraged ETFs
  excluded, same list the ML trainer uses.
* Bars loaded with an explicit ``end=`` anchor so the replay engine
  sees only that window (plus its own indicator warm-up inside the
  window, identical to a single-window run).
* Same strategy parameters in every window — the bot has no per-window
  optimizer, so this is pure rolling out-of-sample evaluation.

Verdict rule (per strategy): ROBUST if profit_factor > 1.0 in >= 4 of
6 windows AND no window with sum_return_pct < -15. FRAGILE otherwise.

Run:
    /home/nvidia/anaconda3/envs/trading-bot/bin/python \
        research/walkforward_2026-06-09.py
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

# The replay engine logs every strategy_decision at INFO — megabytes per
# window. Keep the research run readable.
logging.getLogger("TradingBot").setLevel(logging.WARNING)

from sqlalchemy import create_engine, text

from backtest.engine import DEFAULT_DSN, _run_backtest_with_trades
from train_ml_model import _LEVERAGED_ETFS

WINDOW_DAYS = 60
N_WINDOWS = 6
TOP_N = 30
FREQUENCY = 5
DATA_END = datetime(2026, 4, 17, 20, 0)  # last bar in minute_bars
OUT_PATH = REPO_ROOT / "research" / "walkforward_report_2026-06-09.json"


def window_symbols(dsn: str, start: datetime, end: datetime, n: int) -> list[str]:
    """Top-N by total volume inside [start, end] — the era's hot names."""
    engine = create_engine(dsn)
    sql = text(
        "SELECT symbol, SUM(volume) AS total_vol FROM minute_bars "
        "WHERE ts >= :start AND ts <= :end "
        "GROUP BY symbol ORDER BY total_vol DESC LIMIT :fetch"
    )
    with engine.connect() as conn:
        rows = conn.execute(
            sql, {"start": start, "end": end, "fetch": n * 2}
        ).fetchall()
    symbols = [r[0] for r in rows if r[0] not in _LEVERAGED_ETFS]
    return symbols[:n]


def make_window_loader(dsn: str, window_end: datetime):
    """BarsLoader anchored to window_end instead of 'latest data'."""
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)

    def _load(symbol: str, days: int, frequency: int):
        return provider.get_market_data(
            symbol,
            period_type="month",
            period=max(1, days // 30),
            frequency_type="minute",
            frequency=frequency,
            end=window_end,
        )

    return _load


async def main() -> None:
    dsn = DEFAULT_DSN
    windows = []
    end = DATA_END
    for _ in range(N_WINDOWS):
        start = end - timedelta(days=WINDOW_DAYS)
        windows.append((start, end))
        end = start
    windows.reverse()  # chronological

    results = []
    for i, (start, w_end) in enumerate(windows, 1):
        label = f"{start:%Y-%m-%d} → {w_end:%Y-%m-%d}"
        symbols = window_symbols(dsn, start, w_end, TOP_N)
        print(f"[window {i}/{N_WINDOWS}] {label}  universe={len(symbols)} "
              f"(top: {', '.join(symbols[:5])})", flush=True)
        report, _trades = await _run_backtest_with_trades(
            symbols=symbols,
            days=WINDOW_DAYS,
            frequency=FREQUENCY,
            dsn=dsn,
            bars_loader=make_window_loader(dsn, w_end),
            run_id=f"wf{i}",
            seed=42,
        )
        results.append({
            "window": i,
            "label": label,
            "universe": symbols,
            "total_trades": report.get("total_trades"),
            "per_strategy": report.get("per_strategy", {}),
        })
        for strat, split in sorted(report.get("per_strategy", {}).items()):
            m = split.get("all", split)  # schema: {"all","long","short"}
            if m.get("n_trades", 0) == 0:
                print(f"    {strat:16} n=   0", flush=True)
                continue
            print(f"    {strat:16} n={m['n_trades']:4}  "
                  f"PF={m['profit_factor']:5.2f}  "
                  f"WR={m['win_rate_pct']:5.1f}%  "
                  f"sumRet={m['sum_return_pct']:+7.1f}%", flush=True)

    # ── Verdicts ─────────────────────────────────────────────────────
    strategies = sorted({s for r in results for s in r["per_strategy"]})
    verdicts = {}
    for strat in strategies:
        rows = [r["per_strategy"].get(strat) for r in results]
        rows = [m.get("all", m) for m in rows if m]
        rows = [m for m in rows if m.get("n_trades", 0) > 0]
        pf_wins = sum(1 for m in rows if m["profit_factor"] > 1.0)
        worst = min((m["sum_return_pct"] for m in rows), default=0.0)
        robust = pf_wins >= 4 and worst > -15.0 and len(rows) >= 5
        verdicts[strat] = {
            "windows_with_pf_gt_1": pf_wins,
            "windows_traded": len(rows),
            "worst_window_sum_return_pct": worst,
            "verdict": "ROBUST" if robust else "FRAGILE",
        }

    out = {
        "generated": datetime.now().isoformat(),
        "design": {
            "window_days": WINDOW_DAYS, "n_windows": N_WINDOWS,
            "top_n": TOP_N, "frequency_min": FREQUENCY,
            "data_end": DATA_END.isoformat(),
            "universe": "per-window top-30 by volume (era-appropriate)",
        },
        "windows": results,
        "verdicts": verdicts,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nreport → {OUT_PATH}", flush=True)
    print(json.dumps(verdicts, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
