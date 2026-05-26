"""Command-line entry point for the backtest engine.

Mirrors the CLI that previously lived in ``backtest_strategies.py`` and
adds two MVP affordances:

- ``--out <dir>``: write both ``report.json`` and ``trades.csv`` into
  that directory. Mutually exclusive with ``--report-path``.
- ``--seed <int>``: seed ``random`` and ``numpy.random`` once at the
  start of the run. Used by the determinism CI test; harmless in normal
  use (the replay loop is already deterministic given identical inputs).

Invoke via:
    python -m backtest.cli --symbols NVDA --days 7 --frequency 5 --out /tmp/bt_test/
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from backtest.engine import (
    DEFAULT_DAYS,
    DEFAULT_DSN,
    DEFAULT_FREQUENCY,
    DEFAULT_TOP_N,
    _run_backtest_with_trades,
    logger,
)
from backtest.report_writers import write_report_json, write_trades_csv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-strategy P&L attribution backtest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sym = p.add_mutually_exclusive_group()
    sym.add_argument("--symbols", nargs="+")
    sym.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--frequency", type=int, default=DEFAULT_FREQUENCY)
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    # Output: either a legacy single JSON path or a directory containing both
    # ``report.json`` and ``trades.csv``. Default keeps the old behaviour.
    out = p.add_mutually_exclusive_group()
    out.add_argument("--report-path", default=None,
                     help="Legacy: write only the JSON report to this path.")
    out.add_argument("--out", default=None,
                     help="Directory to write report.json + trades.csv.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for random/numpy.random (deterministic plumbing).")
    return p.parse_args(argv)


def _print_human_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(f"BACKTEST REPORT - {report['symbol_count']} symbols x "
          f"{report['days']} days x {report['frequency_minutes']}min bars")
    print("=" * 70)

    def _print_row(label: str, s: dict) -> None:
        if s["n_trades"] == 0:
            print(f"  {label:<7} (no trades)")
            return
        pf = s.get("profit_factor")
        pf_str = f"{pf:.2f}" if pf is not None else "N/A"
        print(f"  {label:<7} n={s['n_trades']:<5} "
              f"win%={s['win_rate_pct']:<5} "
              f"avg={s['avg_return_pct']:+.3f}% "
              f"pf={pf_str:<5} "
              f"sumR={s['sum_return_pct']:+7.2f}% "
              f"mdd={s['max_drawdown_pct']:+7.2f}% "
              f"sortino={s.get('per_trade_sortino', 0):+.2f} "
              f"calmar={s.get('calmar_ratio', 0):+.2f}")

    for name, by_side in report["per_strategy"].items():
        print(f"\n{name.upper()}")
        _print_row("all", by_side["all"])
        _print_row("long", by_side["long"])
        _print_row("short", by_side["short"])
    print("=" * 70 + "\n")


async def main_async(args: argparse.Namespace) -> int:
    report, trades = await _run_backtest_with_trades(
        symbols=args.symbols,
        top_n=args.top_n if not args.symbols else None,
        days=args.days,
        frequency=args.frequency,
        dsn=args.dsn,
        seed=args.seed,
    )
    _print_human_report(report)

    if args.out is not None:
        out_dir = Path(args.out)
        report_path = write_report_json(report, out_dir / "report.json")
        trades_path = write_trades_csv(trades, out_dir / "trades.csv")
        logger.info("report_written path=%s", report_path)
        logger.info("trades_csv_written path=%s n=%d", trades_path, len(trades))
    else:
        report_path = args.report_path or "backtest_report.json"
        write_report_json(report, report_path)
        logger.info("report_written path=%s", report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
