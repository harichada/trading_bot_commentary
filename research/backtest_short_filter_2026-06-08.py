"""SHORT-branch + rising-peak-filter backtest comparison.

Three runs against the same data window:
  * filter_on   — ENABLE_MEAN_REV_SHORT=True,  ENABLE_RISING_PEAK_FILTER=True
                 (the proposed new behavior)
  * filter_off  — ENABLE_MEAN_REV_SHORT=True,  ENABLE_RISING_PEAK_FILTER=False
                 (matches the 2026-05-11 incident config — baseline)
  * short_off   — ENABLE_MEAN_REV_SHORT=False
                 (current production — sanity / regression check)

Decision criterion (from the 2026-06-08 advisory recommendation):
  filter_on must show ALL of:
    1. profit_factor >= 1.3
    2. n_trades >= 5
    3. max_simultaneous_shorts_per_day < 4
  vs filter_off as the prior-incident baseline.

How config overrides work: each Config() call instantiates a new
ConfigManager, so per-call dict mutation doesn't survive. Instead this
script monkey-patches the two relevant @property descriptors at the
class level, then restores them in a finally block. The strategies in
strategies/builtin.py re-read Config() at signal-generation time, so
they pick up the patched values.

Usage:
    python research/backtest_short_filter_2026-06-08.py \
        --days 60 --top-n 20 \
        --output-dir backtest_results/short_filter_2026-06-08/
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backtest.engine import _run_backtest_with_trades  # noqa: E402
from backtest.trades import Trade  # noqa: E402


@contextlib.contextmanager
def override_config_flags(
    enable_mean_rev_short: bool,
    enable_rising_peak_filter: bool,
):
    """Class-level monkey-patch of Config @property descriptors.

    Restored on exit even if the backtest raises. Failure to restore
    would corrupt subsequent runs (and tests) in the same Python
    process — hence the try/finally pattern.

    NOTE: cannot use `import core.config as ...` because
    `core/__init__.py` does `from core.config import config`, where
    `config` (the lowercase singleton instance) shadows the submodule
    in the `core` namespace. After that, `core.config` resolves to the
    instance, not the module. We bypass by going straight to
    `sys.modules['core.config']` (the actual module object) or by
    importing the Config class directly. The class is the same object
    in both cases — monkey-patching its @property descriptors flips
    the result for ALL existing and future Config() instances.
    """
    from core.config import Config

    orig_short = Config.__dict__.get("ENABLE_MEAN_REV_SHORT")
    orig_filter = Config.__dict__.get("ENABLE_RISING_PEAK_FILTER")

    Config.ENABLE_MEAN_REV_SHORT = property(
        lambda self, _v=enable_mean_rev_short: _v
    )
    Config.ENABLE_RISING_PEAK_FILTER = property(
        lambda self, _v=enable_rising_peak_filter: _v
    )
    try:
        yield
    finally:
        if orig_short is not None:
            Config.ENABLE_MEAN_REV_SHORT = orig_short
        if orig_filter is not None:
            Config.ENABLE_RISING_PEAK_FILTER = orig_filter


def _max_simultaneous_shorts_per_day(trades: list[Trade]) -> dict[str, int]:
    """Daily count of distinct symbols with an open SHORT at any time.

    Returns: {date_iso: count_of_distinct_symbols_held_short_that_day}.
    The relevant safety metric is the MAX across all days — that's what
    the 2026-05-11 incident hit (5 simultaneous SHORTS in one session).
    """
    shorts = [t for t in trades if t.side == "short"]
    if not shorts:
        return {}

    # For each trade, the calendar days it "owned" the symbol short.
    per_day_symbols: dict[date, set[str]] = defaultdict(set)
    for t in shorts:
        try:
            start_date = (t.entry_time.date() if hasattr(t.entry_time, "date")
                          else date.fromisoformat(str(t.entry_time)[:10]))
            end_date = (t.exit_time.date() if hasattr(t.exit_time, "date")
                        else date.fromisoformat(str(t.exit_time)[:10]))
        except (AttributeError, ValueError, TypeError):
            continue
        # Walk forward; cheap for intraday trades (1-2 days).
        d = start_date
        while d <= end_date:
            per_day_symbols[d].add(t.symbol)
            # Guard against absurd hold-period bug
            if (d - start_date).days > 30:
                break
            d = date.fromordinal(d.toordinal() + 1)

    return {d.isoformat(): len(syms) for d, syms in per_day_symbols.items()}


def _short_only_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Extract just the mean-rev SHORT slice from a report."""
    mr = report.get("per_strategy", {}).get("mean_reversion", {})
    short = mr.get("short", {})
    return {
        "n_trades": short.get("n_trades", 0),
        "win_rate_pct": short.get("win_rate_pct", 0),
        "avg_return_pct": short.get("avg_return_pct", 0),
        "profit_factor": short.get("profit_factor"),
        "sum_return_pct": short.get("sum_return_pct", 0),
        "max_drawdown_pct": short.get("max_drawdown_pct", 0),
        "per_trade_sortino": short.get("per_trade_sortino", 0),
        "calmar_ratio": short.get("calmar_ratio", 0),
    }


def _print_comparison(results: dict[str, dict[str, Any]]) -> None:
    """Side-by-side human-readable table."""
    bar = "=" * 86
    print("\n" + bar)
    print("MEAN-REV SHORT — rising-peak filter comparison")
    print(bar)
    cols = ["filter_on", "filter_off", "short_off"]
    labels = {
        "filter_on": "filter ON (new)",
        "filter_off": "filter OFF (May11)",
        "short_off": "SHORT off (prod)",
    }
    metrics = [
        ("n_trades", "n trades", "{:.0f}"),
        ("win_rate_pct", "win %", "{:.1f}"),
        ("avg_return_pct", "avg ret %", "{:+.3f}"),
        ("profit_factor", "profit factor", "{:.2f}"),
        ("sum_return_pct", "sum ret %", "{:+.2f}"),
        ("max_drawdown_pct", "max DD %", "{:+.2f}"),
        ("per_trade_sortino", "sortino", "{:+.2f}"),
        ("calmar_ratio", "calmar", "{:+.2f}"),
    ]
    print(f"{'Metric':<18}" + "".join(f"{labels[c]:>22}" for c in cols))
    print("-" * 86)
    for key, lbl, fmt in metrics:
        row = f"{lbl:<18}"
        for c in cols:
            v = results[c]["short_summary"].get(key)
            if v is None:
                row += f"{'N/A':>22}"
            else:
                try:
                    row += f"{fmt.format(v):>22}"
                except (TypeError, ValueError):
                    row += f"{'?':>22}"
        print(row)

    # Safety metric: max simultaneous SHORTs per day
    print("-" * 86)
    print(f"{'max shorts/day':<18}", end="")
    for c in cols:
        daily = results[c]["daily_short_symbols"]
        peak = max(daily.values()) if daily else 0
        print(f"{peak:>22}", end="")
    print()
    print(bar)

    # Decision card for filter_on
    fo = results["filter_on"]["short_summary"]
    daily = results["filter_on"]["daily_short_symbols"]
    peak = max(daily.values()) if daily else 0
    pf = fo.get("profit_factor") or 0
    n = fo.get("n_trades") or 0
    print("\nDECISION GATE for re-enabling mean-rev SHORT (filter_on):")
    pf_pass = (pf is not None and pf >= 1.3)
    n_pass = n >= 5
    peak_pass = peak < 4
    print(f"  [{'PASS' if pf_pass else 'FAIL'}] profit_factor >= 1.3        "
          f"(actual: {pf:.2f})")
    print(f"  [{'PASS' if n_pass else 'FAIL'}] n_trades >= 5                "
          f"(actual: {n})")
    print(f"  [{'PASS' if peak_pass else 'FAIL'}] max_simultaneous_shorts < 4 "
          f"(actual: {peak})")
    if pf_pass and n_pass and peak_pass:
        print("\n  → ALL GATES PASS. Safe to proceed to sim-mode soak before live.")
    else:
        print("\n  → GATE FAILED. Do NOT enable live SHORT. Iterate on filter "
              "or wait for a more favorable backtest window.")
    print(bar + "\n")


async def _one_run(
    label: str,
    *,
    enable_mean_rev_short: bool,
    enable_rising_peak_filter: bool,
    days: int,
    top_n: int | None,
    symbols: list[str] | None,
    dsn: str | None,
) -> dict[str, Any]:
    print(f"\n[{label}] running backtest "
          f"(SHORT={enable_mean_rev_short}, FILTER={enable_rising_peak_filter}, "
          f"days={days}, top_n={top_n}, symbols={symbols})…")
    with override_config_flags(
        enable_mean_rev_short=enable_mean_rev_short,
        enable_rising_peak_filter=enable_rising_peak_filter,
    ):
        report, trades = await _run_backtest_with_trades(
            symbols=symbols,
            top_n=top_n if not symbols else None,
            days=days,
            dsn=dsn,
        )
    short_summary = _short_only_summary(report)
    daily = _max_simultaneous_shorts_per_day(trades)
    print(f"[{label}] done. SHORT trades: {short_summary['n_trades']}, "
          f"PF: {short_summary['profit_factor']}, "
          f"max shorts/day: {max(daily.values()) if daily else 0}")
    return {
        "label": label,
        "config": {
            "enable_mean_rev_short": enable_mean_rev_short,
            "enable_rising_peak_filter": enable_rising_peak_filter,
        },
        "report": report,
        "short_summary": short_summary,
        "daily_short_symbols": daily,
    }


async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("filter_on", True, True),
        ("filter_off", True, False),
        ("short_off", False, True),  # filter value irrelevant when SHORT off
    ]
    results: dict[str, dict[str, Any]] = {}
    for label, short_on, filter_on in runs:
        try:
            results[label] = await _one_run(
                label,
                enable_mean_rev_short=short_on,
                enable_rising_peak_filter=filter_on,
                days=args.days,
                top_n=(args.top_n if not args.symbols else None),
                symbols=args.symbols,
                dsn=args.dsn,
            )
        except Exception as exc:
            print(f"[{label}] FAILED: {exc}", file=sys.stderr)
            return 1

    _print_comparison(results)

    # Persist each run's full report + the comparison summary
    for label, payload in results.items():
        report_path = out_dir / f"{label}_report.json"
        with open(report_path, "w") as f:
            json.dump(payload["report"], f, default=str, indent=2)
        print(f"saved → {report_path}")

    summary_path = out_dir / "comparison_summary.json"
    summary = {
        label: {
            "config": p["config"],
            "short_summary": p["short_summary"],
            "max_simultaneous_shorts_per_day":
                max(p["daily_short_symbols"].values()) if p["daily_short_symbols"] else 0,
            "daily_short_symbol_counts": p["daily_short_symbols"],
        }
        for label, p in results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, default=str, indent=2)
    print(f"saved → {summary_path}")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sym = p.add_mutually_exclusive_group()
    sym.add_argument("--symbols", nargs="+",
                     help="Explicit symbol list. Mutually exclusive with --top-n.")
    sym.add_argument("--top-n", type=int, default=20,
                     help="Universe size (top-N by volume). Default 20.")
    p.add_argument("--days", type=int, default=60,
                   help="Lookback window in days. Default 60.")
    p.add_argument("--dsn", default=None,
                   help="Postgres DSN. Falls back to POSTGRES_DSN env or built-in default.")
    p.add_argument("--output-dir", default="backtest_results/short_filter_2026-06-08/",
                   help="Where to write per-run reports + comparison summary.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
