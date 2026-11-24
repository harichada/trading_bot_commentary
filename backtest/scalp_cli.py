"""CLI entry point for the scalp reversal backtest.

Usage:
    python -m backtest.scalp_cli [flags]

Example:
    python -m backtest.scalp_cli --symbols NVDA TSLA AMD --days 90
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from backtest.scalp_reversal import (
    CAPITAL_USD,
    DEFAULT_DSN,
    DEFAULT_SYMBOLS,
    MAX_SLOTS,
    ScalpParams,
    run,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scalp reversal backtest — 1-min intraday reversal scalp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                   metavar="SYM", help="Symbols to test (default: %(default)s)")
    p.add_argument("--days", type=int, default=90,
                   help="Lookback window in calendar days (default: %(default)s)")
    p.add_argument("--rsi-long", type=float, default=30.0,
                   dest="rsi_long", help="RSI threshold for long signal (default: %(default)s)")
    p.add_argument("--rsi-short", type=float, default=75.0,
                   dest="rsi_short", help="RSI threshold for short signal (default: %(default)s)")
    p.add_argument("--vol-min", type=float, default=2.0,
                   dest="vol_min", help="Minimum volume ratio to trigger (default: %(default)s)")
    p.add_argument("--rsi-turn", action="store_true", default=False,
                   dest="rsi_turn",
                   help="Only enter when RSI is at extreme AND ticking back (inflection confirmation)")
    p.add_argument("--entry-confirm", action="store_true", default=False,
                   dest="entry_confirm",
                   help="Wait one bar after the signal for price to confirm the turn before entering")
    p.add_argument("--reversal-candle", action="store_true", default=False,
                   dest="reversal_candle",
                   help="Require signal bar to show a reversal candle (hammer/shooting-star or bullish/bearish close)")
    p.add_argument("--candle-wick-ratio", type=float, default=1.5,
                   dest="candle_wick_ratio",
                   help="Wick-to-body ratio to qualify as hammer/shooting-star (default: %(default)s)")
    p.add_argument("--adx-max", type=float, default=999.0,
                   dest="adx_max",
                   help="Skip signals when ADX > this (ranging filter, 999=disabled, try 25-30)")
    p.add_argument("--atr-stop-mult", type=float, default=1.0,
                   dest="atr_stop_mult",
                   help="ATR multiplier for stop distance (default: %(default)s)")
    p.add_argument("--atr-target-mult", type=float, default=2.0,
                   dest="atr_target_mult",
                   help="ATR multiplier for target distance (default: %(default)s)")
    p.add_argument("--quick-exit-bars", type=int, default=3,
                   dest="quick_exit_bars",
                   help="Bars window for quick-profit exit (default: %(default)s)")
    p.add_argument("--min-profit-pct", type=float, default=0.15,
                   dest="min_profit_pct",
                   help="Minimum %% gain to trigger quick exit (default: %(default)s)")
    p.add_argument("--max-hold-bars", type=int, default=15,
                   dest="max_hold_bars",
                   help="Timeout in bars (default: %(default)s)")
    p.add_argument("--max-slots", type=int, default=MAX_SLOTS,
                   dest="max_slots",
                   help="Max concurrent open positions (default: %(default)s)")
    p.add_argument("--capital", type=int, default=CAPITAL_USD,
                   help="Total margin capital in USD (default: %(default)s)")
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN),
                   help="Postgres DSN (default: env POSTGRES_DSN or project default)")
    p.add_argument("--out", default="research/reports/scalp",
                   help="Output directory for JSON report (default: %(default)s)")
    p.add_argument("--sides", default="both",
                   choices=["both", "long_only", "short_only"],
                   help="Which signal sides to trade (default: %(default)s)")
    p.add_argument("--end-date", default=None, dest="end_date",
                   metavar="YYYY-MM-DD",
                   help="End date for backtest window (default: today). "
                        "Use with --days for out-of-sample testing.")
    p.add_argument("--from-open-drop-max", type=float, default=0.0,
                   dest="from_open_drop_max",
                   help="Skip long signals when stock is down > this %% from RTH open "
                        "(0 = disabled, e.g. 1.5 blocks longs after a 1.5%% drop from open)")
    p.add_argument("--direction-gate", action="store_true", default=False,
                   dest="direction_gate",
                   help="Per-stock daily direction gate: gainer (open > prev close) → "
                        "short_only; loser → long_only. Mirrors live screener logic.")
    p.add_argument("--bar-minutes", type=int, default=1,
                   dest="bar_minutes",
                   help="Bar granularity in minutes (default: 1). "
                        "Use 5 for 5-min bars, 15 for 15-min, etc.")
    return p.parse_args(argv)


def _fmt(val: object, width: int = 10) -> str:
    if val is None:
        return "-".rjust(width)
    if isinstance(val, float):
        return f"{val:>{width}.2f}"
    return str(val).rjust(width)


def _print_summary(report: dict) -> None:
    overall = report.get("overall", {})
    by_side = report.get("by_side", {})
    exit_bd = report.get("overall", {}).get("exit_breakdown", {})

    print("\n" + "=" * 60)
    print("  SCALP REVERSAL BACKTEST — SUMMARY")
    print("=" * 60)
    print(f"  Symbols : {', '.join(report.get('symbols', []))}")
    print(f"  Days    : {report.get('days')}  |  Capital: ${report.get('capital_usd'):,}")
    print(f"  Slots   : {report.get('max_slots')} × ${report.get('slot_size_usd'):,.0f}")
    print(f"  Run at  : {report.get('run_date')}")
    print("-" * 60)

    # Header
    hdr = f"{'Side':<8} {'Trades':>7} {'Win%':>7} {'Avg R%':>8} {'Sum R%':>8} {'$PnL':>10}"
    print(hdr)
    print("-" * 60)

    def _row(label: str, m: dict, dollar: float | None = None) -> None:
        n = m.get("n_trades", 0)
        win = m.get("win_rate_pct", 0.0)
        avg = m.get("avg_return_pct", 0.0)
        s = m.get("sum_return_pct", 0.0)
        d = f"${dollar:>+,.0f}" if dollar is not None else "-"
        print(f"{label:<8} {n:>7} {win:>7.1f} {avg:>8.3f} {s:>8.2f} {d:>10}")

    _row("OVERALL", overall, report.get("total_dollar_pnl"))
    _row("  long", by_side.get("long", {}))
    _row("  short", by_side.get("short", {}))

    print("-" * 60)
    print("  Exit breakdown (% of executed trades):")
    for reason, pct in sorted(exit_bd.items()):
        print(f"    {reason:<15} {pct:>5.1f}%")

    print("-" * 60)
    print("  Dropped (over slot cap):", report.get("dropped_trades", 0))

    per_sym = report.get("per_symbol", {})
    if per_sym:
        print("-" * 60)
        print("  Per symbol:")
        sym_hdr = f"  {'Symbol':<8} {'Trades':>7} {'Win%':>7} {'$PnL':>10}"
        print(sym_hdr)
        for sym, m in sorted(per_sym.items(),
                              key=lambda kv: kv[1].get("dollar_pnl", 0), reverse=True):
            d = m.get("dollar_pnl", 0.0)
            print(f"  {sym:<8} {m.get('n_trades',0):>7} "
                  f"{m.get('win_rate_pct',0.0):>7.1f} ${d:>+,.0f}")

    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    params = ScalpParams(
        rsi_long_thresh=args.rsi_long,
        rsi_short_thresh=args.rsi_short,
        vol_min=args.vol_min,
        atr_stop_mult=args.atr_stop_mult,
        atr_target_mult=args.atr_target_mult,
        quick_exit_bars=args.quick_exit_bars,
        min_profit_pct=args.min_profit_pct,
        max_hold_bars=args.max_hold_bars,
        adx_max=args.adx_max,
        require_reversal_candle=args.reversal_candle,
        candle_wick_ratio=args.candle_wick_ratio,
        entry_confirm=args.entry_confirm,
        rsi_turn=args.rsi_turn,
        sides=args.sides,
        direction_gate=args.direction_gate,
        bar_minutes=args.bar_minutes,
        from_open_drop_max=args.from_open_drop_max,
    )
    report = run(
        symbols=args.symbols,
        days=args.days,
        params=params,
        dsn=args.dsn,
        max_slots=args.max_slots,
        capital_usd=args.capital,
        end_date=args.end_date,
    )
    _print_summary(report)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M")
    out_path = out_dir / f"{stamp}-scalp-reversal.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
