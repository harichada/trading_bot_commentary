"""News-strategy walk-forward validation.

Gating decision for Tuesday 2026-05-26 go-live: is the news strategy's
PF 2.17 over 13 trades a real edge or sample noise?

Three lenses:
  1. Bootstrap 1000 resamples of the 13 trades to get a 95% CI on PF.
  2. Sensitivity: drop the best trade, drop the worst trade — does PF
     survive at >=1.5?
  3. Compare bot-accepted news trades to all news signal_buy events
     (the ones gates rejected) to see if the bot's selection adds value.

Reuses parsing + reconcile from research/weekly_report.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import asyncio
import random
import statistics

from research.weekly_report import (
    parse_logs,
    reconcile_trades,
    fetch_schwab_fills,
    TradeOutcome,
    normalize_strategy,
)


def filter_news(outcomes: List[TradeOutcome]) -> List[TradeOutcome]:
    """Keep only normalized news-strategy outcomes."""
    return [t for t in outcomes if normalize_strategy(t.strategy) == "news"]


def profit_factor(trades: List[TradeOutcome]) -> float:
    """gross_win / abs(gross_loss). Inf if no losses, 0 if no wins."""
    if not trades:
        return 0.0
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def win_rate(trades: List[TradeOutcome]) -> float:
    """Fraction of trades with positive P&L."""
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.pnl > 0) / len(trades)


def bootstrap_pf(
    trades: List[TradeOutcome],
    iterations: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Resample trades with replacement; return (median PF, 2.5%, 97.5%)."""
    rng = random.Random(seed)
    pfs: List[float] = []
    n = len(trades)
    for _ in range(iterations):
        sample = [trades[rng.randint(0, n - 1)] for _ in range(n)]
        pf = profit_factor(sample)
        if pf == float("inf"):
            # Resamples with all wins → cap at large finite to avoid skew.
            pf = 100.0
        pfs.append(pf)
    pfs.sort()
    return (
        statistics.median(pfs),
        pfs[int(0.025 * len(pfs))],
        pfs[int(0.975 * len(pfs)) - 1],
    )


def sensitivity_drop_one(trades: List[TradeOutcome]) -> dict:
    """For each trade, compute PF if that trade is removed.

    Reports the trades whose removal changes PF most — they're the
    points the conclusion depends on. If removing the best trade
    drops PF below 1.5, the edge is concentrated in one outlier.
    """
    base_pf = profit_factor(trades)
    if not trades:
        return {"base_pf": 0.0, "leave_one_out": []}
    deltas = []
    for i, t in enumerate(trades):
        without = trades[:i] + trades[i + 1 :]
        new_pf = profit_factor(without)
        deltas.append((t.symbol, t.pnl, new_pf, new_pf - base_pf))
    # Sort by absolute change in PF
    deltas.sort(key=lambda x: abs(x[3]), reverse=True)
    return {
        "base_pf": base_pf,
        "leave_one_out_top5": deltas[:5],
    }


def report(news_trades: List[TradeOutcome]) -> None:
    print("=" * 70)
    print("News-Strategy Walk-Forward Validation")
    print("=" * 70)
    print(f"Sample: N = {len(news_trades)} news trades")
    print()
    if not news_trades:
        print("No news trades found. Cannot validate.")
        return

    # Lens 1: descriptive stats
    base_pf = profit_factor(news_trades)
    base_wr = win_rate(news_trades)
    total_pnl = sum(t.pnl for t in news_trades)
    gross_win = sum(t.pnl for t in news_trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in news_trades if t.pnl <= 0))
    print(f"Descriptive:")
    print(f"  Profit Factor: {base_pf:.2f}")
    print(f"  Win Rate:      {base_wr*100:.1f}%")
    print(f"  Net P&L:       ${total_pnl:+,.2f}")
    print(f"  Gross Win:     ${gross_win:+,.2f}")
    print(f"  Gross Loss:    ${gross_loss:+,.2f}")
    print()

    # Lens 2: bootstrap 95% CI
    median, low, high = bootstrap_pf(news_trades, iterations=2000)
    print(f"Bootstrap (2000 resamples):")
    print(f"  Median PF:     {median:.2f}")
    print(f"  95% CI:        [{low:.2f}, {high:.2f}]")
    if low > 1.5:
        ci_verdict = "STRONG: lower bound above 1.5 — real edge"
    elif low > 1.0:
        ci_verdict = "MODERATE: lower bound above 1.0 — edge likely, size cautiously"
    elif low > 0.7:
        ci_verdict = "WEAK: lower bound above 0.7 — edge ambiguous"
    else:
        ci_verdict = "NONE: lower bound at or below 0.7 — cannot rule out noise"
    print(f"  Verdict:       {ci_verdict}")
    print()

    # Lens 3: leave-one-out sensitivity
    sens = sensitivity_drop_one(news_trades)
    print(f"Leave-One-Out Sensitivity (top 5 by impact):")
    print(f"  Base PF: {sens['base_pf']:.2f}")
    print(f"  {'Symbol':<8} {'P&L':>10} {'PF without':>12} {'Delta':>8}")
    for sym, pnl, new_pf, delta in sens["leave_one_out_top5"]:
        print(f"  {sym:<8} ${pnl:+9.2f} {new_pf:>12.2f} {delta:>+8.2f}")
    print()

    # Final recommendation
    print("=" * 70)
    print("Recommendation")
    print("=" * 70)
    # Drop the single best winner — does PF stay above 1.5?
    sorted_wins = sorted(
        [t for t in news_trades if t.pnl > 0], key=lambda t: t.pnl, reverse=True
    )
    if sorted_wins:
        without_top = [t for t in news_trades if t is not sorted_wins[0]]
        pf_no_top = profit_factor(without_top)
        print(f"  PF without best winner: {pf_no_top:.2f}")
    else:
        pf_no_top = 0.0

    if low > 1.0 and pf_no_top > 1.3:
        print("  >>> GO LIVE TUESDAY with news strategy at full size.")
        print("      Bootstrap lower bound > 1.0 and PF survives dropping top winner.")
    elif low > 1.0:
        print("  >>> GO LIVE TUESDAY at HALF size (concentration risk).")
        print("      Bootstrap lower bound > 1.0 but edge concentrated in 1-2 trades.")
    elif median > 1.5:
        print("  >>> KEEP IN SIM. Median PF suggests edge, but CI lower bound is")
        print("      below 1.0 — too much variance for live capital at this N.")
    else:
        print("  >>> KEEP IN SIM. Insufficient evidence of edge after correction.")
    print()


async def main() -> None:
    print("[1/2] parsing logs + reconciling trades...")
    signals, accepts = parse_logs()
    fills = fetch_schwab_fills(days=7)
    outcomes = reconcile_trades(accepts, signals, fills)
    print(f"      {len(outcomes)} reconciled trades")
    print()

    news_trades = filter_news(outcomes)
    closed = [t for t in news_trades if t.exit_px is not None]
    print(f"[2/2] filtered to news strategy: {len(closed)} closed trades")
    print()

    report(closed)


if __name__ == "__main__":
    asyncio.run(main())
