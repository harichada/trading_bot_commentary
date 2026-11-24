"""
Per-stock intraday direction gate for the scalp reversal strategy.

The daily screener selects stocks with 3%+ absolute moves.  For a
mean-reversion strategy the correct direction is the OPPOSITE of the
daily move:

  Gainer (move_pct > 0) → extended to the upside → RSI elevated →
      SHORT the overbought fade.

  Loser  (move_pct < 0) → extended to the downside → RSI depressed →
      LONG the oversold bounce.

This is the inverse of a momentum/trend-following regime gate.
"Buy panic, sell euphoria" — the best long setups appear when the stock
has already sold off hard (loser), not when the market is calm.
"""
from __future__ import annotations


def stock_sides(move_pct: float) -> str:
    """
    Return "long_only", "short_only", or "both" for a stock's daily % move.

    Args:
        move_pct: today's % change from prior close (positive = gainer).
    """
    if move_pct > 0:
        return "short_only"
    elif move_pct < 0:
        return "long_only"
    else:
        return "both"
