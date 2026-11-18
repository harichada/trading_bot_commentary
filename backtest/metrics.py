"""Metric calculations over a list of ``Trade`` objects.

``summarize`` and ``summarize_per_symbol`` are ported from
``backtest_strategies.py`` (lines 269–324) verbatim — the JSON shape they
produce is what ``api/backtest.py`` and the dashboard already consume.

This module adds two new ratios on top of the original ``summarize``
output: ``per_trade_sortino`` and ``calmar_ratio``. They are appended to
the returned dict as additional keys so all existing consumers keep
working unchanged.

Formulas for the new ratios follow the references in
``risk/backtest.py`` (Sortino: lines 230–238; Calmar: lines 391–397) but
do **not** import from that module — ``risk/backtest.py`` is dead code
in this codebase and is being kept untouched per the MVP scope.

Both new ratios operate on the same per-trade return series the existing
metrics use (one return per trade, expressed as a percentage). They do
*not* annualise — the input is event-based rather than periodic, so an
annualisation factor would be misleading.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from backtest.trades import SymbolMetrics, Trade


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _sortino(returns: np.ndarray) -> float:
    """Per-trade Sortino: mean / downside-deviation.

    Downside deviation is the std of negative returns only. Falls back to
    zero when there are no losing trades (no downside risk to penalise).
    """
    losers = returns[returns < 0]
    if len(losers) == 0:
        return 0.0
    downside_std = float(np.std(losers))
    if downside_std == 0.0:
        return 0.0
    return float(np.mean(returns) / downside_std)


def _calmar(returns: np.ndarray) -> float:
    """Per-trade Calmar: total return / |max drawdown| over the trade-equity curve.

    Equity curve is the cumulative sum of per-trade returns — the same
    series used to derive ``max_drawdown_pct`` so the two metrics are
    consistent. Returns 0.0 when no drawdown has occurred.
    """
    if len(returns) == 0:
        return 0.0
    total = float(np.sum(returns))
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    max_dd = abs(float(drawdown.min()))
    if max_dd == 0.0:
        return 0.0
    return total / max_dd


# ---------------------------------------------------------------------------
# Public API — ported from backtest_strategies.summarize / summarize_per_symbol
# ---------------------------------------------------------------------------

def summarize(trades: list[Trade]) -> dict[str, Any]:
    if not trades:
        return {"n_trades": 0}
    returns = np.array([t.return_pct for t in trades])
    wins = returns > 0

    summed = float(np.sum(returns))
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10))
    reasons = [t.exit_reason for t in trades]

    wins_r = returns[wins]
    losses_r = returns[~wins]

    return {
        "n_trades": len(trades),
        "win_rate_pct": round(100 * float(np.mean(wins)), 2),
        "avg_return_pct": round(float(np.mean(returns)), 3),
        "median_return_pct": round(float(np.median(returns)), 3),
        "avg_win_pct": round(float(np.mean(wins_r)) if len(wins_r) else 0.0, 3),
        "avg_loss_pct": round(float(np.mean(losses_r)) if len(losses_r) else 0.0, 3),
        "sum_return_pct": round(summed, 2),
        "max_drawdown_pct": round(float(drawdown.min()), 2),
        "per_trade_sharpe": round(sharpe, 3),
        "per_trade_sortino": round(_sortino(returns), 3),
        "calmar_ratio": round(_calmar(returns), 3),
        "profit_factor": round(
            float(wins_r.sum()) / max(1e-10, float(-losses_r.sum())), 3
        ) if len(losses_r) > 0 else None,
        "avg_hold_bars": round(float(np.mean([t.hold_bars for t in trades])), 1),
        "exit_breakdown": {
            "stop": round(100 * reasons.count("stop") / len(trades), 1),
            "target": round(100 * reasons.count("target") / len(trades), 1),
            "timeout": round(100 * reasons.count("timeout") / len(trades), 1),
            "eod": round(100 * reasons.count("eod") / len(trades), 1),
        },
    }


def summarize_per_symbol(trades: list[Trade]) -> dict[str, dict[str, Any]]:
    """Per-symbol slice of per-strategy trades, for the drill-down table."""
    by_sym: dict[str, list[Trade]] = {}
    for t in trades:
        by_sym.setdefault(t.symbol, []).append(t)

    out: dict[str, dict[str, Any]] = {}
    for sym, sym_trades in by_sym.items():
        returns = np.array([t.return_pct for t in sym_trades])
        wins = returns > 0
        out[sym] = asdict(SymbolMetrics(
            n_trades=len(sym_trades),
            win_rate_pct=round(100 * float(np.mean(wins)), 2),
            sum_return_pct=round(float(np.sum(returns)), 2),
            avg_return_pct=round(float(np.mean(returns)), 3),
        ))
    return out
