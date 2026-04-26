"""PnL-aware evaluation metrics (PROMPT_PACK P2 §2).

Replaces classification_report with the six metrics that actually tell you
whether a model makes money after costs:

    expectancy per trade, profit factor, Sortino, Calmar,
    max adverse excursion, cost-drag %.

All functions take NumPy arrays of R-multiples (net unless the signature says
``gross=`` explicitly). No class side-effects — pure functions only, so each
fold's grading is trivially reproducible and unit-testable.

Rationale (AFML §14.1, López de Prado): accuracy is decoupled from economics
once costs exist. A 60% accurate model can destroy capital; a 45% accurate
model with asymmetric payouts can compound. The metrics below price in both
the hit rate and the payoff distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoldMetrics:
    """All six cost-aware grades for a single OOS fold.

    ``cost_drag_pct`` is None when gross_mean <= 0 (a percentage of a
    non-positive denominator is meaningless). In that case the absolute
    per-trade cost in R is reported via ``absolute_cost_per_trade_r``,
    which is the raw `gross_mean - net_mean` so the reader can still see
    the scale of the friction. Exactly one of the two fields is non-None.
    """

    n_trades: int
    expectancy_r: float
    profit_factor: float
    sortino: float
    calmar: float
    max_adverse_excursion: float
    cost_drag_pct: float | None
    absolute_cost_per_trade_r: float | None


def expectancy_r(r: np.ndarray) -> float:
    """Mean R-multiple per trade. Zero for empty input."""
    r = np.asarray(r, dtype="float64")
    if r.size == 0:
        return 0.0
    return float(r.mean())


def profit_factor(r: np.ndarray) -> float:
    """Sum of wins / sum of |losses|. Inf if no losses, zero if no wins."""
    r = np.asarray(r, dtype="float64")
    if r.size == 0:
        return 0.0
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses == 0.0:
        return math.inf if wins > 0 else 0.0
    return wins / losses


def sortino(r: np.ndarray, bars_per_year: int) -> float:
    """Mean / downside std, annualized by sqrt(bars_per_year).

    Downside std uses the root-mean-square of min(r, 0) (target = 0) across
    all samples — the conventional LPM-2 formulation. This penalizes variance
    only on the loss side, which matches trader intuition: upside variance
    is a feature, not a bug.
    """
    r = np.asarray(r, dtype="float64")
    if r.size == 0:
        return 0.0
    mean_r = float(r.mean())
    downside = np.minimum(r, 0.0)
    downside_mean_sq = float((downside ** 2).mean())
    if downside_mean_sq == 0.0:
        return math.inf if mean_r > 0 else 0.0
    downside_std = math.sqrt(downside_mean_sq)
    ratio = mean_r / downside_std
    return ratio * math.sqrt(bars_per_year)


def calmar(r: np.ndarray, bars_per_year: int) -> float:
    """Annualized return (R) / max drawdown (R).

    Drawdown is computed on the cumulative-R equity curve, not percent of
    equity, because R-multiples already encode per-trade risk. Inf when the
    curve never draws down. Zero for empty input.
    """
    r = np.asarray(r, dtype="float64")
    if r.size == 0:
        return 0.0
    equity = np.cumsum(r)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    max_dd = float(drawdowns.max())
    annualized_return = float(r.mean()) * bars_per_year
    if max_dd == 0.0:
        return math.inf if annualized_return > 0 else 0.0
    return annualized_return / max_dd


def max_adverse_excursion(r: np.ndarray) -> float:
    """Lower-bound MAE from closed-trade R-multiples.

    True MAE requires the intra-trade price path, which training labels don't
    carry. As a conservative stand-in we take the worst *realized* loss in R
    across the fold: any real MAE must be at least that deep. Downstream
    (regime-gated backtest, P3) refines this with path-aware numbers.
    """
    r = np.asarray(r, dtype="float64")
    if r.size == 0:
        return 0.0
    losses = r[r < 0]
    if losses.size == 0:
        return 0.0
    return float(-losses.min())  # min is most negative; sign-flip gives magnitude


def cost_drag_pct(
    *, gross: np.ndarray, net: np.ndarray
) -> tuple[float | None, float | None]:
    """Cost drag report — robust to non-positive gross.

    Returns ``(cost_drag_pct, absolute_cost_per_trade_r)``. Exactly one of
    the two is non-None:

    * ``gross_mean > 0``  → ``cost_drag_pct = (gross_mean - net_mean) /
      gross_mean * 100``, rounded to 1 decimal; absolute is None.
    * ``gross_mean <= 0`` → ``cost_drag_pct`` is None (a percentage of a
      non-positive denominator is meaningless); ``absolute_cost_per_trade_r =
      gross_mean - net_mean``, rounded to 4 decimals.

    Empty input is treated as gross_mean <= 0 (cost_drag_pct is None;
    absolute is 0.0). The training pipeline rejects on the absence of a
    finite percentage, not on a sentinel infinity.
    """
    gross = np.asarray(gross, dtype="float64")
    net = np.asarray(net, dtype="float64")
    if gross.size == 0:
        return None, 0.0
    gross_mean = float(gross.mean())
    net_mean = float(net.mean())
    if gross_mean <= 0.0:
        return None, round(gross_mean - net_mean, 4)
    return round((gross_mean - net_mean) / gross_mean * 100.0, 1), None


def grade_fold(
    *,
    gross_r: np.ndarray,
    net_r: np.ndarray,
    bars_per_year: int,
) -> FoldMetrics:
    """Bundle all six cost-aware metrics for one OOS fold."""
    drag_pct, abs_cost_r = cost_drag_pct(gross=gross_r, net=net_r)
    return FoldMetrics(
        n_trades=int(net_r.size),
        expectancy_r=expectancy_r(net_r),
        profit_factor=profit_factor(net_r),
        sortino=sortino(net_r, bars_per_year=bars_per_year),
        calmar=calmar(net_r, bars_per_year=bars_per_year),
        max_adverse_excursion=max_adverse_excursion(net_r),
        cost_drag_pct=drag_pct,
        absolute_cost_per_trade_r=abs_cost_r,
    )
