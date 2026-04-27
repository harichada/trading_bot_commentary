"""Per-regime metrics aggregator for P3 nested-CV regime-routing.

Buckets trades by their committed-regime label and reports
PF / expectancy / win-rate / n_trades / per-fold AR(1) / cost_drag PER
REGIME, plus an aggregate-across-regimes block (informational sanity
check vs the P4 baseline).

Wraps ml/evaluation.py and ml/iid_diagnostics.py — does not duplicate
their math. Per-fold AR(1) is suppressed (None) for regimes with fewer
than 30 trades in that fold (small-sample autocorrelation is noise).

Aggregation across folds uses the **median** rather than the mean —
matches the P4 / P2 readout convention (avoid single-fold luck).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from ml.evaluation import (
    cost_drag_pct as _cost_drag_pct,
    expectancy_r as _expectancy_r,
    profit_factor as _profit_factor,
)
from ml.iid_diagnostics import _ar1_coefficient

AR1_MIN_TRADES = 30  # Below this, AR(1) is small-sample noise → None.


def _safe_median(xs: list[float | None]) -> float | None:
    """Median of a list, ignoring None and NaN entries. Preserves +inf
    (a profit-factor of infinity is meaningful — "all wins, no losses"
    — and shouldn't be silently dropped). Returns None if every entry
    is None or NaN."""
    keep = [x for x in xs if x is not None and not (
        isinstance(x, float) and math.isnan(x))]
    if not keep:
        return None
    return float(np.median(keep))


class PerRegimeAggregator:
    """Accumulator across CV folds.

    Usage::

        agg = PerRegimeAggregator(regimes=("trend_up_low_vol", ...))
        for fold_idx, (...) in enumerate(folds):
            agg.add_fold(fold_idx, regime_labels, gross_r, net_r,
                         residuals=resids)
        report = agg.summarize()

    The ``residuals`` argument is the per-trade weighted residual from
    the OOS primary's softprob — used for the per-fold AR(1) gate
    within surviving regimes (P3 acceptance gate B).
    """

    def __init__(self, regimes: tuple[str, ...]) -> None:
        if not regimes:
            raise ValueError("regimes must be non-empty")
        self.regimes = tuple(regimes)
        # Per-(regime, fold) bookkeeping.
        self._fold_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._fold_indices_seen: set[int] = set()

    def add_fold(
        self,
        *,
        fold_idx: int,
        regime_labels: np.ndarray,
        gross_r: np.ndarray,
        net_r: np.ndarray,
        residuals: np.ndarray | None = None,
    ) -> None:
        """Record one fold's trade outcomes, bucketed by regime."""
        regime_labels = np.asarray(regime_labels, dtype=object)
        gross_r = np.asarray(gross_r, dtype="float64")
        net_r = np.asarray(net_r, dtype="float64")
        if not (regime_labels.shape == gross_r.shape == net_r.shape):
            raise ValueError(
                "regime_labels, gross_r, net_r must align row-for-row"
            )
        if residuals is not None:
            residuals = np.asarray(residuals, dtype="float64")
            if residuals.shape != net_r.shape:
                raise ValueError("residuals must align with trades")

        self._fold_indices_seen.add(int(fold_idx))

        # Always record an empty entry for every regime so per_fold_n_trades
        # has the same length across regimes.
        for r in self.regimes:
            mask = regime_labels == r
            n = int(mask.sum())
            sub_gross = gross_r[mask] if n > 0 else np.array([], dtype="float64")
            sub_net = net_r[mask] if n > 0 else np.array([], dtype="float64")
            sub_resid = (residuals[mask] if (residuals is not None and n > 0)
                         else None)

            ar1_val: float | None = None
            if sub_resid is not None and n >= AR1_MIN_TRADES:
                ar1_val = float(_ar1_coefficient(sub_resid))

            wins = int((sub_net > 0).sum())
            win_rate = (wins / n) if n > 0 else None
            pf = _profit_factor(sub_net) if n > 0 else None
            exp_r = _expectancy_r(sub_net) if n > 0 else None
            drag_pct, abs_cost = _cost_drag_pct(gross=sub_gross, net=sub_net)

            self._fold_records[r].append({
                "fold_idx": int(fold_idx),
                "n_trades": n,
                "expectancy_r": exp_r,
                "profit_factor": pf,
                "win_rate": win_rate,
                "ar1_residual_autocorr": ar1_val,
                "cost_drag_pct": drag_pct,
                "absolute_cost_per_trade_r": abs_cost,
                "gross_r": sub_gross,
                "net_r": sub_net,
                "residuals": sub_resid,
            })

    def summarize(self) -> dict[str, Any]:
        """Produce the report dict consumed by the runner."""
        out: dict[str, Any] = {}
        for r in self.regimes:
            recs = self._fold_records.get(r, [])
            per_fold_n = [rec["n_trades"] for rec in recs]
            per_fold_exp = [rec["expectancy_r"] for rec in recs]
            per_fold_pf = [rec["profit_factor"] for rec in recs]
            per_fold_wr = [rec["win_rate"] for rec in recs]
            per_fold_ar1 = [rec["ar1_residual_autocorr"] for rec in recs]
            per_fold_drag = [rec["cost_drag_pct"] for rec in recs]
            total_n = int(sum(per_fold_n))

            out[r] = {
                "total_n_trades": total_n,
                "per_fold_n_trades": per_fold_n,
                "median_expectancy_r": (
                    _safe_median(per_fold_exp) if total_n > 0 else None
                ),
                "median_profit_factor": (
                    _safe_median(per_fold_pf) if total_n > 0 else None
                ),
                "median_pf": (
                    _safe_median(per_fold_pf) if total_n > 0 else None
                ),
                "median_win_rate": (
                    _safe_median(per_fold_wr) if total_n > 0 else None
                ),
                "per_fold_ar1": per_fold_ar1,
                "median_per_fold_ar1": _safe_median(per_fold_ar1),
                "max_per_fold_ar1": (
                    max((abs(x) for x in per_fold_ar1
                         if x is not None and math.isfinite(x)),
                        default=None)
                ),
                "median_cost_drag_pct": _safe_median(per_fold_drag),
                "fold_detail": [
                    {k: v for k, v in rec.items()
                     if k not in ("gross_r", "net_r", "residuals")}
                    for rec in recs
                ],
            }

        # Aggregate-across-regimes (sanity gate D). Excludes chop trades
        # because gate C forbids them — but if any chop trades slipped
        # through (would be a bug), they'd be included here on purpose
        # so the diagnostic surfaces it.
        agg_per_fold_n: list[int] = []
        agg_per_fold_exp: list[float] = []
        agg_per_fold_pf: list[float] = []
        agg_per_fold_wr: list[float] = []
        agg_per_fold_drag: list[float | None] = []
        for fold_idx in sorted(self._fold_indices_seen):
            fold_gross: list[np.ndarray] = []
            fold_net: list[np.ndarray] = []
            for r in self.regimes:
                if r == "chop":
                    continue  # excluded from aggregate-across-regimes
                for rec in self._fold_records[r]:
                    if rec["fold_idx"] == fold_idx:
                        fold_gross.append(rec["gross_r"])
                        fold_net.append(rec["net_r"])
            if fold_gross:
                gross = np.concatenate(fold_gross)
                net = np.concatenate(fold_net)
            else:
                gross = np.array([], dtype="float64")
                net = np.array([], dtype="float64")
            n = int(net.size)
            agg_per_fold_n.append(n)
            if n > 0:
                agg_per_fold_exp.append(_expectancy_r(net))
                pf = _profit_factor(net)
                if math.isfinite(pf):
                    agg_per_fold_pf.append(pf)
                agg_per_fold_wr.append(float((net > 0).mean()))
                drag, _ = _cost_drag_pct(gross=gross, net=net)
                agg_per_fold_drag.append(drag)

        total_n_aggregate = int(sum(agg_per_fold_n))
        out["__aggregate__"] = {
            "total_n_trades": total_n_aggregate,
            "per_fold_n_trades": agg_per_fold_n,
            "median_expectancy_r": (
                _safe_median(agg_per_fold_exp) if total_n_aggregate > 0 else None
            ),
            "median_pf": (
                _safe_median(agg_per_fold_pf) if total_n_aggregate > 0 else None
            ),
            "median_win_rate": (
                _safe_median(agg_per_fold_wr) if total_n_aggregate > 0 else None
            ),
            "median_cost_drag_pct": _safe_median(agg_per_fold_drag),
        }
        return out
