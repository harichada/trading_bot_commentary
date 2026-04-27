"""Shared event-view builder for P4 and P3 nested-CV runners.

Step 1 of P3: extract the event-collection + uniqueness + cost-aware-
weighting pipeline out of ``train_ml_model_v2.main`` so that both the
P4 runner and the P3 regime-routing runner consume identical inputs
into ``purged_cv_evaluate`` / ``refit_primary_on_fold``.

The function signature is intentionally a thin wrapper around the
existing ``collect_v2_samples`` to guarantee byte-identical P4 outputs
(parity-tested in ``tests/test_event_view_parity.py``). The only
additive change is ``EventView.prices_per_symbol`` — needed by the
P3 regime classifier and ignored by the P4 runner.

Methodology references:
    AFML §3.4 — triple-barrier labels (delegated to ml.labels)
    AFML §4.5 — sample uniqueness (delegated to ml.labels.compute_uniqueness)
    P2 §1   — cost-aware sample weights (delegated to ml.pnl_objective.compute_sample_weights)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventView:
    """Immutable bundle of event-aligned arrays consumed by purged-CV
    evaluators (P4) and nested-CV regime routers (P3).

    All array-typed fields are aligned row-for-row by event index; the
    ordering is chronological by ``event_times`` (purged CV requirement).
    """
    X: np.ndarray                     # (n_events, n_features)
    y: np.ndarray                     # (n_events,) in {0=SELL, 1=HOLD, 2=BUY}
    ret: np.ndarray                   # (n_events,) realized return per event
    weights: np.ndarray               # (n_events,) cost-aware × uniqueness OR uniqueness only
    event_times: pd.DatetimeIndex     # (n_events,) event start
    touch_times: pd.DatetimeIndex     # (n_events,) triple-barrier touch
    sample_r_multiples: np.ndarray    # (n_events,) realized R-multiple
    sample_cost_r: np.ndarray         # (n_events,) cost in R
    sym_groups: np.ndarray            # (n_events,) symbol per event (object dtype)
    uniqueness: np.ndarray            # (n_events,) AFML §4.5 uniqueness weights
    per_symbol_samples: dict[str, int] = field(default_factory=dict)
    drop_counts: dict[str, int] = field(default_factory=dict)
    prices_per_symbol: dict[str, pd.DataFrame] | None = None
    """Per-symbol bar DataFrames (indexed by bar timestamp). Required by
    P3 regime classification; ``None`` when the caller does not need it
    (P4 default — saves ~one bar-DF retention per symbol)."""
    timing: dict[str, float] = field(default_factory=dict)
    """Wall-clock seconds for the collection and uniqueness stages.
    Keys: ``collect``, ``uniqueness``."""


def build_event_view(
    *,
    symbols: list[str],
    days: int,
    frequency: int,
    dsn: str,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    event_stride: int,
    cost_aware: bool,
    fee_bps: float,
    slip_bps: float,
    impact_coef: float,
    participation: float,
    event_source: str = "cusum",
    cusum_h: float = 0.005,
    vertical_mult_duration: float = 2.0,
    min_ret_atr_mult: float = 0.5,
    collect_prices: bool = True,
) -> EventView:
    """Build the chronologically-sorted, weight-tagged EventView.

    Equivalent to the inline assembly previously in
    ``train_ml_model_v2.main``: collect samples, sort by event_time,
    compute uniqueness, compute cost-aware sample weights.

    Parameters
    ----------
    collect_prices:
        When True, the per-symbol bar DataFrames are retained in
        ``EventView.prices_per_symbol``. P3 needs them for regime
        classification. P4 doesn't read this field, so the only cost
        is memory (~one DataFrame per symbol). Default True for safety.
    """
    # Lazy import to avoid circular dependency: train_ml_model_v2 imports
    # build_event_view in main(); we import its private helpers here.
    from train_ml_model_v2 import (
        _cost_r_from_constants,
        _r_multiples_from_labels,
        collect_v2_samples,
    )
    from ml.costs import CostModel
    from ml.labels import compute_uniqueness
    from ml.pnl_objective import compute_sample_weights

    drop_acc: dict[str, int] = {}
    prices_acc: dict[str, pd.DataFrame] | None = {} if collect_prices else None

    t_collect_start = datetime.now()
    X, y, ret, event_times, touch_times, sym_groups, per_sym = collect_v2_samples(
        symbols=symbols, days=days, frequency=frequency, dsn=dsn,
        pt_mult=pt_mult, sl_mult=sl_mult,
        max_holding=max_holding, event_stride=event_stride,
        event_source=event_source, cusum_h=cusum_h,
        vertical_mult_duration=vertical_mult_duration,
        min_ret_atr_mult=min_ret_atr_mult,
        drop_accumulator=drop_acc,
        prices_accumulator=prices_acc,
    )
    collect_seconds = (datetime.now() - t_collect_start).total_seconds()

    if len(X) == 0:
        empty_idx = pd.DatetimeIndex([])
        return EventView(
            X=np.array([]), y=np.array([]), ret=np.array([]),
            weights=np.array([]),
            event_times=empty_idx, touch_times=empty_idx,
            sample_r_multiples=np.array([]), sample_cost_r=np.array([]),
            sym_groups=np.array([]), uniqueness=np.array([]),
            per_symbol_samples={}, drop_counts=drop_acc,
            prices_per_symbol=prices_acc,
            timing={"collect": collect_seconds, "uniqueness": 0.0},
        )

    # Chronological sort — purged CV depends on it.
    order = np.argsort(event_times.to_numpy())
    X = X[order]
    y = y[order]
    ret = ret[order]
    event_times = event_times[order]
    touch_times = touch_times[order]
    sym_groups = sym_groups[order]

    # Uniqueness (AFML §4.5), grouped by symbol so cross-symbol overlaps
    # don't depress weights.
    t_unique_start = datetime.now()
    touch_series = pd.Series(touch_times.to_numpy(), index=event_times)
    combined_index = pd.DatetimeIndex(
        np.unique(np.concatenate([event_times.to_numpy(), touch_times.to_numpy()]))
    ).sort_values()
    labels_for_weights = pd.DataFrame(
        {"touch_time": touch_series.values}, index=touch_series.index,
    )
    uniqueness = compute_uniqueness(
        labels_for_weights, combined_index, groups=pd.Series(sym_groups),
    ).to_numpy()
    uniqueness = np.clip(uniqueness, 1e-6, 1.0)
    uniqueness_seconds = (datetime.now() - t_unique_start).total_seconds()

    # Cost-aware sample weights (P2 §1).
    cost_model = CostModel(
        fee_bps=fee_bps, slip_bps=slip_bps, impact_coef=impact_coef,
    )
    sample_r_multiples = _r_multiples_from_labels(
        y=y, ret=ret, pt_mult=pt_mult, sl_mult=sl_mult,
    )
    sample_cost_r = _cost_r_from_constants(
        n=len(y), cost_model=cost_model, participation=participation,
        sl_mult=sl_mult, pt_mult=pt_mult,
    )

    if cost_aware:
        weights = compute_sample_weights(
            r_multiples=sample_r_multiples,
            cost_in_r=sample_cost_r,
            uniqueness=uniqueness,
        )
    else:
        weights = uniqueness.copy()

    return EventView(
        X=X, y=y, ret=ret, weights=weights,
        event_times=event_times, touch_times=touch_times,
        sample_r_multiples=sample_r_multiples,
        sample_cost_r=sample_cost_r,
        sym_groups=sym_groups, uniqueness=uniqueness,
        per_symbol_samples=per_sym, drop_counts=drop_acc,
        prices_per_symbol=prices_acc,
        timing={
            "collect": collect_seconds,
            "uniqueness": uniqueness_seconds,
        },
    )
