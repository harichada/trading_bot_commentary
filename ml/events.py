"""Event-driven sampling for triple-barrier labeling.

López de Prado, *Advances in Financial Machine Learning* (2018), §2.5.2.

Time-driven sampling (one event every N bars) wastes labeling capacity
on quiet periods and under-samples informative regimes. The CUSUM filter
emits an event each time the cumulative log-return since the last event
crosses ±h, then resets. The result is an event timeline aligned with
*meaningful* moves rather than wall-clock minutes.

The ``h`` threshold sets the event density: small ``h`` → many events
near every wiggle; large ``h`` → only major moves. AFML p. 39 recommends
setting ``h`` to a multiple of the bar's rolling volatility so the
filter adapts to regime changes; this module exposes the scalar form
(constant ``h``) since per-bar adaptive thresholds need a separate
volatility estimate that varies by symbol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_filter(prices: pd.Series, h: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM event filter on log-returns.

    Parameters
    ----------
    prices:
        Close-price Series, indexed by timestamp.
    h:
        Trigger threshold on the cumulative log-return since last reset.
        An event fires (and the cumulator resets to 0) whenever
        ``s_pos > h`` or ``s_neg < -h``.

    Returns
    -------
    DatetimeIndex of event timestamps (subset of ``prices.index``).
    """
    if h <= 0:
        raise ValueError("h must be positive")
    if len(prices) < 2:
        return pd.DatetimeIndex([])

    log_close = np.log(prices.to_numpy(dtype="float64"))
    diff = np.diff(log_close)
    timestamps = prices.index[1:]  # log_diff[i] corresponds to timestamps[i]

    s_pos = 0.0
    s_neg = 0.0
    events: list[pd.Timestamp] = []
    for i, d in enumerate(diff):
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_pos > h:
            s_pos = 0.0
            events.append(timestamps[i])
        elif s_neg < -h:
            s_neg = 0.0
            events.append(timestamps[i])

    return pd.DatetimeIndex(events)
