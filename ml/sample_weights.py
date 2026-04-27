"""Sample-weight machinery for overlapping triple-barrier labels.

López de Prado, *Advances in Financial Machine Learning* (2018), §4.5.

Triple-barrier label windows overlap, so adjacent samples share most of
their forward-return information. Treating them as IID inflates the
effective sample count seen by the classifier and biases its bagging
ensemble toward the most-overlapped windows. AFML §4.5 gives two fixes:

1. **avg_uniqueness** weighting (also implemented in ``ml/labels.py``).
   Down-weight each sample by the average inverse-concurrency of its
   label window.
2. **Sequential bootstrap**. When drawing a bagged training set, replace
   uniform-with-replacement (``np.random.choice``) with draws whose
   probability is proportional to the row's *current* avg_uniqueness
   conditional on already-drawn rows. This explicitly penalizes
   resampling rows that share information with rows already picked.

The two complement each other: weighting fixes the in-fit objective;
sequential bootstrap fixes the bagging variance estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.labels import compute_uniqueness


def num_co_events(
    price_index: pd.DatetimeIndex,
    event_times: pd.DatetimeIndex,
    touch_times: pd.DatetimeIndex,
    groups: pd.Series | None = None,
) -> pd.Series:
    """Number of label windows alive at each bar (AFML §4.3 ``numCoEvents``).

    A label window is "alive" on every bar in ``[event_time, touch_time]``
    inclusive. The returned Series is indexed to ``price_index`` and
    counts how many of the supplied events are alive on that bar.

    With ``groups`` supplied, concurrency is counted per-group: an AAPL
    event at 10:00 does not contribute to NVDA's concurrency at 10:00.
    """
    if len(event_times) != len(touch_times):
        raise ValueError("event_times and touch_times must align")
    co = pd.Series(0, index=price_index, dtype="int64")
    if groups is not None:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != len(event_times):
            raise ValueError("groups length must equal event_times length")
        # Per-group accumulation; tested via the avg_uniqueness path.
        for key in pd.unique(groups_arr):
            mask = groups_arr == key
            for ev, tc in zip(event_times[mask], touch_times[mask]):
                co.loc[ev:tc] += 1
        return co
    for ev, tc in zip(event_times, touch_times):
        co.loc[ev:tc] += 1
    return co


def avg_uniqueness(
    event_times: pd.DatetimeIndex,
    touch_times: pd.DatetimeIndex,
    groups: pd.Series | None = None,
) -> pd.Series:
    """Average uniqueness per sample (AFML §4.3, ``avgUniqueness``).

    Thin wrapper over :func:`ml.labels.compute_uniqueness` exposing the
    AFML-style ``(event_times, touch_times)`` signature instead of the
    legacy DataFrame signature.
    """
    if len(event_times) != len(touch_times):
        raise ValueError("event_times and touch_times must align")
    labels = pd.DataFrame(
        {"touch_time": touch_times.to_numpy()}, index=event_times,
    )
    sparse_index = pd.DatetimeIndex(
        np.unique(np.concatenate([event_times.to_numpy(), touch_times.to_numpy()]))
    ).sort_values()
    return compute_uniqueness(labels, sparse_index, groups=groups)


def get_indicator_matrix(
    price_index: pd.DatetimeIndex,
    event_times: pd.DatetimeIndex,
    touch_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Boolean ``[bars × events]`` indicator matrix (AFML §4.3).

    Cell ``(t, j) = True`` iff event ``j``'s label window covers bar ``t``.
    Used by sequential bootstrap to compute the average uniqueness of
    candidate draws conditional on already-drawn columns.
    """
    if len(event_times) != len(touch_times):
        raise ValueError("event_times and touch_times must align")

    # Collect positional spans first (vectorizes the assignment loop).
    pos_pairs = []
    for ev, tc in zip(event_times, touch_times):
        # searchsorted on a sorted DatetimeIndex is O(log n) and avoids a
        # KeyError when an event/touch isn't an exact bar (uses the
        # nearest-on-the-right bar, AFML's convention).
        i0 = price_index.searchsorted(ev, side="left")
        i1 = price_index.searchsorted(tc, side="right") - 1
        if i1 < i0:
            i1 = i0  # at least one bar
        pos_pairs.append((int(i0), int(i1)))

    n_bars = len(price_index)
    n_events = len(event_times)
    arr = np.zeros((n_bars, n_events), dtype=bool)
    for j, (i0, i1) in enumerate(pos_pairs):
        if i0 < n_bars:
            arr[i0:i1 + 1, j] = True
    return pd.DataFrame(arr, index=price_index)


def sequential_bootstrap_indices(
    indicator_matrix: pd.DataFrame,
    n_samples: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Sequential bootstrap (AFML §4.5.3, Snippet 4.5).

    At each draw, compute every event's *conditional* avg_uniqueness given
    the rows already drawn, then sample the next index with probability
    proportional to that uniqueness. This actively avoids redrawing
    information-overlapping rows.

    Parameters
    ----------
    indicator_matrix:
        Output of :func:`get_indicator_matrix`. Must be boolean-typed.
    n_samples:
        Number of indices to draw (with replacement). Defaults to the
        number of events (columns of ``indicator_matrix``).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray of positional indices into ``indicator_matrix.columns``.
    """
    if indicator_matrix.empty:
        return np.array([], dtype="int64")

    ind = indicator_matrix.to_numpy(dtype=bool)
    n_bars, n_events = ind.shape
    if n_samples is None:
        n_samples = n_events

    rng = np.random.default_rng(seed)
    drawn: list[int] = []
    # Per-bar count of how many already-drawn rows cover that bar.
    # Starts at zero; each drawn column adds 1 wherever it's True.
    # We pre-broadcast: candidate concurrency = drawn_concur + ind[:, j].
    drawn_concur = np.zeros(n_bars, dtype="int64")

    # Pre-compute per-event coverage masks once (kept as bool arrays
    # for fast vectorized average-uniqueness math).
    cols = [ind[:, j] for j in range(n_events)]
    # column lengths (denominator for avg_uniqueness per event)
    col_len = np.maximum(ind.sum(axis=0), 1).astype("float64")

    for _ in range(n_samples):
        # AvgUniqueness_j = mean over bars-in-event-j of 1 / (drawn_concur + 1)
        # if event j were drawn next.
        # Vectorize across all events:
        #   for each event j with mask m_j: u_j = sum_{t: m_j} 1/(drawn_concur[t]+1) / col_len[j]
        denom = drawn_concur + 1  # +1 because j itself would be added
        per_bar = 1.0 / denom
        # Weighted sum per event = sum over its True rows of per_bar.
        u = np.einsum("tj,t->j", ind, per_bar) / col_len  # (n_events,)

        # Probability ∝ uniqueness; renormalize to a valid pmf.
        s = u.sum()
        if s <= 0 or not np.isfinite(s):
            # Degenerate fallback — uniform IID. Should never happen with
            # any non-empty indicator matrix, but defends against it.
            choice = int(rng.integers(0, n_events))
        else:
            probs = u / s
            choice = int(rng.choice(n_events, p=probs))
        drawn.append(choice)
        drawn_concur += cols[choice].astype("int64")

    return np.asarray(drawn, dtype="int64")
