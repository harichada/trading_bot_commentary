"""Purged K-Fold cross-validation with embargo (López de Prado, AFML Ch. 7).

Standard K-Fold leaks information in time-series ML because label
windows overlap across the train/test boundary: a training sample
whose outcome period extends into the test block has seen test-period
price action when its label was computed. PurgedKFold removes such
samples from training. An embargo further excludes a trailing slice
after each test fold to guard against short-horizon serial correlation
in features.

The splitter is sklearn-compatible: it exposes ``split`` and
``get_n_splits`` and yields ``(train_idx, test_idx)`` pairs of
positional indices.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


class PurgedKFold:
    """Time-ordered K-Fold with purging and embargo.

    Parameters
    ----------
    n_splits:
        Number of folds. Must be >= 2.
    touch_times:
        Series indexed by event start time, whose values are the
        barrier-touch timestamps. Defines each sample's outcome window
        as ``[index, touch_time]``.
    embargo_pct:
        Fraction of total samples to embargo immediately after each
        test fold. E.g. 0.01 on 10_000 samples embargoes 100 rows.
    """

    def __init__(
        self,
        n_splits: int,
        touch_times: pd.Series,
        embargo_pct: float = 0.01,
    ) -> None:
        if not isinstance(touch_times, pd.Series):
            raise TypeError("touch_times must be a pandas Series")
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        # Sort before validating — elementwise comparison of unsorted
        # index vs values gives wrong answers. Canonical ordering is
        # what the splitter actually uses downstream anyway.
        sorted_touch = touch_times.sort_index()
        if not (sorted_touch.index <= sorted_touch.values).all():
            raise ValueError(
                "touch_time must be >= its event time for every sample"
            )

        self.n_splits = n_splits
        self.touch_times = sorted_touch
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(self.touch_times)
        if len(X) != n:
            raise ValueError(
                f"X has {len(X)} rows but touch_times has {n}; they must align"
            )

        indices = np.arange(n)
        event_times = self.touch_times.index.to_numpy()
        touch_values = self.touch_times.to_numpy()

        embargo_size = int(n * self.embargo_pct)
        fold_edges = np.array_split(indices, self.n_splits)

        for test_idx in fold_edges:
            test_start_time = event_times[test_idx[0]]
            test_end_time = touch_values[test_idx[-1]]

            # Purge: drop training samples whose outcome window overlaps
            # the test window [test_start_time, test_end_time].
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False

            for i in indices[train_mask]:
                # Overlap iff event_i <= test_end AND touch_i >= test_start.
                if event_times[i] <= test_end_time and touch_values[i] >= test_start_time:
                    train_mask[i] = False

            # Embargo: drop the N samples immediately following the test fold.
            if embargo_size > 0:
                embargo_end = min(test_idx[-1] + 1 + embargo_size, n)
                train_mask[test_idx[-1] + 1 : embargo_end] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx
