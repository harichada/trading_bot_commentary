"""Tests for ``ml/sample_weights.py`` (López de Prado AFML §4.5).

The standard bagging draw (random row subsampling) is statistically wrong
for overlapping triple-barrier labels: heavily concurrent rows share most
of their forward-return information, so they are effectively duplicates.
Sequential bootstrap fixes this by drawing rows with probability inversely
proportional to their *current* concurrency with already-drawn rows
(AFML §4.5.3, ``seqBootstrap``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.sample_weights import (
    avg_uniqueness,
    get_indicator_matrix,
    num_co_events,
    sequential_bootstrap_indices,
)


def _idx(n: int, start: str = "2024-01-02 09:30") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="1min")


class TestNumCoEvents:
    def test_non_overlapping_count_one(self):
        bars = _idx(100)
        events = pd.DatetimeIndex([bars[0], bars[50]])
        touches = pd.DatetimeIndex([bars[10], bars[60]])

        co = num_co_events(bars, events, touches)

        # Bars [0..10] should show concurrency=1 (event A).
        assert co.loc[bars[0]] == 1
        assert co.loc[bars[10]] == 1
        # Bars between events show concurrency=0.
        assert co.loc[bars[20]] == 0
        # Bars [50..60] show concurrency=1 (event B).
        assert co.loc[bars[55]] == 1

    def test_fully_overlapping_count_n(self):
        bars = _idx(100)
        events = pd.DatetimeIndex([bars[0], bars[0], bars[0]])
        touches = pd.DatetimeIndex([bars[10], bars[10], bars[10]])

        co = num_co_events(bars, events, touches)

        assert co.loc[bars[5]] == 3


class TestIndicatorMatrix:
    def test_shape_and_dtype(self):
        bars = _idx(20)
        events = pd.DatetimeIndex([bars[0], bars[5]])
        touches = pd.DatetimeIndex([bars[10], bars[15]])

        ind = get_indicator_matrix(bars, events, touches)

        assert ind.shape == (20, 2)
        assert ind.dtypes.iloc[0] == bool

    def test_column_sums_equal_window_lengths(self):
        bars = _idx(20)
        events = pd.DatetimeIndex([bars[0], bars[5]])
        touches = pd.DatetimeIndex([bars[10], bars[15]])

        ind = get_indicator_matrix(bars, events, touches)

        # Event 0: bars [0..10] inclusive = 11 bars.
        # Event 1: bars [5..15] inclusive = 11 bars.
        assert ind.iloc[:, 0].sum() == 11
        assert ind.iloc[:, 1].sum() == 11


class TestAvgUniqueness:
    def test_non_overlapping_uniqueness_one(self):
        bars = _idx(100)
        events = pd.DatetimeIndex([bars[0], bars[50]])
        touches = pd.DatetimeIndex([bars[10], bars[60]])

        u = avg_uniqueness(events, touches)

        assert u.iloc[0] == pytest.approx(1.0)
        assert u.iloc[1] == pytest.approx(1.0)

    def test_fully_overlapping_uniqueness_one_over_n(self):
        bars = _idx(100)
        events = pd.DatetimeIndex([bars[0], bars[0], bars[0]])
        touches = pd.DatetimeIndex([bars[10], bars[10], bars[10]])

        u = avg_uniqueness(events, touches)

        assert u.iloc[0] == pytest.approx(1 / 3)
        assert u.iloc[1] == pytest.approx(1 / 3)
        assert u.iloc[2] == pytest.approx(1 / 3)

    def test_groups_isolate_concurrency(self):
        bars = _idx(100)
        events = pd.DatetimeIndex([bars[0], bars[0]])
        touches = pd.DatetimeIndex([bars[10], bars[10]])
        groups = pd.Series(["NVDA", "AAPL"])

        u_grouped = avg_uniqueness(events, touches, groups=groups)

        assert u_grouped.iloc[0] == pytest.approx(1.0)
        assert u_grouped.iloc[1] == pytest.approx(1.0)


class TestSequentialBootstrap:
    def test_reproducible_with_seed(self):
        bars = _idx(100)
        events = pd.DatetimeIndex([bars[i] for i in range(0, 80, 10)])
        touches = pd.DatetimeIndex([bars[i + 15] for i in range(0, 80, 10)])
        ind = get_indicator_matrix(bars, events, touches)

        a = sequential_bootstrap_indices(ind, n_samples=8, seed=42)
        b = sequential_bootstrap_indices(ind, n_samples=8, seed=42)

        np.testing.assert_array_equal(a, b)

    def test_different_seed_changes_draw(self):
        bars = _idx(100)
        events = pd.DatetimeIndex([bars[i] for i in range(0, 80, 10)])
        touches = pd.DatetimeIndex([bars[i + 15] for i in range(0, 80, 10)])
        ind = get_indicator_matrix(bars, events, touches)

        a = sequential_bootstrap_indices(ind, n_samples=8, seed=1)
        b = sequential_bootstrap_indices(ind, n_samples=8, seed=2)

        assert not np.array_equal(a, b)

    def test_default_n_samples_equals_input_length(self):
        bars = _idx(50)
        events = pd.DatetimeIndex([bars[i] for i in range(0, 40, 5)])
        touches = pd.DatetimeIndex([bars[i + 5] for i in range(0, 40, 5)])
        ind = get_indicator_matrix(bars, events, touches)

        out = sequential_bootstrap_indices(ind, seed=0)

        assert len(out) == ind.shape[1]

    def test_draws_higher_uniqueness_than_iid_on_overlap_heavy(self):
        """On heavily overlapping labels, sequential bootstrap must yield a
        sample whose mean avg_uniqueness exceeds the np.random.choice baseline.
        That's the entire point of the algorithm (AFML §4.5.3 Fig. 4.1)."""
        bars = _idx(200)
        # 50 events all starting at t=0..49, each with a 100-bar window
        # — heavy overlap.
        events = pd.DatetimeIndex([bars[i] for i in range(50)])
        touches = pd.DatetimeIndex([bars[i + 100] for i in range(50)])
        ind = get_indicator_matrix(bars, events, touches)

        rng = np.random.default_rng(0)
        iid_idx = rng.integers(0, ind.shape[1], size=ind.shape[1])
        seq_idx = sequential_bootstrap_indices(ind, seed=0)

        # avg_uniqueness of the resampled set, computed positionally.
        def mean_unique_of(sample_idx: np.ndarray) -> float:
            sub = ind.iloc[:, sample_idx]
            concur = sub.sum(axis=1).replace(0, np.nan)
            per_bar = 1.0 / concur
            # Uniqueness of column j is mean of per_bar over rows where col j is True.
            uniq = []
            for j in range(sub.shape[1]):
                mask = sub.iloc[:, j].to_numpy()
                if mask.any():
                    uniq.append(float(np.nanmean(per_bar.to_numpy()[mask])))
            return float(np.mean(uniq)) if uniq else 0.0

        assert mean_unique_of(seq_idx) > mean_unique_of(iid_idx)

    def test_indices_in_valid_range(self):
        bars = _idx(50)
        events = pd.DatetimeIndex([bars[i] for i in range(0, 40, 5)])
        touches = pd.DatetimeIndex([bars[i + 5] for i in range(0, 40, 5)])
        ind = get_indicator_matrix(bars, events, touches)

        out = sequential_bootstrap_indices(ind, seed=0)

        assert (out >= 0).all()
        assert (out < ind.shape[1]).all()
