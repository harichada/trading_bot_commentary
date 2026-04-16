"""Tests for Purged K-Fold cross-validation (López de Prado, AFML Ch. 7).

Normal K-Fold leaks information when label windows overlap across the
train/test boundary. PurgedKFold removes training samples whose outcome
windows intersect any test sample's outcome window, and embargoes a
trailing slice after each test fold to protect against serial correlation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.purged_cv import PurgedKFold


def _make_labels(n: int, horizon: int, freq: str = "1min") -> pd.DataFrame:
    """Build a labels DataFrame with evenly-spaced events of fixed horizon."""
    idx = pd.date_range("2024-01-02 09:30", periods=n + horizon, freq=freq)
    event_times = idx[:n]
    touch_times = idx[horizon : n + horizon]
    return pd.DataFrame({"touch_time": touch_times}, index=event_times)


class TestPurgedKFoldStructure:
    def test_n_splits_returned(self):
        labels = _make_labels(n=100, horizon=5)
        cv = PurgedKFold(n_splits=5, touch_times=labels["touch_time"])

        assert cv.get_n_splits() == 5

    def test_yields_exactly_n_splits(self):
        labels = _make_labels(n=100, horizon=5)
        cv = PurgedKFold(n_splits=5, touch_times=labels["touch_time"])

        splits = list(cv.split(X=np.zeros((100, 3))))

        assert len(splits) == 5

    def test_each_fold_is_contiguous_test_block(self):
        """Test indices within a fold must be contiguous (time-ordered CV)."""
        labels = _make_labels(n=100, horizon=5)
        cv = PurgedKFold(n_splits=5, touch_times=labels["touch_time"])

        for _, test_idx in cv.split(X=np.zeros((100, 3))):
            # Should be a contiguous slice of positions
            diffs = np.diff(np.sort(test_idx))
            assert (diffs == 1).all(), f"Non-contiguous test fold: {test_idx}"

    def test_test_folds_cover_all_samples(self):
        labels = _make_labels(n=100, horizon=5)
        cv = PurgedKFold(n_splits=5, touch_times=labels["touch_time"])

        all_test = np.concatenate(
            [test_idx for _, test_idx in cv.split(X=np.zeros((100, 3)))]
        )

        assert sorted(all_test.tolist()) == list(range(100))


class TestPurging:
    def test_train_and_test_are_disjoint(self):
        labels = _make_labels(n=100, horizon=5)
        cv = PurgedKFold(n_splits=5, touch_times=labels["touch_time"])

        for train_idx, test_idx in cv.split(X=np.zeros((100, 3))):
            assert len(set(train_idx).intersection(test_idx)) == 0

    def test_training_samples_do_not_overlap_test_windows(self):
        """No train sample's [event, touch] interval may intersect any
        test sample's [event, touch] interval — that would be leakage."""
        n, horizon = 100, 5
        labels = _make_labels(n=n, horizon=horizon)
        cv = PurgedKFold(n_splits=5, touch_times=labels["touch_time"], embargo_pct=0.0)

        event_times = labels.index.to_numpy()
        touch_times = labels["touch_time"].to_numpy()

        for train_idx, test_idx in cv.split(X=np.zeros((n, 3))):
            test_starts = event_times[test_idx]
            test_ends = touch_times[test_idx]

            for t in train_idx:
                tr_start, tr_end = event_times[t], touch_times[t]
                # Overlap iff tr_start <= test_end AND tr_end >= test_start
                overlaps = (tr_start <= test_ends) & (tr_end >= test_starts)
                assert not overlaps.any(), (
                    f"Training sample {t} ({tr_start}..{tr_end}) overlaps "
                    f"test window"
                )

    def test_purging_removes_samples_near_test_boundary(self):
        """Training samples whose windows straddle into the test block
        must be dropped from training (compared to vanilla KFold)."""
        labels = _make_labels(n=50, horizon=10)
        cv = PurgedKFold(n_splits=5, touch_times=labels["touch_time"], embargo_pct=0.0)

        for train_idx, test_idx in cv.split(X=np.zeros((50, 3))):
            # Training set should be smaller than naive (n - test_size)
            # because boundary samples are purged. In the first fold
            # (test=0..9), samples ending within test window are purged.
            if test_idx[0] > 0:  # not the leftmost fold
                # Some samples before the test block should be purged
                before_test = set(range(test_idx[0]))
                train_set = set(train_idx.tolist())
                purged = before_test - train_set
                assert len(purged) > 0, "Expected purging near test boundary"


class TestEmbargo:
    def test_embargo_excludes_post_test_samples(self):
        """Samples immediately after the test block are embargoed from training."""
        n = 100
        labels = _make_labels(n=n, horizon=2)  # short horizon so purge alone is small
        cv = PurgedKFold(
            n_splits=5,
            touch_times=labels["touch_time"],
            embargo_pct=0.05,  # 5 bars
        )

        for train_idx, test_idx in cv.split(X=np.zeros((n, 3))):
            test_end = test_idx.max()
            embargo_zone = set(range(test_end + 1, min(test_end + 1 + 5, n)))
            train_set = set(train_idx.tolist())
            assert len(embargo_zone & train_set) == 0, (
                f"Embargo violated: {embargo_zone & train_set}"
            )

    def test_zero_embargo_matches_pure_purge(self):
        """embargo_pct=0 should yield more training data than non-zero."""
        labels = _make_labels(n=100, horizon=5)

        cv_no_embargo = PurgedKFold(
            n_splits=5, touch_times=labels["touch_time"], embargo_pct=0.0
        )
        cv_with_embargo = PurgedKFold(
            n_splits=5, touch_times=labels["touch_time"], embargo_pct=0.10
        )

        trains_no = [len(tr) for tr, _ in cv_no_embargo.split(np.zeros((100, 3)))]
        trains_with = [len(tr) for tr, _ in cv_with_embargo.split(np.zeros((100, 3)))]

        assert sum(trains_no) > sum(trains_with)


class TestEdgeCases:
    def test_rejects_touch_times_with_missing_events(self):
        """touch_times must be a Series indexed by event time."""
        with pytest.raises((TypeError, ValueError)):
            PurgedKFold(
                n_splits=5,
                touch_times=pd.DataFrame({"foo": [1, 2, 3]}),
            )

    def test_rejects_n_splits_less_than_two(self):
        labels = _make_labels(n=100, horizon=5)
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1, touch_times=labels["touch_time"])

    def test_touch_must_be_later_than_event(self):
        """If any touch_time precedes its event time, the input is corrupt."""
        idx = pd.date_range("2024-01-02 09:30", periods=10, freq="1min")
        bad = pd.Series(idx[::-1].to_numpy(), index=idx)  # reversed
        with pytest.raises(ValueError, match="touch_time"):
            PurgedKFold(n_splits=3, touch_times=bad)
