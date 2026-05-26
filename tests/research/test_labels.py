"""v-triple-barrier-labels-2026-05-13: tests for triple-barrier
labeling.

Boundary semantics matter most — getting the upper/lower comparison
direction or the time-barrier-vs-price-barrier order wrong silently
inverts the label, which is the most common kind of bug in this
class of code.
"""
from __future__ import annotations

import pytest


def _bars(rows):
    """Build a daily-bars DataFrame from a list of (date, o, h, l, c) tuples."""
    import pandas as pd

    df = pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close"],
    )
    df = df.set_index("date")
    df["volume"] = 1_000_000
    return df


class TestTripleBarrier:
    def test_upper_barrier_hit_yields_long_wins(self):
        from research.labels import BarrierLabel, label_triple_barrier

        # Entry at $100, ATR=2 → upper $103, lower $97 (k=1.5).
        # Bar 1: high=104, low=99 → upper hit first.
        rows = [
            ("2024-01-01", 100, 101, 99, 100),    # entry
            ("2024-01-02", 100, 104, 99, 103),    # hits upper
            ("2024-01-03", 103, 105, 102, 104),
        ]
        df = _bars(rows)
        label = label_triple_barrier(df, entry_idx=0, atr_value=2.0)
        assert label == BarrierLabel.LONG_WINS

    def test_lower_barrier_hit_yields_short_wins(self):
        from research.labels import BarrierLabel, label_triple_barrier

        rows = [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 99, 100, 96, 97),     # hits lower (97)
            ("2024-01-03", 97, 98, 96, 97),
        ]
        df = _bars(rows)
        label = label_triple_barrier(df, entry_idx=0, atr_value=2.0)
        assert label == BarrierLabel.SHORT_WINS

    def test_time_barrier_yields_neither(self):
        from research.labels import BarrierLabel, label_triple_barrier

        # All future bars stay inside the corridor → time barrier.
        rows = [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 101, 99, 100),
            ("2024-01-03", 100, 102, 98, 101),
            ("2024-01-04", 101, 102, 98, 100),
            ("2024-01-05", 100, 102, 99, 101),
            ("2024-01-06", 101, 102, 99, 100),
        ]
        df = _bars(rows)
        label = label_triple_barrier(
            df, entry_idx=0, atr_value=2.0, horizon_bars=5,
        )
        assert label == BarrierLabel.NEITHER

    def test_same_bar_both_barriers_tie_broken_by_close_vs_open(self):
        """Intrabar both hit. Up-close ⇒ LONG_WINS, down-close ⇒ SHORT_WINS.
        Eliminates the NaN-label edge case."""
        from research.labels import BarrierLabel, label_triple_barrier

        # Bar 1: open=100, high=105 (upper hit), low=95 (lower hit), close=102.
        # close > open ⇒ LONG_WINS.
        rows_up = [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 105, 95, 102),
        ]
        df_up = _bars(rows_up)
        assert label_triple_barrier(
            df_up, entry_idx=0, atr_value=2.0,
        ) == BarrierLabel.LONG_WINS

        # close < open ⇒ SHORT_WINS.
        rows_down = [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 105, 95, 98),
        ]
        df_down = _bars(rows_down)
        assert label_triple_barrier(
            df_down, entry_idx=0, atr_value=2.0,
        ) == BarrierLabel.SHORT_WINS

    def test_first_barrier_in_time_wins(self):
        """Lower hit on bar 1, upper hit on bar 2 ⇒ SHORT_WINS, not LONG."""
        from research.labels import BarrierLabel, label_triple_barrier

        rows = [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 102, 96, 99),    # lower hit (97)
            ("2024-01-03", 99, 110, 99, 108),    # upper hit but LATER
        ]
        df = _bars(rows)
        label = label_triple_barrier(df, entry_idx=0, atr_value=2.0)
        assert label == BarrierLabel.SHORT_WINS

    def test_invalid_atr_raises(self):
        from research.labels import label_triple_barrier

        rows = [("2024-01-01", 100, 101, 99, 100), ("2024-01-02", 100, 101, 99, 100)]
        df = _bars(rows)
        with pytest.raises(ValueError):
            label_triple_barrier(df, entry_idx=0, atr_value=0)
        with pytest.raises(ValueError):
            label_triple_barrier(df, entry_idx=0, atr_value=-1.0)

    def test_last_bar_entry_returns_neither(self):
        """No future bars to evaluate; label is NEITHER, not an error."""
        from research.labels import BarrierLabel, label_triple_barrier

        rows = [("2024-01-01", 100, 101, 99, 100)]
        df = _bars(rows)
        label = label_triple_barrier(df, entry_idx=0, atr_value=2.0)
        assert label == BarrierLabel.NEITHER


class TestLabelDataset:
    def test_dataset_labels_align_with_index(self):
        from research.labels import label_dataset
        import pandas as pd

        rows = [
            ("2024-01-01", 100, 101, 99, 100),
            ("2024-01-02", 100, 105, 99, 104),
            ("2024-01-03", 104, 106, 103, 105),
            ("2024-01-04", 105, 107, 102, 104),
            ("2024-01-05", 104, 105, 101, 102),
            ("2024-01-06", 102, 103, 99, 100),
            ("2024-01-07", 100, 101, 95, 96),
            ("2024-01-08", 96, 98, 94, 97),
        ]
        df = _bars(rows)
        atr = pd.Series([2.0] * len(df), index=df.index)
        labels = label_dataset(df, atr, k=1.5, horizon_bars=5)
        assert len(labels) == len(df)
        assert (labels.index == df.index).all()
