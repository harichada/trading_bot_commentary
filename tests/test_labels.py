"""Tests for triple-barrier labeling (López de Prado, AFML Ch. 3).

Uses synthetic price paths where the correct barrier is knowable by hand.
"""
import numpy as np
import pandas as pd
import pytest

from ml.labels import apply_triple_barrier, compute_uniqueness


def _make_series(values: list[float], start: str = "2024-01-02 09:30") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="1min")
    return pd.Series(values, index=idx, dtype="float64")


class TestTripleBarrierBasic:
    def test_flat_prices_label_hold(self):
        """Prices never move → time barrier always wins → label 0."""
        prices = _make_series([100.0] * 100)
        atr = pd.Series(1.0, index=prices.index)
        events = prices.index[:5]

        out = apply_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=20
        )

        assert len(out) == 5
        assert (out["label"] == 0).all()
        assert (out["ret"].abs() < 1e-9).all()

    def test_rising_prices_label_buy(self):
        """Prices rise monotonically → upper barrier hit → label +1."""
        # Start at 100, step +0.5 each bar. ATR=1, pt=2 → upper at 102.
        # Hits upper at bar 4 (100 + 4×0.5 = 102).
        prices = _make_series([100.0 + 0.5 * i for i in range(50)])
        atr = pd.Series(1.0, index=prices.index)
        events = [prices.index[0]]

        out = apply_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=30
        )

        assert out.iloc[0]["label"] == 1
        assert out.iloc[0]["touch_time"] == prices.index[4]
        assert out.iloc[0]["ret"] == pytest.approx(0.02, abs=1e-6)

    def test_falling_prices_label_sell(self):
        """Prices fall monotonically → lower barrier hit → label -1."""
        # Start at 100, step -0.5 each bar. ATR=1, sl=1 → lower at 99.
        # Hits lower at bar 2 (100 - 2×0.5 = 99).
        prices = _make_series([100.0 - 0.5 * i for i in range(50)])
        atr = pd.Series(1.0, index=prices.index)
        events = [prices.index[0]]

        out = apply_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=30
        )

        assert out.iloc[0]["label"] == -1
        assert out.iloc[0]["touch_time"] == prices.index[2]
        assert out.iloc[0]["ret"] == pytest.approx(-0.01, abs=1e-6)

    def test_first_touch_wins_when_both_reachable(self):
        """Price dips then rallies → lower barrier hits first → label -1."""
        # Bars: 100, 99.5, 99.0 (hit lower), 101, 103 (would've hit upper)
        prices = _make_series([100.0, 99.5, 99.0, 101.0, 103.0] + [103.0] * 20)
        atr = pd.Series(1.0, index=prices.index)
        events = [prices.index[0]]

        out = apply_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=20
        )

        assert out.iloc[0]["label"] == -1
        assert out.iloc[0]["touch_time"] == prices.index[2]

    def test_time_barrier_when_neither_hit(self):
        """Price wobbles within barriers → time expires → label 0."""
        # Entry at 100, ATR=1, pt=2, sl=1 → barriers at 102 / 99.
        # Price stays in [99.5, 101.5] for 10 bars, then time expires.
        prices = _make_series([100.0, 100.5, 99.5, 101.0, 100.2, 101.5, 99.8,
                               100.0, 100.3, 99.9, 100.1])
        atr = pd.Series(1.0, index=prices.index)
        events = [prices.index[0]]

        out = apply_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=10
        )

        assert out.iloc[0]["label"] == 0
        # Time barrier = entry + max_holding bars → index 10
        assert out.iloc[0]["touch_time"] == prices.index[10]

    def test_atr_scaling_widens_barriers(self):
        """Higher ATR → wider barriers → harder to hit."""
        # Price rises 0.3/bar → at bar 20 is 106.
        # With ATR=1, upper=102, hit at bar 7 → +1.
        # With ATR=5, upper=110, never reached in 20 bars → HOLD.
        prices = _make_series([100.0 + 0.3 * i for i in range(30)])
        events = [prices.index[0]]

        out_tight = apply_triple_barrier(
            prices, events, pd.Series(1.0, index=prices.index),
            pt_mult=2.0, sl_mult=1.0, max_holding=20
        )
        out_wide = apply_triple_barrier(
            prices, events, pd.Series(5.0, index=prices.index),
            pt_mult=2.0, sl_mult=1.0, max_holding=20
        )

        assert out_tight.iloc[0]["label"] == 1
        assert out_wide.iloc[0]["label"] == 0

    def test_asymmetric_barriers(self):
        """sl_mult < pt_mult → easier to hit downside → tests skew."""
        # ATR=1, pt=3 (upper 103), sl=0.5 (lower 99.5).
        # Price: 100, 99.4 → lower hit at bar 1.
        prices = _make_series([100.0, 99.4] + [101.0] * 20)
        atr = pd.Series(1.0, index=prices.index)
        events = [prices.index[0]]

        out = apply_triple_barrier(
            prices, events, atr, pt_mult=3.0, sl_mult=0.5, max_holding=15
        )

        assert out.iloc[0]["label"] == -1

    def test_multiple_events_independent_labels(self):
        """Several events on the same series each get their own label."""
        # Prices: 100, 100, 100, 102.5, 100, 100, 97, 100, 100, 100, ...
        # Event at bar 0 → upper barrier at 102, hit at bar 3 → +1
        # Event at bar 5 → entry 100, ATR=1, hits lower at bar 6 (97) → -1
        prices = _make_series(
            [100.0, 100.0, 100.0, 102.5, 100.0, 100.0, 97.0]
            + [100.0] * 30
        )
        atr = pd.Series(1.0, index=prices.index)
        events = [prices.index[0], prices.index[5]]

        out = apply_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=20
        )

        assert out.iloc[0]["label"] == 1
        assert out.iloc[1]["label"] == -1


class TestTripleBarrierEdgeCases:
    def test_event_near_end_of_series_truncates_max_holding(self):
        """If fewer than max_holding bars remain, use what's available."""
        prices = _make_series([100.0] * 10)  # flat
        atr = pd.Series(1.0, index=prices.index)
        events = [prices.index[8]]  # only 2 bars after

        out = apply_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=20
        )

        assert out.iloc[0]["label"] == 0
        # Touch at last available bar
        assert out.iloc[0]["touch_time"] == prices.index[-1]

    def test_zero_atr_raises(self):
        """ATR of zero would produce degenerate barriers → reject."""
        prices = _make_series([100.0] * 10)
        atr = pd.Series(0.0, index=prices.index)

        with pytest.raises(ValueError, match="ATR"):
            apply_triple_barrier(
                prices, [prices.index[0]], atr,
                pt_mult=2.0, sl_mult=1.0, max_holding=5
            )

    def test_event_not_in_price_index_raises(self):
        """Events outside the price index are invalid."""
        prices = _make_series([100.0] * 10)
        atr = pd.Series(1.0, index=prices.index)
        bad_event = [pd.Timestamp("2099-01-01")]

        with pytest.raises(KeyError):
            apply_triple_barrier(
                prices, bad_event, atr,
                pt_mult=2.0, sl_mult=1.0, max_holding=5
            )


class TestUniqueness:
    def test_non_overlapping_samples_all_weight_one(self):
        """Samples whose windows don't overlap are fully unique."""
        idx = pd.date_range("2024-01-02 09:30", periods=100, freq="1min")
        # Two events 50 bars apart with max_holding=10 → no overlap
        labels = pd.DataFrame(
            {"touch_time": [idx[10], idx[60]]},
            index=[idx[0], idx[50]],
        )

        weights = compute_uniqueness(labels, idx)

        assert weights.iloc[0] == pytest.approx(1.0)
        assert weights.iloc[1] == pytest.approx(1.0)

    def test_fully_overlapping_samples_half_weight(self):
        """Two identical windows → each sample is 50% unique."""
        idx = pd.date_range("2024-01-02 09:30", periods=100, freq="1min")
        # Two events at same time, same touch → 100% overlap
        labels = pd.DataFrame(
            {"touch_time": [idx[10], idx[10]]},
            index=[idx[0], idx[0]],
        )

        weights = compute_uniqueness(labels, idx)

        assert weights.iloc[0] == pytest.approx(0.5)
        assert weights.iloc[1] == pytest.approx(0.5)

    def test_partial_overlap_weighted_correctly(self):
        """Half-overlapping windows should average ~0.75 uniqueness."""
        idx = pd.date_range("2024-01-02 09:30", periods=100, freq="1min")
        # Sample A: bars 0-10 (11 bars). Sample B: bars 5-15 (11 bars).
        # Overlap: bars 5-10 (6 bars shared).
        # For A: 5 unique bars (weight 1) + 6 shared bars (weight 0.5) over 11 bars
        #   → (5*1 + 6*0.5) / 11 = 8/11 ≈ 0.727
        labels = pd.DataFrame(
            {"touch_time": [idx[10], idx[15]]},
            index=[idx[0], idx[5]],
        )

        weights = compute_uniqueness(labels, idx)

        assert weights.iloc[0] == pytest.approx(8 / 11, abs=1e-6)
        assert weights.iloc[1] == pytest.approx(8 / 11, abs=1e-6)

    def test_groups_isolate_uniqueness_across_symbols(self):
        """Samples in different groups must NOT depress each other's uniqueness.

        Two symbols with identical event timestamps and overlapping windows:
        - Without groups: each sample sees 2x concurrency (both symbols share
          the timeline) → uniqueness ~0.5.
        - With groups: each symbol's samples are counted only against its own
          peers → uniqueness should be higher (ideally ~1.0 if within-symbol
          windows don't overlap).
        """
        idx = pd.date_range("2024-01-02 09:30", periods=100, freq="1min")
        # Two symbols, one event each at t=0, both resolving at t=10.
        labels = pd.DataFrame(
            {"touch_time": [idx[10], idx[10]]},
            index=[idx[0], idx[0]],
        )
        groups = pd.Series(["NVDA", "AAPL"])

        weights_no_group = compute_uniqueness(labels, idx)
        weights_grouped = compute_uniqueness(labels, idx, groups=groups)

        # Without groups: treated as two overlapping samples → ~0.5 each.
        assert weights_no_group.iloc[0] == pytest.approx(0.5)
        # With groups: each symbol's sample is alone in its group → 1.0.
        assert weights_grouped.iloc[0] == pytest.approx(1.0)
        assert weights_grouped.iloc[1] == pytest.approx(1.0)

    def test_groups_preserve_within_group_overlap(self):
        """Overlap within the same group must still reduce uniqueness."""
        idx = pd.date_range("2024-01-02 09:30", periods=100, freq="1min")
        # Four samples — two NVDA overlapping, two AAPL overlapping, NVDA pair
        # time-shares with AAPL pair (without groups → heavy overlap).
        labels = pd.DataFrame(
            {"touch_time": [idx[10], idx[10], idx[10], idx[10]]},
            index=[idx[0], idx[0], idx[0], idx[0]],
        )
        groups = pd.Series(["NVDA", "NVDA", "AAPL", "AAPL"])

        weights = compute_uniqueness(labels, idx, groups=groups)

        # Within each group: 2 identical windows → 0.5 each.
        assert weights.iloc[0] == pytest.approx(0.5)
        assert weights.iloc[1] == pytest.approx(0.5)
        assert weights.iloc[2] == pytest.approx(0.5)
        assert weights.iloc[3] == pytest.approx(0.5)

    def test_groups_length_mismatch_raises(self):
        idx = pd.date_range("2024-01-02 09:30", periods=10, freq="1min")
        labels = pd.DataFrame(
            {"touch_time": [idx[5], idx[5]]},
            index=[idx[0], idx[0]],
        )
        with pytest.raises(ValueError, match="groups"):
            compute_uniqueness(labels, idx, groups=pd.Series(["A"]))

    def test_groups_equivalent_to_no_groups_when_single_group(self):
        """With a single group, the grouped path must match the ungrouped
        path when given the same (sparse) per-group price index.

        Note: when callers pass a dense master index to the ungrouped path,
        per-bar weights differ because the grouped path only samples at
        event + touch timestamps. This is an intentional approximation
        documented in compute_uniqueness; for this equivalence test we
        pass both paths the same sparse index.
        """
        idx = pd.date_range("2024-01-02 09:30", periods=100, freq="1min")
        labels = pd.DataFrame(
            {"touch_time": [idx[10], idx[15]]},
            index=[idx[0], idx[5]],
        )
        groups = pd.Series(["NVDA", "NVDA"])
        sparse_idx = pd.DatetimeIndex(
            np.unique(np.concatenate([
                labels.index.to_numpy(),
                labels["touch_time"].to_numpy(),
            ]))
        ).sort_values()

        w_no_group = compute_uniqueness(labels, sparse_idx)
        w_grouped = compute_uniqueness(labels, idx, groups=groups)

        assert w_no_group.iloc[0] == pytest.approx(w_grouped.iloc[0], abs=1e-9)
        assert w_no_group.iloc[1] == pytest.approx(w_grouped.iloc[1], abs=1e-9)

    def test_weights_are_bounded_zero_to_one(self):
        """Sanity: weights must always be in (0, 1]."""
        idx = pd.date_range("2024-01-02 09:30", periods=1000, freq="1min")
        rng = np.random.default_rng(42)
        starts = sorted(rng.choice(900, size=50, replace=False))
        labels = pd.DataFrame(
            {"touch_time": [idx[s + 30] for s in starts]},
            index=[idx[s] for s in starts],
        )

        weights = compute_uniqueness(labels, idx)

        assert (weights > 0).all()
        assert (weights <= 1.0 + 1e-9).all()
