"""Tests for side-aware (meta) labeling.

Meta-labeling assumes the *primary* model has already decided the side
(BUY/SELL). The meta-label is binary: 1 if the primary was right (price
reached profit target before stop), 0 otherwise (stop hit first, or time
expired, or price moved the wrong way).

López de Prado, AFML §3.6-3.7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.meta_labels import apply_meta_triple_barrier


def _series(values: list[float], start: str = "2024-01-02 09:30") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="1min")
    return pd.Series(values, index=idx, dtype="float64")


class TestSideAwareBarriers:
    def test_long_side_win_on_upper_touch(self):
        """Primary said BUY, price hit upper barrier → meta label 1 (win)."""
        prices = _series([100.0 + 0.5 * i for i in range(30)])  # rises
        atr = pd.Series(1.0, index=prices.index)
        events = pd.DataFrame(
            {"side": [1]},
            index=[prices.index[0]],
        )

        out = apply_meta_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=20
        )

        assert out.iloc[0]["bin"] == 1
        assert out.iloc[0]["ret"] > 0

    def test_long_side_loss_on_lower_touch(self):
        """Primary said BUY, price hit stop → meta label 0 (loss)."""
        prices = _series([100.0, 99.5, 99.0] + [101.0] * 20)  # dips then recovers
        atr = pd.Series(1.0, index=prices.index)
        events = pd.DataFrame(
            {"side": [1]},
            index=[prices.index[0]],
        )

        out = apply_meta_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=15
        )

        assert out.iloc[0]["bin"] == 0

    def test_short_side_win_on_lower_touch(self):
        """Primary said SELL, price dropped → meta label 1 (win)."""
        prices = _series([100.0 - 0.5 * i for i in range(30)])  # falls
        atr = pd.Series(1.0, index=prices.index)
        events = pd.DataFrame(
            {"side": [-1]},
            index=[prices.index[0]],
        )

        out = apply_meta_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=20
        )

        assert out.iloc[0]["bin"] == 1
        # For a short, realized PnL is the negative of buy-and-hold return.
        assert out.iloc[0]["ret"] > 0

    def test_short_side_loss_on_upper_touch(self):
        """Primary said SELL, price rose to stop → meta label 0 (loss).

        Upper barrier = 102 must be hit before lower barrier = 99.
        """
        prices = _series([100.0, 101.0, 102.5] + [95.0] * 20)
        atr = pd.Series(1.0, index=prices.index)
        events = pd.DataFrame(
            {"side": [-1]},
            index=[prices.index[0]],
        )

        out = apply_meta_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=15
        )

        assert out.iloc[0]["bin"] == 0

    def test_time_expiry_is_loss(self):
        """If neither barrier hits, the primary's bet didn't pay → bin=0."""
        prices = _series([100.0] * 30)
        atr = pd.Series(1.0, index=prices.index)
        events = pd.DataFrame(
            {"side": [1]},
            index=[prices.index[0]],
        )

        out = apply_meta_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=10
        )

        assert out.iloc[0]["bin"] == 0

    def test_multiple_mixed_sides(self):
        """BUY followed by SELL on different events produce independent labels."""
        # Bar 0 BUY: at 100, rises to 102 by bar 4 (above upper=102) → win
        # Bar 10 SELL: at 99.5, drops to 97 by bar 13 (below lower=97.5) → win
        prices = _series(
            [100.0, 100.5, 101.0, 101.5, 102.5]
            + [101.0] * 5
            + [99.5, 98.5, 97.5, 96.5]
            + [96.5] * 16
        )
        atr = pd.Series(1.0, index=prices.index)
        events = pd.DataFrame(
            {"side": [1, -1]},
            index=[prices.index[0], prices.index[10]],
        )

        out = apply_meta_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=15
        )

        assert out.iloc[0]["bin"] == 1  # long win
        assert out.iloc[1]["bin"] == 1  # short win

    def test_invalid_side_raises(self):
        """Side must be in {-1, 1}; 0 or other values are invalid."""
        prices = _series([100.0] * 20)
        atr = pd.Series(1.0, index=prices.index)
        events = pd.DataFrame({"side": [0]}, index=[prices.index[0]])

        with pytest.raises(ValueError, match="side"):
            apply_meta_triple_barrier(
                prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=5
            )

    def test_return_is_side_signed(self):
        """Return is reported from the strategy's perspective: long loss has
        negative ret; short loss (price went up against us) also negative."""
        # Long entry at 100, stopped at 99 → ret = -0.01
        prices = _series([100.0, 99.0] + [105.0] * 20)
        atr = pd.Series(1.0, index=prices.index)
        long_events = pd.DataFrame({"side": [1]}, index=[prices.index[0]])
        long_out = apply_meta_triple_barrier(
            prices, long_events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=15
        )
        assert long_out.iloc[0]["ret"] == pytest.approx(-0.01, abs=1e-6)

        # Short entry at 100, price rises to upper barrier 102 → stop.
        # raw ret = +0.02, strategy ret = -0.02 (short loses when price rises).
        prices2 = _series([100.0, 101.0, 102.0] + [95.0] * 20)
        short_events = pd.DataFrame({"side": [-1]}, index=[prices2.index[0]])
        short_out = apply_meta_triple_barrier(
            prices2, short_events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=15
        )
        assert short_out.iloc[0]["ret"] == pytest.approx(-0.02, abs=1e-6)

    def test_touch_time_preserved(self):
        """The barrier-touch timestamp must be reported for downstream CV."""
        prices = _series([100.0 + 0.5 * i for i in range(30)])
        atr = pd.Series(1.0, index=prices.index)
        events = pd.DataFrame({"side": [1]}, index=[prices.index[0]])

        out = apply_meta_triple_barrier(
            prices, events, atr, pt_mult=2.0, sl_mult=1.0, max_holding=20
        )

        assert out.iloc[0]["touch_time"] == prices.index[4]
