"""Tests for the new ``triple_barrier_labels`` API (López de Prado AFML §3.4).

The legacy ``apply_triple_barrier`` is kept for the existing meta-labeling path
(``ml/meta_labels.py``). The new ``triple_barrier_labels`` adds:

* asymmetric ``pt_sl`` tuple (``(pt_mult, sl_mult)``),
* per-event ``vertical_barrier`` (Series or scalar bar count),
* ``min_ret`` filter that drops events whose realized |ret| < ``min_ret * atr / entry``
  (AFML p.47 — discards "barely-resolving" events that pollute the training distribution),
* explicit ``side`` column (long=+1, short=-1) so the same row can be reused
  by short strategies without sign-flipping outside.

Returned columns: ``[t1, ret, bin, side]`` indexed by event timestamp.
``bin`` is computed in the *side*-adjusted frame: a short trade whose price
falls is a winner (``bin=+1``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.labels import triple_barrier_labels


def _series(values: list[float], start: str = "2024-01-02 09:30") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="1min")
    return pd.Series(values, index=idx, dtype="float64")


def _events(idx: pd.DatetimeIndex, vertical: int) -> pd.DataFrame:
    """Build an events DataFrame with a uniform vertical barrier."""
    return pd.DataFrame({"vertical": [vertical] * len(idx)}, index=idx)


class TestTripleBarrierBasic:
    def test_returns_required_columns(self):
        close = _series([100.0] * 30)
        atr = pd.Series(1.0, index=close.index)
        events = _events(close.index[:3], vertical=10)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(2.0, 1.0), vertical_barrier=10, min_ret=0.0,
        )

        assert set(out.columns) == {"t1", "ret", "bin", "side"}
        assert out.index.equals(events.index)

    def test_upward_trend_hits_pt_bin_plus_one(self):
        # Step +0.5 each bar, ATR=1, pt=1, sl=1 → upper=101, hit at bar 2.
        close = _series([100.0 + 0.5 * i for i in range(50)])
        atr = pd.Series(1.0, index=close.index)
        events = _events(close.index[[0]], vertical=20)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(1.0, 1.0), vertical_barrier=20, min_ret=0.0,
        )

        assert out.iloc[0]["bin"] == 1
        assert out.iloc[0]["t1"] == close.index[2]
        assert out.iloc[0]["ret"] == pytest.approx(0.01, abs=1e-6)

    def test_downward_trend_hits_sl_bin_minus_one(self):
        close = _series([100.0 - 0.5 * i for i in range(50)])
        atr = pd.Series(1.0, index=close.index)
        events = _events(close.index[[0]], vertical=20)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(1.0, 1.0), vertical_barrier=20, min_ret=0.0,
        )

        assert out.iloc[0]["bin"] == -1

    def test_asymmetric_pt_sl_uses_each_multiple(self):
        # ATR=1, pt_sl=(2,1) → upper=102, lower=99.
        # Slow ramp: +0.6 each bar. Hits 102 at bar 4 (100+4*0.6=102.4).
        # Should NOT hit 99 first.
        close = _series([100.0 + 0.6 * i for i in range(20)])
        atr = pd.Series(1.0, index=close.index)
        events = _events(close.index[[0]], vertical=15)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(2.0, 1.0), vertical_barrier=15, min_ret=0.0,
        )

        assert out.iloc[0]["bin"] == 1
        # bar 4 is first close >= 102.
        assert out.iloc[0]["t1"] == close.index[4]

    def test_vertical_barrier_respected_per_event(self):
        # Flat — no barrier hit, time always wins.
        close = _series([100.0] * 50)
        atr = pd.Series(1.0, index=close.index)
        # Two events with different verticals (5 and 15 bars).
        events = pd.DataFrame(
            {"vertical": [5, 15]},
            index=close.index[[0, 20]],
        )

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(2.0, 1.0),
            vertical_barrier=events["vertical"],
            min_ret=0.0,
        )

        assert out.iloc[0]["t1"] == close.index[5]
        assert out.iloc[1]["t1"] == close.index[20 + 15]

    def test_min_ret_drops_low_magnitude_rows(self):
        # Flat — bin=0, |ret|=0 → dropped by min_ret>0.
        close = _series([100.0] * 30)
        atr = pd.Series(1.0, index=close.index)
        events = _events(close.index[:5], vertical=10)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(2.0, 1.0), vertical_barrier=10,
            min_ret=0.5,  # require |ret| >= 0.5 * atr/entry = 0.005
        )

        assert len(out) == 0, "all flat events should be dropped by min_ret"

    def test_min_ret_keeps_high_magnitude_rows(self):
        # Strong move — |ret|=2% well above min_ret * (atr/entry)=0.5%.
        close = _series([100.0 + 0.5 * i for i in range(50)])
        atr = pd.Series(1.0, index=close.index)
        events = _events(close.index[[0]], vertical=20)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(1.0, 1.0), vertical_barrier=20, min_ret=0.5,
        )

        assert len(out) == 1
        assert out.iloc[0]["bin"] == 1

    def test_short_side_inverts_bin(self):
        # Price falls — long bin=-1, short bin=+1.
        close = _series([100.0 - 0.5 * i for i in range(20)])
        atr = pd.Series(1.0, index=close.index)
        events = pd.DataFrame(
            {"vertical": [10], "side": [-1]},
            index=close.index[[0]],
        )

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(1.0, 1.0), vertical_barrier=10, min_ret=0.0,
        )

        assert out.iloc[0]["side"] == -1
        # Short hits its take-profit when price drops by sl_mult*atr (since short
        # uses the "stop" side as its profit). bin in side-adjusted frame = +1.
        assert out.iloc[0]["bin"] == 1

    def test_default_side_is_long(self):
        close = _series([100.0 + 0.5 * i for i in range(20)])
        atr = pd.Series(1.0, index=close.index)
        events = _events(close.index[[0]], vertical=10)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(1.0, 1.0), vertical_barrier=10, min_ret=0.0,
        )

        assert (out["side"] == 1).all()

    def test_zero_atr_event_dropped_not_raised(self):
        # New API filters degenerate-ATR events instead of raising — the
        # caller is now feeding many candidate events from CUSUM/stride and
        # cannot pre-clean each one. ``apply_triple_barrier`` still raises;
        # this is only the high-level wrapper's behavior.
        close = _series([100.0] * 30)
        atr = pd.Series([1.0] * 30, index=close.index)
        atr.iloc[0] = 0.0  # bad
        events = _events(close.index[[0, 5]], vertical=10)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(2.0, 1.0), vertical_barrier=10, min_ret=0.0,
        )

        assert len(out) == 1
        assert out.index[0] == close.index[5]

    def test_event_outside_price_index_dropped(self):
        close = _series([100.0] * 30)
        atr = pd.Series(1.0, index=close.index)
        good = close.index[[0]]
        bad = pd.DatetimeIndex([pd.Timestamp("2099-01-01")])
        events = pd.DataFrame(
            {"vertical": [10, 10]},
            index=good.append(bad),
        )

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(2.0, 1.0), vertical_barrier=10, min_ret=0.0,
        )

        assert len(out) == 1
        assert out.index[0] == close.index[0]

    def test_t1_never_exceeds_vertical(self):
        # Price would eventually hit upper at bar 10, but vertical=5.
        close = _series([100.0 + 0.2 * i for i in range(40)])
        atr = pd.Series(1.0, index=close.index)
        events = _events(close.index[[0]], vertical=5)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(1.0, 1.0), vertical_barrier=5, min_ret=0.0,
        )

        assert out.iloc[0]["t1"] <= close.index[5]

    def test_min_ret_uses_per_event_atr(self):
        # Two events: one with ATR=1 (loose threshold), one with ATR=10
        # (strict threshold). Same realized return → first survives, second
        # is dropped by min_ret.
        close = _series([100.0, 100.5] + [100.5] * 30)
        atr = pd.Series([1.0] * 32, index=close.index)
        atr.iloc[10] = 10.0
        # Event at 0 → quickly resolves to time-out at bar 5 with ret≈0.5%.
        # Event at 10 → same series shape from bar 10 onward (flat at 100.5)
        # so |ret|≈0; with ATR=10 the threshold is 5% — way above 0.
        events = _events(close.index[[0, 10]], vertical=5)

        out = triple_barrier_labels(
            events, close, atr,
            pt_sl=(5.0, 5.0),  # barriers wide enough not to fire
            vertical_barrier=5,
            min_ret=0.1,  # 0.1 * atr/entry
        )

        # Event 0: threshold = 0.1 * 1/100 = 0.001 → 0.5% > 0.001 → kept.
        # Event 10: threshold = 0.1 * 10/100 = 0.01  → 0% < 0.01 → dropped.
        assert close.index[0] in out.index
        assert close.index[10] not in out.index
