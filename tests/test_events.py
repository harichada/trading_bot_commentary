"""Tests for the CUSUM event filter (López de Prado AFML §2.5.2).

CUSUM accumulates returns and emits an event whenever the running sum
crosses ±h, then resets. It samples bars at *meaningful* moves rather
than at fixed time intervals — much better triple-barrier triggers than
"every 5th bar".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.events import cusum_filter


def _series(values: list[float], start: str = "2024-01-02 09:30") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="1min")
    return pd.Series(values, index=idx, dtype="float64")


class TestCUSUM:
    def test_flat_series_no_events(self):
        prices = _series([100.0] * 100)
        events = cusum_filter(prices, h=0.005)

        assert len(events) == 0

    def test_strong_upward_move_triggers_event(self):
        # Single 1% jump triggers when h=0.005.
        prices = _series([100.0] * 5 + [101.0] * 5)
        events = cusum_filter(prices, h=0.005)

        assert len(events) >= 1
        # First event must occur at or after the jump bar.
        assert events[0] >= prices.index[5]

    def test_resets_after_event(self):
        # Two 1% jumps separated by flat → two events expected.
        prices = _series([100.0] * 5 + [101.0] * 20 + [102.0] * 20)
        events = cusum_filter(prices, h=0.005)

        assert len(events) >= 2

    def test_threshold_controls_event_count(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.005, size=2000)
        prices = pd.Series(
            100 * np.exp(np.cumsum(rets)),
            index=pd.date_range("2024-01-02", periods=2000, freq="1min"),
        )

        loose = cusum_filter(prices, h=0.005)
        tight = cusum_filter(prices, h=0.05)

        assert len(loose) > len(tight)

    def test_returns_datetime_index(self):
        prices = _series([100.0] + [101.0] * 5)
        events = cusum_filter(prices, h=0.005)

        assert isinstance(events, pd.DatetimeIndex)

    def test_h_must_be_positive(self):
        prices = _series([100.0] * 10)
        with pytest.raises(ValueError):
            cusum_filter(prices, h=0.0)

    def test_symmetric_response_to_drops(self):
        # 1% drop also triggers.
        prices = _series([100.0] * 5 + [99.0] * 5)
        events = cusum_filter(prices, h=0.005)

        assert len(events) >= 1
