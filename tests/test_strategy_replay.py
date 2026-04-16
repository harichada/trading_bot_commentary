"""Tests for vectorized historical strategy replay.

The replay functions reproduce each primary strategy's entry rules over
a historical OHLCV DataFrame, returning (timestamp, side) pairs for every
bar that would have fired a signal. These timestamps become the events
to label and train meta-models on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.strategy_replay import replay_mean_reversion


def _ohlcv(closes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with Close=High=Low and ample volume."""
    idx = pd.date_range("2024-01-02 09:30", periods=len(closes), freq="1min")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1000.0] * len(closes),
        },
        index=idx,
    )


class TestMeanReversionReplay:
    def test_no_signal_on_flat_prices(self):
        """RSI≈50 and price in the middle of BB → no mean-reversion trigger."""
        closes = [100.0] * 80
        df = _ohlcv(closes)

        events = replay_mean_reversion(df)

        assert len(events) == 0

    def test_long_signal_fires_after_sharp_selloff(self):
        """Oversold + below lower BB + uptrend intact → BUY events emitted.

        Engineered so that at the signal bar: close > SMA50 (knife filter
        passes), RSI drops sharply (bounded by threshold), and close sits
        below the 20-bar Bollinger lower band. Uses a loose rsi_threshold
        because synthetic piecewise-linear prices need less pressure than
        real markets to cross RSI<X.
        """
        warmup = np.linspace(100, 160, 80).tolist()   # long uptrend; SMA50 rises
        dip = np.linspace(160, 148, 10).tolist()      # sharp but shallow drop
        hold = [148.0] * 10
        df = _ohlcv(warmup + dip + hold)

        events = replay_mean_reversion(df, rsi_threshold=45.0)

        longs = events[events["side"] == 1]
        assert len(longs) >= 1, "expected at least one BUY signal"

    def test_short_signal_fires_after_sharp_rally(self):
        """Overbought + above upper BB → SELL events emitted."""
        base = np.linspace(100, 100, 60).tolist()
        rally = np.linspace(100, 130, 20).tolist()          # rapid rise → overbought
        top = [130.0] * 10
        df = _ohlcv(base + rally + top)

        events = replay_mean_reversion(df)

        shorts = events[events["side"] == -1]
        assert len(shorts) >= 1, "expected at least one SELL signal"

    def test_events_have_side_column(self):
        """Output must be indexed by timestamp with a ``side`` column."""
        base = np.linspace(100, 120, 60).tolist()
        sell_off = np.linspace(120, 100, 20).tolist()
        df = _ohlcv(base + sell_off + [100.0] * 10)

        events = replay_mean_reversion(df)

        assert list(events.columns) == ["side"]
        assert events.index.is_monotonic_increasing

    def test_no_signal_when_insufficient_history(self):
        """Need at least 50 bars of warmup for SMA50; fewer → empty output."""
        df = _ohlcv([100.0] * 30)

        events = replay_mean_reversion(df)

        assert len(events) == 0

    def test_falling_knife_filter_blocks_long_in_downtrend(self):
        """RSI oversold BUT price below SMA50 with bearish MACD → no long."""
        # Long persistent downtrend; no bounce.
        closes = np.linspace(200, 100, 120).tolist()
        df = _ohlcv(closes)

        events = replay_mean_reversion(df)

        # Any longs here would be catching a falling knife — must be zero.
        longs = events[events["side"] == 1]
        assert len(longs) == 0

    def test_respects_rsi_threshold_param(self):
        """Stricter RSI threshold → fewer long signals."""
        base = np.linspace(100, 120, 60).tolist()
        sell_off = np.linspace(120, 95, 20).tolist()
        df = _ohlcv(base + sell_off + [95.0] * 10)

        loose = replay_mean_reversion(df, rsi_threshold=40)
        strict = replay_mean_reversion(df, rsi_threshold=25)

        loose_longs = (loose["side"] == 1).sum()
        strict_longs = (strict["side"] == 1).sum()
        assert loose_longs >= strict_longs
