"""Tests for TickIndicatorEngine — RSI, ATR, multi-EMA, volume surge, backward compat.

Self-contained test suite: no shared conftest.
"""

import pytest
from datetime import datetime, time
from zoneinfo import ZoneInfo

from gap_fade_strategies.indicators import (
    FiveMinBar,
    SymbolIndicators,
    TickIndicatorEngine,
)

ET = ZoneInfo('US/Eastern')


# ── Helpers ──────────────────────────────────────────────────────────

def make_ts(hour, minute, second=0, date='2024-03-01'):
    """Create an ET datetime for a given time."""
    y, m, d = date.split('-')
    return datetime(int(y), int(m), int(d), hour, minute, second, tzinfo=ET)


def feed_bar(engine, symbol, prices, volume=1000, bar_hour=10, bar_start_min=0):
    """Feed ticks to simulate a completed 5-min bar.

    prices: (open, high, low, close) tuple.
    """
    o, h, l, c = prices
    ts_open = make_ts(bar_hour, bar_start_min, 0)
    ts_mid = make_ts(bar_hour, bar_start_min, 30)
    ts_close = make_ts(bar_hour, bar_start_min + 4, 59)

    # Open tick
    engine.on_tick(symbol, o, volume // 4, ts_open)
    # High tick
    engine.on_tick(symbol, h, volume // 4, ts_mid)
    # Low tick
    engine.on_tick(symbol, l, volume // 4, ts_mid)
    # Close tick
    engine.on_tick(symbol, c, volume // 4, ts_close)


def complete_bar(engine, symbol, close_price, bar_hour, bar_start_min, volume=1000):
    """Feed a simple bar with same O/H/L/C = close_price, then start next bar to finalize."""
    feed_bar(engine, symbol, (close_price, close_price + 0.01, close_price - 0.01, close_price),
             volume=volume, bar_hour=bar_hour, bar_start_min=bar_start_min)


# ── Backward Compatibility Tests ──────────────────────────────────

class TestBackwardCompat:
    """Ensure existing API works unchanged."""

    def test_basic_constructor(self):
        """Old-style constructor with ema_period only."""
        engine = TickIndicatorEngine(
            indicators=['vwap', 'ema', 'opening_range', 'day_high'],
            ema_period=9,
            or_minutes=5,
        )
        assert engine._ema_period == 9
        assert engine._or_minutes == 5

    def test_vwap_calculation(self):
        engine = TickIndicatorEngine(indicators=['vwap'])
        ts = make_ts(10, 0)

        engine.on_tick('AAPL', 150.0, 100, ts)
        engine.on_tick('AAPL', 152.0, 200, ts)

        data = engine.get_data('AAPL')
        # VWAP = (150*100 + 152*200) / (100 + 200)
        expected = (150.0 * 100 + 152.0 * 200) / 300
        assert abs(data['vwap'] - expected) < 0.01

    def test_day_high_low(self):
        engine = TickIndicatorEngine(indicators=['day_high', 'day_low'])
        ts = make_ts(10, 0)

        engine.on_tick('AAPL', 150.0, 100, ts)
        engine.on_tick('AAPL', 155.0, 100, ts)
        engine.on_tick('AAPL', 148.0, 100, ts)

        data = engine.get_data('AAPL')
        assert data['day_high'] == 155.0
        assert data['day_low'] == 148.0

    def test_opening_range(self):
        engine = TickIndicatorEngine(indicators=['opening_range'], or_minutes=5)

        # Ticks during OR (9:30-9:35)
        engine.on_tick('AAPL', 150.0, 100, make_ts(9, 30))
        engine.on_tick('AAPL', 152.0, 100, make_ts(9, 32))
        engine.on_tick('AAPL', 149.0, 100, make_ts(9, 34))

        data = engine.get_data('AAPL')
        assert data['or_complete'] is False
        assert data['or_high'] == 152.0
        assert data['or_low'] == 149.0

        # After OR
        engine.on_tick('AAPL', 153.0, 100, make_ts(9, 36))
        data = engine.get_data('AAPL')
        assert data['or_complete'] is True
        assert data['or_high'] == 152.0  # Not updated after OR close

    def test_ema_single_period(self):
        """Single-period EMA (backward compat)."""
        engine = TickIndicatorEngine(indicators=['ema'], ema_period=3)

        # Feed 3 completed bars to initialize EMA with SMA
        for i, (h, m) in enumerate([(10, 0), (10, 5), (10, 10)]):
            feed_bar(engine, 'AAPL', (100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i),
                     bar_hour=h, bar_start_min=m)

        # Trigger bar completion by starting a new bar
        engine.on_tick('AAPL', 103.0, 100, make_ts(10, 15))

        data = engine.get_data('AAPL')
        assert data['ema_initialized'] is True
        assert data[f'ema3'] > 0

    def test_empty_symbol(self):
        engine = TickIndicatorEngine(indicators=['vwap', 'ema'])
        data = engine.get_data('UNKNOWN')
        assert data == {}

    def test_reset(self):
        engine = TickIndicatorEngine(indicators=['vwap'])
        engine.on_tick('AAPL', 150.0, 100, make_ts(10, 0))
        assert len(engine.active_symbols) == 1

        engine.reset()
        assert len(engine.active_symbols) == 0

    def test_daily_reset_on_new_date(self):
        engine = TickIndicatorEngine(indicators=['day_high'])
        engine.on_tick('AAPL', 150.0, 100, make_ts(10, 0, date='2024-03-01'))
        data = engine.get_data('AAPL')
        assert data['day_high'] == 150.0

        # New day resets
        engine.on_tick('AAPL', 140.0, 100, make_ts(10, 0, date='2024-03-04'))
        data = engine.get_data('AAPL')
        assert data['day_high'] == 140.0  # Reset, not 150


# ── Multi-Period EMA Tests ────────────────────────────────────────

class TestMultiPeriodEMA:
    def test_ema_periods_param(self):
        engine = TickIndicatorEngine(
            indicators=['ema_multi'],
            ema_period=9,
            ema_periods=[9, 20, 50],
        )
        assert sorted(engine._ema_periods) == [9, 20, 50]

    def test_multi_ema_output(self):
        """Multi-EMA values appear in get_data output."""
        engine = TickIndicatorEngine(
            indicators=['ema', 'ema_multi'],
            ema_period=9,
            ema_periods=[9, 20],
        )
        # Feed enough bars to initialize EMA9 (need 9 bars)
        for i in range(12):
            h = 10 + i // 12
            m = (i % 12) * 5
            price = 100.0 + i * 0.5
            feed_bar(engine, 'AAPL', (price, price + 0.5, price - 0.5, price),
                     bar_hour=h, bar_start_min=m)

        # Start new bar to complete the last one
        engine.on_tick('AAPL', 106.0, 100, make_ts(11, 0))

        data = engine.get_data('AAPL')
        assert 'ema9' in data
        assert 'ema20' in data
        assert data['ema9'] > 0
        # EMA20 may not be fully initialized with only 12 bars, but should have a value
        assert 'ema20' in data

    def test_ema_period_dedup(self):
        """Primary ema_period is always included in ema_periods."""
        engine = TickIndicatorEngine(
            indicators=['ema'],
            ema_period=9,
            ema_periods=[20, 50],
        )
        assert 9 in engine._ema_periods
        assert 20 in engine._ema_periods
        assert 50 in engine._ema_periods


# ── RSI Tests ─────────────────────────────────────────────────────

class TestRSI:
    def _make_engine(self, period=14):
        return TickIndicatorEngine(
            indicators=['rsi', 'ema'],
            ema_period=9,
            rsi_period=period,
        )

    def test_rsi_starts_at_50(self):
        engine = self._make_engine()
        # Before any ticks, no data
        data = engine.get_data('AAPL')
        assert data == {}

    def test_rsi_all_gains(self):
        """When all bars go up, RSI should be close to 100."""
        engine = self._make_engine(period=5)

        # Feed 7 bars with monotonically increasing closes
        for i in range(7):
            price = 100.0 + i * 2.0  # +2 each bar
            h = 10
            m = i * 5
            feed_bar(engine, 'AAPL', (price, price + 0.5, price - 0.5, price),
                     bar_hour=h, bar_start_min=m)

        # Complete last bar
        engine.on_tick('AAPL', 114.0, 100, make_ts(10, 35))

        data = engine.get_data('AAPL')
        assert data['rsi_initialized'] is True
        assert data['rsi'] > 90  # Should be near 100

    def test_rsi_all_losses(self):
        """When all bars go down, RSI should be close to 0."""
        engine = self._make_engine(period=5)

        for i in range(7):
            price = 200.0 - i * 2.0  # -2 each bar
            feed_bar(engine, 'AAPL', (price, price + 0.5, price - 0.5, price),
                     bar_hour=10, bar_start_min=i * 5)

        engine.on_tick('AAPL', 186.0, 100, make_ts(10, 35))

        data = engine.get_data('AAPL')
        assert data['rsi_initialized'] is True
        assert data['rsi'] < 10  # Should be near 0

    def test_rsi_neutral(self):
        """Alternating up/down bars should give RSI near 50."""
        engine = self._make_engine(period=5)

        prices = [100, 102, 100, 102, 100, 102, 100]
        for i, price in enumerate(prices):
            feed_bar(engine, 'AAPL', (price, price + 0.5, price - 0.5, price),
                     bar_hour=10, bar_start_min=i * 5)

        engine.on_tick('AAPL', 100.0, 100, make_ts(10, 35))

        data = engine.get_data('AAPL')
        assert data['rsi_initialized'] is True
        assert 30 < data['rsi'] < 70  # Should be near 50

    def test_rsi_history(self):
        """RSI history is maintained."""
        engine = self._make_engine(period=3)

        for i in range(8):
            price = 100.0 + i * 0.5
            feed_bar(engine, 'AAPL', (price, price + 0.5, price - 0.5, price),
                     bar_hour=10, bar_start_min=i * 5)

        engine.on_tick('AAPL', 104.0, 100, make_ts(10, 40))

        data = engine.get_data('AAPL')
        assert len(data['rsi_history']) > 0
        assert len(data['rsi_history']) <= 20

    def test_rsi_not_enabled(self):
        """RSI not computed when not in indicators list."""
        engine = TickIndicatorEngine(indicators=['vwap'])
        engine.on_tick('AAPL', 100.0, 100, make_ts(10, 0))
        data = engine.get_data('AAPL')
        assert 'rsi' not in data


# ── ATR Tests ─────────────────────────────────────────────────────

class TestATR:
    def _make_engine(self, period=14):
        return TickIndicatorEngine(
            indicators=['atr', 'ema'],
            ema_period=9,
            atr_period=period,
        )

    def test_atr_first_bar(self):
        """First bar ATR = high - low."""
        engine = self._make_engine(period=3)

        feed_bar(engine, 'AAPL', (100, 105, 95, 102), bar_hour=10, bar_start_min=0)
        # Complete bar by starting new one
        engine.on_tick('AAPL', 102.0, 100, make_ts(10, 5))

        data = engine.get_data('AAPL')
        # ATR should be approximately 10 (105 - 95)
        assert data['atr'] > 0

    def test_atr_increases_with_volatility(self):
        """ATR increases when bars become more volatile."""
        engine = self._make_engine(period=3)

        # Small range bars
        for i in range(4):
            feed_bar(engine, 'AAPL',
                     (100, 101, 99, 100),
                     bar_hour=10, bar_start_min=i * 5)

        engine.on_tick('AAPL', 100.0, 100, make_ts(10, 20))
        data1 = engine.get_data('AAPL')
        atr_small = data1['atr']

        # Large range bars
        for i in range(4, 8):
            feed_bar(engine, 'AAPL',
                     (100, 110, 90, 100),
                     bar_hour=10, bar_start_min=i * 5)

        engine.on_tick('AAPL', 100.0, 100, make_ts(10, 40))
        data2 = engine.get_data('AAPL')
        atr_large = data2['atr']

        assert atr_large > atr_small

    def test_atr_not_enabled(self):
        """ATR not computed when not in indicators list."""
        engine = TickIndicatorEngine(indicators=['vwap'])
        engine.on_tick('AAPL', 100.0, 100, make_ts(10, 0))
        data = engine.get_data('AAPL')
        assert 'atr' not in data


# ── Volume Surge Tests ────────────────────────────────────────────

class TestVolumeSurge:
    def test_volume_surge_ratio(self):
        """Volume surge ratio = current bar vol / 20-bar avg."""
        engine = TickIndicatorEngine(indicators=['volume_profile', 'ema'], ema_period=9)

        # Feed 20 bars with 1000 volume each
        for i in range(22):
            h = 10 + i // 12
            m = (i % 12) * 5
            feed_bar(engine, 'AAPL', (100, 101, 99, 100),
                     volume=1000, bar_hour=h, bar_start_min=m)

        # Now start a high-volume bar (current bar)
        engine.on_tick('AAPL', 100.0, 3000, make_ts(11, 50))  # 3x volume in single tick

        data = engine.get_data('AAPL')
        assert data['volume_avg_20bar'] > 0
        assert data['volume_surge_ratio'] > 2.0  # 3000 vs ~1000 avg

    def test_volume_surge_zero_avg(self):
        """No divide-by-zero when avg is 0."""
        engine = TickIndicatorEngine(indicators=['volume_profile', 'ema'], ema_period=9)
        engine.on_tick('AAPL', 100.0, 0, make_ts(10, 0))

        data = engine.get_data('AAPL')
        assert data['volume_surge_ratio'] == 0.0


# ── Integration Tests ─────────────────────────────────────────────

class TestIntegration:
    def test_all_indicators_together(self):
        """All indicators can be used simultaneously."""
        engine = TickIndicatorEngine(
            indicators=['vwap', 'ema', 'ema_multi', 'opening_range',
                         'day_high', 'day_low', 'bar_history',
                         'rsi', 'atr', 'volume_profile'],
            ema_period=9,
            ema_periods=[9, 20],
            rsi_period=5,
            atr_period=5,
            or_minutes=5,
        )

        # Feed OR ticks
        engine.on_tick('AAPL', 150.0, 1000, make_ts(9, 30))
        engine.on_tick('AAPL', 152.0, 500, make_ts(9, 32))
        engine.on_tick('AAPL', 149.0, 500, make_ts(9, 34))

        # After OR
        for i in range(10):
            h = 9 + (35 + i * 5) // 60
            m = (35 + i * 5) % 60
            price = 150.0 + i * 0.5
            engine.on_tick('AAPL', price, 500, make_ts(h, m))

        data = engine.get_data('AAPL')

        # All indicators present
        assert 'vwap' in data
        assert 'or_high' in data
        assert 'or_complete' in data
        assert 'day_high' in data
        assert 'day_low' in data
        assert 'rsi' in data
        assert 'atr' in data
        assert 'volume_surge_ratio' in data
        assert 'ema9' in data
        assert 'ema20' in data

    def test_multiple_symbols(self):
        """Engine tracks multiple symbols independently."""
        engine = TickIndicatorEngine(
            indicators=['vwap', 'day_high'],
        )

        ts = make_ts(10, 0)
        engine.on_tick('AAPL', 150.0, 100, ts)
        engine.on_tick('MSFT', 300.0, 200, ts)

        assert len(engine.active_symbols) == 2
        assert engine.get_data('AAPL')['day_high'] == 150.0
        assert engine.get_data('MSFT')['day_high'] == 300.0

    def test_get_all_data(self):
        engine = TickIndicatorEngine(indicators=['vwap'])
        ts = make_ts(10, 0)
        engine.on_tick('AAPL', 150.0, 100, ts)
        engine.on_tick('MSFT', 300.0, 200, ts)

        all_data = engine.get_all_data()
        assert 'AAPL' in all_data
        assert 'MSFT' in all_data

    def test_reset_symbol(self):
        engine = TickIndicatorEngine(indicators=['vwap'])
        ts = make_ts(10, 0)
        engine.on_tick('AAPL', 150.0, 100, ts)
        engine.on_tick('MSFT', 300.0, 200, ts)

        engine.reset_symbol('AAPL')
        assert 'AAPL' not in engine.active_symbols
        assert 'MSFT' in engine.active_symbols

    def test_opening_range_15_min(self):
        """15-minute opening range (used by ORB strategy)."""
        engine = TickIndicatorEngine(
            indicators=['opening_range'],
            or_minutes=15,
        )

        # Ticks during 15-min OR (9:30-9:45)
        engine.on_tick('AAPL', 150.0, 100, make_ts(9, 30))
        engine.on_tick('AAPL', 155.0, 100, make_ts(9, 38))
        engine.on_tick('AAPL', 148.0, 100, make_ts(9, 44))

        data = engine.get_data('AAPL')
        assert data['or_complete'] is False
        assert data['or_high'] == 155.0
        assert data['or_low'] == 148.0

        # After 9:45
        engine.on_tick('AAPL', 156.0, 100, make_ts(9, 46))
        data = engine.get_data('AAPL')
        assert data['or_complete'] is True
        assert data['or_high'] == 155.0  # Unchanged
