"""Tests for Range Trade strategy."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from gap_fade_strategies.range_trade import RangeTradeStrategy
from gap_fade_strategies.intraday_registry import IntradayStrategyRegistry

ET = ZoneInfo('US/Eastern')


def make_ts(hour, minute, second=0):
    return datetime(2024, 3, 1, hour, minute, second, tzinfo=ET)


def make_tick_data(day_high=152.0, day_low=148.0, rsi=35.0,
                   rsi_initialized=True, vwap=150.0,
                   volume_surge=0.8, ema9=148.3, atr=1.0, bar_count=10):
    return {
        'day_high': day_high, 'day_low': day_low,
        'rsi': rsi, 'rsi_initialized': rsi_initialized,
        'vwap': vwap, 'atr': atr,
        'volume_surge_ratio': volume_surge,
        'ema9': ema9, 'bar_count': bar_count,
        'rsi_history': [],
    }


class TestRangeRegistry:
    def test_registered(self):
        assert IntradayStrategyRegistry.has_strategy('range_trade')

    def test_create(self):
        strat = IntradayStrategyRegistry.create_strategy('range_trade')
        assert strat.name == 'Range Trade'
        assert strat.strategy_id == 'range_trade'


class TestRangeScan:
    def test_long_at_support(self):
        """Long when price near day low, RSI < 40, low volume, bounce."""
        strat = RangeTradeStrategy()
        tick_data = make_tick_data(
            day_high=152.0, day_low=148.0,
            rsi=32, volume_surge=0.7,
            ema9=148.2,  # Price above EMA9 (bounce)
            bar_count=10,
        )
        # Price near support (within 0.3% of day_low=148.0)
        snapshot = {'price': 148.3}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is not None
        assert setup.direction == 'long'
        assert setup.target_price > setup.entry_price  # Target at resistance

    def test_short_at_resistance(self):
        """Short when price near day high, RSI > 60, low volume."""
        strat = RangeTradeStrategy()
        tick_data = make_tick_data(
            day_high=152.0, day_low=148.0,
            rsi=68, volume_surge=0.8,
            ema9=151.8,  # Price below EMA9 (rejection)
            bar_count=10,
        )
        snapshot = {'price': 151.7}  # Near resistance

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is not None
        assert setup.direction == 'short'

    def test_no_setup_range_too_narrow(self):
        """No setup when range < min_range_pct."""
        strat = RangeTradeStrategy()
        tick_data = make_tick_data(day_high=150.2, day_low=150.0)  # 0.13%
        snapshot = {'price': 150.05}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None

    def test_no_setup_range_too_wide(self):
        """No setup when range > max_range_pct."""
        strat = RangeTradeStrategy()
        tick_data = make_tick_data(day_high=160.0, day_low=140.0)  # ~13%
        snapshot = {'price': 140.3}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None

    def test_no_setup_not_at_boundary(self):
        """No setup when price is in middle of range."""
        strat = RangeTradeStrategy()
        tick_data = make_tick_data()
        snapshot = {'price': 150.0}  # Midpoint
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None

    def test_no_setup_not_enough_bars(self):
        """No setup before range is established."""
        strat = RangeTradeStrategy()
        tick_data = make_tick_data(bar_count=3)  # < 6 bars
        snapshot = {'price': 148.3}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None

    def test_no_setup_high_volume(self):
        """No setup when volume surging (potential breakout)."""
        strat = RangeTradeStrategy()
        tick_data = make_tick_data(volume_surge=2.5)
        snapshot = {'price': 148.3}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None

    def test_no_setup_rsi_not_extreme(self):
        """No long at support when RSI > 40."""
        strat = RangeTradeStrategy()
        tick_data = make_tick_data(rsi=50)  # Not oversold
        snapshot = {'price': 148.3}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None


class TestRangeExit:
    def test_exit_support_break(self):
        strat = RangeTradeStrategy()
        position = {'direction': 'long', 'entry_price': 148.3, 'stop_price': 147.3}
        tick_data = {'day_high': 152.0, 'day_low': 148.0}
        # Price 0.8% below day_low — range broken
        exit_sig = strat.evaluate_exit(position, 146.8, tick_data, make_ts(12, 0))
        assert exit_sig is not None
        assert 'broken' in exit_sig.reason

    def test_no_exit_inside_range(self):
        strat = RangeTradeStrategy()
        position = {'direction': 'long', 'entry_price': 148.3, 'stop_price': 147.3}
        tick_data = {'day_high': 152.0, 'day_low': 148.0}
        exit_sig = strat.evaluate_exit(position, 150.0, tick_data, make_ts(12, 0))
        assert exit_sig is None


class TestRangeTrailing:
    def test_tighten_at_midpoint(self):
        strat = RangeTradeStrategy()
        position = {
            'direction': 'long', 'entry_price': 148.3,
            'stop_price': 147.3,
        }
        tick_data = {'day_high': 152.0, 'day_low': 148.0}
        # Price above midpoint (150.0)
        new_stop = strat.update_trailing_stop(position, 151.0, tick_data, make_ts(12, 0))
        assert new_stop is not None
        assert new_stop > 147.3

    def test_no_tighten_below_mid(self):
        strat = RangeTradeStrategy()
        position = {
            'direction': 'long', 'entry_price': 148.3,
            'stop_price': 147.3,
        }
        tick_data = {'day_high': 152.0, 'day_low': 148.0}
        new_stop = strat.update_trailing_stop(position, 149.0, tick_data, make_ts(12, 0))
        assert new_stop is None


class TestRangeConfig:
    def test_default_config(self):
        strat = RangeTradeStrategy()
        cfg = strat.get_default_config()
        assert cfg['min_range_pct'] == 0.005
        assert cfg['max_entries_per_session'] == 3

    def test_active_window(self):
        assert RangeTradeStrategy().get_active_window() == (10, 30, 15, 0)

    def test_ema_periods(self):
        assert 9 in RangeTradeStrategy().get_ema_periods()
