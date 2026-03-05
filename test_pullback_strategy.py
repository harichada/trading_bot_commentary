"""Tests for Pullback Entry strategy."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from gap_fade_strategies.pullback_entry import PullbackEntryStrategy
from gap_fade_strategies.intraday_registry import IntradayStrategyRegistry

ET = ZoneInfo('US/Eastern')


def make_ts(hour, minute, second=0):
    return datetime(2024, 3, 1, hour, minute, second, tzinfo=ET)


def make_tick_data(ema9=151.0, ema20=150.0, ema50=148.0, rsi=48.0,
                   rsi_initialized=True, vwap=150.2, atr=1.5,
                   volume_surge=0.8, day_high=153.0, day_low=147.0):
    return {
        'ema9': ema9, 'ema20': ema20, 'ema50': ema50,
        'rsi': rsi, 'rsi_initialized': rsi_initialized,
        'vwap': vwap, 'atr': atr,
        'volume_surge_ratio': volume_surge,
        'day_high': day_high, 'day_low': day_low,
        'rsi_history': [],
    }


class TestPullbackRegistry:
    def test_registered(self):
        assert IntradayStrategyRegistry.has_strategy('pullback_entry')

    def test_create(self):
        strat = IntradayStrategyRegistry.create_strategy('pullback_entry')
        assert strat.name == 'Pullback Entry'
        assert strat.strategy_id == 'pullback_entry'


class TestPullbackScan:
    def test_long_pullback(self):
        """Long when EMA20 > EMA50, price near VWAP, RSI 40-55, low volume, bounce."""
        strat = PullbackEntryStrategy()
        tick_data = make_tick_data(
            ema9=150.3, ema20=150.0, ema50=148.0,
            rsi=48, vwap=150.2, volume_surge=0.8,
        )
        # Price near VWAP (150.2), above EMA9 (150.3 < price)
        snapshot = {'price': 150.5}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is not None
        assert setup.direction == 'long'
        assert setup.strategy_id == 'pullback_entry'

    def test_no_setup_wrong_trend(self):
        """No long when EMA20 < EMA50 (downtrend), no short when RSI out of range."""
        strat = PullbackEntryStrategy()
        tick_data = make_tick_data(ema20=148.0, ema50=150.0, rsi=30)  # RSI too low for short (needs 45-60)
        snapshot = {'price': 148.5}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None

    def test_no_setup_high_volume(self):
        """No setup when volume is still surging (selling not done)."""
        strat = PullbackEntryStrategy()
        tick_data = make_tick_data(volume_surge=2.5)
        snapshot = {'price': 150.5}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None

    def test_no_setup_rsi_too_high(self):
        """No setup when RSI > 55 (hasn't pulled back enough)."""
        strat = PullbackEntryStrategy()
        tick_data = make_tick_data(rsi=65)
        snapshot = {'price': 150.5}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None

    def test_no_setup_price_far_from_support(self):
        """No setup when price far from VWAP/EMA20."""
        strat = PullbackEntryStrategy()
        tick_data = make_tick_data(vwap=145.0, ema20=145.0)
        snapshot = {'price': 150.0}  # 3.4% above VWAP
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None

    def test_no_setup_no_bounce(self):
        """No setup when price below EMA9 (no bounce confirmation)."""
        strat = PullbackEntryStrategy()
        tick_data = make_tick_data(ema9=151.0)  # EMA9 above price
        snapshot = {'price': 150.5}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(11, 0))
        assert setup is None


class TestPullbackExit:
    def test_exit_below_ema20(self):
        strat = PullbackEntryStrategy()
        position = {'direction': 'long', 'entry_price': 150.5, 'stop_price': 147.5}
        tick_data = {'ema20': 150.0}
        exit_sig = strat.evaluate_exit(position, 149.0, tick_data, make_ts(11, 30))
        assert exit_sig is not None
        assert 'failed' in exit_sig.reason

    def test_no_exit_above_ema20(self):
        strat = PullbackEntryStrategy()
        position = {'direction': 'long', 'entry_price': 150.5, 'stop_price': 147.5}
        tick_data = {'ema20': 150.0}
        exit_sig = strat.evaluate_exit(position, 151.0, tick_data, make_ts(11, 30))
        assert exit_sig is None


class TestPullbackTrailing:
    def test_trail_after_1r(self):
        strat = PullbackEntryStrategy()
        position = {'direction': 'long', 'entry_price': 150.0, 'stop_price': 147.0}
        tick_data = {'ema9': 153.5}
        # Risk=3, 1R=153. Price 155 = 1.67R
        new_stop = strat.update_trailing_stop(position, 155.0, tick_data, make_ts(12, 0))
        assert new_stop is not None
        assert new_stop > 147.0

    def test_no_trail_before_1r(self):
        strat = PullbackEntryStrategy()
        position = {'direction': 'long', 'entry_price': 150.0, 'stop_price': 147.0}
        tick_data = {'ema9': 151.0}
        new_stop = strat.update_trailing_stop(position, 151.0, tick_data, make_ts(12, 0))
        assert new_stop is None


class TestPullbackConfig:
    def test_default_config(self):
        strat = PullbackEntryStrategy()
        assert strat.get_default_config()['target_rr'] == 2.0

    def test_active_window(self):
        assert PullbackEntryStrategy().get_active_window() == (10, 0, 15, 0)

    def test_ema_periods(self):
        assert 50 in PullbackEntryStrategy().get_ema_periods()

    def test_lifecycle(self):
        strat = PullbackEntryStrategy()
        strat._entered_symbols['AAPL'] = 1
        strat.on_day_start()
        assert len(strat._entered_symbols) == 0
