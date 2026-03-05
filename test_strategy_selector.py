"""Tests for Strategy Selector — condition-based strategy routing."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from gap_fade_strategies.strategy_selector import StrategySelector
from gap_fade_strategies.market_condition import MarketConditionDetector
from gap_fade_strategies.intraday_registry import IntradayStrategyRegistry

ET = ZoneInfo('US/Eastern')


def make_ts(hour, minute, second=0):
    return datetime(2024, 3, 1, hour, minute, second, tzinfo=ET)


def make_spy_data(ema9=451, ema20=449, ema50=446, rsi=60,
                  day_high=452, day_low=447, rsi_history=None, vwap=449):
    return {
        'ema9': ema9, 'ema20': ema20, 'ema50': ema50,
        'rsi': rsi, 'rsi_initialized': True,
        'day_high': day_high, 'day_low': day_low,
        'vwap': vwap,
        'rsi_history': rsi_history or [55, 56, 57, 58, 59],
        'bar_history': [],
    }


def make_selector():
    """Create selector with all available strategies."""
    strategies = {}
    for sid in IntradayStrategyRegistry.get_all_ids():
        strategies[sid] = IntradayStrategyRegistry.create_strategy(sid)
    return StrategySelector(strategies)


class TestSelectorCreation:
    def test_create_with_all_strategies(self):
        selector = make_selector()
        assert len(selector.active_strategy_ids) == 0  # Nothing active yet

    def test_initial_state(self):
        selector = make_selector()
        assert selector.current_condition is None
        assert selector.size_multiplier == 1.0


class TestTrendingUp:
    def test_trending_up_activates_momentum_pullback(self):
        selector = make_selector()
        spy_data = make_spy_data(ema9=451, ema20=449, ema50=446, rsi=60)

        active = selector.update(spy_data, make_ts(12, 0), force=True)

        assert 'momentum_surge' in selector.active_strategy_ids
        assert 'pullback_entry' in selector.active_strategy_ids
        assert selector.size_multiplier == 1.0

    def test_trending_up_orb_available_early(self):
        """ORB should be available during early session even in trending market."""
        selector = make_selector()
        spy_data = make_spy_data(ema9=451, ema20=449, ema50=446, rsi=60)

        selector.update(spy_data, make_ts(10, 0), force=True)

        assert 'orb_breakout' in selector.active_strategy_ids
        assert 'momentum_surge' in selector.active_strategy_ids

    def test_trending_up_no_orb_late(self):
        """ORB not automatically available after 11:30 in trending market."""
        selector = make_selector()
        spy_data = make_spy_data(ema9=451, ema20=449, ema50=446, rsi=60)

        selector.update(spy_data, make_ts(13, 0), force=True)

        assert 'orb_breakout' not in selector.active_strategy_ids


class TestTrendingDown:
    def test_trending_down_activates_momentum_pullback(self):
        selector = make_selector()
        spy_data = make_spy_data(ema9=445, ema20=448, ema50=451, rsi=40)

        selector.update(spy_data, make_ts(12, 0), force=True)

        assert 'momentum_surge' in selector.active_strategy_ids
        assert 'pullback_entry' in selector.active_strategy_ids


class TestChoppy:
    def test_choppy_activates_orb_only(self):
        selector = make_selector()
        rsi_hist = [55, 45, 60, 40, 58, 42, 57, 43, 56, 44]
        spy_data = make_spy_data(
            ema9=449, ema20=448, rsi=50,
            day_high=455, day_low=440,
            rsi_history=rsi_hist,
        )

        selector.update(spy_data, make_ts(12, 0), force=True)

        assert 'orb_breakout' in selector.active_strategy_ids
        assert selector.size_multiplier == 0.5  # Reduced size

    def test_choppy_reduces_size(self):
        selector = make_selector()
        rsi_hist = [55, 45, 60, 40, 58, 42, 57, 43, 56, 44]
        spy_data = make_spy_data(
            ema9=449, ema20=448, rsi=50,
            day_high=455, day_low=440,
            rsi_history=rsi_hist,
        )
        selector.update(spy_data, make_ts(12, 0), force=True)
        assert selector.size_multiplier == 0.5


class TestSideways:
    def test_sideways_activates_range(self):
        selector = make_selector()
        spy_data = make_spy_data(
            ema9=449, ema20=449, ema50=449, rsi=50,
            day_high=450, day_low=449,
            rsi_history=[50, 51, 50, 49, 50],
        )

        selector.update(spy_data, make_ts(12, 0), force=True)

        assert 'range_trade' in selector.active_strategy_ids


class TestORBAlwaysAvailable:
    def test_orb_in_morning(self):
        """ORB available before 11:30 regardless of condition."""
        selector = make_selector()
        spy_data = make_spy_data(ema9=449, ema20=449, rsi=50,
                                 day_high=450, day_low=449,
                                 rsi_history=[50, 51, 50, 49, 50])

        selector.update(spy_data, make_ts(10, 0), force=True)
        assert 'orb_breakout' in selector.active_strategy_ids

    def test_orb_boundary(self):
        """ORB available at exactly 11:30."""
        selector = make_selector()
        spy_data = make_spy_data(ema9=449, ema20=449, rsi=50,
                                 day_high=450, day_low=449,
                                 rsi_history=[50, 51, 50, 49, 50])

        selector.update(spy_data, make_ts(11, 30), force=True)
        assert 'orb_breakout' in selector.active_strategy_ids


class TestStatus:
    def test_get_status(self):
        selector = make_selector()
        spy_data = make_spy_data(rsi=60)
        selector.update(spy_data, make_ts(10, 0), force=True)

        status = selector.get_status()
        assert 'condition' in status
        assert 'active_strategies' in status
        assert 'size_multiplier' in status
        assert status['condition'] != 'unknown'


class TestReset:
    def test_reset_daily(self):
        selector = make_selector()
        spy_data = make_spy_data(rsi=60)
        selector.update(spy_data, make_ts(10, 0), force=True)
        assert len(selector.active_strategy_ids) > 0

        selector.reset_daily()
        assert len(selector.active_strategy_ids) == 0
        assert selector.current_condition is None
        assert selector.size_multiplier == 1.0


class TestCustomMapping:
    def test_custom_strategy_map(self):
        strategies = {}
        for sid in IntradayStrategyRegistry.get_all_ids():
            strategies[sid] = IntradayStrategyRegistry.create_strategy(sid)

        custom_map = {
            'trending_up': ['orb_breakout'],
            'trending_down': ['range_trade'],
            'choppy': [],
            'sideways': ['momentum_surge'],
        }
        selector = StrategySelector(strategies, strategy_map=custom_map)

        spy_data = make_spy_data(ema9=451, ema20=449, ema50=446, rsi=60)
        selector.update(spy_data, make_ts(13, 0), force=True)

        assert 'orb_breakout' in selector.active_strategy_ids
        assert 'momentum_surge' not in selector.active_strategy_ids
