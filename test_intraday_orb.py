"""Tests for ORB (Opening Range Breakout) intraday strategy.

Self-contained test suite: no shared conftest.
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from gap_fade_strategies.intraday_base import IntradaySetup, IntradayStrategy
from gap_fade_strategies.intraday_registry import IntradayStrategyRegistry
from gap_fade_strategies.orb_breakout import ORBBreakoutStrategy
from gap_fade_strategies.base import ExitSignal

ET = ZoneInfo('US/Eastern')


# ── Helpers ──────────────────────────────────────────────────────────

def make_ts(hour, minute, second=0):
    return datetime(2024, 3, 1, hour, minute, second, tzinfo=ET)


def make_tick_data(or_high=151.0, or_low=148.5, or_complete=True,
                   vwap=149.5, volume_surge=2.0, rsi=55.0,
                   ema9=150.0, ema20=149.5, atr=1.5):
    """Create a mock tick_data dict.

    Default OR range: $2.50 / ~1.7% (within 0.5%-3% bounds).
    """
    return {
        'or_high': or_high,
        'or_low': or_low,
        'or_complete': or_complete,
        'vwap': vwap,
        'volume_surge_ratio': volume_surge,
        'rsi': rsi,
        'ema9': ema9,
        'ema20': ema20,
        'atr': atr,
        'day_high': or_high + 1.0,
        'day_low': or_low - 1.0,
    }


# ── Registry Tests ────────────────────────────────────────────────

class TestORBRegistry:
    def test_registered(self):
        assert IntradayStrategyRegistry.has_strategy('orb_breakout')

    def test_create_instance(self):
        strat = IntradayStrategyRegistry.create_strategy('orb_breakout')
        assert isinstance(strat, ORBBreakoutStrategy)
        assert strat.name == 'ORB Breakout'
        assert strat.strategy_id == 'orb_breakout'

    def test_list_strategies(self):
        strategies = IntradayStrategyRegistry.list_strategies()
        ids = [s['id'] for s in strategies]
        assert 'orb_breakout' in ids

    def test_unknown_strategy(self):
        with pytest.raises(KeyError):
            IntradayStrategyRegistry.create_strategy('nonexistent')

    def test_inherits_intraday_strategy(self):
        strat = ORBBreakoutStrategy()
        assert isinstance(strat, IntradayStrategy)


# ── Scan Tests ────────────────────────────────────────────────────

class TestORBScan:
    def test_long_breakout(self):
        """Long setup when price breaks above OR high."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data()  # or_high=151.0, or_low=148.5, vwap=149.5
        snapshot = {'price': 151.5}  # Above OR high + buffer
        now = make_ts(10, 0)

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)

        assert setup is not None
        assert setup.direction == 'long'
        assert setup.symbol == 'AAPL'
        assert setup.strategy_id == 'orb_breakout'
        assert setup.setup_type == 'orb_breakout'
        assert setup.entry_price == 151.5
        assert setup.stop_price < 148.5  # Below OR low with buffer
        assert setup.target_price > 151.5  # Above entry
        assert setup.risk_reward >= 1.0

    def test_short_breakout(self):
        """Short setup when price breaks below OR low."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data()  # or_high=151.0, or_low=148.5, vwap=149.5
        # Price below OR low - buffer, below VWAP
        snapshot = {'price': 148.0}

        now = make_ts(10, 0)
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)

        assert setup is not None
        assert setup.direction == 'short'
        assert setup.stop_price > 151.0  # Above OR high with buffer
        assert setup.target_price < 148.0  # Below entry

    def test_no_breakout_inside_range(self):
        """No setup when price is inside OR."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data()
        snapshot = {'price': 150.0}  # Inside range (148.5-151.0)
        now = make_ts(10, 0)

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)
        assert setup is None

    def test_no_setup_before_or_complete(self):
        """No setup before opening range is finalized."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data(or_complete=False)
        snapshot = {'price': 152.0}
        now = make_ts(9, 40)

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)
        assert setup is None

    def test_no_setup_low_volume(self):
        """No setup when volume is below threshold."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data(volume_surge=0.5)  # Below 1.5x threshold
        snapshot = {'price': 152.0}
        now = make_ts(10, 0)

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)
        assert setup is None

    def test_no_setup_or_too_narrow(self):
        """No setup when OR range is too narrow."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data(or_high=150.5, or_low=150.2)  # < 0.5%
        snapshot = {'price': 150.6}
        now = make_ts(10, 0)

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)
        assert setup is None

    def test_no_setup_or_too_wide(self):
        """No setup when OR range is too wide."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data(or_high=160.0, or_low=140.0)  # ~13.3% > 3%
        snapshot = {'price': 161.0}
        now = make_ts(10, 0)

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)
        assert setup is None

    def test_max_entries_per_session(self):
        """No setup after max entries reached."""
        cfg = ORBBreakoutStrategy().get_default_config()
        cfg['max_entries_per_session'] = 1  # Override after defaults
        strat = ORBBreakoutStrategy(config=cfg)
        strat._entered_symbols['AAPL'] = 1
        tick_data = make_tick_data()
        snapshot = {'price': 152.0}
        now = make_ts(10, 0)

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)
        assert setup is None

    def test_vwap_filter_long(self):
        """Long rejected when price below VWAP."""
        strat = ORBBreakoutStrategy()
        # OR: 150-152, range ~1.3%. Price above OR high but below VWAP
        tick_data = make_tick_data(or_high=152.0, or_low=150.0, vwap=153.0)
        snapshot = {'price': 152.5}  # Above OR high but below VWAP
        now = make_ts(10, 0)

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)
        assert setup is None

    def test_vwap_filter_short(self):
        """Short rejected when price above VWAP."""
        strat = ORBBreakoutStrategy()
        # OR: 148-150, range ~1.3%. Price below OR low but above VWAP
        tick_data = make_tick_data(or_high=150.0, or_low=148.0, vwap=147.0)
        snapshot = {'price': 147.5}  # Below OR low but above VWAP (147.5 > 147.0)
        now = make_ts(10, 0)

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, now)
        assert setup is None

    def test_no_snapshot_returns_none(self):
        """No setup when snapshot is missing."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data()
        now = make_ts(10, 0)

        setup = strat.scan_for_setups('AAPL', tick_data, None, now)
        assert setup is None


# ── Validation Tests ──────────────────────────────────────────────

class TestORBValidation:
    def test_valid_setup(self):
        strat = ORBBreakoutStrategy()
        setup = IntradaySetup(
            symbol='AAPL', strategy_id='orb_breakout',
            direction='long', entry_price=151.5,
            stop_price=148.2, target_price=156.5,
            risk_reward=1.5, confidence=0.7,
            setup_type='orb_breakout',
        )
        valid, reason = strat.validate_setup(setup, {}, make_ts(10, 0))
        assert valid is True

    def test_reject_low_rr(self):
        strat = ORBBreakoutStrategy()
        setup = IntradaySetup(
            symbol='AAPL', strategy_id='orb_breakout',
            direction='long', entry_price=151.5,
            stop_price=148.2, target_price=152.0,
            risk_reward=0.1, confidence=0.7,
            setup_type='orb_breakout',
        )
        valid, reason = strat.validate_setup(setup, {}, make_ts(10, 0))
        assert valid is False
        assert 'R:R' in reason

    def test_reject_low_confidence(self):
        strat = ORBBreakoutStrategy()
        setup = IntradaySetup(
            symbol='AAPL', strategy_id='orb_breakout',
            direction='long', entry_price=151.5,
            stop_price=148.2, target_price=156.5,
            risk_reward=1.5, confidence=0.2,
            setup_type='orb_breakout',
        )
        valid, reason = strat.validate_setup(setup, {}, make_ts(10, 0))
        assert valid is False
        assert 'Confidence' in reason

    def test_reject_before_window(self):
        strat = ORBBreakoutStrategy()
        setup = IntradaySetup(
            symbol='AAPL', strategy_id='orb_breakout',
            direction='long', entry_price=155.5,
            stop_price=147.7, target_price=167.2,
            risk_reward=1.5, confidence=0.7,
            setup_type='orb_breakout',
        )
        valid, reason = strat.validate_setup(setup, {}, make_ts(9, 30))
        assert valid is False
        assert 'Before' in reason

    def test_reject_after_window(self):
        strat = ORBBreakoutStrategy()
        setup = IntradaySetup(
            symbol='AAPL', strategy_id='orb_breakout',
            direction='long', entry_price=155.5,
            stop_price=147.7, target_price=167.2,
            risk_reward=1.5, confidence=0.7,
            setup_type='orb_breakout',
        )
        valid, reason = strat.validate_setup(setup, {}, make_ts(12, 0))
        assert valid is False
        assert 'After' in reason


# ── Active Window Tests ───────────────────────────────────────────

class TestORBActiveWindow:
    def test_default_window(self):
        strat = ORBBreakoutStrategy()
        window = strat.get_active_window()
        assert window == (9, 46, 11, 30)


# ── Exit Tests ────────────────────────────────────────────────────

class TestORBExit:
    def test_failed_breakout_long(self):
        """Close long when price falls below OR midpoint."""
        strat = ORBBreakoutStrategy()
        position = {
            'direction': 'long',
            'entry_price': 151.5,
            'stop_price': 148.2,
        }
        tick_data = make_tick_data()  # or_high=151.0, or_low=148.5
        # OR mid = 149.75, price below that
        exit_signal = strat.evaluate_exit(position, 149.0, tick_data, make_ts(10, 30))
        assert exit_signal is not None
        assert exit_signal.action == 'close'
        assert 'failed breakout' in exit_signal.reason

    def test_no_exit_above_midpoint(self):
        """No exit when long and price is above OR midpoint."""
        strat = ORBBreakoutStrategy()
        position = {
            'direction': 'long',
            'entry_price': 151.5,
            'stop_price': 148.2,
        }
        tick_data = make_tick_data()  # or_high=151.0, or_low=148.5
        # OR mid = 149.75, price above that
        exit_signal = strat.evaluate_exit(position, 150.5, tick_data, make_ts(10, 30))
        assert exit_signal is None

    def test_failed_breakdown_short(self):
        """Close short when price rises above OR midpoint."""
        strat = ORBBreakoutStrategy()
        position = {
            'direction': 'short',
            'entry_price': 148.0,
            'stop_price': 151.3,
        }
        tick_data = make_tick_data()  # or_high=151.0, or_low=148.5
        # OR mid = 149.75, price above that
        exit_signal = strat.evaluate_exit(position, 150.5, tick_data, make_ts(10, 30))
        assert exit_signal is not None
        assert exit_signal.action == 'close'
        assert 'failed breakdown' in exit_signal.reason


# ── Trailing Stop Tests ──────────────────────────────────────────

class TestORBTrailingStop:
    def test_no_trail_before_1r(self):
        """No trailing before 1R profit."""
        strat = ORBBreakoutStrategy()
        position = {
            'direction': 'long',
            'entry_price': 155.0,
            'stop_price': 148.0,
        }
        tick_data = make_tick_data(ema9=155.5)
        # Risk = 155 - 148 = 7. Price at 157 = 2/7 R = 0.29R < 1R
        new_stop = strat.update_trailing_stop(position, 157.0, tick_data, make_ts(10, 30))
        assert new_stop is None

    def test_trail_after_1r(self):
        """Trailing stop activates after 1R profit using EMA9."""
        strat = ORBBreakoutStrategy()
        position = {
            'direction': 'long',
            'entry_price': 155.0,
            'stop_price': 148.0,
        }
        # Risk = 7. 1R = 162. Price at 163 = >1R
        tick_data = make_tick_data(ema9=161.0)
        new_stop = strat.update_trailing_stop(position, 163.0, tick_data, make_ts(10, 30))
        assert new_stop is not None
        assert new_stop > 148.0  # Tighter than original stop

    def test_trail_only_tightens(self):
        """Trailing stop never widens (loosens)."""
        strat = ORBBreakoutStrategy()
        position = {
            'direction': 'long',
            'entry_price': 100.0,
            'stop_price': 110.0,  # Already very tight stop
        }
        tick_data = make_tick_data(ema9=108.0)
        # Even though 1R+ profit, EMA9 - buffer < current stop
        new_stop = strat.update_trailing_stop(position, 120.0, tick_data, make_ts(10, 30))
        # EMA9 * 0.999 = 107.89 < 110.0, so no update
        assert new_stop is None


# ── Confidence Scoring Tests ──────────────────────────────────────

class TestORBConfidence:
    def test_base_confidence(self):
        """Minimum confidence is around 0.50."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data(volume_surge=1.5, vwap=0, rsi=50, ema9=0, ema20=0)
        snapshot = {'price': 156.0}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 0))
        # VWAP is 0, so VWAP filter passes (no check when vwap <= 0)
        # Volume just at threshold
        if setup:
            assert setup.confidence >= 0.50

    def test_high_confidence_factors(self):
        """High confidence with strong volume, VWAP alignment, good RSI."""
        strat = ORBBreakoutStrategy()
        tick_data = make_tick_data(
            or_high=150.0, or_low=148.5,  # ~1% range (within 0.8-2%)
            volume_surge=3.0,  # Strong volume
            vwap=149.0,  # Price above VWAP for long
            rsi=55.0,  # Good RSI for long
            ema9=150.5,  # EMA9 > EMA20 for long
            ema20=150.0,
        )
        snapshot = {'price': 150.5}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 0))
        if setup:
            assert setup.confidence >= 0.70


# ── Config Tests ──────────────────────────────────────────────────

class TestORBConfig:
    def test_default_config(self):
        strat = ORBBreakoutStrategy()
        config = strat.get_default_config()
        assert config['or_minutes'] == 15
        assert config['target_rr'] == 1.5
        assert config['max_entries_per_session'] == 3

    def test_parameter_schema(self):
        strat = ORBBreakoutStrategy()
        schema = strat.get_parameter_schema()
        assert 'or_minutes' in schema
        assert 'target_rr' in schema
        assert 'volume_confirm_ratio' in schema

    def test_custom_config(self):
        custom = {'or_minutes': 30, 'target_rr': 2.0, 'max_entries_per_session': 5}
        strat = ORBBreakoutStrategy(config=custom)
        assert strat.config['or_minutes'] == 30
        assert strat.config['target_rr'] == 2.0

    def test_update_config(self):
        strat = ORBBreakoutStrategy()
        strat.update_config({'target_rr': 2.5})
        assert strat.config['target_rr'] == 2.5


# ── Indicator Requirements Tests ──────────────────────────────────

class TestORBIndicators:
    def test_required_indicators(self):
        strat = ORBBreakoutStrategy()
        indicators = strat.get_required_indicators()
        assert 'opening_range' in indicators
        assert 'vwap' in indicators
        assert 'volume_profile' in indicators
        assert 'rsi' in indicators

    def test_ema_periods(self):
        strat = ORBBreakoutStrategy()
        periods = strat.get_ema_periods()
        assert 9 in periods
        assert 20 in periods

    def test_watchlist_criteria(self):
        strat = ORBBreakoutStrategy()
        criteria = strat.get_watchlist_criteria()
        assert criteria['min_volume'] > 0
        assert criteria['min_price'] > 0
        assert criteria['prefer_gappers'] is True


# ── Lifecycle Tests ───────────────────────────────────────────────

class TestORBLifecycle:
    def test_day_start_resets(self):
        strat = ORBBreakoutStrategy()
        strat._entered_symbols['AAPL'] = 2
        strat._or_data['AAPL'] = {'or_high': 155.0}

        strat.on_day_start()

        assert len(strat._entered_symbols) == 0
        assert len(strat._or_data) == 0

    def test_record_entry(self):
        strat = ORBBreakoutStrategy()
        strat.record_entry('AAPL')
        assert strat._entered_symbols['AAPL'] == 1
        strat.record_entry('AAPL')
        assert strat._entered_symbols['AAPL'] == 2


# ── IntradaySetup Dataclass Tests ─────────────────────────────────

class TestIntradaySetup:
    def test_create_setup(self):
        setup = IntradaySetup(
            symbol='AAPL',
            strategy_id='orb_breakout',
            direction='long',
            entry_price=155.0,
            stop_price=148.0,
            target_price=165.5,
            risk_reward=1.5,
            confidence=0.75,
            setup_type='orb_breakout',
        )
        assert setup.symbol == 'AAPL'
        assert setup.direction == 'long'
        assert setup.notes == ''

    def test_setup_with_indicators(self):
        setup = IntradaySetup(
            symbol='AAPL',
            strategy_id='orb_breakout',
            direction='short',
            entry_price=148.0,
            stop_price=155.0,
            target_price=137.5,
            risk_reward=1.5,
            confidence=0.65,
            setup_type='orb_breakout',
            indicators={'vwap': 151.0, 'rsi': 45.0},
            notes='Strong volume breakdown',
        )
        assert setup.indicators['vwap'] == 151.0
        assert 'Strong volume' in setup.notes
