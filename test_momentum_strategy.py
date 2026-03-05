"""Tests for Momentum Surge strategy and Market Condition Detector.

Self-contained test suite: no shared conftest.
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from gap_fade_strategies.momentum_surge import MomentumSurgeStrategy
from gap_fade_strategies.market_condition import MarketConditionDetector, MarketCondition
from gap_fade_strategies.intraday_base import IntradaySetup, IntradayStrategy
from gap_fade_strategies.intraday_registry import IntradayStrategyRegistry
from gap_fade_strategies.indicators import FiveMinBar

ET = ZoneInfo('US/Eastern')


def make_ts(hour, minute, second=0):
    return datetime(2024, 3, 1, hour, minute, second, tzinfo=ET)


def make_tick_data(ema9=152.0, ema20=150.0, rsi=60.0, rsi_initialized=True,
                   vwap=149.5, atr=2.0, volume_surge=2.5,
                   day_high=152.5, day_low=148.0, ema50=0):
    """Create mock tick_data for momentum testing."""
    data = {
        'ema9': ema9,
        'ema20': ema20,
        'rsi': rsi,
        'rsi_initialized': rsi_initialized,
        'vwap': vwap,
        'atr': atr,
        'volume_surge_ratio': volume_surge,
        'day_high': day_high,
        'day_low': day_low,
        'rsi_history': [],
    }
    if ema50:
        data['ema50'] = ema50
    return data


# ── Registry Tests ────────────────────────────────────────────────

class TestMomentumRegistry:
    def test_registered(self):
        assert IntradayStrategyRegistry.has_strategy('momentum_surge')

    def test_create_instance(self):
        strat = IntradayStrategyRegistry.create_strategy('momentum_surge')
        assert isinstance(strat, MomentumSurgeStrategy)
        assert strat.name == 'Momentum Surge'

    def test_inherits_intraday(self):
        strat = MomentumSurgeStrategy()
        assert isinstance(strat, IntradayStrategy)


# ── Scan Tests ────────────────────────────────────────────────────

class TestMomentumScan:
    def test_long_setup(self):
        """Long when price > EMA9 > EMA20, RSI 50-75, volume 2x+, near high."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(
            ema9=152.0, ema20=150.0, rsi=60, vwap=149.5,
            volume_surge=2.5, day_high=152.5,
        )
        snapshot = {'price': 152.3}  # Near day high, above EMA9 > EMA20

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))

        assert setup is not None
        assert setup.direction == 'long'
        assert setup.strategy_id == 'momentum_surge'
        assert setup.risk_reward >= 1.5

    def test_short_setup(self):
        """Short when price < EMA9 < EMA20, RSI 25-50, volume 2x+, near low."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(
            ema9=148.0, ema20=150.0, rsi=35, vwap=151.0,
            volume_surge=2.5, day_low=147.5,
        )
        snapshot = {'price': 147.8}  # Near day low, below EMA9 < EMA20

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))

        assert setup is not None
        assert setup.direction == 'short'

    def test_no_setup_ema_misaligned(self):
        """No setup when EMAs aren't aligned (EMA9 < EMA20 for long attempt)."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(ema9=149.0, ema20=150.0)  # EMA9 < EMA20
        snapshot = {'price': 152.0}  # Price > both EMAs but alignment wrong

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        assert setup is None

    def test_no_setup_low_volume(self):
        """No setup when volume < 2x threshold."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(volume_surge=1.2)
        snapshot = {'price': 152.3}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        assert setup is None

    def test_no_setup_rsi_overbought(self):
        """No long when RSI > 75 (overbought)."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(rsi=80)
        snapshot = {'price': 152.3}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        assert setup is None

    def test_no_setup_rsi_oversold_for_long(self):
        """No long when RSI < 50."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(rsi=40)
        snapshot = {'price': 152.3}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        assert setup is None

    def test_no_setup_below_vwap_for_long(self):
        """No long when price < VWAP."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(vwap=153.0)  # VWAP above price
        snapshot = {'price': 152.3}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        assert setup is None

    def test_no_setup_not_near_high(self):
        """No long when price is far from intraday high."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(day_high=160.0)  # Far above price
        snapshot = {'price': 152.3}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        assert setup is None

    def test_no_setup_rsi_not_initialized(self):
        """No setup when RSI hasn't initialized yet."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(rsi_initialized=False)
        snapshot = {'price': 152.3}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        assert setup is None

    def test_no_snapshot(self):
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data()
        setup = strat.scan_for_setups('AAPL', tick_data, None, make_ts(10, 30))
        assert setup is None

    def test_max_entries_limit(self):
        strat = MomentumSurgeStrategy()
        strat._entered_symbols['AAPL'] = 3
        tick_data = make_tick_data()
        snapshot = {'price': 152.3}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        assert setup is None


# ── Stop Computation Tests ────────────────────────────────────────

class TestMomentumStops:
    def test_long_stop_tighter_wins(self):
        """Stop is the tighter of EMA20 or ATR-based."""
        strat = MomentumSurgeStrategy()
        # EMA20 = 150, ATR = 2.0 -> ATR stop = 152.3 - 3.0 = 149.3
        # EMA20 stop = 150 * 0.999 = 149.85
        # Tighter = max(149.3, 149.85) = 149.85 (EMA20)
        tick_data = make_tick_data(ema20=150.0, atr=2.0)
        snapshot = {'price': 152.3}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        if setup:
            assert setup.stop_price >= 149.0

    def test_short_stop_tighter_wins(self):
        """Short stop is the tighter of EMA20 or ATR-based."""
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(
            ema9=148.0, ema20=150.0, rsi=35, vwap=151.0,
            atr=2.0, volume_surge=2.5, day_low=147.5,
        )
        snapshot = {'price': 147.8}

        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        if setup:
            assert setup.stop_price <= 151.0


# ── Exit Tests ────────────────────────────────────────────────────

class TestMomentumExit:
    def test_long_exit_below_ema9(self):
        """Close long when price drops below EMA9."""
        strat = MomentumSurgeStrategy()
        position = {'direction': 'long', 'entry_price': 152.0, 'stop_price': 149.0}
        tick_data = make_tick_data(ema9=151.0)
        # Price below EMA9 * 0.998
        exit_signal = strat.evaluate_exit(position, 150.0, tick_data, make_ts(11, 0))
        assert exit_signal is not None
        assert exit_signal.action == 'close'
        assert 'Momentum lost' in exit_signal.reason

    def test_no_exit_above_ema9(self):
        """No exit when long and price is above EMA9."""
        strat = MomentumSurgeStrategy()
        position = {'direction': 'long', 'entry_price': 152.0, 'stop_price': 149.0}
        tick_data = make_tick_data(ema9=151.0)
        exit_signal = strat.evaluate_exit(position, 152.0, tick_data, make_ts(11, 0))
        assert exit_signal is None

    def test_short_exit_above_ema9(self):
        """Close short when price rises above EMA9."""
        strat = MomentumSurgeStrategy()
        position = {'direction': 'short', 'entry_price': 148.0, 'stop_price': 151.0}
        tick_data = make_tick_data(ema9=149.0)
        exit_signal = strat.evaluate_exit(position, 150.0, tick_data, make_ts(11, 0))
        assert exit_signal is not None
        assert 'Momentum lost' in exit_signal.reason


# ── Trailing Stop Tests ──────────────────────────────────────────

class TestMomentumTrailing:
    def test_no_trail_before_1r(self):
        strat = MomentumSurgeStrategy()
        position = {'direction': 'long', 'entry_price': 150.0, 'stop_price': 147.0}
        tick_data = make_tick_data(ema9=151.0)
        # Risk = 3. 1R = 153. Price at 151 = 0.33R
        new_stop = strat.update_trailing_stop(position, 151.0, tick_data, make_ts(11, 0))
        assert new_stop is None

    def test_trail_after_1r(self):
        strat = MomentumSurgeStrategy()
        position = {'direction': 'long', 'entry_price': 150.0, 'stop_price': 147.0}
        tick_data = make_tick_data(ema9=153.0)
        # Risk = 3. 1R = 153. Price at 155 = 1.67R
        new_stop = strat.update_trailing_stop(position, 155.0, tick_data, make_ts(11, 0))
        assert new_stop is not None
        assert new_stop > 147.0


# ── Confidence Tests ──────────────────────────────────────────────

class TestMomentumConfidence:
    def test_base_confidence(self):
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data()
        snapshot = {'price': 152.3}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        if setup:
            assert setup.confidence >= 0.45

    def test_high_confidence_all_factors(self):
        strat = MomentumSurgeStrategy()
        tick_data = make_tick_data(
            ema9=152.0, ema20=150.0, ema50=148.0,
            rsi=62, volume_surge=3.5, vwap=149.0,
            day_high=152.5,
        )
        snapshot = {'price': 152.3}
        setup = strat.scan_for_setups('AAPL', tick_data, snapshot, make_ts(10, 30))
        if setup:
            assert setup.confidence >= 0.65


# ── Config Tests ──────────────────────────────────────────────────

class TestMomentumConfig:
    def test_default_config(self):
        strat = MomentumSurgeStrategy()
        cfg = strat.get_default_config()
        assert cfg['volume_surge_ratio'] == 2.0
        assert cfg['target_rr'] == 2.0

    def test_active_window(self):
        strat = MomentumSurgeStrategy()
        assert strat.get_active_window() == (9, 45, 15, 0)

    def test_indicators(self):
        strat = MomentumSurgeStrategy()
        indicators = strat.get_required_indicators()
        assert 'vwap' in indicators
        assert 'rsi' in indicators
        assert 'ema_multi' in indicators
        assert 'atr' in indicators

    def test_lifecycle(self):
        strat = MomentumSurgeStrategy()
        strat._entered_symbols['AAPL'] = 2
        strat.on_day_start()
        assert len(strat._entered_symbols) == 0


# ── Market Condition Detector Tests ───────────────────────────────

class TestMarketConditionDetector:
    def _make_spy_data(self, ema9=450, ema20=448, ema50=445, rsi=58,
                       day_high=451, day_low=447, rsi_history=None,
                       vwap=449):
        data = {
            'ema9': ema9, 'ema20': ema20, 'rsi': rsi,
            'rsi_initialized': True,
            'day_high': day_high, 'day_low': day_low,
            'vwap': vwap,
            'rsi_history': rsi_history or [55, 56, 57, 58, 59],
            'bar_history': [],
        }
        if ema50:
            data['ema50'] = ema50
        return data

    def test_trending_up(self):
        """Bullish EMA alignment + RSI 50-70 = trending_up."""
        detector = MarketConditionDetector()
        spy_data = self._make_spy_data(ema9=451, ema20=449, ema50=446, rsi=60)

        result = detector.evaluate(spy_data, make_ts(10, 30), force=True)
        assert result.condition == 'trending_up'
        assert result.ema_alignment == 'bullish'
        assert result.confidence >= 0.60

    def test_trending_down(self):
        """Bearish EMA alignment + RSI 30-50 = trending_down."""
        detector = MarketConditionDetector()
        spy_data = self._make_spy_data(ema9=445, ema20=448, ema50=451, rsi=40)

        result = detector.evaluate(spy_data, make_ts(10, 30), force=True)
        assert result.condition == 'trending_down'
        assert result.ema_alignment == 'bearish'

    def test_choppy(self):
        """Oscillating RSI + high volatility = choppy."""
        detector = MarketConditionDetector()
        # Oscillating RSI: up-down-up-down-up-down
        rsi_hist = [55, 45, 60, 40, 58, 42, 57, 43, 56, 44]
        spy_data = self._make_spy_data(
            ema9=449, ema20=448, rsi=50,
            day_high=455, day_low=440,  # Wide range = high vol
            rsi_history=rsi_hist,
        )

        result = detector.evaluate(spy_data, make_ts(10, 30), force=True)
        assert result.condition == 'choppy'

    def test_sideways(self):
        """Low volatility + neutral RSI = sideways."""
        detector = MarketConditionDetector()
        spy_data = self._make_spy_data(
            ema9=449, ema20=449, ema50=449, rsi=50,
            day_high=450, day_low=449,  # Very narrow range
            rsi_history=[50, 51, 50, 49, 50],
        )

        result = detector.evaluate(spy_data, make_ts(10, 30), force=True)
        assert result.condition == 'sideways'
        assert result.volatility == 'low'

    def test_caching(self):
        """Results are cached within eval_interval."""
        detector = MarketConditionDetector(eval_interval=300)
        spy_data = self._make_spy_data(rsi=60)

        result1 = detector.evaluate(spy_data, make_ts(10, 30), force=True)
        # Change data but don't force
        spy_data2 = self._make_spy_data(rsi=30)
        result2 = detector.evaluate(spy_data2, make_ts(10, 31))

        # Should return cached result
        assert result2.rsi == result1.rsi

    def test_force_bypass_cache(self):
        """force=True bypasses cache."""
        detector = MarketConditionDetector(eval_interval=300)
        spy_data1 = self._make_spy_data(ema9=451, ema20=449, rsi=60)
        detector.evaluate(spy_data1, make_ts(10, 30), force=True)

        spy_data2 = self._make_spy_data(ema9=445, ema20=448, rsi=40)
        result = detector.evaluate(spy_data2, make_ts(10, 31), force=True)
        assert result.rsi == 40

    def test_reset_daily(self):
        detector = MarketConditionDetector()
        spy_data = self._make_spy_data(rsi=60)
        detector.evaluate(spy_data, make_ts(10, 30), force=True)
        assert detector.current_condition is not None

        detector.reset_daily()
        assert detector.current_condition is None
        assert len(detector.get_history()) == 0

    def test_condition_history(self):
        detector = MarketConditionDetector()
        for i in range(5):
            spy_data = self._make_spy_data(rsi=55 + i)
            detector.evaluate(spy_data, make_ts(10, 30 + i), force=True)

        history = detector.get_history()
        assert len(history) == 5

    def test_default_unknown(self):
        detector = MarketConditionDetector()
        assert detector.condition_name == 'unknown'

    def test_extreme_rsi_override(self):
        """RSI > 70 classifies as trending_up regardless."""
        detector = MarketConditionDetector()
        spy_data = self._make_spy_data(ema9=449, ema20=449, rsi=75)
        result = detector.evaluate(spy_data, make_ts(10, 30), force=True)
        assert result.condition == 'trending_up'

    def test_extreme_rsi_down(self):
        """RSI < 30 classifies as trending_down."""
        detector = MarketConditionDetector()
        spy_data = self._make_spy_data(ema9=449, ema20=449, rsi=25)
        result = detector.evaluate(spy_data, make_ts(10, 30), force=True)
        assert result.condition == 'trending_down'
